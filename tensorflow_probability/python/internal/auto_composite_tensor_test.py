# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for auto_composite_tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import test_util

tf.enable_v2_behavior()

flags.DEFINE_string(
    'model_output_path',
    None,
    'If defined, serialize a `tf.Module` instance to this directory with '
    '`tf.saved_model`.')

FLAGS = flags.FLAGS

TFP_PYTHON_DIR = 'tensorflow_probability/tensorflow_probability/python'

tfb = tfp.bijectors
tfd = tfp.distributions


AutoIdentity = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorIdentity, non_identifying_kwargs=('name',))
AutoDiag = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorDiag, non_identifying_kwargs=('name',))
AutoBlockDiag = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorBlockDiag, non_identifying_kwargs=('name',))
AutoTriL = tfp.experimental.auto_composite_tensor(
    tf.linalg.LinearOperatorLowerTriangular, non_identifying_kwargs=('name',))

AutoNormal = tfp.experimental.auto_composite_tensor(
    tfd.Normal, non_identifying_kwargs=('name',))
AutoIndependent = tfp.experimental.auto_composite_tensor(
    tfd.Independent, non_identifying_kwargs=('name',))
AutoReshape = tfp.experimental.auto_composite_tensor(
    tfb.Reshape, non_identifying_kwargs=('name',))


class Model(tf.Module):

  def __init__(self):
    self.scale = tf.Variable([0., 1.], shape=[None])

  @tf.function(input_signature=(
      tfb.Scale([1., 2.], validate_args=True)._type_spec,))
  def make_bij(self, b):
    return tfb.Scale(
        tf.convert_to_tensor(self.scale) + b.scale,
        validate_args=True)


@tfp.experimental.auto_composite_tensor
class ThingWithCallableArg(tfp.experimental.AutoCompositeTensor):

  def __init__(self, a, f):
    self.a = tf.convert_to_tensor(a, dtype_hint=tf.float32, name='a')
    self.f = f
    self.parameters = dict(a=self.a, b=self.f)

  def call(self):
    return self.f(self.a)


def tearDownModule():
  # If `FLAGS.model_output_path` is set, serialize a `Model` instance to disk.
  # To update the serialized data read by `test_saved_model_from_disk`, pass
  # the local path to
  # `tensorflow_probability/python/internal/testdata/auto_composite_tensor`.
  # You may need to pass `--test_strategy=local` to avoid permissions errors.
  if FLAGS.model_output_path is not None:
    model = Model()
    tf.saved_model.save(model, FLAGS.model_output_path)


@test_util.test_all_tf_execution_regimes
class AutoCompositeTensorTest(test_util.TestCase):

  def test_example(self):
    @tfp.experimental.auto_composite_tensor(non_identifying_kwargs=('name',))
    class Adder(object):

      def __init__(self, x, y, name=None):
        with tf.name_scope(name or 'Adder') as name:
          self._x = tensor_util.convert_nonref_to_tensor(x)
          self._y = tensor_util.convert_nonref_to_tensor(y)
          self._name = name

      def xpy(self):
        return self._x + self._y

    x = 1.
    y = tf.Variable(1.)
    self.evaluate(y.initializer)

    def body(obj):
      return Adder(obj.xpy(), y),

    result, = tf.while_loop(
        cond=lambda _: True,
        body=body,
        loop_vars=(Adder(x, y),),
        maximum_iterations=3)
    self.assertAllClose(5., result.xpy())

  def test_function(self):
    lop = AutoDiag(2. * tf.ones([3]))
    self.assertAllClose(
        6. * tf.ones([3]),
        tf.function(lambda lop: lop.matvec(3. * tf.ones([3])))(lop))

  def test_loop(self):
    def body(lop):
      return AutoDiag(lop.matvec(tf.ones([3]) * 2.)),
    init_lop = AutoDiag(tf.ones([3]))
    lop, = tf.while_loop(
        cond=lambda _: True,
        body=body,
        loop_vars=(init_lop,),
        maximum_iterations=3)
    self.assertAllClose(2.**3 * tf.ones([3]), lop.matvec(tf.ones([3])))

  def test_shape_parameters(self):
    dist = AutoIndependent(AutoNormal(0, tf.ones([1])),
                           reinterpreted_batch_ndims=1)
    stream = test_util.test_seed_stream()
    lp = dist.log_prob(dist.sample(seed=stream()))
    lp, _ = tf.while_loop(
        lambda *_: True,
        lambda lp, d: (d.log_prob(d.sample(seed=stream())), d),
        (lp, dist),
        maximum_iterations=2)
    self.evaluate(lp)

  def test_prefer_static_shape_params(self):
    @tf.function
    def f(b):
      return b
    b = AutoReshape(
        event_shape_out=[2, 3],
        event_shape_in=[tf.reduce_prod([2, 3])])  # Tensor in a list.
    f(b)

  def test_nested(self):
    lop = AutoBlockDiag([AutoDiag(tf.ones([2]) * 2), AutoIdentity(1)])
    self.assertAllClose(
        tf.constant([6., 6, 3]),
        tf.function(lambda lop: lop.matvec(3. * tf.ones([3])))(lop))

  def test_preconditioner(self):
    xs = self.evaluate(tf.random.uniform([30, 30], seed=test_util.test_seed()))
    cov_linop = tf.linalg.LinearOperatorFullMatrix(
        tf.matmul(xs, xs, transpose_b=True) + tf.linalg.eye(30) * 1e-3,
        is_self_adjoint=True,
        is_positive_definite=True)

    tfed = tfp.experimental.distributions
    auto_ct_mvn_prec_linop = tfp.experimental.auto_composite_tensor(
        tfed.MultivariateNormalPrecisionFactorLinearOperator,
        non_identifying_kwargs=('name',))
    tril = AutoTriL(**cov_linop.cholesky().parameters)
    momentum_distribution = auto_ct_mvn_prec_linop(precision_factor=tril)
    def body(d):
      return d.copy(precision_factor=AutoTriL(
          **dict(d.precision_factor.parameters,
                 tril=d.precision_factor.to_dense() + tf.linalg.eye(30),))),
    after_loop = tf.while_loop(lambda d: True, body, (momentum_distribution,),
                               maximum_iterations=1)
    tf.nest.map_structure(self.evaluate,
                          after_loop,
                          expand_composites=True)

  def test_already_ct_subclass(self):

    @tfp.experimental.auto_composite_tensor
    class MyCT(tfp.experimental.AutoCompositeTensor):

      def __init__(self, tensor_param, non_tensor_param, maybe_tensor_param):
        self._tensor_param = tf.convert_to_tensor(tensor_param)
        self._non_tensor_param = non_tensor_param
        self._maybe_tensor_param = maybe_tensor_param

    def body(obj):
      return MyCT(obj._tensor_param + 1,
                  obj._non_tensor_param,
                  obj._maybe_tensor_param),

    init = MyCT(0., 0, 0)
    result, = tf.while_loop(
        cond=lambda *_: True,
        body=body,
        loop_vars=(init,),
        maximum_iterations=3)
    self.assertAllClose(3., result._tensor_param)

    init = MyCT(0., 0, tf.constant(0))
    result, = tf.while_loop(
        cond=lambda *_: True,
        body=body,
        loop_vars=(init,),
        maximum_iterations=3)
    self.assertAllClose(3., result._tensor_param)

  def test_parameters_lookup(self):

    @tfp.experimental.auto_composite_tensor
    class ThingWithParametersButNoAttrs(tfp.experimental.AutoCompositeTensor):

      def __init__(self, a, b):
        self.a = tf.convert_to_tensor(a, dtype_hint=tf.float32, name='a')
        self.b = tf.convert_to_tensor(b, dtype_hint=tf.float32, name='a')
        self.parameters = dict(a=self.a, b=self.b)

    t = ThingWithParametersButNoAttrs(1., 2.)
    self.assertIsInstance(t, tf.__internal__.CompositeTensor)

    ts = t._type_spec
    components = ts._to_components(t)
    self.assertAllEqualNested(components, dict(a=1., b=2.))

    t2 = ts._from_components(components)
    self.assertIsInstance(t2, ThingWithParametersButNoAttrs)

  def test_wrapped_constructor(self):
    def add_tag(f):
      @functools.wraps(f)
      def wrapper(*args, **kwargs):
        args[0]._tag = 'tagged'
        return f(*args, **kwargs)
      return wrapper

    @tfp.experimental.auto_composite_tensor
    class ThingWithWrappedInit(tfp.experimental.AutoCompositeTensor):

      @add_tag
      def __init__(self, value):
        self.value = tf.convert_to_tensor(value)

    init = ThingWithWrappedInit(3)
    def body(obj):
      return ThingWithWrappedInit(value=obj.value + 1),

    out, = tf.while_loop(
        cond=lambda *_: True,
        body=body,
        loop_vars=(init,),
        maximum_iterations=3)
    self.assertEqual(self.evaluate(out.value), 6)

  def test_deferred_assertion_context(self):
    # If `validate_args` assertions in `__init__` are not deferred, a graph
    # cycle is created when `d._type_spec` calls `__init__` and this test fails.
    d = AutoNormal(0., 1., validate_args=True)

    @tf.function
    def f(d):
      return d

    f(d)

  def test_function_with_variable(self):
    loc = tf.Variable(3.)
    dist = AutoIndependent(
        AutoNormal(loc, scale=tf.ones([3])), reinterpreted_batch_ndims=1)

    new_loc = 32.
    @tf.function
    def f(d):
      d.distribution.loc.assign(new_loc)
      self.assertLen(d.trainable_variables, 1)
      return d

    dist_ = f(dist)
    self.evaluate(loc.initializer)
    self.assertEqual(self.evaluate(dist_.distribution.loc), new_loc)
    self.assertEqual(self.evaluate(dist.distribution.loc), new_loc)
    self.assertLen(dist.trainable_variables, 1)

  def test_export_import(self):
    path = self.create_tempdir().full_path

    m1 = Model()
    self.evaluate([v.initializer for v in m1.variables])
    self.evaluate(m1.scale.assign(m1.scale + 1.))

    tf.saved_model.save(m1, os.path.join(path, 'saved_model1'))
    m2 = tf.saved_model.load(os.path.join(path, 'saved_model1'))
    self.evaluate(m2.scale.initializer)
    b = tfb.Scale([5., 9.], validate_args=True)
    self.evaluate(m2.make_bij(b).forward(2.))
    self.evaluate(m2.scale.assign(m2.scale + [1., 2.]))
    self.evaluate(m2.make_bij(b).forward(2.))

    self.evaluate(m2.scale.assign([1., 2., 3.]))
    tf.saved_model.save(m2, os.path.join(path, 'saved_model2'))
    m3 = tf.saved_model.load(os.path.join(path, 'saved_model2'))
    self.evaluate(m3.scale.initializer)
    with self.assertRaisesOpError('compatible shape'):
      self.evaluate(m3.make_bij(b).forward([3.]))

  def test_saved_model_from_disk(self):

    test_srcdir = absltest.get_default_test_srcdir()
    relative_testdata_path = os.path.join(
        TFP_PYTHON_DIR, 'internal/testdata/auto_composite_tensor')
    absolute_testdata_path = os.path.join(test_srcdir, relative_testdata_path)

    m = tf.saved_model.load(absolute_testdata_path)
    self.evaluate(m.scale.initializer)
    b = tfb.Scale([5., 9.], validate_args=True)
    self.assertAllClose(self.evaluate(m.make_bij(b).forward(2.)), [10., 20.])
    self.evaluate(m.scale.assign(m.scale + [1., 2.]))
    self.assertAllClose(self.evaluate(m.make_bij(b).forward(2.)), [12., 24.])

  def test_callable_arg(self):

    t = ThingWithCallableArg(1., lambda x: x + 2.)
    self.assertIsInstance(t, tf.__internal__.CompositeTensor)

    ts = t._type_spec
    components = ts._to_components(t)
    self.assertAllEqualNested(components, dict(a=1.))

    t2 = ts._from_components(components)
    self.assertIsInstance(t2, ThingWithCallableArg)

    self.assertAllClose(tf.function(lambda t: t.call())(t2), 3.)

  def test_different_names_type_specs_equal(self):

    dist_1 = AutoNormal([0., 2.], scale=1., name='FirstNormal')
    dist_2 = AutoNormal([1., 3.], scale=2., name='SecondNormal')
    self.assertEqual(dist_1._type_spec, dist_2._type_spec)

  def test_save_restore_functor(self):

    f = lambda x: x ** 2
    a = tf.constant([3., 2.])
    ct = ThingWithCallableArg(a, f=f)

    struct_coder = tf.__internal__.saved_model.StructureCoder()
    with self.assertRaisesRegex(ValueError, 'Cannot serialize'):
      struct_coder.encode_structure(ct._type_spec)  # pylint: disable=protected-access

    @tfp.experimental.auto_composite_tensor(module_name='my.module')
    class F(tfp.experimental.AutoCompositeTensor):

      def __call__(self, *args, **kwargs):
        return f(*args, **kwargs)

    ct_functor = ThingWithCallableArg(a, f=F())
    enc = struct_coder.encode_structure(ct_functor._type_spec)
    dec = struct_coder.decode_proto(enc)
    self.assertEqual(dec, ct_functor._type_spec)

  def test_composite_tensor_callable_arg(self):
    # Parameters that are both `CompositeTensor` and callable should be
    # handled by the `_type_spec` as `CompositeTensor`.
    inner_bij = tfb.Scale([[1., 3.]], validate_args=True)
    bij = tfb.TransformDiagonal(inner_bij, validate_args=True)
    self.assertLen(tf.nest.flatten(bij), 1)
    self.assertLen(bij._type_spec._callable_params, 0)  # pylint: disable=protected-access
    self.assertIn('diag_bijector', bij._type_spec._param_specs)  # pylint: disable=protected-access

  def test_subclass_with_inherited_type_spec_raises(self):

    @tfp.experimental.auto_composite_tensor(
        omit_kwargs=('parameters',), non_identifying_kwargs=('name',))
    class ParentBijector(
        tfb.Bijector, tfp.experimental.AutoCompositeTensor):
      """Minimal specification of a `Bijector`.

      We do not subclass `AutoCompositeTensorBijector` since its metaclass
      would make subclasses automatically re-generate their `TypeSpec`.
      """

      def __init__(self, a):
        parameters = dict(locals())
        self.a = a
        super(ParentBijector, self).__init__(
            forward_min_event_ndims=0,
            parameters=parameters)

    class ChildBijector(ParentBijector):

      def __init__(self, b):
        self.b = b
        super(ChildBijector, self).__init__(a=b+1)

    b = ChildBijector(b=4)
    with self.assertRaisesRegex(
        ValueError,
        '`ChildBijector` has inherited the `_type_spec` of `ParentBijector`'):
      tf.nest.flatten(b, expand_composites=True)

    AutoChildBijector = tfp.experimental.auto_composite_tensor(ChildBijector)  # pylint: disable=invalid-name
    b_ct = AutoChildBijector(b=2)
    self.assertLen(tf.nest.flatten(b_ct, expand_composites=True), 0)

  def test_names_preserved_through_flatten(self):

    dist = AutoNormal(0., scale=3., name='ScaleThreeNormal')
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    unflat_name = ('ScaleThreeNormal' if tf.executing_eagerly()
                   else 'ScaleThreeNormal_1')
    self.assertEqual(unflat.name, unflat_name)


class _TestTypeSpec(auto_composite_tensor._AutoCompositeTensorTypeSpec):

  def __init__(self, param_specs, non_tensor_params=None, omit_kwargs=(),
               prefer_static_value=(), non_identifying_kwargs=(),
               callable_params=None):
    non_tensor_params = {} if non_tensor_params is None else non_tensor_params
    super(_TestTypeSpec, self).__init__(
        param_specs, non_tensor_params=non_tensor_params,
        omit_kwargs=omit_kwargs, prefer_static_value=prefer_static_value,
        non_identifying_kwargs=non_identifying_kwargs,
        callable_params=callable_params)

  @property
  def value_type(self):
    """Unused `value_type` to allow the `TypeSpec` to be instantiated."""
    pass


@test_util.test_all_tf_execution_regimes
class AutoCompositeTensorTypeSpecTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('WithoutCallable',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32)},
           non_tensor_params={'validate_args': True},
           omit_kwargs=('name',),
           prefer_static_value=('a',)),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32)},
           non_tensor_params={'validate_args': True},
           omit_kwargs=('name',),
           prefer_static_value=('a',))),
      ('WithCallable',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32),
                        'b': tfb.Scale(3.)._type_spec},
           omit_kwargs=('name', 'foo'),
           prefer_static_value=('a',),
           callable_params={'f': tf.math.exp}),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32),
                        'b': tfb.Scale(3.)._type_spec},
           omit_kwargs=('name', 'foo'),
           prefer_static_value=('a',),
           callable_params={'f': tf.math.exp})),
      ('DifferentNonIdentifyingKwargsValues',
       _TestTypeSpec(
           param_specs={'x': tf.TensorSpec([], tf.float64)},
           non_tensor_params={'name': 'MyAutoCT'},
           non_identifying_kwargs=('name')),
       _TestTypeSpec(
           param_specs={'x': tf.TensorSpec([], tf.float64)},
           non_tensor_params={'name': 'OtherAutoCT'},
           non_identifying_kwargs=('name'))),
      )
  def testEquality(self, v1, v2):
    # pylint: disable=g-generic-assert
    self.assertEqual(v1, v2)
    self.assertEqual(v2, v1)
    self.assertFalse(v1 != v2)
    self.assertFalse(v2 != v1)
    self.assertEqual(hash(v1), hash(v2))

  @parameterized.named_parameters(
      ('DifferentTensorSpecs',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, 2], tf.float32)},
           non_tensor_params={'validate_args': True},
           omit_kwargs=('name',)),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32)},
           non_tensor_params={'validate_args': True},
           omit_kwargs=('name',))),
      ('DifferentCallables',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32)},
           omit_kwargs=('name', 'foo'),
           callable_params={'f': tf.math.exp}),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32)},
           omit_kwargs=('name', 'foo'),
           callable_params={'f': tf.math.sigmoid})),
      ('DifferentMetadata',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, 2], tf.float32)},
           non_tensor_params={'validate_args': True},
           non_identifying_kwargs=('name',)),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32)},
           non_tensor_params={'validate_args': True})),
      )
  def testInequality(self, v1, v2):
    # pylint: disable=g-generic-assert
    self.assertNotEqual(v1, v2)
    self.assertNotEqual(v2, v1)
    self.assertFalse(v1 == v2)
    self.assertFalse(v2 == v1)

  @parameterized.named_parameters(
      ('WithoutCallable',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, 2], tf.float32)},
           non_tensor_params={'validate_args': True, 'b': 3.},
           omit_kwargs=('name',),
           prefer_static_value=('b',)),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, None], tf.float32)},
           non_tensor_params={'validate_args': True, 'b': 3.},
           omit_kwargs=('name',),
           prefer_static_value=('b',))),
      ('WithCallable',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32),
                        'b': tfb.Scale(
                            tf.Variable(2., shape=None))._type_spec},
           omit_kwargs=('name', 'foo'),
           callable_params={'f': tf.math.exp}),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32),
                        'b': tfb.Scale(3.)._type_spec},
           omit_kwargs=('name', 'foo'),
           callable_params={'f': tf.math.exp})),
      ('DifferentNonIdentifyingKwargsValues',
       _TestTypeSpec(
           param_specs={'x': tf.TensorSpec(None, tf.float64)},
           non_tensor_params={'name': 'MyAutoCT'},
           non_identifying_kwargs=('name')),
       _TestTypeSpec(
           param_specs={'x': tf.TensorSpec([], tf.float64)},
           non_tensor_params={'name': 'OtherAutoCT'},
           non_identifying_kwargs=('name'))),
      )
  def testIsCompatibleWith(self, v1, v2):
    self.assertTrue(v1.is_compatible_with(v2))
    self.assertTrue(v2.is_compatible_with(v1))

  @parameterized.named_parameters(
      ('IncompatibleTensorSpecs',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, 2, 3], tf.float32)},
           non_tensor_params={'validate_args': True, 'b': [3, 2]},
           omit_kwargs=('name',),
           prefer_static_value=('b',)),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, None], tf.float32)},
           non_tensor_params={'validate_args': True, 'b': [3, 2]},
           omit_kwargs=('name',),
           prefer_static_value=('b',))),
      ('DifferentMetadataSameCallables',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, 2], tf.float32)},
           non_tensor_params={'validate_args': True},
           omit_kwargs=('name',),
           callable_params={'g': tf.math.softplus}),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, None], tf.float32)},
           non_tensor_params={'validate_args': False},
           omit_kwargs=('name',),
           callable_params={'g': tf.math.softplus})),
      ('DifferentCallables',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32),
                        'b': tfb.Scale(
                            tf.Variable(2., shape=None))._type_spec},
           omit_kwargs=('name', 'foo'),
           callable_params={'f': tf.math.exp}),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32),
                        'b': tfb.Scale(3.)._type_spec},
           omit_kwargs=('name', 'foo'),
           callable_params={'f': tf.math.sigmoid}))
      )
  def testIsNotCompatibleWith(self, v1, v2):
    self.assertFalse(v1.is_compatible_with(v2))
    self.assertFalse(v2.is_compatible_with(v1))

  @parameterized.named_parameters(
      ('WithoutCallable',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, 2], tf.float32)},
           omit_kwargs=('name',)),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, None], tf.float32)},
           omit_kwargs=('name',)),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, None], tf.float32)},
           omit_kwargs=('name',))),
      ('WithCallable',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec(None, tf.float32),
                        'b': tfb.Scale(
                            tf.Variable(2., shape=None))._type_spec},
           callable_params={'f': tf.math.exp}),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32),
                        'b': tfb.Scale(tf.Variable(3.))._type_spec},
           callable_params={'f': tf.math.exp}),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec(None, tf.float32),
                        'b': tfb.Scale(
                            tf.Variable(2., shape=None))._type_spec},
           callable_params={'f': tf.math.exp})),
      )
  def testMostSpecificCompatibleType(self, v1, v2, expected):
    self.assertEqual(v1.most_specific_compatible_type(v2), expected)
    self.assertEqual(v2.most_specific_compatible_type(v1), expected)

  @parameterized.named_parameters(
      ('DifferentParamSpecs',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, 2], tf.float32)},
           omit_kwargs=('foo',)),
       _TestTypeSpec(
           param_specs={'b': tf.TensorSpec([5, None], tf.float32)},
           omit_kwargs=('foo',))),
      ('DifferentMetadata',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, 2], tf.float32)},
           omit_kwargs=('foo',)),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, None], tf.float32)},
           omit_kwargs=('bar',))),
      ('DifferentCallables',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec(None, tf.float32),
                        'b': tfb.Scale(
                            tf.Variable(2., shape=None))._type_spec},
           callable_params={'f': tf.math.exp}),
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([3, None], tf.float32),
                        'b': tfb.Scale(tf.Variable(3.))._type_spec},
           callable_params={'f': tf.math.softplus})),
      )
  def testMostSpecificCompatibleTypeException(self, v1, v2):
    with self.assertRaises(ValueError):
      v1.most_specific_compatible_type(v2)
    with self.assertRaises(ValueError):
      v2.most_specific_compatible_type(v1)

  @parameterized.named_parameters(
      ('WithoutCallable',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec([4, 2], tf.float32)},
           omit_kwargs=('parameters',), non_identifying_kwargs=('name',))),
      ('WithCallable',
       _TestTypeSpec(
           param_specs={'a': tf.TensorSpec(None, tf.float32),
                        'b': tfb.Scale(
                            tf.Variable(2., shape=None))._type_spec},
           callable_params={'f': tf.math.exp})),
      )
  def testRepr(self, spec):
    spec_data = (auto_composite_tensor._AUTO_COMPOSITE_TENSOR_VERSION,
                 spec._param_specs, spec._non_tensor_params, spec._omit_kwargs,
                 spec._prefer_static_value, spec._non_identifying_kwargs,
                 spec._callable_params)
    self.assertEqual(repr(spec), f'_TestTypeSpec{spec_data}')

if __name__ == '__main__':
  tf.test.main()
