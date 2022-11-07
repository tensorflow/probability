# Copyright 2018 The TensorFlow Probability Authors.
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
"""Tests for Bijector."""

# Dependency imports
from absl.testing import parameterized
import mock
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import chain as chain_lib
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import matrix_inverse_tril
from tensorflow_probability.python.bijectors import power
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import restructure
from tensorflow_probability.python.bijectors import scale as scale_lib
from tensorflow_probability.python.bijectors import scale_matvec_diag
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_lib
from tensorflow_probability.python.bijectors import sinh_arcsinh
from tensorflow_probability.python.bijectors import softmax_centered
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.bijectors import split
from tensorflow_probability.python.internal import batch_shape_lib
from tensorflow_probability.python.internal import cache_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import test_util


JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class BaseBijectorTest(test_util.TestCase):
  """Tests properties of the Bijector base-class."""

  def testIsAbstract(self):
    with self.assertRaisesRegexp(TypeError,
                                 ('Can\'t instantiate abstract class Bijector '
                                  'with abstract methods? __init__')):
      bijector_lib.Bijector()  # pylint: disable=abstract-class-instantiated

  def testDefaults(self):

    class _BareBonesBijector(bijector_lib.Bijector):
      """Minimal specification of a `Bijector`."""

      def __init__(self):
        parameters = dict(locals())
        super(_BareBonesBijector, self).__init__(
            forward_min_event_ndims=0,
            parameters=parameters)

    bij = _BareBonesBijector()
    self.assertFalse(bij.is_constant_jacobian)
    self.assertFalse(bij.validate_args)
    self.assertIsNone(bij.dtype)
    self.assertStartsWith(bij.name, 'bare_bones_bijector')

    for shape in [[], [1, 2], [1, 2, 3]]:
      forward_event_shape_ = self.evaluate(
          bij.inverse_event_shape_tensor(shape))
      inverse_event_shape_ = self.evaluate(
          bij.forward_event_shape_tensor(shape))
      self.assertAllEqual(shape, forward_event_shape_)
      self.assertAllEqual(shape, bij.forward_event_shape(shape))
      self.assertAllEqual(shape, inverse_event_shape_)
      self.assertAllEqual(shape, bij.inverse_event_shape(shape))

    with self.assertRaisesRegexp(NotImplementedError,
                                 'inverse not implemented'):
      bij.inverse(0)

    with self.assertRaisesRegexp(NotImplementedError,
                                 'forward not implemented'):
      bij.forward(0)

    with self.assertRaisesRegexp(
        NotImplementedError,
        'Cannot derive `inverse_log_det_jacobian`'):
      bij.inverse_log_det_jacobian(0, event_ndims=0)

    with self.assertRaisesRegexp(
        NotImplementedError,
        'Cannot derive `forward_log_det_jacobian`'):
      bij.forward_log_det_jacobian(0, event_ndims=0)

  def testVariableEq(self):
    # Testing for b/186021261, bijector equality in the presence of TF
    # Variables.
    v1 = tf.Variable(3, dtype=tf.float32)
    v2 = tf.Variable(4, dtype=tf.float32)
    self.assertNotEqual(
        sinh_arcsinh.SinhArcsinh(v1),
        sinh_arcsinh.SinhArcsinh(v2))

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='`convert_to_tensor` casts instead of raising')
  def testChecksDType(self):

    class _TypedIdentity(bijector_lib.Bijector):
      """Bijector with an explicit dtype."""

      def __init__(self, dtype):
        parameters = dict(locals())
        super(_TypedIdentity, self).__init__(
            forward_min_event_ndims=0,
            dtype=dtype,
            parameters=parameters)

      def _forward(self, x):
        return x

    x32 = tf.constant(0, dtype=tf.float32)
    x64 = tf.constant(0, dtype=tf.float64)
    error_clazz = TypeError if JAX_MODE else ValueError

    b32 = _TypedIdentity(tf.float32)
    self.assertEqual(tf.float32, b32(0).dtype)
    self.assertEqual(tf.float32, b32(x32).dtype)
    with self.assertRaisesRegexp(
        error_clazz, 'Tensor conversion requested dtype'):
      b32.forward(x64)

    b64 = _TypedIdentity(tf.float64)
    self.assertEqual(tf.float64, b64(0).dtype)
    self.assertEqual(tf.float64, b64(x64).dtype)
    with self.assertRaisesRegexp(
        error_clazz, 'Tensor conversion requested dtype'):
      b64.forward(x32)

  @parameterized.named_parameters(
      ('no_batch_shape', 1.4),
      ('with_batch_shape', [[[2., 3.], [5., 7.]]]))
  @test_util.numpy_disable_gradient_test
  def testAutodiffLogDetJacobian(self, bijector_scale):

    class NoJacobianBijector(bijector_lib.Bijector):
      """Bijector with no log det jacobian methods."""

      def __init__(self, scale=2.):
        parameters = dict(locals())
        self._scale = tensor_util.convert_nonref_to_tensor(scale)
        super(NoJacobianBijector, self).__init__(
            validate_args=True,
            forward_min_event_ndims=0,
            parameters=parameters)

      def _forward(self, x):
        return tf.exp(self._scale * x)

      def _inverse(self, y):
        return tf.math.log(y) / self._scale

      @classmethod
      def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            scale=parameter_properties.ParameterProperties(event_ndims=0))

    b = NoJacobianBijector(scale=bijector_scale)
    x = tf.convert_to_tensor([2., -3.])
    [
        fldj,
        true_fldj,
        ildj
    ] = self.evaluate([
        b.forward_log_det_jacobian(x, event_ndims=0),
        tf.math.log(b._scale) + b._scale * x,
        b.inverse_log_det_jacobian(b.forward(x), event_ndims=0)
    ])
    self.assertAllClose(fldj, true_fldj)
    self.assertAllClose(fldj, -ildj)

    y = tf.convert_to_tensor([27., 5.])
    [
        ildj,
        true_ildj,
        fldj
    ] = self.evaluate([
        b.inverse_log_det_jacobian(y, event_ndims=0),
        -tf.math.log(tf.abs(y * b._scale)),
        b.forward_log_det_jacobian(b.inverse(y), event_ndims=0)
    ])
    self.assertAllClose(ildj, true_ildj)
    self.assertAllClose(ildj, -fldj)

  def testCopyExtraArgs(self):
    # Note: we cannot easily test all bijectors since each requires
    # different initialization arguments. We therefore spot test a few.
    sigmoid = sigmoid_lib.Sigmoid(low=-1., high=2., validate_args=True)
    self.assertEqual(sigmoid.parameters, sigmoid.copy().parameters)
    chain = chain_lib.Chain([
        softplus.Softplus(hinge_softness=[1., 2.], validate_args=True),
        matrix_inverse_tril.MatrixInverseTriL(validate_args=True)
    ], validate_args=True)
    self.assertEqual(chain.parameters, chain.copy().parameters)

  def testCopyOverride(self):
    sigmoid = sigmoid_lib.Sigmoid(low=-1., high=2., validate_args=True)
    self.assertEqual(sigmoid.parameters, sigmoid.copy().parameters)
    unused_sigmoid_copy = sigmoid.copy(validate_args=False)
    base_params = sigmoid.parameters.copy()
    copy_params = sigmoid.copy(validate_args=False).parameters.copy()
    self.assertNotEqual(
        base_params.pop('validate_args'), copy_params.pop('validate_args'))
    self.assertEqual(base_params, copy_params)

  def testNameScopeRefersToInitialScope(self):
    if tf.executing_eagerly():
      self.skipTest('Eager mode.')

    outer_bijector = exp.Exp(name='Exponential')
    self.assertStartsWith(outer_bijector.name, 'Exponential')

    with tf.name_scope('inside'):
      inner_bijector = exp.Exp(name='Exponential')
      self.assertStartsWith(inner_bijector.name, 'Exponential')

      self.assertStartsWith(inner_bijector.forward(0., name='x').name,
                            'inside/Exponential/x')
      self.assertStartsWith(outer_bijector.forward(0., name='x').name,
                            'inside/Exponential_CONSTRUCTED_AT_top_level/x')

      meta_bijector = chain_lib.Chain([inner_bijector], name='meta_bijector')
      # Check for spurious `_CONSTRUCTED_AT_`.
      self.assertStartsWith(
          meta_bijector.forward(0., name='x').name,
          'inside/meta_bijector/x/Exponential/forward')

    # Outside the scope.
    self.assertStartsWith(inner_bijector.forward(0., name='x').name,
                          'Exponential_CONSTRUCTED_AT_inside/x')
    self.assertStartsWith(outer_bijector.forward(0., name='x').name,
                          'Exponential/x')
    # Check that init scope is annotated only for the toplevel bijector.
    self.assertStartsWith(
        meta_bijector.forward(0., name='x').name,
        'meta_bijector_CONSTRUCTED_AT_inside/x/Exponential/forward')


@test_util.test_graph_and_eager_modes
class BijectorStringReprTest(test_util.TestCase):

  def _tensor(self, x, dynamic=True):
    if dynamic:
      return tf.Variable(x, shape=tf.TensorShape(None))
    return tf.convert_to_tensor(x)

  def test_single_part_str_repr_match_expected(self):
    bij = exp.Exp()
    self.assertContainsInOrder(
        ['tfp.bijectors.Exp("exp", batch_shape=[], min_event_ndims=0)'],
        str(bij))
    self.assertContainsInOrder(
        ["<tfp.bijectors.Exp 'exp' batch_shape=[] forward_min_event_ndims=0 "
         "inverse_min_event_ndims=0 dtype_x=? dtype_y=?>"],
        repr(bij))

    bij = scale_lib.Scale([1., 1.], name='myscale')
    self.assertContainsInOrder(
        ['tfp.bijectors.Scale("myscale", batch_shape=[2], min_event_ndims=0, '
         'dtype=float32)'],
        str(bij))
    self.assertContainsInOrder(
        ["<tfp.bijectors.Scale 'myscale' batch_shape=[2] "
         "forward_min_event_ndims=0 inverse_min_event_ndims=0 dtype_x=float32 "
         "dtype_y=float32>"],
        repr(bij))

    bij = split.Split([3, 4, 2], name='s_p_l_i_t')
    self.assertContainsInOrder(
        ['tfp.bijectors.Split("s_p_l_i_t", batch_shape=[], '
         'forward_min_event_ndims=1, inverse_min_event_ndims=[1, 1, 1])'],
        str(bij))
    self.assertContainsInOrder(
        ["<tfp.bijectors.Split 's_p_l_i_t' batch_shape=[] "
         "forward_min_event_ndims=1 inverse_min_event_ndims=[1, 1, 1] "
         "dtype_x=? dtype_y=[?, ?, ?]>"
         ], repr(bij))

  @test_util.jax_disable_test_missing_functionality('dynamic shape')
  @test_util.numpy_disable_test_missing_functionality('dynamic shape')
  def test_single_part_str_repr_match_expected_dynamic_shape(self):
    bij = scale_lib.Scale(self._tensor([1., 1.]), name='dynamic_shape_scale')
    self.assertContainsInOrder(
        ['tfp.bijectors.Scale("dynamic_shape_scale", min_event_ndims=0, '
         'dtype=float32)'],
        str(bij))
    self.assertContainsInOrder(
        ["<tfp.bijectors.Scale 'dynamic_shape_scale' batch_shape=? "
         "forward_min_event_ndims=0 inverse_min_event_ndims=0 dtype_x=float32 "
         "dtype_y=float32>"],
        repr(bij))

  def test_invert_str_and_repr_match_expected(self):
    bij = invert.Invert(split.Split([3, 4, 2]))
    self.assertContainsInOrder(
        ['tfp.bijectors.Invert("invert_split", batch_shape=[], '
         'forward_min_event_ndims=[1, 1, 1], inverse_min_event_ndims=1, '
         'bijector=Split)'],
        str(bij))
    self.assertContainsInOrder(
        ["<tfp.bijectors.Invert 'invert_split' batch_shape=[] "
         "forward_min_event_ndims=[1, 1, 1] inverse_min_event_ndims=1 "
         "dtype_x=[?, ?, ?] dtype_y=? "
         "bijector=<tfp.bijectors.Split 'split' batch_shape=[] "
         "forward_min_event_ndims=1 inverse_min_event_ndims=[1, 1, 1] "
         "dtype_x=? dtype_y=[?, ?, ?]>>"
         ],
        repr(bij))

  def test_composition_str_and_repr_match_expected(self):
    bij = chain_lib.Chain(
        [exp.Exp(),
         shift.Shift([1., 2.]),
         softmax_centered.SoftmaxCentered()])
    self.assertContainsInOrder(
        ['tfp.bijectors.Chain(',
         ('batch_shape=[], min_event_ndims=1, '
          'bijectors=[Exp, Shift, SoftmaxCentered])')],
        str(bij))
    self.assertContainsInOrder(
        ['<tfp.bijectors.Chain ',
         ('batch_shape=[] forward_min_event_ndims=1 inverse_min_event_ndims=1 '
          'dtype_x=float32 dtype_y=float32 bijectors=[<tfp.bijectors.Exp'),
         '>, <tfp.bijectors.Shift',
         '>, <tfp.bijectors.SoftmaxCentered',
         '>]>'],
        repr(bij))

    bij = chain_lib.Chain([
        joint_map.JointMap({
            'a': exp.Exp(),
            'b': scale_matvec_diag.ScaleMatvecDiag([2., 2.])
        }),
        restructure.Restructure({
            'a': 0,
            'b': 1
        }, [0, 1]),
        split.Split(2),
        invert.Invert(softmax_centered.SoftmaxCentered()),
    ])
    self.assertContainsInOrder(
        ['tfp.bijectors.Chain(',
         ('batch_shape=[], forward_min_event_ndims=1, '
          'inverse_min_event_ndims={a: 1, b: 1}, '
          'bijectors=[JointMap({a: Exp, b: ScaleMatvecDiag}), '
          'Restructure, Split, Invert(SoftmaxCentered)])')],
        str(bij))
    self.assertContainsInOrder(
        ['<tfp.bijectors.Chain ',
         ('batch_shape=[] forward_min_event_ndims=1 '
          "inverse_min_event_ndims={'a': 1, 'b': 1} dtype_x=float32 "
          "dtype_y={'a': ?, 'b': float32} "
          "bijectors=[<tfp.bijectors.JointMap "),
         '>, <tfp.bijectors.Restructure',
         '>, <tfp.bijectors.Split',
         '>, <tfp.bijectors.Invert',
         '>]>'],
        repr(bij))

  @test_util.jax_disable_test_missing_functionality('dynamic shape')
  @test_util.numpy_disable_test_missing_functionality('dynamic shape')
  def test_composition_str_and_repr_match_expected_dynamic_shape(self):
    bij = chain_lib.Chain([
        exp.Exp(),
        shift.Shift(self._tensor([1., 2.])),
        softmax_centered.SoftmaxCentered()
    ])
    self.assertContainsInOrder(
        ['tfp.bijectors.Chain(',
         ('min_event_ndims=1, bijectors=[Exp, Shift, SoftmaxCentered])')],
        str(bij))
    self.assertContainsInOrder(
        ['<tfp.bijectors.Chain ',
         ('batch_shape=? forward_min_event_ndims=1 inverse_min_event_ndims=1 '
          'dtype_x=float32 dtype_y=float32 bijectors=[<tfp.bijectors.Exp'),
         '>, <tfp.bijectors.Shift',
         '>, <tfp.bijectors.SoftmaxCentered',
         '>]>'],
        repr(bij))

    bij = chain_lib.Chain([
        joint_map.JointMap({
            'a': exp.Exp(),
            'b': scale_matvec_diag.ScaleMatvecDiag(self._tensor([2., 2.]))
        }),
        restructure.Restructure({
            'a': 0,
            'b': 1
        }, [0, 1]),
        split.Split(2),
        invert.Invert(softmax_centered.SoftmaxCentered()),
    ])
    self.assertContainsInOrder(
        ['tfp.bijectors.Chain(',
         ('forward_min_event_ndims=1, '
          'inverse_min_event_ndims={a: 1, b: 1}, '
          'bijectors=[JointMap({a: Exp, b: ScaleMatvecDiag}), '
          'Restructure, Split, Invert(SoftmaxCentered)])')],
        str(bij))
    self.assertContainsInOrder(
        ['<tfp.bijectors.Chain ',
         ('batch_shape=? forward_min_event_ndims=1 '
          "inverse_min_event_ndims={'a': 1, 'b': 1} dtype_x=float32 "
          "dtype_y={'a': ?, 'b': float32} "
          "bijectors=[<tfp.bijectors.JointMap "),
         '>, <tfp.bijectors.Restructure',
         '>, <tfp.bijectors.Split',
         '>, <tfp.bijectors.Invert',
         '>]>'],
        repr(bij))


class IntentionallyMissingError(Exception):
  pass


class ForwardOnlyBijector(bijector_lib.Bijector):
  """Bijector with no inverse methods at all."""

  def __init__(self, scale=2., validate_args=False, name=None):
    parameters = dict(locals())
    with tf.name_scope(name or 'forward_only') as name:
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale,
          dtype_hint=tf.float32)
      super(ForwardOnlyBijector, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          parameters=parameters,
          name=name)

  def _forward(self, x):
    return self._scale * x

  def _forward_log_det_jacobian(self, _):
    return tf.math.log(self._scale)


class InverseOnlyBijector(bijector_lib.Bijector):
  """Bijector with no forward methods at all."""

  def __init__(self, scale=2., validate_args=False, name=None):
    parameters = dict(locals())
    with tf.name_scope(name or 'inverse_only') as name:
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale,
          dtype_hint=tf.float32)
      super(InverseOnlyBijector, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          parameters=parameters,
          name=name)

  def _inverse(self, y):
    return y / self._scale

  def _inverse_log_det_jacobian(self, _):
    return -tf.math.log(self._scale)


class ExpOnlyJacobian(bijector_lib.Bijector):
  """Only used for jacobian calculations."""

  def __init__(self, validate_args=False, forward_min_event_ndims=0):
    parameters = dict(locals())
    super(ExpOnlyJacobian, self).__init__(
        validate_args=validate_args,
        is_constant_jacobian=False,
        forward_min_event_ndims=forward_min_event_ndims,
        parameters=parameters,
        name='exp')

  def _inverse_log_det_jacobian(self, y):
    return -tf.math.log(y)

  def _forward_log_det_jacobian(self, x):
    return tf.math.log(x)


class VectorExpOnlyJacobian(bijector_lib.Bijector):
  """An Exp bijector that operates only on vector (or higher-order) events."""

  def __init__(self):
    parameters = dict(locals())
    super(VectorExpOnlyJacobian, self).__init__(
        validate_args=False,
        is_constant_jacobian=False,
        forward_min_event_ndims=1,
        parameters=parameters,
        name='vector_exp')

  def _inverse_log_det_jacobian(self, y):
    return -tf.reduce_sum(tf.math.log(y), axis=-1)

  def _forward_log_det_jacobian(self, x):
    return tf.reduce_sum(tf.math.log(x), axis=-1)


class ConstantJacobian(bijector_lib.Bijector):
  """Only used for jacobian calculations."""

  def __init__(self, forward_min_event_ndims=0):
    parameters = dict(locals())
    super(ConstantJacobian, self).__init__(
        validate_args=False,
        is_constant_jacobian=True,
        forward_min_event_ndims=forward_min_event_ndims,
        parameters=parameters,
        name='c')

  def _inverse_log_det_jacobian(self, y):
    return tf.constant(2., y.dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.constant(-2., x.dtype)


class UniqueCacheKey(bijector_lib.Bijector):
  """Used to test instance-level caching."""

  def __init__(self, forward_min_event_ndims=0):
    parameters = dict(locals())
    super(UniqueCacheKey, self).__init__(
        validate_args=False,
        is_constant_jacobian=True,
        forward_min_event_ndims=forward_min_event_ndims,
        parameters=parameters,
        name='instance_cache')

  def _forward(self, x):
    return x + tf.constant(1., x.dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.constant(0., x.dtype)

  def _get_parameterization(self):
    return id(self)


class UnspecifiedParameters(bijector_lib.Bijector):
  """A bijector that fails to pass `parameters` to the base class."""

  def __init__(self, loc):
    self._loc = loc
    super(UnspecifiedParameters, self).__init__(
        validate_args=False,
        is_constant_jacobian=True,
        forward_min_event_ndims=0,
        name='unspecified_parameters')

  def _forward(self, x):
    return x + self._loc

  def _forward_log_det_jacobian(self, x):
    return tf.constant(0., x.dtype)


@test_util.test_all_tf_execution_regimes
class BijectorTestEventNdims(test_util.TestCase):

  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

  def testBijectorNonIntegerEventNdims(self):
    bij = ExpOnlyJacobian()
    with self.assertRaisesRegexp(ValueError, 'Expected integer'):
      bij.forward_log_det_jacobian(1., event_ndims=1.5)
    with self.assertRaisesRegexp(ValueError, 'Expected integer'):
      bij.inverse_log_det_jacobian(1., event_ndims=1.5)

  def testBijectorArrayEventNdims(self):
    bij = ExpOnlyJacobian()
    with self.assertRaisesRegexp(ValueError, 'Expected scalar'):
      bij.forward_log_det_jacobian(1., event_ndims=(1, 2))
    with self.assertRaisesRegexp(ValueError, 'Expected scalar'):
      bij.inverse_log_det_jacobian(1., event_ndims=(1, 2))

  def testBijectorDynamicEventNdims(self):
    with self.assertRaisesError('Expected scalar'):
      bij = ExpOnlyJacobian(validate_args=True)
      event_ndims = tf1.placeholder_with_default((1, 2), shape=None)
      self.evaluate(
          bij.forward_log_det_jacobian(1., event_ndims=event_ndims))
    with self.assertRaisesError('Expected scalar'):
      bij = ExpOnlyJacobian(validate_args=True)
      event_ndims = tf1.placeholder_with_default((1, 2), shape=None)
      self.evaluate(
          bij.inverse_log_det_jacobian(1., event_ndims=event_ndims))


@test_util.test_all_tf_execution_regimes
class BijectorBatchShapesTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('exp', exp.Exp, None),
      ('scale', lambda: scale_lib.Scale(tf.ones([4, 2])), None),
      ('sigmoid',
       lambda: sigmoid_lib.Sigmoid(  # pylint: disable=g-long-lambda
           low=tf.zeros([3]), high=tf.ones([4, 1])), None),
      ('scale_matvec',
       lambda: scale_matvec_diag.ScaleMatvecDiag([[0.], [3.]]), None),
      ('invert', lambda: invert.Invert(  # pylint: disable=g-long-lambda
          scale_matvec_diag.ScaleMatvecDiag(tf.ones([2, 1]))), None),
      ('reshape', lambda: reshape.Reshape([1], event_shape_in=[1, 1]), None),
      (
          'chain',
          lambda: chain_lib.Chain([  # pylint: disable=g-long-lambda
              power.Power(power=[[2.], [3.]]),
              invert.Invert(split.Split(2))
          ]),
          None),
      ('jointmap_01', lambda: joint_map.JointMap(  # pylint: disable=g-long-lambda
          [scale_lib.Scale([5, 3]), scale_lib.Scale([1, 4])]), [0, 1]),
      ('jointmap_11', lambda: joint_map.JointMap(  # pylint: disable=g-long-lambda
          [scale_lib.Scale([5, 3]), scale_lib.Scale([1, 4])]), [1, 1]),
      ('jointmap_20', lambda: joint_map.JointMap(  # pylint: disable=g-long-lambda
          [scale_lib.Scale([5, 3]), scale_lib.Scale([1, 4])]), [2, 0]),
      ('jointmap_22', lambda: joint_map.JointMap(  # pylint: disable=g-long-lambda
          [scale_lib.Scale([5, 3]), scale_lib.Scale([1, 4])]), [2, 2]),
      (
          'restructure_with_ragged_event_ndims',
          lambda: restructure.Restructure(  # pylint: disable=g-long-lambda
              input_structure=[0, 1],
              output_structure={
                  'a': 0,
                  'b': 1
              }),
          [0, 1]))
  def test_batch_shape_matches_output_shapes(self,
                                             bijector_fn,
                                             override_x_event_ndims=None):
    bijector = bijector_fn()
    if override_x_event_ndims is None:
      x_event_ndims = bijector.forward_min_event_ndims
      y_event_ndims = bijector.inverse_min_event_ndims
    else:
      x_event_ndims = override_x_event_ndims
      y_event_ndims = bijector.forward_event_ndims(x_event_ndims)

    # All ways of calculating the batch shape should yield the same result.
    batch_shape_x = bijector.experimental_batch_shape(
        x_event_ndims=x_event_ndims)
    batch_shape_y = bijector.experimental_batch_shape(
        y_event_ndims=y_event_ndims)
    self.assertEqual(batch_shape_x, batch_shape_y)

    batch_shape_tensor_x = bijector.experimental_batch_shape_tensor(
        x_event_ndims=x_event_ndims)
    batch_shape_tensor_y = bijector.experimental_batch_shape_tensor(
        y_event_ndims=y_event_ndims)
    self.assertAllEqual(batch_shape_tensor_x, batch_shape_tensor_y)
    self.assertAllEqual(batch_shape_tensor_x, batch_shape_x)

    # Check that we're robust to integer type.
    batch_shape_tensor_x64 = bijector.experimental_batch_shape_tensor(
        x_event_ndims=tf.nest.map_structure(np.int64, x_event_ndims))
    batch_shape_tensor_y64 = bijector.experimental_batch_shape_tensor(
        y_event_ndims=tf.nest.map_structure(np.int64, y_event_ndims))
    self.assertAllEqual(batch_shape_tensor_x64, batch_shape_tensor_y64)
    self.assertAllEqual(batch_shape_tensor_x64, batch_shape_x)

    # Pushing a value through the bijector should return a Tensor(s) with
    # the expected batch shape...
    xs = tf.nest.map_structure(lambda nd: tf.ones([1] * nd), x_event_ndims)
    ys = bijector.forward(xs)
    for y_part, nd in zip(tf.nest.flatten(ys), tf.nest.flatten(y_event_ndims)):
      part_batch_shape = ps.shape(y_part)[:ps.rank(y_part) - nd]
      self.assertAllEqual(batch_shape_y,
                          ps.broadcast_shape(batch_shape_y, part_batch_shape))

    # ... which should also be the shape of the fldj.
    fldj = bijector.forward_log_det_jacobian(xs, event_ndims=x_event_ndims)
    self.assertAllEqual(batch_shape_y, ps.shape(fldj))

    # Also check the inverse case.
    xs = bijector.inverse(tf.nest.map_structure(tf.identity, ys))
    for x_part, nd in zip(tf.nest.flatten(xs), tf.nest.flatten(x_event_ndims)):
      part_batch_shape = ps.shape(x_part)[:ps.rank(x_part) - nd]
      self.assertAllEqual(batch_shape_x,
                          ps.broadcast_shape(batch_shape_x, part_batch_shape))
    ildj = bijector.inverse_log_det_jacobian(ys, event_ndims=y_event_ndims)
    self.assertAllEqual(batch_shape_x, ps.shape(ildj))

  @parameterized.named_parameters(('scale', lambda: scale_lib.Scale([3.14159])),
                                  ('chain', lambda: exp.Exp()  # pylint: disable=g-long-lambda
                                   (scale_lib.Scale([3.14159]))))
  def test_ndims_specification(self, bijector_fn):
    bijector = bijector_fn()

    # If no `event_ndims` is passed, should assume min_event_ndims.
    self.assertAllEqual(bijector.experimental_batch_shape(), [1])
    self.assertAllEqual(bijector.experimental_batch_shape_tensor(), [1])

    with self.assertRaisesRegex(
        ValueError, 'Only one of `x_event_ndims` and `y_event_ndims`'):
      bijector.experimental_batch_shape(x_event_ndims=0, y_event_ndims=0)

    with  self.assertRaisesRegex(
        ValueError, 'Only one of `x_event_ndims` and `y_event_ndims`'):
      bijector.experimental_batch_shape_tensor(x_event_ndims=0, y_event_ndims=0)

  @parameterized.named_parameters(
      ('scale', lambda: scale_lib.Scale(tf.ones([4, 2])), None),
      ('sigmoid',
       lambda: sigmoid_lib.Sigmoid(  # pylint: disable=g-long-lambda
           low=tf.zeros([3]), high=tf.ones([4, 1])), None),
      ('invert', lambda: invert.Invert(  # pylint: disable=g-long-lambda
          scale_matvec_diag.ScaleMatvecDiag(tf.ones([2, 1]))), None),
      (
          'chain',
          lambda: chain_lib.Chain([  # pylint: disable=g-long-lambda
              power.Power(power=[[2.], [3.]]),
              invert.Invert(split.Split(2))
          ]),
          None),
      (
          'jointmap_01',
          lambda: joint_map.JointMap(  # pylint: disable=g-long-lambda
              [scale_lib.Scale([5, 3]), scale_lib.Scale([1, 4])]),
          [0, 1]),
      (
          'jointmap_11',
          lambda: joint_map.JointMap(  # pylint: disable=g-long-lambda
              [scale_lib.Scale([5, 3]), scale_lib.Scale([1, 4])]),
          [1, 1]),
      (
          'jointmap_20',
          lambda: joint_map.JointMap(  # pylint: disable=g-long-lambda
              [scale_lib.Scale([5, 3]), scale_lib.Scale([1, 4])]),
          [2, 0]),
      (
          'jointmap_22',
          lambda: joint_map.JointMap(  # pylint: disable=g-long-lambda
              [scale_lib.Scale([5, 3]), scale_lib.Scale([1, 4])]),
          [2, 2]),
      (
          'nested_jointmap',
          lambda: joint_map.JointMap([  # pylint: disable=g-long-lambda
              joint_map.JointMap({
                  'a': scale_lib.Scale([1.]),
                  'b': exp.Exp()
              }),
              scale_lib.Scale([1, 4])(invert.Invert(split.Split(2)))
          ]),
          [{
              'a': 0,
              'b': 0
          }, [2, 2]]))
  def test_with_broadcast_batch_shape(self, bijector_fn, x_event_ndims=None):
    bijector = bijector_fn()
    if x_event_ndims is None:
      x_event_ndims = bijector.forward_min_event_ndims
    batch_shape = bijector.experimental_batch_shape(x_event_ndims=x_event_ndims)
    param_batch_shapes = batch_shape_lib.batch_shape_parts(
        bijector, bijector_x_event_ndims=x_event_ndims)

    new_batch_shape = [4, 2, 1, 1, 1]
    broadcast_bijector = bijector._broadcast_parameters_with_batch_shape(
        new_batch_shape, x_event_ndims)
    broadcast_batch_shape = broadcast_bijector.experimental_batch_shape_tensor(
        x_event_ndims=x_event_ndims)
    self.assertAllEqual(broadcast_batch_shape,
                        ps.broadcast_shape(batch_shape, new_batch_shape))

    # Check that all params have the expected batch shape.
    broadcast_param_batch_shapes = batch_shape_lib.batch_shape_parts(
        broadcast_bijector, bijector_x_event_ndims=x_event_ndims)

    def _maybe_broadcast_param_batch_shape(p, s):
      if isinstance(p, invert.Invert) and not p.bijector._params_event_ndims():
        # Split has a shape parameter that has no batch shape.
        if isinstance(p.bijector, split.Split):
          return s
        if not p.bijector._params_event_ndims():
          return s  # Can't broadcast a bijector that doesn't have params.
      return ps.broadcast_shape(s, new_batch_shape)
    expected_broadcast_param_batch_shapes = tf.nest.map_structure(
        _maybe_broadcast_param_batch_shape,
        {param: getattr(bijector, param) for param in param_batch_shapes},
        param_batch_shapes)
    self.assertAllEqualNested(broadcast_param_batch_shapes,
                              expected_broadcast_param_batch_shapes)


@test_util.test_all_tf_execution_regimes
class BijectorCachingTest(test_util.TestCase):

  def testCachingOfForwardResults(self):
    forward_only_bijector = ForwardOnlyBijector()
    x = tf.constant(1.1)
    y = tf.constant(2.2)

    with self.assertRaises(NotImplementedError):
      forward_only_bijector.inverse(y)

    with self.assertRaises(NotImplementedError):
      forward_only_bijector.inverse_log_det_jacobian(y, event_ndims=0)

    # Call forward and forward_log_det_jacobian one-by-one (not together).
    y = forward_only_bijector.forward(x)
    _ = forward_only_bijector.forward_log_det_jacobian(x, event_ndims=0)
    self.assertIs(y, forward_only_bijector.forward(x))
    # Now, everything should be cached if the argument `is y`, so these are ok.
    forward_only_bijector.inverse(y)
    forward_only_bijector.inverse_log_det_jacobian(y, event_ndims=0)

  def testCachingOfInverseResults(self):
    inverse_only_bijector = InverseOnlyBijector()
    x = tf.constant(1.1)
    y = tf.constant(2.2)

    with self.assertRaises(NotImplementedError):
      inverse_only_bijector.forward(x)

    with self.assertRaises(NotImplementedError):
      inverse_only_bijector.forward_log_det_jacobian(x, event_ndims=0)

    # Call inverse and inverse_log_det_jacobian one-by-one (not together).
    x = inverse_only_bijector.inverse(y)
    _ = inverse_only_bijector.inverse_log_det_jacobian(y, event_ndims=0)
    self.assertIs(x, inverse_only_bijector.inverse(y))

    # Now, everything should be cached if the argument `is x`.
    inverse_only_bijector.forward(x)
    inverse_only_bijector.forward_log_det_jacobian(x, event_ndims=0)

  def testCachingGarbageCollection(self):
    bijector = ForwardOnlyBijector()
    bijector._cache.clear()
    niters = 6
    for i in range(niters):
      x = tf.constant(i, dtype=tf.float32)
      y = bijector.forward(x)  # pylint: disable=unused-variable

    # We tolerate leaking tensor references in graph mode only.
    expected_live = 1 if tf.executing_eagerly() else niters
    self.assertEqual(
        expected_live, len(bijector._cache.weak_keys(direction='forward')))

  def testSharedCacheForward(self):

    # Test that shared caching behaves as expected when bijectors are
    # parameterized by Python floats, Tensors, and np arrays.
    f = lambda x: x
    g = lambda x: tf.constant(x, dtype=tf.float32)
    h = lambda x: np.array(x).astype(np.float32)

    scale_1 = 2.
    scale_2 = 3.
    x = tf.constant(3., dtype=tf.float32)

    for fn in [f, g, h]:
      s_1 = fn(scale_1)
      s_2 = fn(scale_2)
      bijector_1a = ForwardOnlyBijector(scale=s_1)
      bijector_1b = ForwardOnlyBijector(scale=s_1)
      bijector_2 = ForwardOnlyBijector(scale=s_2)

      y = bijector_1a.forward(x)

      # Different bijector instances with the same type/parameterization
      # => cache hit.
      self.assertIs(y, bijector_1b.forward(x))

      # Bijectors with different parameterizations => cache miss.
      self.assertIsNot(y, bijector_2.forward(x))

  def testSharedCacheInverse(self):
    # Test that shared caching behaves as expected when bijectors are
    # parameterized by Python floats, Tensors, and np arrays.
    f = lambda x: x
    g = lambda x: tf.constant(x, dtype=tf.float32)
    h = lambda x: np.array(x).astype(np.float32)

    scale_1 = 2.
    scale_2 = 3.
    y = tf.constant(3., dtype=tf.float32)

    for fn in [f, g, h]:
      s_1 = fn(scale_1)
      s_2 = fn(scale_2)
      InverseOnlyBijector._cache.clear()
      bijector_1a = InverseOnlyBijector(scale=s_1)
      bijector_1b = InverseOnlyBijector(scale=s_1)
      bijector_2 = InverseOnlyBijector(scale=s_2)

      x = bijector_1a.inverse(y)

      # Different bijector instances with the same type/parameterization
      # => cache hit.
      self.assertIs(x, bijector_1b.inverse(y))

      # Bijectors with different parameterizations => cache miss.
      self.assertIsNot(x, bijector_2.inverse(y))

      # There is only one entry in the cache corresponding to each fn call
      self.assertLen(bijector_1a._cache.weak_keys(direction='forward'), 1)
      self.assertLen(bijector_2._cache.weak_keys(direction='inverse'), 1)

  def testUniqueCacheKey(self):
    bijector_1 = UniqueCacheKey()
    bijector_2 = UniqueCacheKey()

    x = tf.constant(3., dtype=tf.float32)
    y_1 = bijector_1.forward(x)
    y_2 = bijector_2.forward(x)

    self.assertIsNot(y_1, y_2)
    self.assertLen(bijector_1._cache.weak_keys(direction='forward'), 1)
    self.assertLen(bijector_2._cache.weak_keys(direction='forward'), 1)

  def testBijectorsWithUnspecifiedParametersDoNotShareCache(self):
    bijector_1 = UnspecifiedParameters(loc=tf.constant(1., dtype=tf.float32))
    bijector_2 = UnspecifiedParameters(loc=tf.constant(2., dtype=tf.float32))

    x = tf.constant(3., dtype=tf.float32)
    y_1 = bijector_1.forward(x)
    y_2 = bijector_2.forward(x)

    self.assertIsNot(y_1, y_2)
    self.assertLen(bijector_1._cache.weak_keys(direction='forward'), 1)
    self.assertLen(bijector_2._cache.weak_keys(direction='forward'), 1)

  def testInstanceCache(self):
    instance_cache_bijector = exp.Exp()
    instance_cache_bijector._cache = cache_util.BijectorCache(
        bijector=instance_cache_bijector)
    global_cache_bijector = exp.Exp()

    # Ensure the global cache does not persist between tests in different
    # execution regimes.
    exp.Exp._cache.clear()

    x = tf.constant(0., dtype=tf.float32)
    y = global_cache_bijector.forward(x)

    # Instance-level cache doesn't store values from calls to an identical but
    # globally-cached bijector.
    self.assertLen(
        global_cache_bijector._cache.weak_keys(direction='forward'), 1)
    self.assertLen(
        instance_cache_bijector._cache.weak_keys(direction='forward'), 0)

    # Bijector with instance-level cache performs a globally-cached
    # transformation => cache miss. (Implying global cache did not pick it up.)
    z = instance_cache_bijector.forward(x)
    self.assertIsNot(y, z)

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True, reason='keras')
  @parameterized.named_parameters(
      ('Keras', True),
      ('NoKeras', False))
  def testJacobianRespectsCache(self, keras):
    bijector = InverseOnlyBijector(scale=2.)
    y = tf.constant(10.)
    if keras:
      y = tf.keras.layers.Input(shape=(), dtype=tf.float32, tensor=y)
    x = bijector.inverse(y)
    # Forward computation should work here because it should look up
    # `y` in the cache and call `inverse_log_det_jacobian`.
    fldj = bijector.forward_log_det_jacobian(x)
    self.assertAllClose(fldj, np.log(2.))


@test_util.test_all_tf_execution_regimes
class BijectorReduceEventDimsTest(test_util.TestCase):
  """Test reducing of event dims."""

  def testReduceEventNdimsForward(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian()
    self.assertAllClose(
        np.log(x), self.evaluate(
            bij.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        np.sum(np.log(x), axis=-1),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(np.log(x), axis=(-1, -2)),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=2)))

  def testNoReductionWhenEventNdimsIsOmitted(self):
    x = [[[1., 2.], [3., 4.]]]

    bij = ExpOnlyJacobian()
    self.assertAllClose(
        np.log(x),
        self.evaluate(bij.forward_log_det_jacobian(x)))
    self.assertAllClose(
        -np.log(x),
        self.evaluate(bij.inverse_log_det_jacobian(x)))

    bij = VectorExpOnlyJacobian()
    self.assertAllClose(
        np.sum(np.log(x), axis=-1),
        self.evaluate(bij.forward_log_det_jacobian(x)))
    self.assertAllClose(
        np.sum(-np.log(x), axis=-1),
        self.evaluate(bij.inverse_log_det_jacobian(x)))

  def testInverseWithEventDimsOmitted(self):
    bij = split.Split(2)

    self.assertAllEqual(
        0.0,
        self.evaluate(bij.inverse_log_det_jacobian(
            [tf.ones((3, 4, 5)), tf.ones((3, 4, 5))])))

  def testReduceEventNdimsForwardRaiseError(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    with self.assertRaisesRegexp(ValueError, 'must be at least'):
      bij.forward_log_det_jacobian(x, event_ndims=0)
    with self.assertRaisesRegexp(ValueError, 'Input must have rank at least'):
      bij.forward_log_det_jacobian(x, event_ndims=4)

  def testReduceEventNdimsInverse(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian()
    self.assertAllClose(
        -np.log(x), self.evaluate(
            bij.inverse_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        np.sum(-np.log(x), axis=-1),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(-np.log(x), axis=(-1, -2)),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=2)))

  def testReduceEventNdimsInverseRaiseError(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    with self.assertRaisesRegexp(ValueError, 'must be at least'):
      bij.inverse_log_det_jacobian(x, event_ndims=0)
    with self.assertRaisesRegexp(ValueError, 'Input must have rank at least'):
      bij.inverse_log_det_jacobian(x, event_ndims=4)

  def testReduceEventNdimsForwardConstJacobian(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ConstantJacobian()
    self.assertAllClose(
        -2., self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        -4., self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        -8., self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=2)))

  def testReduceEventNdimsInverseConstJacobian(self):
    x = [[[1., 2.], [3., 4.]]]
    bij = ConstantJacobian()
    self.assertAllClose(
        2., self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        4., self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        8., self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=2)))

  def testHandlesNonStaticEventNdims(self):
    x_ = [[[1., 2.], [3., 4.]]]
    x = tf1.placeholder_with_default(x_, shape=None)
    event_ndims = tf1.placeholder_with_default(1, shape=None)
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    bij.inverse_log_det_jacobian(x, event_ndims=event_ndims)
    ildj = self.evaluate(
        bij.inverse_log_det_jacobian(x, event_ndims=event_ndims))
    self.assertAllClose(-np.log(x_), ildj)


class BijectorLDJCachingTest(test_util.TestCase):

  def testShapeCachingIssue(self):
    if tf.executing_eagerly(): return
    # Exercise the scenario outlined in
    # https://github.com/tensorflow/probability/issues/253 (originally reported
    # internally as b/119756336).
    x1_value = np.random.uniform(size=[10, 2])
    x1 = tf1.placeholder_with_default(x1_value, shape=[None, 2], name='x1')
    x2 = tf1.placeholder(tf.float32, shape=[None, 2], name='x2')

    bij = ConstantJacobian()

    bij.forward_log_det_jacobian(x2, event_ndims=1)
    a = bij.forward_log_det_jacobian(x1, event_ndims=1, name='a_fldj')

    self.evaluate(a)


@test_util.test_all_tf_execution_regimes
class NumpyArrayCaching(test_util.TestCase):

  def test_caches(self):
    # We need to call convert_to_tensor on outputs to make sure scalar
    # outputs from the numpy backend are wrapped correctly. We could just
    # directly wrap numpy scalars with np.array, but it would look pretty
    # out of place, considering that the numpy backend is still private.
    if mock is None:
      return

    x_ = np.array([[-0.1, 0.2], [0.3, -0.4]], np.float32)
    y_ = np.exp(x_)
    b = exp.Exp()

    # Ensure the global cache does not persist between tests in different
    # execution regimes.
    exp.Exp._cache.clear()

    # We will intercept calls to TF to ensure np.array objects don't get
    # converted to tf.Tensor objects.

    with mock.patch.object(tf, 'convert_to_tensor', return_value=x_):
      with mock.patch.object(tf, 'exp', return_value=y_):
        y = b.forward(x_)
        self.assertIsInstance(y, np.ndarray)
        self.assertAllEqual(
            [x_], [k() for k in b._cache.weak_keys(direction='forward')])

    with mock.patch.object(tf, 'convert_to_tensor', return_value=y_):
      with mock.patch.object(tf.math, 'log', return_value=x_):
        x = b.inverse(y_)
        self.assertIsInstance(x, np.ndarray)
        self.assertIs(x, b.inverse(y))
        self.assertAllEqual(
            [y_], [k() for k in b._cache.weak_keys(direction='inverse')])

    yt_ = y_.T
    xt_ = x_.T
    with mock.patch.object(tf, 'convert_to_tensor', return_value=yt_):
      with mock.patch.object(tf.math, 'log', return_value=xt_):
        xt = b.inverse(yt_)
        self.assertIsNot(x, xt)
        self.assertIs(xt_, xt)


@test_util.test_all_tf_execution_regimes
class TfModuleTest(test_util.TestCase):

  @test_util.numpy_disable_variable_test
  @test_util.jax_disable_variable_test
  def test_variable_tracking(self):
    x = tf.Variable(1.)
    b = ForwardOnlyBijector(scale=x, validate_args=True)
    self.assertIsInstance(b, tf.Module)
    self.assertEqual((x,), b.trainable_variables)

  @test_util.numpy_disable_variable_test
  @test_util.jax_disable_variable_test
  def test_gradient(self):
    x = tf.Variable(1.)
    b = InverseOnlyBijector(scale=x, validate_args=True)
    with tf.GradientTape() as tape:
      loss = b.inverse(1.)
    g = tape.gradient(loss, b.trainable_variables)
    self.evaluate(tf1.global_variables_initializer())
    self.assertEqual((-1.,), self.evaluate(g))


class _ConditionalBijector(bijector_lib.Bijector):

  def __init__(self):
    parameters = dict(locals())
    super(_ConditionalBijector, self).__init__(
        forward_min_event_ndims=0,
        is_constant_jacobian=True,
        validate_args=False,
        dtype=tf.float32,
        parameters=parameters,
        name='test_bijector')

  # These are not implemented in the base class, but we need to write a stub in
  # order to mock them out.
  def _inverse_log_det_jacobian(self, _, arg1, arg2):
    pass

  def _forward_log_det_jacobian(self, _, arg1, arg2):
    pass


# Test that ensures kwargs from public methods are passed in to
# private methods.
@test_util.test_all_tf_execution_regimes
class ConditionalBijectorTest(test_util.TestCase):

  def testConditionalBijector(self):
    b = _ConditionalBijector()
    arg1 = 'b1'
    arg2 = 'b2'
    retval = tf.constant(1.)
    for name in ['forward', 'inverse']:
      method = getattr(b, name)
      with mock.patch.object(b, '_' + name, return_value=retval) as mock_method:
        method(1., arg1=arg1, arg2=arg2)
      mock_method.assert_called_once_with(mock.ANY, arg1=arg1, arg2=arg2)

    for name in ['inverse_log_det_jacobian', 'forward_log_det_jacobian']:
      method = getattr(b, name)
      with mock.patch.object(b, '_' + name, return_value=retval) as mock_method:
        method(1., event_ndims=0, arg1=arg1, arg2=arg2)
      mock_method.assert_called_once_with(mock.ANY, arg1=arg1, arg2=arg2)

  def testNestedCondition(self):
    b = _ConditionalBijector()
    arg1 = {'b1': 'c1'}
    arg2 = {'b2': 'c2'}
    retval = tf.constant(1.)
    for name in ['forward', 'inverse']:
      method = getattr(b, name)
      with mock.patch.object(b, '_' + name, return_value=retval) as mock_method:
        method(1., arg1=arg1, arg2=arg2)
      mock_method.assert_called_once_with(mock.ANY, arg1=arg1, arg2=arg2)

    for name in ['inverse_log_det_jacobian', 'forward_log_det_jacobian']:
      method = getattr(b, name)
      with mock.patch.object(b, '_' + name, return_value=retval) as mock_method:
        method(1., event_ndims=0, arg1=arg1, arg2=arg2)
      mock_method.assert_called_once_with(mock.ANY, arg1=arg1, arg2=arg2)


class CompositeForwardBijector(bijector_lib.AutoCompositeTensorBijector):

  def __init__(self, scale=2., validate_args=False, parameters=None, name=None):
    parameters = dict(locals()) if parameters is None else parameters
    with tf.name_scope(name or 'forward_only') as name:
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale,
          dtype_hint=tf.float32)
      super(CompositeForwardBijector, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          parameters=parameters,
          name=name)

  def _forward(self, x):
    return self._scale * x

  def _forward_log_det_jacobian(self, _):
    return tf.math.log(self._scale)


class CompositeForwardScaleThree(CompositeForwardBijector):

  def __init__(self, name='scale_three'):
    parameters = dict(locals())
    super(CompositeForwardScaleThree, self).__init__(
        scale=3., parameters=parameters, name=name)


@test_util.test_all_tf_execution_regimes
class AutoCompositeTensorBijectorTest(test_util.TestCase):

  def test_disable_ct_bijector(self):

    ct_bijector = CompositeForwardBijector()
    self.assertIsInstance(ct_bijector, tf.__internal__.CompositeTensor)

    non_ct_bijector = ForwardOnlyBijector()
    self.assertNotIsInstance(non_ct_bijector, tf.__internal__.CompositeTensor)

    flat = tf.nest.flatten(ct_bijector, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(
        ct_bijector, flat, expand_composites=True)

    x = tf.constant([2., 3.])
    self.assertAllClose(
        non_ct_bijector.forward(x),
        tf.function(lambda b: b.forward(x))(unflat))

  def test_composite_tensor_subclass(self):

    bij = CompositeForwardScaleThree()
    self.assertIs(bij._type_spec.value_type, type(bij))

    flat = tf.nest.flatten(bij, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(bij, flat, expand_composites=True)
    self.assertIsInstance(unflat, CompositeForwardScaleThree)


if __name__ == '__main__':
  test_util.main()
