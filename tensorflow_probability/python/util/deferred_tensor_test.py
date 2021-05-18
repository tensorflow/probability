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
"""Tests for `tfp.util.DeferredTensor`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.util import deferred_tensor
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=g-direct-tensorflow-import


tfb = tfp.bijectors
tfd = tfp.distributions

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class DeferredTensorTest(test_util.TestCase):

  # This needs to match the doc example, so we won't port it to use the backend
  # agnostic code.
  @test_util.jax_disable_test_missing_functionality('GradientTape')
  @test_util.numpy_disable_test_missing_functionality('GradientTape')
  def test_docstring_example(self):
    trainable_normal = tfd.Normal(
        loc=tf.Variable(0.),
        scale=tfp.util.DeferredTensor(
            tf.Variable(0., name='raw_scale'), tf.math.exp))
    with tf.GradientTape() as tape:
      negloglik = -trainable_normal.log_prob(0.5)
    g = tape.gradient(negloglik, trainable_normal.trainable_variables)
    self.evaluate([v.initializer for v in trainable_normal.trainable_variables])
    self.assertEqual((-1. / (2. * 1.), (1. - 0.5**2) / 1.), self.evaluate(g))
    self.assertIsInstance(trainable_normal.scale, tfp.util.DeferredTensor)
    self.assertEqual(1., self.evaluate(trainable_normal.scale**2.))
    # For speed, we don't bother testing the optimization part of the example.

  def test_properties(self):
    x = tfp.util.DeferredTensor(tf.Variable(0.), tf.math.exp)
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertEqual((), x.shape)
    self.assertEqual(tf.float32, x.dtype)
    if tf.executing_eagerly():
      self.assertEqual(
          repr(x),
          '<DeferredTensor: dtype=float32, shape=[], fn=exp, numpy=1.0>')
    else:
      self.assertStartsWith(x.name, 'exp')
      self.assertEqual(
          repr(x),
          '<DeferredTensor: dtype=float32, shape=[], fn=exp>')

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_variable_test
  def test_retains_trainable_variables_from_bijector(self):
    m = tf.Variable(0., name='m')
    x = tfp.util.DeferredTensor(1., tfb.Scale(m))
    self.assertIn(m, x.trainable_variables)

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_variable_test
  def test_retains_trainable_variables_from_also_track(self):
    m = tf.Variable(0., name='m')
    x = tfp.util.DeferredTensor(1., lambda x: x * m, also_track=m)
    self.assertIn(m, x.trainable_variables)

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_variable_test
  def test_variable_shape_changes(self):
    v = tf.Variable(np.zeros((3, 2, 3)), shape=tf.TensorShape((None, 2, None)))
    self.evaluate(v.initializer)
    x = tfp.util.DeferredTensor(v, tf.math.softmax)

    self.assertAllEqual((None, 2, None), x.shape.as_list())
    self.assertAllEqual((3, 2, 3), self.evaluate(tf.shape(x)))

    with tf.control_dependencies([v.assign(np.zeros((1, 2, 4)))]):
      self.assertAllEqual((1, 2, 4), self.evaluate(tf.shape(x)))
      self.assertAllEqual((None, 2, None), x.shape.as_list())

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_variable_test
  def test_variable_rank_changes(self):
    def f(x):
      shape = tf.shape(x)
      return tf.reshape(
          x, tf.concat([[2, (shape[0] * shape[1]) // 2], shape[2:]], axis=0))

    v = tf.Variable(np.zeros((3, 4, 3)), shape=tf.TensorShape(None))
    self.evaluate(v.initializer)
    x = tfp.util.DeferredTensor(v, f)

    self.assertIsNone(x.shape.rank)
    self.assertAllEqual((2, 6, 3), self.evaluate(tf.shape(x)))

    with tf.control_dependencies([v.assign(np.zeros((4, 5, 1, 1)))]):
      self.assertAllEqual((2, 10, 1, 1), self.evaluate(tf.shape(x)))
      self.assertIsNone(x.shape.rank)

  def test_from_bijector_with_inverted_assignment(self):
    x = tfp.util.DeferredTensor(tf.Variable([[1.], [2.], [3.]]),
                                tfb.Pad(validate_args=True))
    self.assertEqual(tf.float32, x.dtype)
    self.assertAllEqual([3, 1], x.pretransformed_input.shape)
    self.assertAllEqual([3, 2], x.shape)
    if tf.executing_eagerly():
      self.assertEqual(
          repr(x),
          '<DeferredTensor: dtype=float32, shape=[3, 2], fn="pad", '
          'numpy=\narray([[1., 0.],\n       [2., 0.],\n       [3., 0.]], '
          'dtype=float32)>')
    else:
      self.assertEqual(
          repr(x),
          '<DeferredTensor: dtype=float32, shape=[3, 2], fn="pad">')

    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertAllEqual([[1., 0.], [2., 0.], [3., 0.]],
                        self.evaluate(tf.convert_to_tensor(x)))

    assign_op = x.pretransformed_input.assign([[-1.], [-2.], [-3.]])
    with tf.control_dependencies([assign_op]):
      self.assertAllEqual([3, 1], x.pretransformed_input.shape)
      self.assertAllEqual([3, 2], x.shape)
      v_, y_ = self.evaluate([
          tf.convert_to_tensor(x.pretransformed_input),
          tf.convert_to_tensor(x)])
      self.assertAllEqual([[-1.], [-2.], [-3.]], v_)
      self.assertAllEqual([[-1., 0.], [-2., 0.], [-3., 0.]], y_)


@test_util.test_all_tf_execution_regimes
class TransformedVariableTest(test_util.TestCase):

  # This needs to match the doc example, so we won't port it to use the backend
  # agnostic code.
  @test_util.jax_disable_test_missing_functionality('GradientTape')
  @test_util.numpy_disable_test_missing_functionality('GradientTape')
  def test_docstring_1(self):
    trainable_normal = tfd.Normal(
        loc=tf.Variable(0.),
        scale=tfp.util.TransformedVariable(1., tfb.Exp()))
    self.evaluate([v.initializer for v in trainable_normal.trainable_variables])
    self.assertAllEqual(
        1., self.evaluate(tf.convert_to_tensor(trainable_normal.scale)))
    self.assertAllEqual(
        2., self.evaluate(trainable_normal.scale + 1.))
    with tf.GradientTape() as tape:
      negloglik = -trainable_normal.log_prob(0.5)
    g = tape.gradient(negloglik, trainable_normal.trainable_variables)
    self.assertAllEqual([-0.5, 0.75], self.evaluate(g))

  def test_docstring_2(self):
    d = tfd.Normal(loc=tf.Variable(0.),
                   scale=tfp.util.TransformedVariable(
                       [1., 2.], tfb.Softplus(validate_args=True)))
    self.evaluate([v.initializer for v in d.trainable_variables])
    self.assertAllClose([1., 2.], self.evaluate(d.stddev()), atol=0., rtol=1e-5)
    with tf.control_dependencies([d.scale.assign_add([0.5, 1.])]):
      self.assertAllClose([1.5, 3.], self.evaluate(d.stddev()),
                          atol=0., rtol=1e-5)

  def test_assign_ops_work_correctly(self):
    x = tfp.util.TransformedVariable(
        [[0.25, 0.75], [0.2, 0.8], [0.66, 0.34]],
        tfb.SoftmaxCentered(validate_args=True))
    self.evaluate([v.initializer for v in x.trainable_variables])

    assign_op = x.assign([[0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    with tf.control_dependencies([assign_op]):
      self.assertAllClose([[0.3, 0.7], [0.6, 0.4], [0.1, 0.9]],
                          self.evaluate(tf.convert_to_tensor(x)),
                          atol=0., rtol=1e-5)

    assign_op = x.assign_add([[0.5, -0.5], [-0.25, 0.25], [0.2, -0.2]])
    with tf.control_dependencies([assign_op]):
      self.assertAllClose([[0.8, 0.2], [0.35, 0.65], [0.3, 0.7]],
                          self.evaluate(tf.convert_to_tensor(x)),
                          atol=0., rtol=1e-5)

    assign_op = x.assign_sub([[0.5, -0.5], [-0.25, 0.25], [0.2, -0.2]])
    with tf.control_dependencies([assign_op]):
      self.assertAllClose([[0.3, 0.7], [0.6, 0.4], [0.1, 0.9]],
                          self.evaluate(tf.convert_to_tensor(x)),
                          atol=0., rtol=1e-5)

  def test_properties(self):
    x = tfp.util.TransformedVariable(1., tfb.Exp())
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertEqual((), x.shape)
    self.assertEqual(tf.float32, x.dtype)
    if tf.executing_eagerly():
      self.assertEqual(
          repr(x),
          '<TransformedVariable: dtype=float32, shape=[], fn="exp", numpy=1.0>')
    else:
      self.assertStartsWith(x.name, 'exp')
      self.assertEqual(
          repr(x),
          '<TransformedVariable: dtype=float32, shape=[], fn="exp">')

  def test_shape_changing_bijector(self):
    num_tril_nonzero = lambda num_rows: num_rows * (num_rows + 1) // 2
    num_tril_rows = lambda nnz: (  # pylint: disable=g-long-lambda
        np.sqrt(0.25 + 2. * nnz) - 0.5).astype(np.int32)
    pad_eye = tfb.Inline(
        forward_fn=lambda x: tf.concat([  # pylint: disable=g-long-lambda
            tfb.FillScaleTriL().inverse(
                tf.eye(num_tril_rows(tf.compat.dimension_value(x.shape[-1])),
                       batch_shape=tf.shape(x)[:-2]))[..., tf.newaxis, :],
            x,
        ], axis=tf.rank(x) - 2),
        inverse_fn=lambda y: y[..., 1:, :],
        inverse_log_det_jacobian_fn=lambda y, event_ndims: 0.,
        forward_event_shape_fn=lambda in_shape: in_shape + tf.one_hot(  # pylint: disable=g-long-lambda
            tf.size(in_shape) - 2, depth=tf.size(in_shape), dtype=tf.int32),
        inverse_event_shape_fn=lambda out_shape: out_shape - tf.one_hot(  # pylint: disable=g-long-lambda
            tf.size(out_shape) - 2, depth=tf.size(out_shape), dtype=tf.int32),
        forward_min_event_ndims=2,
        inverse_min_event_ndims=2,
        is_constant_jacobian=True,
        name='PadEyeBijector')
    scale_tril = tfp.util.TransformedVariable(
        tf.eye(3, batch_shape=[5, 1, 4]),
        bijector=tfb.Chain([tfb.FillScaleTriL(), pad_eye]))
    self.assertAllEqual((5, 1, 4, 3, 3), scale_tril.shape)
    self.assertAllEqual((5, 1, 4 - 1, num_tril_nonzero(3)),
                        scale_tril.pretransformed_input.shape)
    self.evaluate([v.initializer for v in scale_tril.trainable_variables])
    shape_, scale_tril_ = self.evaluate([
        tf.shape(scale_tril), tf.convert_to_tensor(scale_tril)])
    self.assertAllEqual((5, 1, 4, 3, 3), shape_)
    self.assertAllEqual((5, 1, 4, 3, 3), scale_tril_.shape)

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_variable_test
  def test_nested_transformed_variable(self):
    x = tfp.util.TransformedVariable(0.25, tfb.Exp())
    self.evaluate(x.initializer)
    y = tfp.util.TransformedVariable(x, tfb.Invert(tfb.Square(), name='Sqrt'))
    self.evaluate(y.initializer)
    self.assertLen(y.trainable_variables, 1)
    y_, x_, vy_, vx_ = self.evaluate([
        tf.convert_to_tensor(y),
        tf.convert_to_tensor(x),
        tf.convert_to_tensor(y.pretransformed_input),
        tf.convert_to_tensor(x.pretransformed_input),
    ])
    self.assertNear(np.log(0.25), vx_, err=1e-3)
    self.assertNear(np.square(0.25), vy_, err=1e-3)
    self.assertNear(0.25, x_, err=1e-3)
    self.assertNear(0.25, y_, err=1e-3)
    self.assertIsNot(x.pretransformed_input, y.pretransformed_input)
    # Different vars have no deps so we needn't test cross-talk.


@test_util.test_all_tf_execution_regimes
class DeferredTensorBehavesLikeTensorTest(test_util.TestCase):

  def testArrayPriority(self):
    x = tfp.util.DeferredTensor(tf.Variable(0.), tf.math.exp)
    y = np.array(3., dtype=np.float32)
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertEqual(3., self.evaluate(y / x))

  @parameterized.parameters(
      operator.add,
      operator.sub,
      operator.mul,
      operator.floordiv,
      operator.truediv,
      operator.pow,
      operator.mod,
      operator.gt,
      operator.ge,
      operator.lt,
      operator.le,
  )
  def testOperatorBinary(self, op):
    x = tfp.util.DeferredTensor(tf.Variable(0.), tf.math.exp)
    # Left operand does not support corresponding op and the operands are of
    # different types. Eg: `__radd__`.
    y1 = op(2., x)
    # Left operand supports op since right operand is implicitly converted by
    # usual `convert_to_tensor` semantics. Eg: `__add__`.
    y2 = op(x, 3.)
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertAllClose([op(2., 1.), op(1., 3.)],
                        self.evaluate([y1, y2]),
                        atol=0., rtol=1e-5)

  @parameterized.parameters(
      operator.abs,
      operator.neg,
  )
  def testOperatorUnary(self, op):
    x = tfp.util.DeferredTensor(tf.Variable(-1.), tf.identity)
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertAllEqual(op(x), self.evaluate(op(x)))

  @parameterized.parameters(
      operator.and_,
      operator.or_,
      operator.xor,
  )
  def testOperatorBinaryLogical(self, op):
    x_ = False
    x = tfp.util.DeferredTensor(
        tf.Variable(0.), lambda x: tf.cast(x, tf.bool), dtype=tf.bool)
    y1 = op(True, x)
    y2 = op(x, False)
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertAllEqual([op(True, x_), op(x_, False)],
                        self.evaluate([y1, y2]))

  # `~` is the only supported unary logical operator.
  # Note: 'boolean operator' is distinct from 'logical operator'. (The former
  # generally being not overrideable.)
  def testOperatorUnaryLogical(self):
    x = tfp.util.DeferredTensor(
        tf.Variable(0), lambda x: tf.cast(x, tf.bool), dtype=tf.bool)
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertAllEqual(*self.evaluate([~tf.convert_to_tensor(x), ~x]))

  def testOperatorBoolNonzero(self):
    x = tfp.util.DeferredTensor(
        tf.Variable(0.), lambda x: tf.cast(x, tf.bool), dtype=tf.bool)
    self.evaluate([v.initializer for v in x.trainable_variables])
    with self.assertRaises(TypeError):
      _ = not x

  def testOperatorGetitem(self):
    x = tfp.util.DeferredTensor(tf.Variable([1., 2.]), tf.math.exp)
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertAllClose([np.exp(1.)], self.evaluate(x[:1]), atol=0., rtol=1e-5)

  def testOperatorIter(self):
    x_ = [0., 1.]
    x = tfp.util.DeferredTensor(tf.Variable(x_), tf.math.exp)
    self.evaluate([v.initializer for v in x.trainable_variables])
    if tf.executing_eagerly():
      for expected_, actual_ in zip(x_, iter(x)):
        self.assertNear(np.exp(expected_), actual_, err=1e-5)
    else:
      with self.assertRaises(TypeError):
        for _ in iter(x):
          pass

  def testMethodNumpy(self):
    x_ = np.array([0., 1.])
    x = tfp.util.DeferredTensor(tf.Variable(x_), tf.math.exp)

    self.evaluate([v.initializer for v in x.trainable_variables])
    if tf.executing_eagerly():
      self.assertAllEqual(tf.math.exp(x_), x.numpy())
    else:
      with self.assertRaises(NotImplementedError):
        x.numpy()


class DeferredTensorBehavesLikeTensorInXLATest(test_util.TestCase):
  # This is entirely for the benefit of JAX, which behaves quite differently
  # inside its jit context than TF.

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='NumPy has no XLA.')
  @parameterized.parameters(
      operator.add,
      operator.sub,
      operator.mul,
      operator.floordiv,
      operator.truediv,
      operator.pow,
      operator.mod,
      operator.gt,
      operator.ge,
      operator.lt,
      operator.le,
  )
  def testOperatorBinary(self, op):
    @tf.function(autograph=False, jit_compile=True)
    def fn(y):
      x = tfp.util.DeferredTensor(y, tf.math.exp)
      y1 = op(2., x)
      y2 = op(x, 3.)
      return y1, y2

    y1, y2 = fn(0.)
    self.assertAllClose([op(2., 1.), op(1., 3.)],
                        self.evaluate([y1, y2]),
                        atol=0., rtol=1e-5)


def _make_transformed_variable_spec(
    input_spec, transform_or_spec, dtype=None, name=None):
  """Returns a `_TransformedVariableSpec` instance."""
  dtype = dtype or input_spec.dtype
  return deferred_tensor._TransformedVariableSpec(
      input_spec=input_spec, transform_or_spec=transform_or_spec, dtype=dtype,
      name=name)


def _make_bijector_spec(
    bijector_class, param, use_variable=False, variable_shape=None):
  """Returns the `TypeSpec` of a Bijector with one Tensor-valued parameter.

  This utility avoids errors in the JAX backend due to instantiation of a
  bijector before `app.run` is called.

  Args:
    bijector_class: Subclass of `tfp.bijectors.Bijector`.
    param: `Tensor`-like parameter of the bijector.
    use_variable: Python `bool`. If True, `param` is converted to a
      `tf.Variable`.
    variable_shape: `tf.TensorShape` or list of `int`s. Static shape of the
      `tf.Variable`, if `use_variable` is True.

  Returns:
    bijector_spec: `TypeSpec` for a `Bijector` instance, or None if the test is
      running in JAX mode.
  """
  if JAX_MODE:
    return None
  if use_variable:
    param = tf.Variable(param, shape=variable_shape)
  return bijector_class(param)._type_spec


@test_util.test_all_tf_execution_regimes
@test_util.disable_test_for_backend(
    disable_numpy=True, disable_jax=True,
    reason='JAX and Numpy have no notion of `TypeSpec`.')
class DeferredTensorSpecTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('TransformedVariableBijector',
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, None], tf.float32),
           transform_or_spec=_make_bijector_spec(tfb.Scale, [3.])),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, None], tf.float32),
           transform_or_spec=_make_bijector_spec(tfb.Scale, [3.]))),
      ('TranformedVariableCallable',
       _make_transformed_variable_spec(
           input_spec=resource_variable_ops.VariableSpec(None, tf.float64),
           transform_or_spec=tf.math.sigmoid,
           dtype=tf.float64,
           name='one'),
       _make_transformed_variable_spec(
           input_spec=resource_variable_ops.VariableSpec(None, tf.float64),
           transform_or_spec=tf.math.sigmoid,
           dtype=tf.float64,
           name='two')),
      )
  def testEquality(self, v1, v2):
    # pylint: disable=g-generic-assert
    self.assertEqual(v1, v2)
    self.assertEqual(v2, v1)
    self.assertFalse(v1 != v2)
    self.assertFalse(v2 != v1)
    self.assertEqual(hash(v1), hash(v2))

  @parameterized.named_parameters(
      ('DifferentDtypes',
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float64),
           transform_or_spec=tfb.Sigmoid(validate_args=True)._type_spec,
           dtype=tf.float64),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float32),
           transform_or_spec=tfb.Sigmoid(validate_args=True)._type_spec)),
      ('DifferentCallables',
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float64),
           transform_or_spec=tf.math.sigmoid,
           dtype=tf.float64,
           name='one'),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float64),
           transform_or_spec=tf.math.softplus,
           dtype=tf.float64,
           name='two')),
      )
  def testInequality(self, v1, v2):
    # pylint: disable=g-generic-assert
    self.assertNotEqual(v1, v2)
    self.assertNotEqual(v2, v1)
    self.assertFalse(v1 == v2)
    self.assertFalse(v2 == v1)

  @parameterized.named_parameters(
      ('TransformedVariableBijector',
       _make_transformed_variable_spec(
           input_spec=resource_variable_ops.VariableSpec([4, 2], tf.float32),
           transform_or_spec=tfb.Sigmoid(validate_args=True)._type_spec),
       _make_transformed_variable_spec(
           input_spec=resource_variable_ops.VariableSpec([4, 2], tf.float32),
           transform_or_spec=tfb.Sigmoid(validate_args=True)._type_spec)),
      ('TransformedVariableCallable',
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float32),
           transform_or_spec=tf.math.sigmoid,
           name='one'),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float32),
           transform_or_spec=tf.math.sigmoid,
           name='two')),
      )
  def testIsCompatibleWith(self, v1, v2):
    self.assertTrue(v1.is_compatible_with(v2))
    self.assertTrue(v2.is_compatible_with(v1))

  @parameterized.named_parameters(
      ('DifferentDtypes',
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float32),
           transform_or_spec=tfb.Sigmoid(validate_args=True)._type_spec,
           dtype=tf.float64),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float32),
           transform_or_spec=tfb.Sigmoid(validate_args=True)._type_spec)),
      ('DifferentCallables',
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float64),
           transform_or_spec=tf.math.sigmoid,
           dtype=tf.float64,
           name='one'),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float64),
           transform_or_spec=tf.math.softplus,
           dtype=tf.float64,
           name='two')),
      )
  def testIsNotCompatibleWith(self, v1, v2):
    self.assertFalse(v1.is_compatible_with(v2))
    self.assertFalse(v2.is_compatible_with(v1))

  @parameterized.named_parameters(
      ('TransformedVariableBijector',
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float32),
           transform_or_spec=_make_bijector_spec(
               tfb.Shift, [[2.]], use_variable=True, variable_shape=[1, 1])),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float32),
           transform_or_spec=_make_bijector_spec(
               tfb.Shift, [[3.]], use_variable=True, variable_shape=[1, None])),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float32),
           transform_or_spec=_make_bijector_spec(
               tfb.Shift, [[3.]], use_variable=True, variable_shape=[1, None]))
       ),
      ('TransformedVariableCallable',
       _make_transformed_variable_spec(
           input_spec=resource_variable_ops.VariableSpec([4, 2], tf.float32),
           transform_or_spec=tf.math.sigmoid),
       _make_transformed_variable_spec(
           input_spec=resource_variable_ops.VariableSpec(None, tf.float32),
           transform_or_spec=tf.math.sigmoid),
       _make_transformed_variable_spec(
           input_spec=resource_variable_ops.VariableSpec(None, tf.float32),
           transform_or_spec=tf.math.sigmoid))
      )
  def testMostSpecificCompatibleType(self, v1, v2, expected):
    self.assertEqual(v1.most_specific_compatible_type(v2), expected)
    self.assertEqual(v2.most_specific_compatible_type(v1), expected)

  @parameterized.named_parameters(
      ('DifferentDtypes',
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([], tf.float32),
           transform_or_spec=tfb.Sigmoid()._type_spec,
           dtype=tf.float64),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([], tf.float32),
           transform_or_spec=tfb.Sigmoid()._type_spec)),
      ('DifferentCallables',
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float64),
           transform_or_spec=tf.math.sigmoid,
           dtype=tf.float64,
           name='one'),
       _make_transformed_variable_spec(
           input_spec=tf.TensorSpec([4, 2], tf.float64),
           transform_or_spec=tf.math.softplus,
           dtype=tf.float64,
           name='two')),
      )
  def testMostSpecificCompatibleTypeException(self, v1, v2):
    with self.assertRaises(ValueError):
      v1.most_specific_compatible_type(v2)
    with self.assertRaises(ValueError):
      v2.most_specific_compatible_type(v1)

  def testRepr(self):
    spec = _make_transformed_variable_spec(
        input_spec=tf.TensorSpec([4, 2], tf.float32),
        transform_or_spec=tfb.Sigmoid(validate_args=True)._type_spec,
        dtype=tf.float64)
    expected = (
        "_TransformedVariableSpec(input_spec=TensorSpec(shape=(4, 2), "
        "dtype=tf.float32, name=None), "
        "transform_or_spec=Sigmoid_ACTTypeSpec(2, {}, "
        "{'low': None, 'high': None, 'validate_args': True}, ('name',), (), "
        "{}), dtype=<dtype: 'float64'>, name=None)")
    self.assertEqual(repr(spec), expected)


if __name__ == '__main__':
  tf.test.main()
