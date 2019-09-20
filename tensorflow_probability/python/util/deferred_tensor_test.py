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
from tensorflow_probability.python.internal import test_case

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class DeferredTensorTest(test_case.TestCase):

  def test_docstring_example(self):
    trainable_normal = tfd.Normal(
        loc=tf.Variable(0.),
        scale=tfp.util.DeferredTensor(
            tf.math.exp, tf.Variable(0., name='raw_scale')))
    with tf.GradientTape() as tape:
      negloglik = -trainable_normal.log_prob(0.5)
    g = tape.gradient(negloglik, trainable_normal.trainable_variables)
    self.evaluate([v.initializer for v in trainable_normal.trainable_variables])
    self.assertEqual((-1. / (2. * 1.), (1. - 0.5**2) / 1.), self.evaluate(g))
    self.assertIsInstance(trainable_normal.scale, tfp.util.DeferredTensor)
    self.assertEqual(1., self.evaluate(trainable_normal.scale**2.))
    # For speed, we don't bother testing the optimization part of the example.

  def test_properties(self):
    x = tfp.util.DeferredTensor(tf.math.exp, tf.Variable(0.))
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

  def test_variable_shape_changes(self):
    v = tf.Variable(np.zeros((3, 2, 3)), shape=tf.TensorShape((None, 2, None)))
    self.evaluate(v.initializer)
    x = tfp.util.DeferredTensor(tf.math.softmax, v)

    self.assertAllEqual((None, 2, None), x.shape.as_list())
    self.assertAllEqual((3, 2, 3), self.evaluate(tf.shape(x)))

    with tf.control_dependencies([v.assign(np.zeros((1, 2, 4)))]):
      self.assertAllEqual((1, 2, 4), self.evaluate(tf.shape(x)))
      self.assertAllEqual((None, 2, None), x.shape.as_list())

  def test_variable_rank_changes(self):
    def f(x):
      shape = tf.shape(x)
      return tf.reshape(
          x, tf.concat([[2, (shape[0] * shape[1]) // 2], shape[2:]], axis=0))

    v = tf.Variable(np.zeros((3, 4, 3)), shape=tf.TensorShape(None))
    self.evaluate(v.initializer)
    x = tfp.util.DeferredTensor(f, v)

    self.assertIsNone(x.shape.rank)
    self.assertAllEqual((2, 6, 3), self.evaluate(tf.shape(x)))

    with tf.control_dependencies([v.assign(np.zeros((4, 5, 1, 1)))]):
      self.assertAllEqual((2, 10, 1, 1), self.evaluate(tf.shape(x)))
      self.assertIsNone(x.shape.rank)

  def test_from_bijector_with_inverted_assignment(self):
    x = tfp.util.DeferredTensor(tfb.Pad(validate_args=True),
                                tf.Variable([[1.], [2.], [3.]]))
    self.assertIs(tf.float32, x.dtype)
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


@test_util.run_all_in_graph_and_eager_modes
class TransformedVariableTest(test_case.TestCase):

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
    self.assertAllEqual([1., 2.], self.evaluate(d.stddev()))
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


@test_util.run_all_in_graph_and_eager_modes
class DeferredTensorBehavesLikeTensorTest(test_case.TestCase,
                                          parameterized.TestCase):

  def testArrayPriority(self):
    x = tfp.util.DeferredTensor(tf.math.exp, tf.Variable(0.))
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
    x = tfp.util.DeferredTensor(tf.math.exp, tf.Variable(0.))
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
    x = tfp.util.DeferredTensor(tf.identity, tf.Variable(-1.))
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
        lambda x: tf.cast(x, tf.bool), tf.Variable(0.), dtype=tf.bool)
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
        lambda x: tf.cast(x, tf.bool), tf.Variable(0), dtype=tf.bool)
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertAllEqual(*self.evaluate([~tf.convert_to_tensor(x), ~x]))

  def testOperatorBoolNonzero(self):
    x = tfp.util.DeferredTensor(
        lambda x: tf.cast(x, tf.bool), tf.Variable(0.), dtype=tf.bool)
    self.evaluate([v.initializer for v in x.trainable_variables])
    with self.assertRaises(TypeError):
      _ = not x

  def testOperatorGetitem(self):
    x = tfp.util.DeferredTensor(tf.math.exp, tf.Variable([1., 2.]))
    self.evaluate([v.initializer for v in x.trainable_variables])
    self.assertAllClose([np.exp(1.)], self.evaluate(x[:1]), atol=0., rtol=1e-5)

  def testOperatorIter(self):
    x_ = [0., 1.]
    x = tfp.util.DeferredTensor(tf.math.exp, tf.Variable(x_))
    self.evaluate([v.initializer for v in x.trainable_variables])
    if tf.executing_eagerly():
      for expected_, actual_ in zip(x_, iter(x)):
        self.assertNear(np.exp(expected_), actual_.numpy(), err=1e-5)
    else:
      with self.assertRaises(TypeError):
        for _ in iter(x):
          pass


if __name__ == '__main__':
  tf.test.main()
