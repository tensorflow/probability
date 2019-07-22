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

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class DeferredTensorTest(tf.test.TestCase):

  def test_docstring_example(self):
    trainable_normal = tfd.Normal(
        loc=tf.Variable(0.),
        scale=tfp.util.DeferredTensor(
            tf.math.exp, tf.Variable(0., name='raw_scale')))
    with tf.GradientTape() as tape:
      negloglik = -trainable_normal.log_prob(0.5)
    g = tape.gradient(negloglik, trainable_normal.trainable_variables)
    self.evaluate(tf1.global_variables_initializer())
    self.assertEqual((-1. / (2. * 1.), (1. - 0.5**2) / 1.), self.evaluate(g))
    self.assertIsInstance(trainable_normal.scale, tfp.util.DeferredTensor)
    self.assertEqual(1., self.evaluate(trainable_normal.scale**2.))
    # For speed, we don't bother testing the optimization part of the example.

  def test_properties(self):
    x = tfp.util.DeferredTensor(tf.math.exp, tf.Variable(0.))
    self.evaluate(tf1.global_variables_initializer())
    self.assertEqual((), x.shape)
    self.assertEqual(tf.float32, x.dtype)
    if not tf.executing_eagerly():
      self.assertStartsWith(x.name, 'exp_variable')
    self.assertEqual(
        '<DeferredTensor: dtype=float32, shape=[], fn=exp>',
        repr(x))


@test_util.run_all_in_graph_and_eager_modes
class DeferredTensorBehavesLikeTensorTest(tf.test.TestCase,
                                          parameterized.TestCase):

  def testArrayPriority(self):
    x = tfp.util.DeferredTensor(tf.math.exp, tf.Variable(0.))
    y = np.array(3., dtype=np.float32)
    self.evaluate(tf1.global_variables_initializer())
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
    self.evaluate(tf1.global_variables_initializer())
    self.assertAllClose([op(2., 1.), op(1., 3.)],
                        self.evaluate([y1, y2]),
                        atol=0., rtol=1e-5)

  @parameterized.parameters(
      operator.abs,
      operator.neg,
  )
  def testOperatorUnary(self, op):
    x = tfp.util.DeferredTensor(tf.identity, tf.Variable(-1.))
    self.evaluate(tf1.global_variables_initializer())
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
    self.evaluate(tf1.global_variables_initializer())
    self.assertAllEqual([op(True, x_), op(x_, False)],
                        self.evaluate([y1, y2]))

  # `~` is the only supported unary logical operator.
  # Note: 'boolean operator' is distinct from 'logical operator'. (The former
  # generally being not overrideable.)
  def testOperatorUnaryLogical(self):
    x = tfp.util.DeferredTensor(
        lambda x: tf.cast(x, tf.bool), tf.Variable(0), dtype=tf.bool)
    self.evaluate(tf1.global_variables_initializer())
    self.assertAllEqual(*self.evaluate([~tf.convert_to_tensor(x), ~x]))

  def testOperatorBoolNonzero(self):
    x = tfp.util.DeferredTensor(
        lambda x: tf.cast(x, tf.bool), tf.Variable(0.), dtype=tf.bool)
    self.evaluate(tf1.global_variables_initializer())
    with self.assertRaises(TypeError):
      _ = not x

  def testOperatorGetitem(self):
    x = tfp.util.DeferredTensor(tf.math.exp, tf.Variable([1., 2.]))
    self.evaluate(tf1.global_variables_initializer())
    self.assertAllClose([np.exp(1.)], self.evaluate(x[:1]), atol=0., rtol=1e-5)

  def testOperatorIter(self):
    x_ = [0., 1.]
    x = tfp.util.DeferredTensor(tf.math.exp, tf.Variable(x_))
    self.evaluate(tf1.global_variables_initializer())
    if tf.executing_eagerly():
      for expected_, actual_ in zip(x_, iter(x)):
        self.assertNear(np.exp(expected_), actual_.numpy(), err=1e-5)
    else:
      with self.assertRaises(TypeError):
        for _ in iter(x):
          pass


if __name__ == '__main__':
  tf.test.main()
