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
"""Tests CategoricalToDiscrete bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import categorical_to_discrete
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class CategoricalToDiscreteTest(test_util.TestCase):

  def testUnsortedValuesRaises(self):
    with self.assertRaisesOpError('map_values is not strictly increasing'):
      bijector = categorical_to_discrete.CategoricalToDiscrete(
          map_values=[1, 3, 2], validate_args=True)
      self.evaluate(bijector.forward([0, 1, 2]))

  def testMapValuesRankNotEqualToOneRaises(self):
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             'Rank of map_values must be 1'):
      bijector = categorical_to_discrete.CategoricalToDiscrete(
          map_values=[[1, 2], [3, 4]], validate_args=True)
      self.evaluate(bijector.map_values)

  def testMapValuesSizeZeroRaises(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Size of map_values must be greater than 0'):
      bijector = categorical_to_discrete.CategoricalToDiscrete(
          map_values=[], validate_args=True)
      self.evaluate(bijector.map_values)

  def testBijectorForward(self):
    bijector = categorical_to_discrete.CategoricalToDiscrete(
        map_values=[0.1, 0.2, 0.3, 0.4], validate_args=True)
    self.assertAllClose([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                        self.evaluate(
                            bijector.forward([[0, 1, 2, 3], [3, 2, 1, 0]])))

  def testBijectorForwardOutOfBoundIndicesRaises(self):
    with self.assertRaisesOpError('indices out of bound'):
      bijector = categorical_to_discrete.CategoricalToDiscrete(
          map_values=[0.1, 0.2, 0.3, 0.4], validate_args=True)
      self.evaluate(bijector.forward([5]))

  def testBijectorInverse(self):
    bijector = categorical_to_discrete.CategoricalToDiscrete(
        map_values=[0.1, 0.2, 0.3, 0.4], validate_args=True)
    self.assertAllEqual([[3, 3, 3], [0, 1, 2]],
                        self.evaluate(
                            bijector.inverse([[0.400001, 0.4, 0.399999],
                                              [0.1, 0.2, 0.3]])))

  def testBijectorInverseValueNotFoundRaises(self):
    with self.assertRaisesOpError('inverse value not found'):
      bijector = categorical_to_discrete.CategoricalToDiscrete(
          map_values=[0.1, 0.2, 0.3, 0.4], validate_args=True)
      self.evaluate(bijector.inverse([0.21, 0.4]))

  def testInverseLogDetJacobian(self):
    bijector = categorical_to_discrete.CategoricalToDiscrete(
        map_values=[0.1, 0.2], validate_args=True)
    self.assertAllClose(
        0,
        self.evaluate(
            bijector.inverse_log_det_jacobian([0.1, 0.2], event_ndims=0)))

  def testBijectiveAndFinite32bit(self):
    x = np.arange(100).astype(np.int32)
    y = np.logspace(-10, 10, 100).astype(np.float32)
    bijector = categorical_to_discrete.CategoricalToDiscrete(map_values=y)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0)

  def testBijectiveAndFinite16bit(self):
    x = np.arange(100).astype(np.int32)
    y = np.logspace(-5, 4, 100).astype(np.float16)
    bijector = categorical_to_discrete.CategoricalToDiscrete(map_values=y)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0)

  def testVariableGradients(self):
    map_values = tf.Variable([0.3, 0.5])
    b = categorical_to_discrete.CategoricalToDiscrete(map_values=map_values,
                                                      validate_args=True)
    with tf.GradientTape() as tape:
      y = tf.reduce_sum(b.forward([0, 1]))
    grads = tape.gradient(y, [map_values])
    self.assertAllNotNone(grads)

  def testNonVariableGradients(self):
    map_values = tf.convert_to_tensor([0.3, 0.5])
    b = categorical_to_discrete.CategoricalToDiscrete(map_values=map_values,
                                                      validate_args=True)
    with tf.GradientTape() as tape:
      tape.watch(map_values)
      y = tf.reduce_sum(b.forward([0, 1]))
    grads = tape.gradient(y, [map_values])
    self.assertAllNotNone(grads)

  def testModifiedMapValuesIncreasingAssertion(self):
    map_values = tf.Variable([0.1, 0.2])
    b = categorical_to_discrete.CategoricalToDiscrete(map_values=map_values,
                                                      validate_args=True)
    self.evaluate(map_values.initializer)
    with self.assertRaisesOpError('map_values is not strictly increasing.'):
      with tf.control_dependencies([map_values.assign([0.2, 0.1])]):
        self.evaluate(b.forward([0, 1]))


if __name__ == '__main__':
  tf.test.main()
