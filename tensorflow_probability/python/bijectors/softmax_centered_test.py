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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


rng = np.random.RandomState(42)


@test_util.test_all_tf_execution_regimes
class SoftmaxCenteredBijectorTest(test_util.TestCase):
  """Tests correctness of the Y = g(X) = exp(X) / sum(exp(X)) transformation."""

  def testBijectorVector(self):
    softmax = tfb.SoftmaxCentered()
    self.assertStartsWith(softmax.name, 'softmax_centered')
    x = np.log([[2., 3, 4], [4., 8, 12]])
    y = [[0.2, 0.3, 0.4, 0.1], [0.16, 0.32, 0.48, 0.04]]
    self.assertAllClose(y, self.evaluate(softmax.forward(x)))
    self.assertAllClose(x, self.evaluate(softmax.inverse(y)))
    self.assertAllClose(
        -np.sum(np.log(y), axis=1) - 0.5 * np.log(4.)[np.newaxis, ...],
        self.evaluate(softmax.inverse_log_det_jacobian(y, event_ndims=1)),
        atol=0.,
        rtol=1e-7)
    self.assertAllClose(
        self.evaluate(-softmax.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(softmax.forward_log_det_jacobian(x, event_ndims=1)),
        atol=0.,
        rtol=1e-7)

  def testBijectorUnknownShape(self):
    softmax = tfb.SoftmaxCentered()
    self.assertStartsWith(softmax.name, 'softmax_centered')
    x_ = np.log([[2., 3, 4], [4., 8, 12]]).astype(np.float32)
    y_ = np.array(
        [[0.2, 0.3, 0.4, 0.1], [0.16, 0.32, 0.48, 0.04]], dtype=np.float32)
    x = tf1.placeholder_with_default(x_, shape=[2, None])
    y = tf1.placeholder_with_default(y_, shape=[2, None])
    self.assertAllClose(y_, self.evaluate(softmax.forward(x)))
    self.assertAllClose(x_, self.evaluate(softmax.inverse(y)))
    self.assertAllClose(
        -np.sum(np.log(y_), axis=1) - 0.5 * np.log(4.)[np.newaxis, ...],
        self.evaluate(softmax.inverse_log_det_jacobian(y, event_ndims=1)),
        atol=0.,
        rtol=1e-7)
    self.assertAllClose(
        -self.evaluate(softmax.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(softmax.forward_log_det_jacobian(x, event_ndims=1)),
        atol=0.,
        rtol=1e-7)

  def testShapeGetters(self):
    x = tf.TensorShape([4])
    y = tf.TensorShape([5])
    bijector = tfb.SoftmaxCentered(validate_args=True)
    self.assertAllEqual(y, bijector.forward_event_shape(x))
    self.assertAllEqual(
        tensorshape_util.as_list(y),
        self.evaluate(
            bijector.forward_event_shape_tensor(tensorshape_util.as_list(x))))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(
        tensorshape_util.as_list(x),
        self.evaluate(
            bijector.inverse_event_shape_tensor(tensorshape_util.as_list(y))))

  def testShapeGetersWithBatchShape(self):
    x = tf.TensorShape([2, 4])
    y = tf.TensorShape([2, 5])
    bijector = tfb.SoftmaxCentered(validate_args=True)
    self.assertAllEqual(y, bijector.forward_event_shape(x))
    self.assertAllEqual(
        tensorshape_util.as_list(y),
        self.evaluate(
            bijector.forward_event_shape_tensor(tensorshape_util.as_list(x))))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(
        tensorshape_util.as_list(x),
        self.evaluate(
            bijector.inverse_event_shape_tensor(tensorshape_util.as_list(y))))

  def testShapeGettersWithDynamicShape(self):
    x = tf1.placeholder_with_default([2, 4], shape=None)
    y = tf1.placeholder_with_default([2, 5], shape=None)
    bijector = tfb.SoftmaxCentered(validate_args=True)
    self.assertAllEqual(
        [2, 5], self.evaluate(bijector.forward_event_shape_tensor(x)))
    self.assertAllEqual(
        [2, 4], self.evaluate(bijector.inverse_event_shape_tensor(y)))

  def testBijectiveAndFinite(self):
    softmax = tfb.SoftmaxCentered()
    x = np.linspace(-50, 50, num=10).reshape(5, 2).astype(np.float32)
    # Make y values on the simplex with a wide range.
    y_0 = np.ones(5).astype(np.float32)
    y_1 = (1e-5 * rng.rand(5)).astype(np.float32)
    y_2 = (1e1 * rng.rand(5)).astype(np.float32)
    y = np.array([y_0, y_1, y_2])
    y /= y.sum(axis=0)
    y = y.T  # y.shape = [5, 3]
    bijector_test_util.assert_bijective_and_finite(
        softmax, x, y, eval_func=self.evaluate, event_ndims=1)

  def testAssertsValidArgToInverse(self):
    softmax = tfb.SoftmaxCentered(validate_args=True)
    with self.assertRaisesOpError('must sum to `1`'):
      self.evaluate(softmax.inverse([0.03, 0.7, 0.4]))

    with self.assertRaisesOpError(
        'must be less than or equal to `1`|must sum to `1`'):
      self.evaluate(softmax.inverse([0.06, 0.4, 1.02]))

    with self.assertRaisesOpError('must be non-negative'):
      self.evaluate(softmax.inverse([0.4, 0.5, 0.3, -0.2]))

  @test_util.numpy_disable_gradient_test
  def testTheoreticalFldj(self):
    softmax = tfb.SoftmaxCentered()
    x = np.linspace(-15, 15, num=10).reshape(5, 2).astype(np.float64)

    fldj = softmax.forward_log_det_jacobian(x, event_ndims=1)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        softmax, x, event_ndims=1)
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)


if __name__ == '__main__':
  tf.test.main()
