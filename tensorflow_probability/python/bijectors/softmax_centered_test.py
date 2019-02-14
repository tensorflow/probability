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
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


rng = np.random.RandomState(42)


@test_util.run_all_in_graph_and_eager_modes
class SoftmaxCenteredBijectorTest(tf.test.TestCase):
  """Tests correctness of the Y = g(X) = exp(X) / sum(exp(X)) transformation."""

  def testBijectorVector(self):
    softmax = tfb.SoftmaxCentered()
    self.assertEqual("softmax_centered", softmax.name)
    x = np.log([[2., 3, 4], [4., 8, 12]])
    y = [[0.2, 0.3, 0.4, 0.1], [0.16, 0.32, 0.48, 0.04]]
    self.assertAllClose(y, self.evaluate(softmax.forward(x)))
    self.assertAllClose(x, self.evaluate(softmax.inverse(y)))
    self.assertAllClose(
        -np.sum(np.log(y), axis=1),
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
    self.assertEqual("softmax_centered", softmax.name)
    x_ = np.log([[2., 3, 4], [4., 8, 12]]).astype(np.float32)
    y_ = np.array(
        [[0.2, 0.3, 0.4, 0.1], [0.16, 0.32, 0.48, 0.04]], dtype=np.float32)
    x = tf.compat.v1.placeholder_with_default(x_, shape=[2, None])
    y = tf.compat.v1.placeholder_with_default(y_, shape=[2, None])
    self.assertAllClose(y_, self.evaluate(softmax.forward(x)))
    self.assertAllClose(x_, self.evaluate(softmax.inverse(y)))
    self.assertAllClose(
        -np.sum(np.log(y_), axis=1),
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
    self.assertAllEqual(y.as_list(),
                        self.evaluate(
                            bijector.forward_event_shape_tensor(
                                x.as_list())))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(x.as_list(),
                        self.evaluate(
                            bijector.inverse_event_shape_tensor(
                                y.as_list())))

  def testShapeGetersWithBatchShape(self):
    x = tf.TensorShape([2, 4])
    y = tf.TensorShape([2, 5])
    bijector = tfb.SoftmaxCentered(validate_args=True)
    self.assertAllEqual(y, bijector.forward_event_shape(x))
    self.assertAllEqual(y.as_list(),
                        self.evaluate(
                            bijector.forward_event_shape_tensor(
                                x.as_list())))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(x.as_list(),
                        self.evaluate(
                            bijector.inverse_event_shape_tensor(
                                y.as_list())))

  def testShapeGettersWithDynamicShape(self):
    x = tf.compat.v1.placeholder_with_default(input=[2, 4], shape=None)
    y = tf.compat.v1.placeholder_with_default(input=[2, 5], shape=None)
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


if __name__ == "__main__":
  tf.test.main()
