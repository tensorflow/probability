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
import tensorflow_probability as tfp

from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow.python.framework import test_util

tfd = tfp.distributions
tfb = tfp.bijectors


class OrderedBijectorTest(tf.test.TestCase):
  """Tests correctness of the ordered transformation."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  @test_util.run_in_graph_and_eager_modes()
  def testBijectorVector(self):
    ordered = tfb.Ordered()
    self.assertEqual("ordered", ordered.name)
    x = np.asarray([[2., 3, 4], [4., 8, 13]])
    y = [[2., 0, 0], [4., np.log(4.), np.log(5.)]]
    self.assertAllClose(y, self.evaluate(ordered.forward(x)))
    self.assertAllClose(x, self.evaluate(ordered.inverse(y)))
    self.assertAllClose(
        np.sum(np.asarray(y)[..., 1:], axis=-1),
        self.evaluate(ordered.inverse_log_det_jacobian(y, event_ndims=1)),
        atol=0.,
        rtol=1e-7)
    self.assertAllClose(
        self.evaluate(-ordered.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(ordered.forward_log_det_jacobian(x, event_ndims=1)),
        atol=0.,
        rtol=1e-7)

  def testBijectorUnknownShape(self):
    with self.test_session():
      ordered = tfb.Ordered()
      self.assertEqual("ordered", ordered.name)
      x = tf.placeholder(shape=[2, None], dtype=tf.float32)
      real_x = np.asarray([[2., 3, 4], [4., 8, 13]])
      y = tf.placeholder(shape=[2, None], dtype=tf.float32)
      real_y = [[2., 0, 0], [4., np.log(4.), np.log(5.)]]
      self.assertAllClose(real_y, ordered.forward(x).eval(
          feed_dict={x: real_x}))
      self.assertAllClose(real_x, ordered.inverse(y).eval(
          feed_dict={y: real_y}))
      self.assertAllClose(
          np.sum(np.asarray(real_y)[..., 1:], axis=-1),
          ordered.inverse_log_det_jacobian(y, event_ndims=1).eval(
              feed_dict={y: real_y}),
          atol=0.,
          rtol=1e-7)
      self.assertAllClose(
          -ordered.inverse_log_det_jacobian(y, event_ndims=1).eval(
              feed_dict={y: real_y}),
          ordered.forward_log_det_jacobian(x, event_ndims=1).eval(
              feed_dict={x: real_x}),
          atol=0.,
          rtol=1e-7)

  @test_util.run_in_graph_and_eager_modes()
  def testShapeGetters(self):
    x = tf.TensorShape([4])
    y = tf.TensorShape([4])
    bijector = tfb.Ordered(validate_args=True)
    self.assertAllEqual(y, bijector.forward_event_shape(x))
    self.assertAllEqual(y.as_list(),
                        self.evaluate(bijector.forward_event_shape_tensor(
                            x.as_list())))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(x.as_list(),
                        self.evaluate(bijector.inverse_event_shape_tensor(
                            y.as_list())))

  def testBijectiveAndFinite(self):
    ordered = tfb.Ordered()
    x = np.sort(self._rng.randn(3, 10), axis=-1).astype(np.float32)
    y = (self._rng.randn(3, 10)).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        ordered, x, y, eval_func=self.evaluate, event_ndims=1)


if __name__ == "__main__":
  tf.test.main()
