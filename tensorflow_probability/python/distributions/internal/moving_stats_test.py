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
"""Tests for computing moving-average statistics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions.internal import moving_stats
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class MovingReduceMeanVarianceTest(tf.test.TestCase):

  def test_assign_moving_mean_variance(self):
    shape = [1, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])
    # Start "x" out with this mean.
    mean_var = tf.compat.v2.Variable(tf.zeros_like(true_mean))
    variance_var = tf.compat.v2.Variable(tf.ones_like(true_stddev))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    def body(it):
      x = tf.random.normal(shape, dtype=np.float64, seed=17)
      x = true_stddev * x + true_mean
      ema, emv = moving_stats.assign_moving_mean_variance(
          mean_var, variance_var, x, decay=0.999)
      self.assertEqual(ema.dtype.base_dtype, tf.float64)
      self.assertEqual(emv.dtype.base_dtype, tf.float64)
      with tf.control_dependencies([ema, emv]):
        return it + 1

    def cond(it):
      return it < 10000

    # Run 10000 updates; moving averages should be near the true values.
    self.evaluate(tf.while_loop(
        cond=cond, body=body, loop_vars=[0], parallel_iterations=1))

    [mean_var_, variance_var_] = self.evaluate([mean_var, variance_var])
    # Test that values are as expected.
    # What tolerance should we set for the mean?  The expential moving average
    # of iid Gaussians is itself a Gaussian with the same mean.  If you assume
    # the average has converged (i.e., forgotten its initialization), then you
    # can work out that the variance is multiplied by a factor of
    # (1-decay)/(1+decay); that is, the effective sample size is
    # (1+decay)/(1-decay).  To assure low flakiness, set the tolerance
    # to > 4 * stddev.  We compute:
    # `4 * stddev = 4 * true_stddev * sqrt((1-decay)/(1+decay))
    #             = 4 * true_stddev * sqrt(0.001/1.999)
    #             = 4 * [1.1, 0.5]  * 0.0223
    #             < 0.1`
    # so `atol=0.15` ensures failures are highly unlikely.
    self.assertAllClose(true_mean, mean_var_, atol=0.15)
    # The tolerance for the variance is more annoying to derive.  The handwave
    # justiying `atol=0.15` goes like this: The variance is the mean of the
    # square; so one would think the variance of the estimator of the variance
    # is around the square of the variance.  Since the variance is around 1,
    # that square doesn't matter, and the same tolerance should do.  </handwave>
    self.assertAllClose(true_stddev**2., variance_var_, atol=0.15)

    # Change the mean, var then update some more. Moving averages should
    # re-converge.
    self.evaluate([
        mean_var.assign(np.array([[-1., 2.]])),
        variance_var.assign(np.array([[2., 1.]])),
    ])
    # Run 10000 updates; moving averages should be near the true values.
    self.evaluate(tf.while_loop(
        cond=cond, body=body, loop_vars=[0], parallel_iterations=1))

    [mean_var_, variance_var_] = self.evaluate([mean_var, variance_var])
    # Test that values are as expected.
    self.assertAllClose(true_mean, mean_var_, atol=0.15)
    self.assertAllClose(true_stddev**2., variance_var_, atol=0.15)


@test_util.run_all_in_graph_and_eager_modes
class MovingLogExponentialMovingMeanExpTest(tf.test.TestCase):

  def test_assign_log_moving_mean_exp(self):
    shape = [1, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])
    decay = 0.99
    # Start "x" out with this mean.
    log_mean_exp_var = tf.compat.v2.Variable(tf.zeros_like(true_mean))
    expected_var = tf.compat.v2.Variable(tf.zeros_like(true_mean))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    def body(it):
      x = tf.random.normal(shape, dtype=np.float64, seed=0)
      x = true_stddev * x + true_mean
      log_mean_exp = moving_stats.assign_log_moving_mean_exp(
          log_mean_exp_var, x, decay=decay)
      expected = tf.math.log(
          decay * tf.math.exp(expected_var) + (1 - decay) * tf.math.exp(x))
      expected = tf.compat.v1.assign(expected_var, expected)
      relerr = tf.abs((log_mean_exp - expected) / expected)
      op = tf.compat.v1.assert_less(relerr, np.array(1e-6, dtype=np.float64))
      with tf.control_dependencies([op]):
        return it + 1

    def cond(it):
      return it < 2000

    self.evaluate(tf.while_loop(
        cond=cond, body=body, loop_vars=[0], parallel_iterations=1))


if __name__ == "__main__":
  tf.test.main()
