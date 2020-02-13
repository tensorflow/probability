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
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class MovingReduceMeanVarianceTest(test_util.TestCase):

  def test_assign_moving_mean_variance(self):
    shape = [3, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])

    mean_var = tf.Variable(tf.zeros_like(true_mean))
    variance_var = tf.Variable(tf.zeros_like(true_stddev))
    zero_debias_count = tf.Variable(0)
    self.evaluate([
        mean_var.initializer,
        variance_var.initializer,
        zero_debias_count.initializer,
    ])

    def body(it):
      x = tf.random.normal(shape, dtype=np.float64, seed=17)
      x = true_stddev * x + true_mean
      ema, emv = tfp.stats.assign_moving_mean_variance(
          x, mean_var, variance_var, zero_debias_count, decay=0.999, axis=-2)
      with tf.control_dependencies([ema, emv]):
        return [it + 1]

    # Run 10000 updates; moving averages should be near the true values.
    run_op = tf.while_loop(
        cond=lambda it: it < 10000,
        body=body,
        loop_vars=[0],
        parallel_iterations=1)

    with tf.control_dependencies(run_op):
      [
          final_unbiased_mean,
          final_ubiased_variance,
      ] = tfp.stats.moving_mean_variance_zero_debiased(
          mean_var,
          variance_var,
          zero_debias_count,
          decay=0.999)
      final_biased_mean = tf.convert_to_tensor(mean_var)
      final_biased_variance = tf.convert_to_tensor(variance_var)
      read_zero_debias_count = tf.convert_to_tensor(zero_debias_count)

    [
        final_biased_mean_,
        final_biased_variance_,
        final_unbiased_mean_,
        final_unbiased_variance_,
        read_zero_debias_count_,
    ] = self.evaluate([
        final_biased_mean,
        final_biased_variance,
        final_unbiased_mean,
        final_ubiased_variance,
        read_zero_debias_count,
    ])

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
    self.assertAllClose(true_mean, final_biased_mean_, atol=0.15)
    # The tolerance for the variance is more annoying to derive.  The handwave
    # justiying `atol=0.15` goes like this: The variance is the mean of the
    # square; so one would think the variance of the estimator of the variance
    # is around the square of the variance.  Since the variance is around 1,
    # that square doesn't matter, and the same tolerance should do.  </handwave>
    self.assertAllClose(true_stddev**2., final_biased_variance_, atol=0.15)

    self.assertEqual(10000, read_zero_debias_count_)
    self.assertAllClose(final_biased_mean_, final_unbiased_mean_,
                        atol=0.01, rtol=0.01)
    self.assertAllClose(final_biased_variance_, final_unbiased_variance_,
                        atol=0.01, rtol=0.01)

    # Change the mean, var then update some more. Moving averages should
    # re-converge.
    self.evaluate([
        mean_var.assign(np.array([[-1., 2.]])),
        variance_var.assign(np.array([[2., 1.]])),
    ])
    # Run 10000 updates; moving averages should be near the true values.
    run_op = tf.while_loop(
        cond=lambda it: it < 10000,
        body=body,
        loop_vars=[0],
        parallel_iterations=1)
    self.evaluate(run_op)

    [final_biased_mean_, final_biased_variance_] = self.evaluate(
        [mean_var, variance_var])
    # Test that values are as expected.
    self.assertAllClose(true_mean, final_biased_mean_, atol=0.15)
    self.assertAllClose(true_stddev**2., final_biased_variance_, atol=0.15)


@test_util.test_all_tf_execution_regimes
class MovingLogExponentialMovingMeanExpTest(test_util.TestCase):

  def test_assign_log_moving_mean_exp(self):
    shape = [1, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])
    decay = 0.99
    log_mean_exp_var = tf.Variable(tf.zeros_like(true_mean))
    expected_var = tf.Variable(tf.zeros_like(true_mean))
    self.evaluate([log_mean_exp_var.initializer, expected_var.initializer])

    def body(it):
      x = tf.random.normal(shape, dtype=np.float64, seed=0)
      x = true_stddev * x + true_mean
      log_mean_exp = tfp.stats.assign_log_moving_mean_exp(
          x, log_mean_exp_var, decay=decay)
      expected = tf.math.log(
          decay * tf.math.exp(expected_var) + (1 - decay) * tf.math.exp(x))
      expected = expected_var.assign(expected)
      relerr = tf.abs((log_mean_exp - expected) / expected)
      op = tf.debugging.assert_less(relerr, np.array(1e-6, dtype=np.float64))
      with tf.control_dependencies([op]):
        return [it + 1]

    self.evaluate(tf.while_loop(
        cond=lambda it: it < 2000,
        body=body,
        loop_vars=[0],
        parallel_iterations=1))


if __name__ == '__main__':
  tf.test.main()
