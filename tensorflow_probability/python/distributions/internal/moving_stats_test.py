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

rng = np.random.RandomState(0)


class MovingReduceMeanVarianceTest(tf.test.TestCase):

  def test_assign_moving_mean_variance(self):
    tf.reset_default_graph()
    shape = [1, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])
    # Start "x" out with this mean.
    mean_var = tf.Variable(tf.zeros_like(true_mean))
    variance_var = tf.Variable(tf.ones_like(true_stddev))
    x = tf.random_normal(shape, dtype=np.float64, seed=0)
    x = true_stddev * x + true_mean
    ema, emv = moving_stats.assign_moving_mean_variance(
        mean_var, variance_var, x, decay=0.99)

    self.assertEqual(ema.dtype.base_dtype, tf.float64)
    self.assertEqual(emv.dtype.base_dtype, tf.float64)

    # Run 1000 updates; moving averages should be near the true values.
    self.evaluate(tf.global_variables_initializer())
    for _ in range(2000):
      self.evaluate([ema, emv])

    [mean_var_, variance_var_, ema_,
     emv_] = self.evaluate([mean_var, variance_var, ema, emv])
    # Test that variables are passed-through.
    self.assertAllEqual(mean_var_, ema_)
    self.assertAllEqual(variance_var_, emv_)
    # Test that values are as expected.
    self.assertAllClose(true_mean, ema_, rtol=0.005, atol=0.015)
    self.assertAllClose(true_stddev**2., emv_, rtol=0.06, atol=0.)

    # Change the mean, var then update some more. Moving averages should
    # re-converge.
    self.evaluate([
        mean_var.assign(np.array([[-1., 2.]])),
        variance_var.assign(np.array([[2., 1.]])),
    ])
    for _ in range(2000):
      self.evaluate([ema, emv])

    [mean_var_, variance_var_, ema_,
     emv_] = self.evaluate([mean_var, variance_var, ema, emv])
    # Test that variables are passed-through.
    self.assertAllEqual(mean_var_, ema_)
    self.assertAllEqual(variance_var_, emv_)
    # Test that values are as expected.
    self.assertAllClose(true_mean, ema_, rtol=0.005, atol=0.015)
    self.assertAllClose(true_stddev**2., emv_, rtol=0.1, atol=0.)

  def test_moving_mean_variance(self):
    tf.reset_default_graph()
    shape = [1, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])
    # Start "x" out with this mean.
    x = tf.random_normal(shape, dtype=np.float64, seed=0)
    x = true_stddev * x + true_mean
    ema, emv = moving_stats.moving_mean_variance(x, decay=0.99)

    self.assertEqual(ema.dtype.base_dtype, tf.float64)
    self.assertEqual(emv.dtype.base_dtype, tf.float64)

    # Run 1000 updates; moving averages should be near the true values.
    self.evaluate(tf.global_variables_initializer())
    for _ in range(2000):
      self.evaluate([ema, emv])

    [ema_, emv_] = self.evaluate([ema, emv])
    self.assertAllClose(true_mean, ema_, rtol=0.005, atol=0.015)
    self.assertAllClose(true_stddev**2., emv_, rtol=0.06, atol=0.)


class MovingLogExponentialMovingMeanExpTest(tf.test.TestCase):

  def test_assign_log_moving_mean_exp(self):
    tf.reset_default_graph()
    shape = [1, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])
    decay = 0.99
    # Start "x" out with this mean.
    x = tf.random_normal(shape, dtype=np.float64, seed=0)
    x = true_stddev * x + true_mean
    log_mean_exp_var = tf.Variable(tf.zeros_like(true_mean))
    self.evaluate(tf.global_variables_initializer())
    log_mean_exp = moving_stats.assign_log_moving_mean_exp(
        log_mean_exp_var, x, decay=decay)
    expected_ = np.zeros_like(true_mean)
    for _ in range(2000):
      x_, log_mean_exp_ = self.evaluate([x, log_mean_exp])
      expected_ = np.log(decay * np.exp(expected_) + (1 - decay) * np.exp(x_))
      self.assertAllClose(expected_, log_mean_exp_, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
  tf.test.main()
