# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for TruncatedCauchy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class TruncatedCauchyTest(test_util.TestCase):

  def testBatchSampling(self):
    """Check (empirically) that different parameters in a batch are respected.
    """
    n = int(1e5)
    lb = [[-1.0, 9.0], [0., 8.]]
    ub = [[1.0, 11.0], [5., 20.]]
    dist = tfd.TruncatedCauchy(
        loc=[[0., 10.], [0., 10.]],
        scale=[[1., 1.], [5., 5.]],
        low=lb,
        high=ub,
        validate_args=True)
    x = self.evaluate(dist.sample(n, seed=test_util.test_seed()))
    self.assertEqual(x.shape, (n, 2, 2))

    empirical_lb = np.min(x, axis=0)
    self.assertAllClose(empirical_lb, lb, atol=0.1)
    empirical_ub = np.max(x, axis=0)
    self.assertAllClose(empirical_ub, ub, atol=0.1)

    self.assertAllClose(np.mean(x, axis=0), dist.mean(), rtol=1e-2, atol=1e-2)
    self.assertAllClose(np.var(x, axis=0), dist.variance(),
                        rtol=1e-2, atol=1e-2)

  def testSampleShape(self):
    loc = tf.zeros((1, 1, 3), dtype=tf.float64)
    scale = tf.ones((2, 3), dtype=tf.float64)
    low = -tf.ones((4, 1, 1), dtype=tf.float64)
    high = 1.0
    d = tfd.TruncatedCauchy(loc, scale, low, high)
    self.assertAllEqual([7, 2, 4, 2, 3],
                        d.sample([7, 2], seed=test_util.test_seed()).shape)

  def testMode(self):
    loc = np.array([0., 1., -0.5, 10., 2., -2], dtype=np.float32)
    scale = np.array([1., 1., 0.5, 3., 1.5, 0.2], dtype=np.float32)
    low = np.array([-1., 0., -0.9, 9.9, 0.1, -1.5], dtype=np.float32)
    high = np.array([1., 2., -0.5, 25., 1.9, -0.5], dtype=np.float32)
    dist = tfd.TruncatedCauchy(loc, scale, low, high, validate_args=True)
    self.assertAllClose([0., 1., -0.5, 10., 1.9, -1.5],
                        self.evaluate(dist.mode()))

  @parameterized.parameters(
      (0., 1., -1., 1.),
      (1., 1., 0., 2.),
      (-0.5, 0.5, -0.9, -0.4),
      (10., 3.0, 9.9, 25.),
      (2., 1.5, 0.1, 1.9),
      (-2., 0.2, -1.5, -0.5))
  def testLogProb(self, loc, scale, low, high):
    tc = tfd.TruncatedCauchy(loc, scale, low, high, validate_args=False)
    c = tfd.Cauchy(loc, scale)

    x = tf.concat([tf.range(low, high, 0.1), [high]], axis=0)
    self.assertAllClose(
        self.evaluate(c.log_prob(x) - tf.math.log(c.cdf(high) - c.cdf(low))),
        self.evaluate(tc.log_prob(x)),
        rtol=1e-5, atol=1e-5)

    self.assertAllEqual([-np.inf, -np.inf],
                        tc.log_prob([low - 0.1, high + 0.1]))

  @parameterized.parameters(
      (0., 1., -1., 1.),
      (1., 1., 0., 2.),
      (-0.5, 0.5, -0.9, -0.4),
      (10., 3.0, 9.9, 25.),
      (2., 1.5, 0.1, 1.9),
      (-2., 0.2, -1.5, -0.5))
  def testCdf(self, loc, scale, low, high):
    tc = tfd.TruncatedCauchy(loc, scale, low, high, validate_args=False)
    c = tfd.Cauchy(loc, scale)

    x = tf.concat([tf.range(low, high, 0.1), [high]], axis=0)
    self.assertAllClose(
        self.evaluate((c.cdf(x) - c.cdf(low)) / (c.cdf(high) - c.cdf(low))),
        tc.cdf(x),
        rtol=1e-5, atol=1e-5)

    self.assertAllClose(x, tc.quantile(tc.cdf(x)))

    self.assertAllEqual([0., 1.], tc.cdf([low - 0.1, high + 0.1]))

  @parameterized.parameters(
      (0., 1., -1., 1.),
      (1., 1., 0., 2.),
      (-0.5, 0.5, -0.9, -0.4),
      (10., 3.0, 9.9, 25.),
      (2., 1.5, 0.1, 1.9),
      (-2., 0.2, -1.5, -0.5))
  def testQuantile(self, loc, scale, low, high):
    tc = tfd.TruncatedCauchy(loc, scale, low, high, validate_args=True)
    c = tfd.Cauchy(np.float64(loc), scale)

    p = np.arange(0.025, 1., step=0.025)
    self.assertAllClose(p, tc.cdf(tc.quantile(p.astype(np.float32))))

    self.assertAllClose(
        c.quantile(p * (c.cdf(high) - c.cdf(low)) + c.cdf(low)),
        tc.quantile(p.astype(np.float32)))

    self.assertAllClose([low, high], tc.quantile([0., 1.]))

  @parameterized.parameters(np.float32, np.float64)
  def testSampleMean(self, dtype):
    seed_stream = test_util.test_seed_stream()
    loc = tf.random.normal([20], dtype=dtype, seed=seed_stream())
    scale = tf.random.uniform(
        [20], minval=0.5, maxval=2., dtype=dtype, seed=seed_stream())
    low = tf.random.uniform(
        [20], minval=-50., maxval=50., dtype=dtype, seed=seed_stream())
    high = tf.random.uniform(
        [20], minval=-50., maxval=50., dtype=dtype, seed=seed_stream())
    low, high = tf.math.minimum(low, high), tf.math.maximum(low, high)
    tc = tfd.TruncatedCauchy(loc, scale, low, high, validate_args=True)
    samples_, mean_ = self.evaluate([
        tc.sample(int(1e6), seed=seed_stream()), tc.mean()])
    self.assertAllClose(mean_, np.mean(samples_, axis=0), rtol=0.08)

  @parameterized.parameters(np.float32, np.float64)
  def testSampleVariance(self, dtype):
    seed_stream = test_util.test_seed_stream()
    loc = tf.random.normal([20], dtype=dtype, seed=seed_stream())
    scale = tf.random.uniform(
        [20], minval=0.5, maxval=2., dtype=dtype, seed=seed_stream())
    low = tf.random.uniform(
        [20], minval=-50., maxval=50., dtype=dtype, seed=seed_stream())
    high = tf.random.uniform(
        [20], minval=-50., maxval=50., dtype=dtype, seed=seed_stream())
    low, high = tf.math.minimum(low, high), tf.math.maximum(low, high)
    tc = tfd.TruncatedCauchy(loc, scale, low, high, validate_args=True)
    samples_, variance_ = self.evaluate([
        tc.sample(int(1e6), seed=seed_stream()), tc.variance()])
    self.assertAllClose(
        variance_, np.var(samples_, axis=0), rtol=0.1, atol=0.1)

  def testNegativeScaleFails(self):
    with self.assertRaisesOpError('`scale` must be positive'):
      dist = tfd.TruncatedCauchy(
          loc=0., scale=-0.1, low=-1.0, high=1.0, validate_args=True)
      self.evaluate(dist.mode())

  def testIncorrectBoundsFails(self):
    with self.assertRaisesOpError('`low >= high`'):
      dist = tfd.TruncatedCauchy(
          loc=0., scale=0.1, low=1.0, high=-1.0, validate_args=True)
      self.evaluate(dist.mode())

    with self.assertRaisesOpError('`low >= high`'):
      dist = tfd.TruncatedCauchy(
          loc=0., scale=0.1, low=1.0, high=1.0, validate_args=True)
      self.evaluate(dist.mode())

  def testAssertValidSample(self):
    dist = tfd.TruncatedCauchy(
        loc=0., scale=2., low=-4., high=3., validate_args=True)
    with self.assertRaisesOpError('must be greater than or equal to `low`'):
      self.evaluate(dist.cdf([-4.2, 1.7, 2.3]))
    with self.assertRaisesOpError('must be less than or equal to `high`'):
      self.evaluate(dist.survival_function([2.3, -3.2, 4.]))


if __name__ == '__main__':
  test_util.main()
