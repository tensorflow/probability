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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class LogisticTest(test_util.TestCase):

  def testReparameterizable(self):
    batch_size = 6
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5
    dist = tfd.Logistic(loc, scale, validate_args=True)
    self.assertEqual(tfd.FULLY_REPARAMETERIZED, dist.reparameterization_type)

  def testLogisticLogProb(self):
    batch_size = 6
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    dist = tfd.Logistic(loc, scale, validate_args=True)
    expected_log_prob = stats.logistic.logpdf(x, np_loc, scale)

    log_prob = dist.log_prob(x)
    self.assertEqual(log_prob.shape, (6,))
    self.assertAllClose(self.evaluate(log_prob), expected_log_prob)

    prob = dist.prob(x)
    self.assertEqual(prob.shape, (6,))
    self.assertAllClose(self.evaluate(prob), np.exp(expected_log_prob))

  def testLogisticCDF(self):
    batch_size = 6
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5

    dist = tfd.Logistic(loc, scale, validate_args=True)
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    cdf = dist.cdf(x)
    expected_cdf = stats.logistic.cdf(x, np_loc, scale)

    self.assertEqual(cdf.shape, (6,))
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testLogisticLogCDF(self):
    batch_size = 6
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5

    dist = tfd.Logistic(loc, scale, validate_args=True)
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    logcdf = dist.log_cdf(x)
    expected_logcdf = stats.logistic.logcdf(x, np_loc, scale)

    self.assertEqual(logcdf.shape, (6,))
    self.assertAllClose(self.evaluate(logcdf), expected_logcdf)

  def testLogisticSurvivalFunction(self):
    batch_size = 6
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5

    dist = tfd.Logistic(loc, scale, validate_args=True)
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    survival_function = dist.survival_function(x)
    expected_survival_function = stats.logistic.sf(x, np_loc, scale)

    self.assertEqual(survival_function.shape, (6,))
    self.assertAllClose(
        self.evaluate(survival_function), expected_survival_function)

  def testLogisticLogSurvivalFunction(self):
    batch_size = 6
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5

    dist = tfd.Logistic(loc, scale, validate_args=True)
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    logsurvival_function = dist.log_survival_function(x)
    expected_logsurvival_function = stats.logistic.logsf(x, np_loc, scale)

    self.assertEqual(logsurvival_function.shape, (6,))
    self.assertAllClose(
        self.evaluate(logsurvival_function), expected_logsurvival_function)

  def testLogisticMean(self):
    loc = [2.0, 1.5, 1.0]
    scale = 1.5
    expected_mean = stats.logistic.mean(loc, scale)
    dist = tfd.Logistic(loc, scale, validate_args=True)
    self.assertAllClose(self.evaluate(dist.mean()), expected_mean)

  def testLogisticVariance(self):
    loc = [2.0, 1.5, 1.0]
    scale = 1.5
    expected_variance = stats.logistic.var(loc, scale)
    dist = tfd.Logistic(loc, scale, validate_args=True)
    self.assertAllClose(self.evaluate(dist.variance()), expected_variance)

  def testLogisticEntropy(self):
    batch_size = 3
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5
    expected_entropy = stats.logistic.entropy(np_loc, scale)
    dist = tfd.Logistic(loc, scale, validate_args=True)
    self.assertAllClose(self.evaluate(dist.entropy()), expected_entropy)

  def testLogisticSample(self):
    loc_ = [3.0, 4.0, 2.0]
    scale_ = 1.0
    dist = tfd.Logistic(loc_, scale_, validate_args=True)
    n = int(15e3)
    samples = dist.sample(n, seed=test_util.test_seed())
    self.assertEqual(samples.shape, (n, 3))
    samples_ = self.evaluate(samples)
    for i in range(3):
      self.assertLess(
          stats.kstest(
              samples_[:, i],
              stats.logistic(loc=loc_[i], scale=scale_).cdf)[0],
          0.013)

  def testLogisticQuantile(self):
    loc = [3.0, 4.0, 2.0]
    scale = np.random.randn(3)**2 + 1e-3
    x = [.2, .5, .99]
    expected_quantile = stats.logistic.ppf(x, loc=loc, scale=scale)
    dist = tfd.Logistic(loc, scale, validate_args=True)
    self.assertAllClose(self.evaluate(dist.quantile(x)), expected_quantile)

  def testDtype(self):
    loc = tf.constant([0.1, 0.4], dtype=tf.float32)
    scale = tf.constant(1.0, dtype=tf.float32)
    dist = tfd.Logistic(loc, scale, validate_args=True)
    self.assertEqual(dist.dtype, tf.float32)
    self.assertEqual(dist.loc.dtype, dist.scale.dtype)
    self.assertEqual(dist.dtype, dist.sample(
        5, seed=test_util.test_seed()).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.loc.dtype, dist.mean().dtype)
    self.assertEqual(dist.loc.dtype, dist.variance().dtype)
    self.assertEqual(dist.loc.dtype, dist.stddev().dtype)
    self.assertEqual(dist.loc.dtype, dist.entropy().dtype)
    self.assertEqual(dist.loc.dtype, dist.prob(0.2).dtype)
    self.assertEqual(dist.loc.dtype, dist.log_prob(0.2).dtype)

    loc = tf.constant([0.1, 0.4], dtype=tf.float64)
    scale = tf.constant(1.0, dtype=tf.float64)
    dist64 = tfd.Logistic(loc, scale, validate_args=True)
    self.assertEqual(dist64.dtype, tf.float64)
    self.assertEqual(dist64.dtype, dist64.sample(
        5, seed=test_util.test_seed()).dtype)

  @test_util.tf_tape_safety_test
  def testGradientThroughParams(self):
    loc = tf.Variable([-5., 0., 5.])
    scale = tf.Variable(2.)
    d = tfd.Logistic(loc=loc, scale=scale, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 3.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 2)
    self.assertAllNotNone(grad)

  def testAssertsPositiveScale(self):
    scale = tf.Variable([1., 2., -3.])
    with self.assertRaisesOpError("Argument `scale` must be positive."):
      d = tfd.Logistic(loc=0, scale=scale, validate_args=True)
      self.evaluate([v.initializer for v in d.variables])
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveScaleAfterMutation(self):
    scale = tf.Variable([1., 2., 3.])
    self.evaluate(scale.initializer)
    d = tfd.Logistic(loc=0., scale=scale, validate_args=True)
    with self.assertRaisesOpError("Argument `scale` must be positive."):
      with tf.control_dependencies([scale.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertParamsAreFloats(self):
    loc = tf.convert_to_tensor(0, dtype=tf.int32)
    scale = tf.convert_to_tensor(1, dtype=tf.int32)
    with self.assertRaisesRegexp(ValueError, "Expected floating point"):
      tfd.Logistic(loc=loc, scale=scale)


if __name__ == "__main__":
  tf.test.main()
