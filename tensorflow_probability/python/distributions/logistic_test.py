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

from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class LogisticTest(tf.test.TestCase):

  def testReparameterizable(self):
    batch_size = 6
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5
    dist = tfd.Logistic(loc, scale)
    self.assertTrue(
        dist.reparameterization_type == tfd.FULLY_REPARAMETERIZED)

  def testLogisticLogProb(self):
    batch_size = 6
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    dist = tfd.Logistic(loc, scale)
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

    dist = tfd.Logistic(loc, scale)
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

    dist = tfd.Logistic(loc, scale)
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

    dist = tfd.Logistic(loc, scale)
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

    dist = tfd.Logistic(loc, scale)
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
    dist = tfd.Logistic(loc, scale)
    self.assertAllClose(self.evaluate(dist.mean()), expected_mean)

  def testLogisticVariance(self):
    loc = [2.0, 1.5, 1.0]
    scale = 1.5
    expected_variance = stats.logistic.var(loc, scale)
    dist = tfd.Logistic(loc, scale)
    self.assertAllClose(self.evaluate(dist.variance()), expected_variance)

  def testLogisticEntropy(self):
    batch_size = 3
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = tf.constant(np_loc)
    scale = 1.5
    expected_entropy = stats.logistic.entropy(np_loc, scale)
    dist = tfd.Logistic(loc, scale)
    self.assertAllClose(self.evaluate(dist.entropy()), expected_entropy)

  def testLogisticSample(self):
    loc = [3.0, 4.0, 2.0]
    scale = 1.0
    dist = tfd.Logistic(loc, scale)
    sample = dist.sample(
        seed=tfp_test_util.test_seed(hardcoded_seed=100, set_eager_seed=False))
    self.assertEqual(sample.shape, (3,))
    self.assertAllClose(
        self.evaluate(sample), [6.22460556, 3.79602098, 2.05084133])

  def testDtype(self):
    loc = tf.constant([0.1, 0.4], dtype=tf.float32)
    scale = tf.constant(1.0, dtype=tf.float32)
    dist = tfd.Logistic(loc, scale)
    self.assertEqual(dist.dtype, tf.float32)
    self.assertEqual(dist.loc.dtype, dist.scale.dtype)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.loc.dtype, dist.mean().dtype)
    self.assertEqual(dist.loc.dtype, dist.variance().dtype)
    self.assertEqual(dist.loc.dtype, dist.stddev().dtype)
    self.assertEqual(dist.loc.dtype, dist.entropy().dtype)
    self.assertEqual(dist.loc.dtype, dist.prob(0.2).dtype)
    self.assertEqual(dist.loc.dtype, dist.log_prob(0.2).dtype)

    loc = tf.constant([0.1, 0.4], dtype=tf.float64)
    scale = tf.constant(1.0, dtype=tf.float64)
    dist64 = tfd.Logistic(loc, scale)
    self.assertEqual(dist64.dtype, tf.float64)
    self.assertEqual(dist64.dtype, dist64.sample(5).dtype)


if __name__ == "__main__":
  tf.test.main()
