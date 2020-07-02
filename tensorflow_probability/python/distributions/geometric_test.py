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
"""Tests for the Geometric distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


# In all tests that follow, we use scipy.stats.geom, which
# represents the "Shifted" Geometric distribution. Hence, loc=-1 is passed
# in to each scipy function for testing.
@test_util.test_all_tf_execution_regimes
class GeometricTest(test_util.TestCase):

  def testGeometricShape(self):
    probs = tf.constant([.1] * 5)
    geom = tfd.Geometric(probs=probs, validate_args=True)

    self.assertEqual([
        5,
    ], self.evaluate(geom.batch_shape_tensor()))
    self.assertAllEqual([], self.evaluate(geom.event_shape_tensor()))
    self.assertEqual(tf.TensorShape([5]), geom.batch_shape)
    self.assertEqual(tf.TensorShape([]), geom.event_shape)

  def testInvalidP(self):
    invalid_ps = [-.01, -0.01, -2.]
    with self.assertRaisesOpError('must be positive'):
      geom = tfd.Geometric(probs=invalid_ps, validate_args=True)
      self.evaluate(geom.mean())

    invalid_ps = [1.1, 3., 5.]
    with self.assertRaisesOpError('must be less than or equal to 1'):
      geom = tfd.Geometric(probs=invalid_ps, validate_args=True)
      self.evaluate(geom.mean())

  def testGeomLogPmf(self):
    batch_size = 6
    probs = tf.constant([.2] * batch_size)
    probs_v = .2
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=np.float32)
    geom = tfd.Geometric(probs=probs, validate_args=True)
    expected_log_prob = stats.geom.logpmf(x, probs_v, loc=-1)
    log_prob = geom.log_prob(x)
    self.assertEqual([
        6,
    ], log_prob.shape)
    self.assertAllClose(expected_log_prob, self.evaluate(log_prob))

    pmf = geom.prob(x)
    self.assertEqual([
        6,
    ], pmf.shape)
    self.assertAllClose(np.exp(expected_log_prob), self.evaluate(pmf))

  def testGeometricLogPmf_validate_args(self):
    batch_size = 6
    probs = tf.constant([.9] * batch_size)
    x = tf1.placeholder_with_default(
        [2.5, 3.2, 4.3, 5.1, 6., 7.], shape=[6])
    geom = tfd.Geometric(probs=probs, validate_args=True)

    with self.assertRaisesOpError('cannot contain fractional components'):
      self.evaluate(geom.log_prob(x))

    with self.assertRaisesOpError('Condition x >= 0'):
      self.evaluate(geom.log_prob([-1.]))

    geom = tfd.Geometric(probs=probs, validate_args=False)
    log_prob = geom.log_prob(x)
    self.assertEqual([
        6,
    ], log_prob.shape)
    pmf = geom.prob(x)
    self.assertEqual([
        6,
    ], pmf.shape)

  def testGeometricLogPmfMultidimensional(self):
    batch_size = 6
    probs = tf.constant([[.2, .3, .5]] * batch_size)
    probs_v = np.array([.2, .3, .5])
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T
    geom = tfd.Geometric(probs=probs, validate_args=True)
    expected_log_prob = stats.geom.logpmf(x, probs_v, loc=-1)
    log_prob = geom.log_prob(x)
    log_prob_values = self.evaluate(log_prob)
    self.assertEqual([6, 3], log_prob.shape)
    self.assertAllClose(expected_log_prob, log_prob_values)

    pmf = geom.prob(x)
    pmf_values = self.evaluate(pmf)
    self.assertEqual([6, 3], pmf.shape)
    self.assertAllClose(np.exp(expected_log_prob), pmf_values)

  def testGeometricCDF(self):
    batch_size = 6
    probs = tf.constant([[.2, .4, .5]] * batch_size)
    probs_v = np.array([.2, .4, .5])
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

    geom = tfd.Geometric(probs=probs, validate_args=True)
    expected_cdf = stats.geom.cdf(x, probs_v, loc=-1)

    cdf = geom.cdf(x)
    self.assertEqual([6, 3], cdf.shape)
    self.assertAllClose(expected_cdf, self.evaluate(cdf))

  def testGeometricSurvivalFunction(self):
    batch_size = 6
    probs = tf.constant([[.2, .4, .5]] * batch_size)
    probs_v = np.array([.2, .4, .5])
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

    geom = tfd.Geometric(probs=probs, validate_args=True)
    expected_sf = stats.geom.sf(x, probs_v, loc=-1)

    sf = geom.survival_function(x)
    self.assertEqual([6, 3], sf.shape)
    self.assertAllClose(expected_sf, self.evaluate(sf))

  def testGeometricEntropy(self):
    probs_v = np.array([.1, .3, .25], dtype=np.float32)
    geom = tfd.Geometric(probs=probs_v, validate_args=True)
    expected_entropy = stats.geom.entropy(probs_v, loc=-1)
    self.assertEqual([3], geom.entropy().shape)
    self.assertAllClose(expected_entropy, self.evaluate(geom.entropy()))

  def testGeometricMean(self):
    probs_v = np.array([.1, .3, .25])
    geom = tfd.Geometric(probs=probs_v, validate_args=True)
    expected_means = stats.geom.mean(probs_v, loc=-1)
    self.assertEqual([3], geom.mean().shape)
    self.assertAllClose(expected_means, self.evaluate(geom.mean()))

  def testGeometricVariance(self):
    probs_v = np.array([.1, .3, .25])
    geom = tfd.Geometric(probs=probs_v, validate_args=True)
    expected_vars = stats.geom.var(probs_v, loc=-1)
    self.assertEqual([3], geom.variance().shape)
    self.assertAllClose(expected_vars, self.evaluate(geom.variance()))

  def testGeometricStddev(self):
    probs_v = np.array([.1, .3, .25])
    geom = tfd.Geometric(probs=probs_v, validate_args=True)
    expected_stddevs = stats.geom.std(probs_v, loc=-1)
    self.assertEqual([3], geom.stddev().shape)
    self.assertAllClose(self.evaluate(geom.stddev()), expected_stddevs)

  def testGeometricMode(self):
    probs_v = np.array([.1, .3, .25])
    geom = tfd.Geometric(probs=probs_v, validate_args=True)
    self.assertEqual([
        3,
    ],
                     geom.mode().shape)
    self.assertAllClose([0.] * 3, self.evaluate(geom.mode()))

  def testGeometricSample(self):
    probs_v = [.3, .9]
    probs = tf.constant(probs_v)
    n = tf.constant(100000)
    geom = tfd.Geometric(probs=probs, validate_args=True)

    samples = geom.sample(n, seed=test_util.test_seed())
    self.assertEqual([100000, 2], samples.shape)

    sample_values = self.evaluate(samples)
    self.assertFalse(np.any(sample_values < 0.0))
    for i in range(2):
      self.assertAllClose(
          sample_values[:, i].mean(),
          stats.geom.mean(probs_v[i], loc=-1),
          rtol=.04)
      self.assertAllClose(
          sample_values[:, i].var(),
          stats.geom.var(probs_v[i], loc=-1),
          rtol=.04)

  def testGeometricSampleMultiDimensional(self):
    batch_size = 2
    probs_v = [.3, .9]
    probs = tf.constant([probs_v] * batch_size)

    geom = tfd.Geometric(probs=probs, validate_args=True)

    n = 400000
    samples = geom.sample(n, seed=test_util.test_seed())
    self.assertEqual([n, batch_size, 2], samples.shape)

    sample_values = self.evaluate(samples)

    self.assertFalse(np.any(sample_values < 0.0))
    for i in range(2):
      self.assertAllClose(
          sample_values[:, 0, i].mean(),
          stats.geom.mean(probs_v[i], loc=-1),
          rtol=.04)
      self.assertAllClose(
          sample_values[:, 0, i].var(),
          stats.geom.var(probs_v[i], loc=-1),
          rtol=.04)
      self.assertAllClose(
          sample_values[:, 1, i].mean(),
          stats.geom.mean(probs_v[i], loc=-1),
          rtol=.04)
      self.assertAllClose(
          sample_values[:, 1, i].var(),
          stats.geom.var(probs_v[i], loc=-1),
          rtol=.04)

  def testGeometricAtBoundary(self):
    geom = tfd.Geometric(probs=1., validate_args=True)

    x = np.array([0., 2., 3., 4., 5., 6., 7.], dtype=np.float32)
    expected_log_prob = stats.geom.logpmf(x, [1.], loc=-1)
    # Scipy incorrectly returns nan.
    expected_log_prob[np.isnan(expected_log_prob)] = 0.

    log_prob = geom.log_prob(x)
    self.assertEqual([
        7,
    ], log_prob.shape)
    self.assertAllClose(expected_log_prob, self.evaluate(log_prob))

    pmf = geom.prob(x)
    self.assertEqual([
        7,
    ], pmf.shape)
    self.assertAllClose(np.exp(expected_log_prob), self.evaluate(pmf))

    expected_log_cdf = stats.geom.logcdf(x, 1., loc=-1)

    log_cdf = geom.log_cdf(x)
    self.assertEqual([
        7,
    ], log_cdf.shape)
    self.assertAllClose(expected_log_cdf, self.evaluate(log_cdf))

    cdf = geom.cdf(x)
    self.assertEqual([
        7,
    ], cdf.shape)
    self.assertAllClose(np.exp(expected_log_cdf), self.evaluate(cdf))

    expected_mean = stats.geom.mean(1., loc=-1)
    self.assertEqual([], geom.mean().shape)
    self.assertAllClose(expected_mean, self.evaluate(geom.mean()))

    expected_variance = stats.geom.var(1., loc=-1)
    self.assertEqual([], geom.variance().shape)
    self.assertAllClose(expected_variance, self.evaluate(geom.variance()))

    with self.assertRaisesOpError('Entropy is undefined'):
      self.evaluate(geom.entropy())

  def testParamTensorFromLogits(self):
    x = tf.constant([-1., 0.5, 1.])
    d = tfd.Geometric(logits=x, validate_args=True)
    logit = lambda x: tf.math.log(x) - tf.math.log1p(-x)
    self.assertAllClose(
        *self.evaluate([logit(d.prob(0.)), d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([d.prob(0.), d.probs_parameter()]),
        atol=0, rtol=1e-4)

  def testParamTensorFromProbs(self):
    x = tf.constant([0.1, 0.5, 0.4])
    d = tfd.Geometric(probs=x, validate_args=True)
    logit = lambda x: tf.math.log(x) - tf.math.log1p(-x)
    self.assertAllClose(
        *self.evaluate([logit(d.prob(0.)), d.logits_parameter()]),
        atol=0, rtol=1e-4)
    self.assertAllClose(
        *self.evaluate([d.prob(0.), d.probs_parameter()]),
        atol=0, rtol=1e-4)

  def testModifiedVariableProbAssertion(self):
    v = tf.Variable(0.9)
    self.evaluate(v.initializer)
    d = tfd.Geometric(probs=v, validate_args=True)
    with self.assertRaisesOpError('must be less than or equal to 1'):
      with tf.control_dependencies([v.assign(1.2)]):
        self.evaluate(d.mean())


if __name__ == '__main__':
  tf.test.main()
