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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_case
from tensorflow.python.framework import test_util

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class PoissonTest(test_case.TestCase):

  def _make_poisson(self,
                    rate,
                    validate_args=False,
                    interpolate_nondiscrete=True):
    return tfd.Poisson(rate=rate,
                       validate_args=validate_args,
                       interpolate_nondiscrete=interpolate_nondiscrete)

  def testPoissonShape(self):
    lam = tf.constant([3.0] * 5)
    poisson = self._make_poisson(rate=lam)

    self.assertEqual(self.evaluate(poisson.batch_shape_tensor()), (5,))
    self.assertEqual(poisson.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(poisson.event_shape_tensor()), [])
    self.assertEqual(poisson.event_shape, tf.TensorShape([]))

  def testInvalidLam(self):
    invalid_lams = [-.01, 0., -2.]
    for lam in invalid_lams:
      with self.assertRaisesOpError("Condition x > 0"):
        poisson = self._make_poisson(rate=lam, validate_args=True)
        self.evaluate(poisson.rate)

  def testPoissonLogPmfDiscreteMatchesScipy(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    x = [-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.]
    poisson = self._make_poisson(rate=lam,
                                 interpolate_nondiscrete=False)
    log_pmf = poisson.log_prob(x)
    self.assertEqual(log_pmf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(log_pmf), stats.poisson.logpmf(x, lam_v))

    pmf = poisson.prob(x)
    self.assertEqual(pmf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(pmf), stats.poisson.pmf(x, lam_v))

  def testPoissonLogPmfContinuousRelaxation(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    x = np.array([-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.]).astype(
        np.float32)
    poisson = self._make_poisson(rate=lam,
                                 interpolate_nondiscrete=True)

    expected_continuous_log_pmf = (x * poisson.log_rate - tf.lgamma(1. + x)
                                   - poisson.rate)
    neg_inf = tf.fill(
        tf.shape(expected_continuous_log_pmf),
        value=np.array(-np.inf,
                       dtype=expected_continuous_log_pmf.dtype.as_numpy_dtype))
    expected_continuous_log_pmf = tf.where(x >= 0.,
                                           expected_continuous_log_pmf,
                                           neg_inf)
    expected_continuous_pmf = tf.exp(expected_continuous_log_pmf)

    log_pmf = poisson.log_prob(x)
    self.assertEqual(log_pmf.get_shape(), (batch_size,))
    self.assertAllClose(self.evaluate(log_pmf),
                        self.evaluate(expected_continuous_log_pmf))

    pmf = poisson.prob(x)
    self.assertEqual(pmf.get_shape(), (batch_size,))
    self.assertAllClose(self.evaluate(pmf),
                        self.evaluate(expected_continuous_pmf))

  def testPoissonLogPmfGradient(self):
    batch_size = 6
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    # Only non-negative values, as negative ones cause nans in the expected
    # value.
    x = [0., 2., 3., 4., 5., 6.]

    dlog_pmf_dlam = self.compute_gradients(
        lambda lam: self._make_poisson(rate=lam).log_prob(x), [lam])[0]

    # A finite difference approximation of the derivative.
    eps = 1e-6
    expected = (stats.poisson.logpmf(x, lam_v + eps)
                - stats.poisson.logpmf(x, lam_v - eps)) / (2 * eps)

    self.assertEqual(dlog_pmf_dlam.shape, (batch_size,))
    self.assertAllClose(dlog_pmf_dlam, expected)

  def testPoissonLogPmfGradientAtZeroPmf(self):
    # Check that the derivative wrt parameter at the zero-prob points is zero.
    batch_size = 6
    lam = tf.constant([3.0] * batch_size)
    x = [-2., -1., -0.5, 0.2, 1.5, 10.5]

    def poisson_log_prob(lam):
      return self._make_poisson(
          rate=lam, interpolate_nondiscrete=False).log_prob(x)
    dlog_pmf_dlam = self.compute_gradients(poisson_log_prob, [lam])[0]

    self.assertEqual(dlog_pmf_dlam.shape, (batch_size,))
    print(dlog_pmf_dlam)
    self.assertAllClose(dlog_pmf_dlam, np.zeros([batch_size]))

  def testPoissonLogPmfMultidimensional(self):
    batch_size = 6
    lam = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    lam_v = [2.0, 4.0, 5.0]
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

    poisson = self._make_poisson(rate=lam)
    log_pmf = poisson.log_prob(x)
    self.assertEqual(log_pmf.shape, (6, 3))
    self.assertAllClose(self.evaluate(log_pmf), stats.poisson.logpmf(x, lam_v))

    pmf = poisson.prob(x)
    self.assertEqual(pmf.shape, (6, 3))
    self.assertAllClose(self.evaluate(pmf), stats.poisson.pmf(x, lam_v))

  def testPoissonCdf(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    x = [-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.]

    poisson = self._make_poisson(rate=lam, interpolate_nondiscrete=False)
    log_cdf = poisson.log_cdf(x)
    self.assertEqual(log_cdf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(log_cdf), stats.poisson.logcdf(x, lam_v))

    cdf = poisson.cdf(x)
    self.assertEqual(cdf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(cdf), stats.poisson.cdf(x, lam_v))

  def testPoissonCdfContinuousRelaxation(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    x = np.array(
        [-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.]).astype(
            np.float32)

    expected_continuous_cdf = tf.igammac(1. + x, lam)
    expected_continuous_cdf = tf.where(x >= 0.,
                                       expected_continuous_cdf,
                                       tf.zeros_like(expected_continuous_cdf))
    expected_continuous_log_cdf = tf.log(expected_continuous_cdf)

    poisson = self._make_poisson(rate=lam, interpolate_nondiscrete=True)
    log_cdf = poisson.log_cdf(x)
    self.assertEqual(log_cdf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(log_cdf),
                        self.evaluate(expected_continuous_log_cdf))

    cdf = poisson.cdf(x)
    self.assertEqual(cdf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(cdf),
                        self.evaluate(expected_continuous_cdf))

  def testPoissonCdfGradient(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    x = [-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.]

    def cdf(lam):
      return self._make_poisson(rate=lam, interpolate_nondiscrete=False).cdf(x)
    dcdf_dlam = self.compute_gradients(cdf, [lam])[0]

    # A finite difference approximation of the derivative.
    eps = 1e-6
    expected = (stats.poisson.cdf(x, lam_v + eps)
                - stats.poisson.cdf(x, lam_v - eps)) / (2 * eps)

    self.assertEqual(dcdf_dlam.shape, (batch_size,))
    self.assertAllClose(dcdf_dlam, expected)

  def testPoissonCdfMultidimensional(self):
    batch_size = 6
    lam = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    lam_v = [2.0, 4.0, 5.0]
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

    poisson = self._make_poisson(rate=lam, interpolate_nondiscrete=False)
    log_cdf = poisson.log_cdf(x)
    self.assertEqual(log_cdf.shape, (6, 3))
    self.assertAllClose(self.evaluate(log_cdf), stats.poisson.logcdf(x, lam_v))

    cdf = poisson.cdf(x)
    self.assertEqual(cdf.shape, (6, 3))
    self.assertAllClose(self.evaluate(cdf), stats.poisson.cdf(x, lam_v))

  def testPoissonMean(self):
    lam_v = [1.0, 3.0, 2.5]
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.mean().shape, (3,))
    self.assertAllClose(
        self.evaluate(poisson.mean()), stats.poisson.mean(lam_v))
    self.assertAllClose(self.evaluate(poisson.mean()), lam_v)

  def testPoissonVariance(self):
    lam_v = [1.0, 3.0, 2.5]
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.variance().shape, (3,))
    self.assertAllClose(
        self.evaluate(poisson.variance()), stats.poisson.var(lam_v))
    self.assertAllClose(self.evaluate(poisson.variance()), lam_v)

  def testPoissonStd(self):
    lam_v = [1.0, 3.0, 2.5]
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.stddev().shape, (3,))
    self.assertAllClose(
        self.evaluate(poisson.stddev()), stats.poisson.std(lam_v))
    self.assertAllClose(self.evaluate(poisson.stddev()), np.sqrt(lam_v))

  def testPoissonMode(self):
    lam_v = [1.0, 3.0, 2.5, 3.2, 1.1, 0.05]
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.mode().shape, (6,))
    self.assertAllClose(self.evaluate(poisson.mode()), np.floor(lam_v))

  def testPoissonMultipleMode(self):
    lam_v = [1.0, 3.0, 2.0, 4.0, 5.0, 10.0]
    poisson = self._make_poisson(rate=lam_v)
    # For the case where lam is an integer, the modes are: lam and lam - 1.
    # In this case, we get back the larger of the two modes.
    self.assertEqual((6,), poisson.mode().shape)
    self.assertAllClose(lam_v, self.evaluate(poisson.mode()))

  def testPoissonSample(self):
    lam_v = 4.0
    lam = tf.constant(lam_v)
    # Choosing `n >= (k/rtol)**2, roughly ensures our sample mean should be
    # within `k` std. deviations of actual up to rtol precision.
    n = int(100e3)
    poisson = self._make_poisson(rate=lam)
    samples = poisson.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.mean(), stats.poisson.mean(lam_v), rtol=.01)
    self.assertAllClose(sample_values.var(), stats.poisson.var(lam_v), rtol=.01)

  def testPoissonSampleMultidimensionalMean(self):
    lam_v = np.array([np.arange(1, 51, dtype=np.float32)])  # 1 x 50
    poisson = self._make_poisson(rate=lam_v)
    # Choosing `n >= (k/rtol)**2, roughly ensures our sample mean should be
    # within `k` std. deviations of actual up to rtol precision.
    n = int(100e3)
    samples = poisson.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 1, 50))
    self.assertEqual(sample_values.shape, (n, 1, 50))
    self.assertAllClose(
        sample_values.mean(axis=0), stats.poisson.mean(lam_v), rtol=.01, atol=0)

  def testPoissonSampleMultidimensionalVariance(self):
    lam_v = np.array([np.arange(5, 15, dtype=np.float32)])  # 1 x 10
    poisson = self._make_poisson(rate=lam_v)
    # Choosing `n >= 2 * lam * (k/rtol)**2, roughly ensures our sample
    # variance should be within `k` std. deviations of actual up to rtol
    # precision.
    n = int(300e3)
    samples = poisson.sample(n, seed=123456)
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 1, 10))
    self.assertEqual(sample_values.shape, (n, 1, 10))

    self.assertAllClose(
        sample_values.var(axis=0), stats.poisson.var(lam_v), rtol=.03, atol=0)


@test_util.run_all_in_graph_and_eager_modes
class PoissonLogRateTest(PoissonTest):

  def _make_poisson(self,
                    rate,
                    validate_args=False,
                    interpolate_nondiscrete=True):
    return tfd.Poisson(log_rate=tf.log(rate),
                       validate_args=validate_args,
                       interpolate_nondiscrete=interpolate_nondiscrete)

  def testInvalidLam(self):
    # No need to worry about the non-negativity of `rate` when using the
    # `log_rate` parameterization.
    pass


if __name__ == "__main__":
  tf.test.main()
