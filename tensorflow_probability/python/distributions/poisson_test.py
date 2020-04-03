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
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import poisson as poisson_dist
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_util
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class PoissonTest(test_util.TestCase):

  def _make_poisson(self,
                    rate,
                    validate_args=True,
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
      with self.assertRaisesOpError('Argument `rate` must be positive.'):
        poisson = self._make_poisson(rate=lam)
        self.evaluate(poisson.rate_parameter())

  def testPoissonLogPmfDiscreteMatchesScipy(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    x = np.array([-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.],
                 dtype=np.float32)
    poisson = self._make_poisson(
        rate=lam, interpolate_nondiscrete=False, validate_args=False)
    log_pmf = poisson.log_prob(x)
    self.assertEqual(log_pmf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(log_pmf), stats.poisson.logpmf(x, lam_v))

    pmf = poisson.prob(x)
    self.assertEqual(pmf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(pmf), stats.poisson.pmf(x, lam_v))

  def testPoissonLogPmfContinuousRelaxation(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    x = tf.constant([-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.])
    poisson = self._make_poisson(
        rate=lam, interpolate_nondiscrete=True, validate_args=False)

    expected_continuous_log_pmf = (
        x * poisson.log_rate_parameter()
        - tf.math.lgamma(1. + x) - poisson.rate_parameter())
    expected_continuous_log_pmf = tf.where(
        x >= 0., expected_continuous_log_pmf,
        dtype_util.as_numpy_dtype(
            expected_continuous_log_pmf.dtype)(-np.inf))
    expected_continuous_pmf = tf.exp(expected_continuous_log_pmf)

    log_pmf = poisson.log_prob(x)
    self.assertEqual((batch_size,), log_pmf.shape)
    self.assertAllClose(self.evaluate(log_pmf),
                        self.evaluate(expected_continuous_log_pmf))

    pmf = poisson.prob(x)
    self.assertEqual((batch_size,), pmf.shape)
    self.assertAllClose(self.evaluate(pmf),
                        self.evaluate(expected_continuous_pmf))

  @test_util.numpy_disable_gradient_test
  def testPoissonLogPmfGradient(self):
    batch_size = 6
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    # Only non-negative values, as negative ones cause nans in the expected
    # value.
    x = np.array([0., 2., 3., 4., 5., 6.], dtype=np.float32)

    _, dlog_pmf_dlam = self.evaluate(tfp.math.value_and_gradient(
        lambda lam: self._make_poisson(rate=lam).log_prob(x), lam))

    # A finite difference approximation of the derivative.
    eps = 1e-6
    expected = (stats.poisson.logpmf(x, lam_v + eps)
                - stats.poisson.logpmf(x, lam_v - eps)) / (2 * eps)

    self.assertEqual(dlog_pmf_dlam.shape, (batch_size,))
    self.assertAllClose(dlog_pmf_dlam, expected)

  @test_util.numpy_disable_gradient_test
  def testPoissonLogPmfGradientAtZeroPmf(self):
    # Check that the derivative wrt parameter at the zero-prob points is zero.
    batch_size = 6
    lam = tf.constant([3.0] * batch_size)
    x = tf.constant([-2., -1., -0.5, 0.2, 1.5, 10.5])

    def poisson_log_prob(lam):
      return self._make_poisson(
          rate=lam, interpolate_nondiscrete=False, validate_args=False
          ).log_prob(x)
    _, dlog_pmf_dlam = self.evaluate(tfp.math.value_and_gradient(
        poisson_log_prob, lam))

    self.assertEqual(dlog_pmf_dlam.shape, (batch_size,))
    self.assertAllClose(dlog_pmf_dlam, np.zeros([batch_size]))

  def testPoissonLogPmfMultidimensional(self):
    batch_size = 6
    lam = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    lam_v = np.array([2.0, 4.0, 5.0], dtype=np.float32)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

    poisson = self._make_poisson(rate=lam)
    log_pmf = poisson.log_prob(x)
    self.assertEqual(log_pmf.shape, (6, 3))
    self.assertAllClose(self.evaluate(log_pmf), stats.poisson.logpmf(x, lam_v))

    pmf = poisson.prob(x)
    self.assertEqual(pmf.shape, (6, 3))
    self.assertAllClose(self.evaluate(pmf), stats.poisson.pmf(x, lam_v))

  @test_util.jax_disable_test_missing_functionality(
      '`tf.math.igammac` is unimplemented in JAX backend.')
  def testPoissonCdf(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    x = np.array([-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.],
                 dtype=np.float32)

    poisson = self._make_poisson(
        rate=lam, interpolate_nondiscrete=False, validate_args=False)
    log_cdf = poisson.log_cdf(x)
    self.assertEqual(log_cdf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(log_cdf), stats.poisson.logcdf(x, lam_v))

    cdf = poisson.cdf(x)
    self.assertEqual(cdf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(cdf), stats.poisson.cdf(x, lam_v))

  @test_util.jax_disable_test_missing_functionality(
      '`tf.math.igammac` is unimplemented in JAX backend.')
  def testPoissonCdfContinuousRelaxation(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    x = np.array([-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.],
                 dtype=np.float32)

    expected_continuous_cdf = tf.math.igammac(1. + x, lam)
    expected_continuous_cdf = tf.where(x >= 0., expected_continuous_cdf,
                                       tf.zeros_like(expected_continuous_cdf))
    expected_continuous_log_cdf = tf.math.log(expected_continuous_cdf)

    poisson = self._make_poisson(
        rate=lam, interpolate_nondiscrete=True, validate_args=False)
    log_cdf = poisson.log_cdf(x)
    self.assertEqual(log_cdf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(log_cdf),
                        self.evaluate(expected_continuous_log_cdf))

    cdf = poisson.cdf(x)
    self.assertEqual(cdf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(cdf),
                        self.evaluate(expected_continuous_cdf))

  @test_util.jax_disable_test_missing_functionality(
      '`tf.math.igammac` is unimplemented in JAX backend.')
  @test_util.numpy_disable_gradient_test
  def testPoissonCdfGradient(self):
    batch_size = 12
    lam = tf.constant([3.0] * batch_size)
    lam_v = 3.0
    x = np.array([-3., -0.5, 0., 2., 2.2, 3., 3.1, 4., 5., 5.5, 6., 7.],
                 dtype=np.float32)

    def cdf(lam):
      return self._make_poisson(
          rate=lam, interpolate_nondiscrete=False, validate_args=False).cdf(x)
    _, dcdf_dlam = self.evaluate(tfp.math.value_and_gradient(cdf, lam))

    # A finite difference approximation of the derivative.
    eps = 1e-6
    expected = (stats.poisson.cdf(x, lam_v + eps)
                - stats.poisson.cdf(x, lam_v - eps)) / (2 * eps)

    self.assertEqual(dcdf_dlam.shape, (batch_size,))
    self.assertAllClose(dcdf_dlam, expected)

  @test_util.jax_disable_test_missing_functionality(
      '`tf.math.igammac` is unimplemented in JAX backend.')
  def testPoissonCdfMultidimensional(self):
    batch_size = 6
    lam = tf.constant([[2.0, 4.0, 5.0]] * batch_size)
    lam_v = np.array([2.0, 4.0, 5.0], dtype=np.float32)
    x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

    poisson = self._make_poisson(rate=lam, interpolate_nondiscrete=False)
    log_cdf = poisson.log_cdf(x)
    self.assertEqual(log_cdf.shape, (6, 3))
    self.assertAllClose(self.evaluate(log_cdf), stats.poisson.logcdf(x, lam_v))

    cdf = poisson.cdf(x)
    self.assertEqual(cdf.shape, (6, 3))
    self.assertAllClose(self.evaluate(cdf), stats.poisson.cdf(x, lam_v))

  def testPoissonMean(self):
    lam_v = np.array([1.0, 3.0, 2.5], dtype=np.float32)
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.mean().shape, (3,))
    self.assertAllClose(
        self.evaluate(poisson.mean()), stats.poisson.mean(lam_v))
    self.assertAllClose(self.evaluate(poisson.mean()), lam_v)

  def testPoissonVariance(self):
    lam_v = np.array([1.0, 3.0, 2.5], dtype=np.float32)
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.variance().shape, (3,))
    self.assertAllClose(
        self.evaluate(poisson.variance()), stats.poisson.var(lam_v))
    self.assertAllClose(self.evaluate(poisson.variance()), lam_v)

  def testPoissonStd(self):
    lam_v = np.array([1.0, 3.0, 2.5], dtype=np.float32)
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.stddev().shape, (3,))
    self.assertAllClose(
        self.evaluate(poisson.stddev()), stats.poisson.std(lam_v))
    self.assertAllClose(self.evaluate(poisson.stddev()), np.sqrt(lam_v))

  def testPoissonMode(self):
    lam_v = np.array([1.0, 3.0, 2.5, 3.2, 1.1, 0.05], dtype=np.float32)
    poisson = self._make_poisson(rate=lam_v)
    self.assertEqual(poisson.mode().shape, (6,))
    self.assertAllClose(self.evaluate(poisson.mode()), np.floor(lam_v))

  def testPoissonMultipleMode(self):
    lam_v = np.array([1.0, 3.0, 2.0, 4.0, 5.0, 10.0], dtype=np.float32)
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
    samples = poisson.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.mean(), stats.poisson.mean(lam_v), rtol=.01)
    self.assertAllClose(sample_values.var(), stats.poisson.var(lam_v),
                        rtol=.013)

  def testAssertValidSample(self):
    lam_v = np.array([1.0, 3.0, 2.5], dtype=np.float32)
    poisson = self._make_poisson(rate=lam_v)
    with self.assertRaisesOpError('Condition x >= 0'):
      self.evaluate(poisson.cdf([-1.2, 3., 4.2]))

  def testPoissonSampleMultidimensionalMean(self):
    lam_v = np.array([np.arange(1, 51, dtype=np.float32)])  # 1 x 50
    poisson = self._make_poisson(rate=lam_v)
    # Choosing `n >= (k/rtol)**2, roughly ensures our sample mean should be
    # within `k` std. deviations of actual up to rtol precision.
    n = int(100e3)
    samples = poisson.sample(n, seed=test_util.test_seed())
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
    samples = poisson.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 1, 10))
    self.assertEqual(sample_values.shape, (n, 1, 10))

    self.assertAllClose(
        sample_values.var(axis=0), stats.poisson.var(lam_v), rtol=.03, atol=0)

  @test_util.tf_tape_safety_test
  def testGradientThroughRate(self):
    rate = tf.Variable(3.)
    dist = self._make_poisson(rate=rate)
    with tf.GradientTape() as tape:
      loss = -dist.log_prob([1., 2., 4.])
    grad = tape.gradient(loss, dist.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveRate(self):
    rate = tf.Variable([1., 2., -3.])
    self.evaluate(rate.initializer)
    with self.assertRaisesOpError('Argument `rate` must be positive.'):
      dist = self._make_poisson(rate=rate, validate_args=True)
      self.evaluate(dist.sample(seed=test_util.test_seed()))

  def testAssertsPositiveRateAfterMutation(self):
    rate = tf.Variable([1., 2., 3.])
    self.evaluate(rate.initializer)
    dist = self._make_poisson(rate=rate, validate_args=True)
    self.evaluate(dist.mean())
    with self.assertRaisesOpError('Argument `rate` must be positive.'):
      with tf.control_dependencies([rate.assign([1., 2., -3.])]):
        self.evaluate(dist.sample(seed=test_util.test_seed()))


@test_util.test_all_tf_execution_regimes
class PoissonLogRateTest(PoissonTest):

  def _make_poisson(self,
                    rate,
                    validate_args=True,
                    interpolate_nondiscrete=True):
    return tfd.Poisson(
        log_rate=tf.math.log(rate),
        validate_args=validate_args,
        interpolate_nondiscrete=interpolate_nondiscrete)

  # No need to worry about the non-negativity of `rate` when using the
  # `log_rate` parameterization.
  def testInvalidLam(self):
    pass

  def testAssertsPositiveRate(self):
    pass

  def testAssertsPositiveRateAfterMutation(self):
    pass

  # The gradient is not tracked through tf.math.log(rate) in _make_poisson(),
  # so log_rate needs to be defined as a Variable and passed directly.
  @test_util.tf_tape_safety_test
  def testGradientThroughRate(self):
    log_rate = tf.Variable(3.)
    dist = tfd.Poisson(log_rate=log_rate, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -dist.log_prob([1., 2., 4.])
    grad = tape.gradient(loss, dist.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)


@test_util.test_all_tf_execution_regimes
class PoissonSampleLogRateTest(test_util.TestCase):

  def testSamplePoissonLowRates(self):
    # Low log rate (< log(10.)) samples would use Knuth's algorithm.
    rate = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    log_rate = np.log(rate)
    num_samples = int(1e5)
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    samples = self.evaluate(
        poisson_dist.random_poisson_rejection_sampler(
            [num_samples, 10], log_rate, seed=test_util.test_seed()))

    poisson = tfd.Poisson(log_rate=log_rate, validate_args=True)
    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(
            samples,
            poisson.cdf,
            st.left_continuous_cdf_discrete_distribution(poisson),
            false_fail_rate=1e-9))

    self.assertAllClose(
        self.evaluate(tf.math.reduce_mean(samples, axis=0)),
        stats.poisson.mean(rate),
        rtol=0.01)
    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        stats.poisson.var(rate),
        rtol=0.05)

  def testSamplePoissonHighRates(self):
    # High rate (>= log(10.)) samples would use rejection sampling.
    rate = [10., 10.5, 11., 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5]
    log_rate = np.log(rate)
    num_samples = int(1e5)
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    samples = self.evaluate(
        poisson_dist.random_poisson_rejection_sampler(
            [num_samples, 10], log_rate, seed=test_util.test_seed()))

    poisson = tfd.Poisson(log_rate=log_rate, validate_args=True)
    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(
            samples,
            poisson.cdf,
            st.left_continuous_cdf_discrete_distribution(poisson),
            false_fail_rate=1e-9))

    self.assertAllClose(
        self.evaluate(tf.math.reduce_mean(samples, axis=0)),
        stats.poisson.mean(rate),
        rtol=0.01)
    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        stats.poisson.var(rate),
        rtol=0.05)

  def testSamplePoissonLowAndHighRates(self):
    rate = [1., 3., 5., 6., 7., 10., 13.0, 14., 15., 18.]
    log_rate = np.log(rate)
    num_samples = int(1e5)
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    samples = self.evaluate(
        poisson_dist.random_poisson_rejection_sampler(
            [num_samples, 10], log_rate, seed=test_util.test_seed()))

    poisson = tfd.Poisson(log_rate=log_rate, validate_args=True)
    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(
            samples,
            poisson.cdf,
            st.left_continuous_cdf_discrete_distribution(poisson),
            false_fail_rate=1e-9))

    self.assertAllClose(
        self.evaluate(tf.math.reduce_mean(samples, axis=0)),
        stats.poisson.mean(rate),
        rtol=0.01)
    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        stats.poisson.var(rate),
        rtol=0.05)

  def testSamplePoissonInvalidRates(self):
    rate = [np.nan, -1., 0., 5., 7., 10., 13.0, 14., 15., 18.]
    log_rate = np.log(rate)
    samples = self.evaluate(
        poisson_dist.random_poisson_rejection_sampler(
            [int(1e5), 10], log_rate, seed=test_util.test_seed()))
    self.assertAllClose(
        self.evaluate(tf.math.reduce_mean(samples, axis=0)),
        stats.poisson.mean(rate),
        rtol=0.01)
    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        stats.poisson.var(rate),
        rtol=0.05)


if __name__ == '__main__':
  tf.test.main()
