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
from absl.testing import parameterized
import numpy as np
from scipy import misc as sp_misc
from scipy import special as sp_special
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import implementation_selection
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class GammaTest(test_util.TestCase):

  def testGammaShape(self):
    concentration = tf.constant([3.0] * 5)
    rate = tf.constant(11.0)
    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)

    self.assertEqual(self.evaluate(gamma.batch_shape_tensor()), (5,))
    self.assertEqual(gamma.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(gamma.event_shape_tensor()), [])
    self.assertEqual(gamma.event_shape, tf.TensorShape([]))

  def testGammaLogPDF(self):
    batch_size = 6
    concentration = tf.constant([2.0] * batch_size)
    rate = tf.constant([3.0] * batch_size)
    concentration_v = 2.0
    rate_v = 3.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)
    log_pdf = gamma.log_prob(x)
    self.assertEqual(log_pdf.shape, (6,))
    pdf = gamma.prob(x)
    self.assertEqual(pdf.shape, (6,))
    expected_log_pdf = sp_stats.gamma.logpdf(
        x, concentration_v, scale=1 / rate_v)
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

  def testGammaLogPDFBoundary(self):
    # When concentration = 1, we have an exponential distribution. Check that at
    # 0 we have finite log prob.
    rate = np.array([0.1, 0.5, 1., 2., 5., 10.], dtype=np.float32)
    gamma = tfd.Gamma(concentration=1., rate=rate, validate_args=True)
    log_pdf = gamma.log_prob(0.)
    self.assertAllClose(np.log(rate), self.evaluate(log_pdf))

    gamma = tfd.Gamma(concentration=[2., 4., 0.5], rate=6., validate_args=True)
    log_pdf = self.evaluate(gamma.log_prob(0.))
    self.assertAllNegativeInf(log_pdf[:2])
    self.assertAllPositiveInf(log_pdf[2])

    pdf = self.evaluate(gamma.prob(0.))
    self.assertAllPositiveInf(pdf[2])
    self.assertAllFinite(pdf[:2])

  def testAssertsValidSample(self):
    g = tfd.Gamma(concentration=2., rate=3., validate_args=True)
    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(g.log_prob(-.1))

  def testSampleWithPartiallyDefinedShapeEndingInOne(self):
    param = tf.Variable(np.ones((8, 16, 16, 1)),
                        shape=tf.TensorShape([None, 16, 16, 1]))
    self.evaluate(param.initializer)
    samples = self.evaluate(
        tfd.Gamma(param, rate=param).sample(seed=test_util.test_seed()))
    self.assertEqual(samples.shape, (8, 16, 16, 1))
    samples = self.evaluate(
        tfd.Gamma(param, log_rate=param).sample(seed=test_util.test_seed()))
    self.assertEqual(samples.shape, (8, 16, 16, 1))

  def testGammaLogPDFMultidimensional(self):
    batch_size = 6
    concentration = tf.constant([[2.0, 4.0]] * batch_size)
    rate = tf.constant([[3.0, 4.0]] * batch_size)
    concentration_v = np.array([2.0, 4.0])
    rate_v = np.array([3.0, 4.0])
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)
    gamma_lr = tfd.Gamma(
        concentration=concentration, log_rate=tf.math.log(rate),
        validate_args=True)
    log_pdf = gamma.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    pdf = gamma.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))
    expected_log_pdf = sp_stats.gamma.logpdf(
        x, concentration_v, scale=1 / rate_v)
    self.assertAllClose(log_pdf_values, expected_log_pdf)
    self.assertAllClose(pdf_values, np.exp(expected_log_pdf))
    self.assertAllClose(gamma_lr.log_prob(x), expected_log_pdf)
    self.assertAllClose(gamma_lr.prob(x), np.exp(expected_log_pdf))

  def testGammaLogPDFMultidimensionalBroadcasting(self):
    batch_size = 6
    concentration = tf.constant([[2.0, 4.0]] * batch_size)
    rate = tf.constant(3.0)
    concentration_v = np.array([2.0, 4.0])
    rate_v = 3.0
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)
    log_pdf = gamma.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    pdf = gamma.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))

    expected_log_pdf = sp_stats.gamma.logpdf(
        x, concentration_v, scale=1 / rate_v)
    self.assertAllClose(log_pdf_values, expected_log_pdf)
    self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testGammaCDF(self):
    batch_size = 6
    concentration = tf.constant([2.0] * batch_size)
    rate = tf.constant([3.0] * batch_size)
    concentration_v = 2.0
    rate_v = 3.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)
    gamma_lr = tfd.Gamma(
        concentration=concentration, log_rate=tf.math.log(rate),
        validate_args=True)
    cdf = gamma.cdf(x)
    self.assertEqual(cdf.shape, (6,))
    expected_cdf = sp_stats.gamma.cdf(x, concentration_v, scale=1 / rate_v)
    self.assertAllClose(cdf, expected_cdf)
    self.assertAllClose(gamma_lr.cdf(x), expected_cdf)

  def testGammaQuantile(self):
    batch_size = 6
    concentration = np.linspace(
        1, 10., batch_size).astype(np.float32)[..., np.newaxis]
    rate = np.linspace(3., 7., batch_size).astype(np.float32)[..., np.newaxis]
    x = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.5, 0.7], dtype=np.float32)

    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)
    gamma_lr = tfd.Gamma(
        concentration=concentration, log_rate=tf.math.log(rate),
        validate_args=True)
    quantile = gamma.quantile(x)
    self.assertEqual(quantile.shape, (6, 7))
    expected_quantile = sp_stats.gamma.ppf(x, concentration, scale=1 / rate)
    self.assertAllClose(quantile, expected_quantile)
    self.assertAllClose(gamma_lr.quantile(x), expected_quantile)

  def testGammaMean(self):
    concentration_v = np.array([1.0, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    gamma = tfd.Gamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)
    gamma_lr = tfd.Gamma(
        concentration=concentration_v, log_rate=np.log(rate_v),
        validate_args=True)
    self.assertEqual(gamma.mean().shape, (3,))
    expected_means = sp_stats.gamma.mean(concentration_v, scale=1 / rate_v)
    self.assertAllClose(self.evaluate(gamma.mean()), expected_means)
    self.assertAllClose(self.evaluate(gamma_lr.mean()), expected_means)

  def testGammaModeAllowNanStatsIsFalseWorksWhenAllBatchMembersAreDefined(self):
    concentration_v = np.array([5.5, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    gamma = tfd.Gamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)
    gamma_lr = tfd.Gamma(
        concentration=concentration_v, log_rate=np.log(rate_v),
        validate_args=True)
    expected_modes = (concentration_v - 1) / rate_v
    self.assertEqual(gamma.mode().shape, (3,))
    self.assertAllClose(self.evaluate(gamma.mode()), expected_modes)
    self.assertAllClose(self.evaluate(gamma_lr.mode()), expected_modes)

  def testGammaModeAllowNanStatsFalseRaisesForUndefinedBatchMembers(self):
    # Mode will not be defined for the first entry.
    concentration_v = np.array([0.5, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    gamma = tfd.Gamma(concentration=concentration_v, rate=rate_v,
                      allow_nan_stats=False, validate_args=True)
    gamma_lr = tfd.Gamma(concentration=concentration_v, log_rate=np.log(rate_v),
                         allow_nan_stats=False, validate_args=True)
    with self.assertRaisesOpError(
        'Mode not defined when any concentration <= 1.'):
      self.evaluate(gamma.mode())
    with self.assertRaisesOpError(
        'Mode not defined when any concentration <= 1.'):
      self.evaluate(gamma_lr.mode())

  def testGammaModeAllowNanStatsIsTrueReturnsNaNforUndefinedBatchMembers(self):
    # Mode will not be defined for the first entry.
    concentration_v = np.array([0.5, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    gamma = tfd.Gamma(concentration=concentration_v, rate=rate_v,
                      allow_nan_stats=True, validate_args=True)
    gamma_lr = tfd.Gamma(concentration=concentration_v, log_rate=np.log(rate_v),
                         allow_nan_stats=True, validate_args=True)
    expected_modes = (concentration_v - 1) / rate_v
    expected_modes[0] = np.nan
    self.assertEqual(gamma.mode().shape, (3,))
    self.assertAllClose(self.evaluate(gamma.mode()), expected_modes)
    self.assertAllClose(self.evaluate(gamma_lr.mode()), expected_modes)

  def testGammaVariance(self):
    concentration_v = np.array([1.0, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    gamma = tfd.Gamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)
    gamma_lr = tfd.Gamma(
        concentration=concentration_v, log_rate=np.log(rate_v),
        validate_args=True)
    self.assertEqual(gamma.variance().shape, (3,))
    expected_variances = sp_stats.gamma.var(concentration_v, scale=1 / rate_v)
    self.assertAllClose(self.evaluate(gamma.variance()), expected_variances)
    self.assertAllClose(self.evaluate(gamma_lr.variance()), expected_variances)

  def testGammaStd(self):
    concentration_v = np.array([1.0, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    gamma = tfd.Gamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)
    gamma_lr = tfd.Gamma(
        concentration=concentration_v, log_rate=np.log(rate_v),
        validate_args=True)
    self.assertEqual(gamma.stddev().shape, (3,))
    expected_stddev = sp_stats.gamma.std(concentration_v, scale=1. / rate_v)
    self.assertAllClose(self.evaluate(gamma.stddev()), expected_stddev)
    self.assertAllClose(self.evaluate(gamma_lr.stddev()), expected_stddev)

  def testGammaEntropy(self):
    concentration_v = np.array([1.0, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    gamma = tfd.Gamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)
    gamma_lr = tfd.Gamma(
        concentration=concentration_v, log_rate=np.log(rate_v),
        validate_args=True)
    self.assertEqual(gamma.entropy().shape, (3,))
    expected_entropy = sp_stats.gamma.entropy(concentration_v, scale=1 / rate_v)
    self.assertAllClose(self.evaluate(gamma.entropy()), expected_entropy)
    self.assertAllClose(self.evaluate(gamma_lr.entropy()), expected_entropy)

  def testGammaSampleSmallconcentration(self):
    concentration_v = 0.05
    rate_v = 1.0
    concentration = tf.constant(concentration_v)
    rate = tf.constant(rate_v)
    n = 100000
    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)
    samples = gamma.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertTrue(self._kstest(concentration_v, rate_v, sample_values))
    self.assertAllClose(
        sample_values.mean(),
        sp_stats.gamma.mean(concentration_v, scale=1 / rate_v),
        atol=.01)
    self.assertAllClose(
        sample_values.var(),
        sp_stats.gamma.var(concentration_v, scale=1 / rate_v),
        atol=.15)

  def testGammaSample(self):
    concentration_v = 4.0
    rate_v = 3.0
    concentration = tf.constant(concentration_v)
    rate = tf.constant(rate_v)
    n = 100000
    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)
    samples = gamma.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertTrue(self._kstest(concentration_v, rate_v, sample_values))
    self.assertAllClose(
        sample_values.mean(),
        sp_stats.gamma.mean(concentration_v, scale=1 / rate_v),
        atol=.01)
    self.assertAllClose(
        sample_values.var(),
        sp_stats.gamma.var(concentration_v, scale=1 / rate_v),
        atol=.15)

  def testGammaSampleZeroAndNegativeParameters(self):
    gamma = tfd.Gamma([1., 2.], 1., validate_args=False)
    seed_stream = test_util.test_seed_stream()
    samples = self.evaluate(gamma.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllFinite(samples)

    gamma = tfd.Gamma([0., 2.], 1., validate_args=False)
    samples = self.evaluate(gamma.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllEqual([s in [0, np.finfo(np.float32).tiny]
                         for s in samples], [True, False])

    gamma = tfd.Gamma([-0.001, 2.], 1., validate_args=False)
    samples = self.evaluate(gamma.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllEqual([np.isnan(s) for s in samples], [True, False])

    gamma = tfd.Gamma([1., -1.], 1., validate_args=False)
    samples = self.evaluate(gamma.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllEqual([np.isnan(s) for s in samples], [False, True])

    gamma = tfd.Gamma([1., 2.], 0., validate_args=False)
    samples = self.evaluate(gamma.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllPositiveInf(samples)

    gamma = tfd.Gamma([1., 2.], -1., validate_args=False)
    samples = self.evaluate(gamma.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllNan(samples)

  @test_util.numpy_disable_gradient_test
  def testGammaFullyReparameterized(self):
    concentration = tf.constant(4.0)
    rate = tf.constant(3.0)
    _, [grad_concentration, grad_rate] = tfp.math.value_and_gradient(
        lambda a, b: tfd.Gamma(concentration=a, rate=b, validate_args=True).  # pylint: disable=g-long-lambda
        sample(100, seed=test_util.test_seed()), [concentration, rate])
    self.assertIsNotNone(grad_concentration)
    self.assertIsNotNone(grad_rate)

  @test_util.numpy_disable_gradient_test
  def testCompareGradientToTfRandomGammaGradient(self):
    n_concentration = 4
    concentration_v = tf.constant(
        np.array([np.arange(1, n_concentration+1, dtype=np.float32)]))
    n_rate = 2
    rate_v = tf.constant(
        np.array([np.arange(1, n_rate+1, dtype=np.float32)]).T)
    num_samples = int(1e5)

    def tfp_gamma(a, b):
      return tfd.Gamma(concentration=a, rate=b, validate_args=True).sample(
          num_samples, seed=test_util.test_seed())

    _, [grad_concentration, grad_rate] = self.evaluate(
        tfp.math.value_and_gradient(tfp_gamma, [concentration_v, rate_v]))

    def tf_gamma(a, b):
      return tf.random.gamma(
          shape=[num_samples], alpha=a, beta=b, seed=test_util.test_seed())

    _, [grad_concentration_tf, grad_rate_tf] = self.evaluate(
        tfp.math.value_and_gradient(tf_gamma, [concentration_v, rate_v]))

    self.assertEqual(grad_concentration.shape, grad_concentration_tf.shape)
    self.assertEqual(grad_rate.shape, grad_rate_tf.shape)
    self.assertAllClose(grad_concentration, grad_concentration_tf, rtol=1e-2)
    self.assertAllClose(grad_rate, grad_rate_tf, rtol=1e-2)

  @test_util.numpy_disable_gradient_test
  def testCompareGradientToTfRandomGammaGradientWithNonLinearity(self):
    # Test that the gradient is correctly computed through a non-linearity.

    n_concentration = 4
    concentration_v = tf.constant(
        np.array([np.arange(1, n_concentration + 1, dtype=np.float32)]))
    n_rate = 2
    rate_v = tf.constant(
        np.array([np.arange(1, n_rate + 1, dtype=np.float32)]).T)
    num_samples = int(1e5)

    def tfp_gamma(a, b):
      return tf.math.square(
          tfd.Gamma(concentration=a, rate=b, validate_args=True).sample(
              num_samples, seed=test_util.test_seed()))

    _, [grad_concentration, grad_rate] = self.evaluate(
        tfp.math.value_and_gradient(tfp_gamma, [concentration_v, rate_v]))

    def tf_gamma(a, b):
      return tf.math.square(tf.random.gamma(
          shape=[num_samples],
          alpha=a,
          beta=b,
          seed=test_util.test_seed()))

    _, [grad_concentration_tf, grad_rate_tf] = self.evaluate(
        tfp.math.value_and_gradient(tf_gamma, [concentration_v, rate_v]))

    self.assertEqual(grad_concentration.shape, grad_concentration_tf.shape)
    self.assertEqual(grad_rate.shape, grad_rate_tf.shape)
    self.assertAllClose(grad_concentration, grad_concentration_tf, rtol=2e-2)
    self.assertAllClose(grad_rate, grad_rate_tf, rtol=2e-2)

  def testGammaSampleMultiDimensional(self):
    n_concentration = 50
    concentration_v = np.array(
        [np.arange(1, n_concentration+1, dtype=np.float32)])  # 1 x 50
    n_rate = 10
    rate_v = np.array([np.arange(1, n_rate+1, dtype=np.float32)]).T  # 10 x 1
    gamma = tfd.Gamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)

    n = 10000

    samples = gamma.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    self.assertEqual(samples.shape, (n, n_rate, n_concentration))
    self.assertEqual(sample_values.shape, (n, n_rate, n_concentration))
    zeros = np.zeros_like(concentration_v + rate_v)  # 10 x 100
    concentration_bc = concentration_v + zeros
    rate_bc = rate_v + zeros
    self.assertAllClose(
        sample_values.mean(axis=0),
        sp_stats.gamma.mean(concentration_bc, scale=1 / rate_bc),
        atol=0.,
        rtol=.05)
    self.assertAllClose(
        sample_values.var(axis=0),
        sp_stats.gamma.var(concentration_bc, scale=1 / rate_bc),
        atol=10.0,
        rtol=0.)
    fails = 0
    trials = 0
    for ai, a in enumerate(np.reshape(concentration_v, [-1])):
      for bi, b in enumerate(np.reshape(rate_v, [-1])):
        s = sample_values[:, bi, ai]
        trials += 1
        fails += 0 if self._kstest(a, b, s) else 1
    self.assertLess(fails, trials * 0.03)

  def _kstest(self, concentration, rate, samples):
    # Uses the Kolmogorov-Smirnov test for goodness of fit.
    ks, _ = sp_stats.kstest(
        samples, sp_stats.gamma(concentration, scale=1 / rate).cdf)
    # Return True when the test passes.
    return ks < 0.02

  def testGammaPdfOfSampleMultiDims(self):
    gamma = tfd.Gamma(
        concentration=[7., 11.], rate=[[5.], [6.]], validate_args=True)
    num = 50000
    samples = gamma.sample(num, seed=test_util.test_seed())
    pdfs = gamma.prob(samples)
    sample_vals, pdf_vals = self.evaluate([samples, pdfs])
    self.assertEqual(samples.shape, (num, 2, 2))
    self.assertEqual(pdfs.shape, (num, 2, 2))
    self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.02)
    self.assertAllClose(
        sp_stats.gamma.mean([[7., 11.], [7., 11.]],
                            scale=1 / np.array([[5., 5.], [6., 6.]])),
        sample_vals.mean(axis=0),
        atol=.1)
    self.assertAllClose(
        sp_stats.gamma.var([[7., 11.], [7., 11.]],
                           scale=1 / np.array([[5., 5.], [6., 6.]])),
        sample_vals.var(axis=0),
        atol=.1)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1e-3):
    s_p = zip(sample_vals, pdf_vals)
    prev = (0, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testGammaNonPositiveInitializationParamsRaises(self):
    concentration_v = tf.constant(0.0, name='concentration')
    rate_v = tf.constant(1.0, name='rate')
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      gamma = tfd.Gamma(
          concentration=concentration_v, rate=rate_v, validate_args=True)
      self.evaluate(gamma.mean())
    concentration_v = tf.constant(1.0, name='concentration')
    rate_v = tf.constant(0.0, name='rate')
    with self.assertRaisesOpError('Argument `rate` must be positive.'):
      gamma = tfd.Gamma(
          concentration=concentration_v, rate=rate_v, validate_args=True)
      self.evaluate(gamma.mean())

  def testGammaGammaKL(self):
    concentration0 = np.array([3.])
    rate0 = np.array([1., 2., 3., 1.5, 2.5, 3.5])

    concentration1 = np.array([0.4])
    rate1 = np.array([0.5, 1., 1.5, 2., 2.5, 3.])

    # Build graph.
    g0 = tfd.Gamma(concentration=concentration0, rate=rate0, validate_args=True)
    g0lr = tfd.Gamma(concentration=concentration0, log_rate=np.log(rate0),
                     validate_args=True)
    g1 = tfd.Gamma(concentration=concentration1, rate=rate1, validate_args=True)
    g1lr = tfd.Gamma(concentration=concentration1, log_rate=np.log(rate1),
                     validate_args=True)

    for d0, d1 in (g0, g1), (g0lr, g1), (g0, g1lr), (g0lr, g1lr):
      x = d0.sample(int(1e4), seed=test_util.test_seed())
      kl_sample = tf.reduce_mean(d0.log_prob(x) - d1.log_prob(x), axis=0)
      kl_actual = tfd.kl_divergence(d0, d1)

      # Execute graph.
      [kl_sample_, kl_actual_] = self.evaluate([kl_sample, kl_actual])

      self.assertEqual(rate0.shape, kl_actual.shape)

      kl_expected = ((
          concentration0 - concentration1) * sp_special.digamma(concentration0)
                     + sp_special.gammaln(concentration1)
                     - sp_special.gammaln(concentration0)
                     + concentration1 * np.log(rate0)
                     - concentration1 * np.log(rate1)
                     + concentration0 * (rate1 / rate0 - 1.))

      self.assertAllClose(kl_expected, kl_actual_, atol=0., rtol=1e-6)
      self.assertAllClose(kl_sample_, kl_actual_, atol=0., rtol=1e-1)

  @test_util.tf_tape_safety_test
  def testGradientThroughConcentration(self):
    concentration = tf.Variable(3.)
    d = tfd.Gamma(concentration=concentration, rate=5., validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 4.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  @test_util.jax_disable_variable_test
  def testAssertsPositiveConcentration(self):
    concentration = tf.Variable([1., 2., -3.])
    self.evaluate(concentration.initializer)
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      d = tfd.Gamma(concentration=concentration, rate=[5.], validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveConcentrationAfterMutation(self):
    concentration = tf.Variable([1., 2., 3.])
    self.evaluate(concentration.initializer)
    d = tfd.Gamma(concentration=concentration, rate=[5.], validate_args=True)
    self.evaluate(d.sample(seed=test_util.test_seed()))
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      with tf.control_dependencies([concentration.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  @test_util.tf_tape_safety_test
  def testGradientThroughRate(self):
    rate = tf.Variable(3.)
    d = tfd.Gamma(concentration=1., rate=rate, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 4.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveRate(self):
    rate = tf.Variable([1., 2., -3.])
    self.evaluate(rate.initializer)
    with self.assertRaisesOpError('Argument `rate` must be positive.'):
      d = tfd.Gamma(concentration=[5.], rate=rate, validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveRateAfterMutation(self):
    rate = tf.Variable([1., 2., 3.])
    self.evaluate(rate.initializer)
    d = tfd.Gamma(concentration=[3.], rate=rate, validate_args=True)
    self.evaluate(d.sample(seed=test_util.test_seed()))
    with self.assertRaisesOpError('Argument `rate` must be positive.'):
      with tf.control_dependencies([rate.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testSupportBijectorOutsideRange(self):
    dist = tfd.Gamma(
        concentration=[3.], rate=[3., 2., 5.4], validate_args=True)
    x = np.array([-4.2, -0.3, -1e-6])
    bijector_inverse_x = dist.experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))


@test_util.test_graph_and_eager_modes
class GammaSamplingTest(test_util.TestCase):

  @test_util.jax_disable_test_missing_functionality('tf stateless_gamma')
  def testSampleCPU(self):
    self.skipTest('b/179283344')
    with tf.device('CPU'):
      _, runtime = self.evaluate(
          gamma_lib.random_gamma_with_runtime(
              shape=tf.constant([], dtype=tf.int32),
              concentration=tf.constant(1.),
              seed=test_util.test_seed()))
    self.assertEqual(implementation_selection._RUNTIME_CPU, runtime)

  def testSampleGPU(self):
    if not tf.test.is_gpu_available():
      self.skipTest('no GPU')
    with tf.device('GPU'):
      _, runtime = self.evaluate(gamma_lib.random_gamma_with_runtime(
          shape=tf.constant([], dtype=tf.int32),
          concentration=tf.constant(1.),
          seed=test_util.test_seed()))
    self.assertEqual(implementation_selection._RUNTIME_DEFAULT, runtime)

  def testSampleXLA(self):
    self.skip_if_no_xla()
    if not tf.executing_eagerly(): return  # jit_compile is eager-only.
    concentration = np.exp(np.random.rand(4, 3).astype(np.float32))
    rate = np.exp(np.random.rand(4, 3).astype(np.float32))
    dist = tfd.Gamma(concentration=concentration, rate=rate, validate_args=True)
    # Verify the compile succeeds going all the way through the distribution.
    self.evaluate(
        tf.function(lambda: dist.sample(5, seed=test_util.test_seed()),
                    jit_compile=True)())
    # Also test the low-level sampler and verify the XLA-friendly variant.
    # TODO(bjp): functools.partial, after eliminating PY2 which breaks
    # tf_inspect in interesting ways:
    # ValueError: Some arguments ['concentration', 'rate'] do not have default
    # value, but they are positioned after those with default values. This can
    # not be expressed with ArgSpec.
    scalar_gamma = tf.function(
        lambda **kwds: gamma_lib.random_gamma_with_runtime(shape=[], **kwds),
        jit_compile=True)
    _, runtime = self.evaluate(
        scalar_gamma(
            concentration=tf.constant(1.),
            seed=test_util.test_seed()))
    self.assertEqual(implementation_selection._RUNTIME_DEFAULT, runtime)

  def testSampleGammaLowConcentration(self):
    concentration = np.linspace(0.1, 1., 10)
    rate = np.float64(1.)
    num_samples = int(1e5)
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    samples = gamma_lib._random_gamma_noncpu(
        shape=[num_samples, 10],
        concentration=concentration,
        rate=rate,
        seed=test_util.test_seed())

    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)
    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(
            samples,
            gamma.cdf,
            false_fail_rate=1e-9))

    self.assertAllClose(
        self.evaluate(tf.math.reduce_mean(samples, axis=0)),
        sp_stats.gamma.mean(concentration, scale=1 / rate),
        rtol=0.04)
    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        sp_stats.gamma.var(concentration, scale=1 / rate),
        rtol=0.07)

  def testSampleGammaHighConcentration(self):
    concentration = np.linspace(10., 20., 10)
    rate = np.float64(1.)
    num_samples = int(1e5)
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    samples = gamma_lib._random_gamma_noncpu(
        shape=[num_samples, 10],
        concentration=concentration,
        rate=rate,
        seed=test_util.test_seed())

    gamma = tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True)
    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(
            samples,
            gamma.cdf,
            false_fail_rate=1e-9))

    self.assertAllClose(
        self.evaluate(tf.math.reduce_mean(samples, axis=0)),
        sp_stats.gamma.mean(concentration, scale=1 / rate),
        rtol=0.01)
    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        sp_stats.gamma.var(concentration, scale=1 / rate),
        rtol=0.05)

  @test_util.numpy_disable_gradient_test
  def testSampleGammaLogRateLogSpaceDerivatives(self):
    conc = tf.constant(np.linspace(.8, 1.2, 5), tf.float64)
    rate = np.linspace(.5, 2, 5)
    np.random.shuffle(rate)
    rate = tf.constant(rate, tf.float64)
    n = int(1e5)

    seed = test_util.test_seed()
    # pylint: disable=g-long-lambda
    lambdas = [  # Each should sample the same distribution.
        lambda c, r: gamma_lib.random_gamma(
            [n], c, r, seed=seed, log_space=True),
        lambda c, r: gamma_lib.random_gamma(
            [n], c, log_rate=tf.math.log(r), seed=seed, log_space=True),
        lambda c, r: tf.math.log(gamma_lib.random_gamma(
            [n], c, r, seed=seed)),
        lambda c, r: tf.math.log(gamma_lib.random_gamma(
            [n], c, log_rate=tf.math.log(r), seed=seed)),
    ]
    # pylint: enable=g-long-lambda
    samps = []
    dconc = []
    drate = []
    for fn in lambdas:
      # Take samples without the nonlinearity.
      samps.append(fn(conc, rate))
      # We compute gradient through a nonlinearity to catch a class of errors.
      _, (dc_i, dr_i) = tfp.math.value_and_gradient(
          lambda c, r: tf.reduce_mean(tf.square(fn(c, r))), (conc, rate))  # pylint: disable=cell-var-from-loop
      dconc.append(dc_i)
      drate.append(dr_i)

    # Assert d rate correctness. Note that the non-logspace derivative for rate
    # depends on the realized sample whereas the logspace one does not. Also,
    # comparing grads with differently-placed log/exp is numerically perilous.
    self.assertAllClose(drate[0], drate[1], rtol=0.06)
    self.assertAllClose(drate[0], drate[2], rtol=0.06)
    self.assertAllClose(drate[1], drate[3], rtol=0.06)

    # Assert sample correctness. If incorrect, dconc will be incorrect.
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        n)
    equiv_dist = tfb.Log()(tfd.Gamma(conc, rate))
    self.evaluate(st.assert_true_cdf_equal_by_dkwm(
        samps[0], equiv_dist.cdf, false_fail_rate=1e-9))
    self.evaluate(st.assert_true_cdf_equal_by_dkwm(
        samps[1], equiv_dist.cdf, false_fail_rate=1e-9))
    self.evaluate(st.assert_true_cdf_equal_by_dkwm(
        samps[2], equiv_dist.cdf, false_fail_rate=1e-9))
    self.evaluate(st.assert_true_cdf_equal_by_dkwm(
        samps[3], equiv_dist.cdf, false_fail_rate=1e-9))

    # Assert d concentration correctness. These are sensitive to sample values,
    # which are more strongly effected by the log/exp, thus looser tolerances.
    self.assertAllClose(dconc[0], dconc[1], rtol=0.06)
    self.assertAllClose(dconc[0], dconc[2], rtol=0.06)
    self.assertAllClose(dconc[1], dconc[3], rtol=0.06)

  def testSampleGammaLogSpace(self):
    concentration = np.linspace(.1, 2., 10)
    rate = np.linspace(.5, 2, 10)
    np.random.shuffle(rate)
    num_samples = int(1e5)
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    samples = gamma_lib.random_gamma(
        [num_samples],
        concentration,
        rate,
        seed=test_util.test_seed(),
        log_space=True)

    exp_gamma = tfb.Log()(tfd.Gamma(
        concentration=concentration, rate=rate, validate_args=True))
    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(
            samples,
            exp_gamma.cdf,
            false_fail_rate=1e-9))

    self.assertAllClose(
        self.evaluate(tf.math.reduce_mean(samples, axis=0)),
        tf.math.digamma(concentration) - tf.math.log(rate),
        rtol=0.02, atol=0.01)
    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        tf.math.polygamma(1., concentration),
        rtol=0.05)

  @parameterized.named_parameters(
      dict(testcase_name='_float32', dtype=np.float32),
      dict(testcase_name='_float64', dtype=np.float64))
  @test_util.numpy_disable_gradient_test
  def testCompareToExplicitGradient(self, dtype):
    """Compare to the explicit reparameterization derivative.

    Defining x to be the output from a gamma sampler with rate=1, and defining y
    to be the actual gamma sample (defined by y = x / rate), we have:

    dx / dconcentration = d igammainv(concentration, x) / dconcentration,
    where u = igamma(concentration, x).

    Therefore, we have:

    dy / dconcentration = (1 / rate) * d igammainv(
      concentration, y * rate) / dconcentration,
    where u = igamma(concentration, y * rate)

    We also have dy / drate = -(x / rate^2) = -y / rate.

    Args:
      dtype: TensorFlow dtype to perform the computations in.
    """
    concentration_n = 4
    concentration_np = np.arange(
        concentration_n).astype(dtype)[..., np.newaxis] + 1.
    concentration = tf.constant(concentration_np)
    rate_n = 3
    rate_np = np.arange(rate_n).astype(dtype) + 1.
    rate = tf.constant(rate_np)
    num_samples = 2

    def gen_samples(concentration, rate):
      return tfd.Gamma(concentration, rate).sample(
          num_samples, seed=test_util.test_seed())

    samples, [concentration_grad, rate_grad] = self.evaluate(
        tfp.math.value_and_gradient(gen_samples, [concentration, rate]))
    self.assertEqual(samples.shape, (num_samples, concentration_n, rate_n))
    self.assertEqual(concentration_grad.shape, concentration.shape)
    self.assertEqual(rate_grad.shape, rate.shape)
    # Sum over the first 2 dimensions since these are batch dimensions.
    self.assertAllClose(rate_grad, np.sum(-samples / rate_np, axis=(0, 1)))
    # Compute the gradient by computing the derivative of gammaincinv
    # over each entry and summing.
    def expected_grad(s, c, r):
      u = sp_special.gammainc(c, s * r)
      delta = 1e-4
      return sp_misc.derivative(
          lambda x: sp_special.gammaincinv(x, u), c, dx=delta * c) / r

    self.assertAllClose(
        concentration_grad,
        np.sum(expected_grad(
            samples,
            concentration_np,
            rate_np), axis=(0, 2))[..., np.newaxis], rtol=1e-3)

  def testPdfOutsideSupport(self):
    def mk_gamma(c):
      return tfd.Gamma(c, 1, force_probs_to_zero_outside_support=True)
    self.assertAllEqual(mk_gamma(.99).log_prob(-.1), -float('inf'))
    self.assertAllEqual(mk_gamma(.99).log_prob(0), float('inf'))
    self.assertAllGreater(mk_gamma(.99).log_prob(0.1), -float('inf'))
    self.assertAllEqual(mk_gamma(1).log_prob(-.1), -float('inf'))
    self.assertAllClose(mk_gamma(1).log_prob(0), 0.)
    self.assertAllGreater(mk_gamma(1).log_prob(.1), -float('inf'))
    self.assertAllEqual(mk_gamma(1.01).log_prob(-.1), -float('inf'))
    self.assertAllEqual(mk_gamma(1.01).log_prob(0), -float('inf'))
    self.assertAllGreater(mk_gamma(1.01).log_prob(.1), -float('inf'))


if __name__ == '__main__':
  test_util.main()
