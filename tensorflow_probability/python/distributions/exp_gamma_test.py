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
# Dependency imports
from absl.testing import parameterized
import numpy as np
from scipy import misc as sp_misc
from scipy import special as sp_special
from scipy import stats as sp_stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import exp_gamma as exp_gamma_lib
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class ExpGammaTest(test_util.TestCase):

  def testShape(self):
    concentration = tf.constant([3.0] * 5)
    rate = tf.constant(11.0)
    d = tfd.ExpGamma(concentration=concentration, rate=rate, validate_args=True)

    self.assertEqual(self.evaluate(d.batch_shape_tensor()), (5,))
    self.assertEqual(d.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(d.event_shape_tensor()), [])
    self.assertEqual(d.event_shape, tf.TensorShape([]))

  def testLogPDF(self):
    batch_size = 6
    concentration = tf.constant([2.0] * batch_size)
    rate = tf.constant([3.0] * batch_size)
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    d = tfd.ExpGamma(concentration=concentration, rate=rate, validate_args=True)
    log_pdf = d.log_prob(x)
    self.assertEqual(log_pdf.shape, (6,))
    pdf = d.prob(x)
    self.assertEqual(pdf.shape, (6,))
    oracle_d = tfb.Log()(tfd.Gamma(concentration=concentration, rate=rate))
    expected_log_pdf = oracle_d.log_prob(x)
    expected_pdf = oracle_d.prob(x)
    self.assertAllClose(log_pdf, expected_log_pdf)
    self.assertAllClose(pdf, expected_pdf)

  def testSampleWithPartiallyDefinedShapeEndingInOne(self):
    param = tf.Variable(np.ones((8, 16, 16, 1)),
                        shape=tf.TensorShape([None, 16, 16, 1]))
    self.evaluate(param.initializer)
    samples = self.evaluate(
        tfd.ExpGamma(param, rate=param).sample(seed=test_util.test_seed()))
    self.assertEqual(samples.shape, (8, 16, 16, 1))
    samples = self.evaluate(
        tfd.ExpGamma(param, log_rate=param).sample(seed=test_util.test_seed()))
    self.assertEqual(samples.shape, (8, 16, 16, 1))

  def testLogPDFMultidimensional(self):
    batch_size = 6
    concentration = tf.constant([[2.0, 4.0]] * batch_size)
    rate = tf.constant([[3.0, 4.0]] * batch_size)
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    d = tfd.ExpGamma(concentration=concentration, rate=rate, validate_args=True)
    d_lr = tfd.ExpGamma(
        concentration=concentration,
        log_rate=tf.math.log(rate),
        validate_args=True)
    log_pdf = d.log_prob(x)
    self.assertEqual(log_pdf.shape, (6, 2))
    pdf = d.prob(x)
    self.assertEqual(pdf.shape, (6, 2))
    oracle_d = tfb.Log()(tfd.Gamma(concentration=concentration, rate=rate))
    expected_log_pdf = oracle_d.log_prob(x)
    expected_pdf = oracle_d.prob(x)
    self.assertAllClose(log_pdf, expected_log_pdf)
    self.assertAllClose(pdf, expected_pdf)
    self.assertAllClose(d_lr.log_prob(x), expected_log_pdf)
    self.assertAllClose(d_lr.prob(x), tf.math.exp(expected_log_pdf))

  def testLogPDFMultidimensionalBroadcasting(self):
    batch_size = 6
    concentration = tf.constant([[2.0, 4.0]] * batch_size)
    rate = tf.constant(3.0)
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    d = tfd.ExpGamma(
        concentration=concentration, rate=rate, validate_args=True)
    log_pdf = d.log_prob(x)
    self.assertEqual(log_pdf.shape, (6, 2))
    pdf = d.prob(x)
    self.assertEqual(pdf.shape, (6, 2))

    oracle_d = tfb.Log()(tfd.Gamma(concentration=concentration, rate=rate))
    expected_log_pdf = oracle_d.log_prob(x)
    expected_pdf = oracle_d.prob(x)
    self.assertAllClose(log_pdf, expected_log_pdf)
    self.assertAllClose(pdf, expected_pdf)

  def testCDF(self):
    batch_size = 6
    concentration = tf.constant([2.0] * batch_size)
    rate = tf.constant([3.0] * batch_size)
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

    d = tfd.ExpGamma(
        concentration=concentration, rate=rate, validate_args=True)
    d_lr = tfd.ExpGamma(
        concentration=concentration, log_rate=tf.math.log(rate),
        validate_args=True)
    cdf = d.cdf(x)
    self.assertEqual(cdf.shape, (6,))
    expected_cdf = tfb.Log()(tfd.Gamma(concentration=concentration,
                                       rate=rate)).cdf(x)
    self.assertAllClose(cdf, expected_cdf)
    self.assertAllClose(d_lr.cdf(x), expected_cdf)

  def testQuantile(self):
    batch_size = 6
    concentration = np.linspace(
        1., 12., batch_size).astype(np.float32)[..., np.newaxis]
    rate = np.linspace(3., 18., batch_size).astype(np.float32)[..., np.newaxis]
    x = np.linspace(0., 1., 13).astype(np.float32)
    d = tfd.ExpGamma(
        concentration=concentration, rate=rate, validate_args=True)
    d_lr = tfd.ExpGamma(
        concentration=concentration, log_rate=tf.math.log(rate),
        validate_args=True)
    quantile = d.quantile(x)
    self.assertEqual(quantile.shape, (6, 13))
    expected_quantile = tfb.Log()(tfd.Gamma(concentration=concentration,
                                            rate=rate)).quantile(x)
    self.assertAllClose(quantile, expected_quantile)
    self.assertAllClose(d_lr.quantile(x), expected_quantile)

  def testMean(self):
    concentration_v = np.array([1.0, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    d = tfd.ExpGamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)
    d_lr = tfd.ExpGamma(
        concentration=concentration_v, log_rate=np.log(rate_v),
        validate_args=True)
    self.assertEqual(d.mean().shape, (3,))
    expected_means = sp_stats.loggamma.mean(concentration_v,
                                            loc=-np.log(rate_v))
    self.assertAllClose(d.mean(), expected_means)
    self.assertAllClose(d_lr.mean(), expected_means)

  def testVariance(self):
    concentration_v = np.array([1.0, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    d = tfd.ExpGamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)
    d_lr = tfd.ExpGamma(
        concentration=concentration_v, log_rate=np.log(rate_v),
        validate_args=True)
    self.assertEqual(d.variance().shape, (3,))
    expected_variances = sp_stats.loggamma.var(concentration_v,
                                               loc=-np.log(rate_v))
    self.assertAllClose(self.evaluate(d.variance()), expected_variances)
    self.assertAllClose(self.evaluate(d_lr.variance()), expected_variances)

  def testStd(self):
    concentration_v = np.array([1.0, 3.0, 2.5])
    rate_v = np.array([1.0, 4.0, 5.0])
    d = tfd.ExpGamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)
    d_lr = tfd.ExpGamma(
        concentration=concentration_v, log_rate=np.log(rate_v),
        validate_args=True)
    self.assertEqual(d.stddev().shape, (3,))
    expected_stddev = sp_stats.loggamma.std(concentration_v,
                                            loc=-np.log(rate_v))
    self.assertAllClose(self.evaluate(d.stddev()), expected_stddev)
    self.assertAllClose(self.evaluate(d_lr.stddev()), expected_stddev)

  def testSampleSmallConcentration(self):
    concentration_v = 0.05
    rate_v = 1.0
    concentration = tf.constant(concentration_v)
    rate = tf.constant(rate_v)
    n = 100000
    d = tfd.ExpGamma(
        concentration=concentration, rate=rate, validate_args=True)
    samples = d.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertTrue(self._kstest(concentration_v, rate_v, sample_values))
    self.assertAllClose(
        sample_values.mean(),
        d.mean(),
        rtol=.01)
    self.assertAllClose(
        sample_values.var(),
        d.variance(),
        rtol=.025)

  def testSample(self):
    concentration_v = 4.0
    rate_v = 3.0
    concentration = tf.constant(concentration_v)
    rate = tf.constant(rate_v)
    n = 100000
    d = tfd.ExpGamma(concentration=concentration, rate=rate, validate_args=True)
    samples = d.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertTrue(self._kstest(concentration_v, rate_v, sample_values))
    self.assertAllClose(
        sample_values.mean(),
        d.mean(),
        atol=.01)
    self.assertAllClose(
        sample_values.var(),
        d.variance(),
        atol=.15)

  def testSampleNonPositiveParameters(self):
    d = tfd.ExpGamma([1., 2.], 1., validate_args=False)
    seed_stream = test_util.test_seed_stream()
    samples = self.evaluate(d.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllFinite(samples)

    d = tfd.ExpGamma([0., 2.], 1., validate_args=False)
    samples = self.evaluate(d.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllEqual([s == -np.inf for s in samples], [True, False])

    d = tfd.ExpGamma([-0.001, 2.], 1., validate_args=False)
    samples = self.evaluate(d.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllEqual([np.isnan(s) for s in samples], [True, False])

    d = tfd.ExpGamma([1., -1.], 1., validate_args=False)
    samples = self.evaluate(d.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllEqual([np.isnan(s) for s in samples], [False, True])

    d = tfd.ExpGamma([1., 2.], 0., validate_args=False)
    samples = self.evaluate(d.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllPositiveInf(samples)

    d = tfd.ExpGamma([1., 2.], -1., validate_args=False)
    samples = self.evaluate(d.sample(seed=seed_stream()))
    self.assertEqual(samples.shape, (2,))
    self.assertAllNan(samples)

  @test_util.numpy_disable_gradient_test
  def testFullyReparameterized(self):
    concentration = tf.constant(4.0)
    rate = tf.constant(3.0)
    _, [grad_concentration, grad_rate] = tfp.math.value_and_gradient(
        lambda a, b: tfd.ExpGamma(concentration=a, rate=b, validate_args=True).  # pylint: disable=g-long-lambda
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

    def tfp_exp_gamma(a, b):
      return tfd.ExpGamma(concentration=a, rate=b, validate_args=True).sample(
          num_samples, seed=test_util.test_seed())

    _, [grad_concentration, grad_rate] = self.evaluate(
        tfp.math.value_and_gradient(tfp_exp_gamma, [concentration_v, rate_v]))

    def tf_exp_gamma(a, b):
      return tf.math.log(tf.random.gamma(
          shape=[num_samples], alpha=a, beta=b, seed=test_util.test_seed()))

    _, [grad_concentration_tf, grad_rate_tf] = self.evaluate(
        tfp.math.value_and_gradient(tf_exp_gamma, [concentration_v, rate_v]))

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

    def tfp_exp_gamma(a, b):
      return tf.math.square(
          tfd.ExpGamma(concentration=a, rate=b, validate_args=True).sample(
              num_samples, seed=test_util.test_seed()))

    _, [grad_concentration, grad_rate] = self.evaluate(
        tfp.math.value_and_gradient(tfp_exp_gamma, [concentration_v, rate_v]))

    def tf_exp_gamma(a, b):
      return tf.math.square(tf.math.log(tf.random.gamma(
          shape=[num_samples],
          alpha=a,
          beta=b,
          seed=test_util.test_seed())))

    _, [grad_concentration_tf, grad_rate_tf] = self.evaluate(
        tfp.math.value_and_gradient(tf_exp_gamma, [concentration_v, rate_v]))

    self.assertEqual(grad_concentration.shape, grad_concentration_tf.shape)
    self.assertEqual(grad_rate.shape, grad_rate_tf.shape)
    self.assertAllClose(grad_concentration, grad_concentration_tf, rtol=.06)
    self.assertAllClose(grad_rate, grad_rate_tf, rtol=2e-2)

  def testSampleMultiDimensional(self):
    n_concentration = 50
    concentration_v = np.array(
        [np.arange(1, n_concentration + 1, dtype=np.float32)])  # 1 x 50
    n_rate = 10
    rate_v = np.array([np.arange(1, n_rate + 1, dtype=np.float32)]).T  # 10 x 1
    d = tfd.ExpGamma(
        concentration=concentration_v, rate=rate_v, validate_args=True)

    n = 10000

    samples = d.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    self.assertEqual(samples.shape, (n, n_rate, n_concentration))
    self.assertEqual(sample_values.shape, (n, n_rate, n_concentration))
    self.assertAllClose(
        sample_values.mean(axis=0),
        d.mean(),
        rtol=.2)
    self.assertAllClose(
        sample_values.var(axis=0),
        d.variance(),
        rtol=0.09)
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
    oracle_dist = tfd.ExpGamma(concentration=concentration, rate=rate)
    ks, _ = sp_stats.kstest(
        samples,
        lambda x: self.evaluate(oracle_dist.cdf(x)))
    # Return True when the test passes.
    return ks < 0.02

  def testPdfOfSampleMultiDims(self):
    d = tfd.ExpGamma(
        concentration=[7., 11.], rate=[[5.], [6.]], validate_args=True)
    num = 50000
    samples = d.sample(num, seed=test_util.test_seed())
    pdfs = d.prob(samples)
    sample_vals, pdf_vals = self.evaluate([samples, pdfs])
    self.assertEqual(samples.shape, (num, 2, 2))
    self.assertEqual(pdfs.shape, (num, 2, 2))
    self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.02)
    self.assertAllClose(
        d.mean(),
        sample_vals.mean(axis=0),
        atol=.1)
    self.assertAllClose(
        d.variance(),
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

  def testNonPositiveInitializationParamsRaises(self):
    concentration_v = tf.constant(0.0, name='concentration')
    rate_v = tf.constant(1.0, name='rate')
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      d = tfd.ExpGamma(
          concentration=concentration_v, rate=rate_v, validate_args=True)
      self.evaluate(d.mean())
    concentration_v = tf.constant(1.0, name='concentration')
    rate_v = tf.constant(0.0, name='rate')
    with self.assertRaisesOpError('Argument `rate` must be positive.'):
      d = tfd.ExpGamma(
          concentration=concentration_v, rate=rate_v, validate_args=True)
      self.evaluate(d.mean())

  def testExpGammaExpGammaKL(self):  # Same as Gamma-Gamma KL.
    concentration0 = np.array([3.])
    rate0 = np.array([1., 2., 3., 1.5, 2.5, 3.5])

    concentration1 = np.array([0.4])
    rate1 = np.array([0.5, 1., 1.5, 2., 2.5, 3.])

    # Build graph.
    g0 = tfd.ExpGamma(
        concentration=concentration0, rate=rate0, validate_args=True)
    g0lr = tfd.ExpGamma(
        concentration=concentration0,
        log_rate=np.log(rate0),
        validate_args=True)
    g1 = tfd.ExpGamma(
        concentration=concentration1, rate=rate1, validate_args=True)
    g1lr = tfd.ExpGamma(
        concentration=concentration1,
        log_rate=np.log(rate1),
        validate_args=True)

    for d0, d1 in (g0, g1), (g0lr, g1), (g0, g1lr), (g0lr, g1lr):
      x = d0.sample(int(1e4), seed=test_util.test_seed())
      kl_samples = d0.log_prob(x) - d1.log_prob(x)
      kl_actual = tfd.kl_divergence(d0, d1)

      # Execute graph.
      [kl_samples_, kl_actual_] = self.evaluate([kl_samples, kl_actual])

      self.assertEqual(rate0.shape, kl_actual.shape)

      kl_expected = ((
          concentration0 - concentration1) * sp_special.digamma(concentration0)
                     + sp_special.gammaln(concentration1)
                     - sp_special.gammaln(concentration0)
                     + concentration1 * np.log(rate0)
                     - concentration1 * np.log(rate1)
                     + concentration0 * (rate1 / rate0 - 1.))

      self.assertAllClose(kl_expected, kl_actual_, atol=0., rtol=1e-6)
      self.assertAllMeansClose(
          kl_samples_, kl_actual_, axis=0, atol=0., rtol=1e-1)

  @test_util.tf_tape_safety_test
  def testGradientThroughConcentration(self):
    concentration = tf.Variable(3.)
    d = tfd.ExpGamma(concentration=concentration, rate=5., validate_args=True)
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
      d = tfd.ExpGamma(
          concentration=concentration, rate=[5.], validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveConcentrationAfterMutation(self):
    concentration = tf.Variable([1., 2., 3.])
    self.evaluate(concentration.initializer)
    d = tfd.ExpGamma(concentration=concentration, rate=[5.], validate_args=True)
    self.evaluate(d.sample(seed=test_util.test_seed()))
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      with tf.control_dependencies([concentration.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  @test_util.tf_tape_safety_test
  def testGradientThroughRate(self):
    rate = tf.Variable(3.)
    d = tfd.ExpGamma(concentration=1., rate=rate, validate_args=True)
    with tf.GradientTape() as tape:
      loss = -d.log_prob([1., 2., 4.])
    grad = tape.gradient(loss, d.trainable_variables)
    self.assertLen(grad, 1)
    self.assertAllNotNone(grad)

  def testAssertsPositiveRate(self):
    rate = tf.Variable([1., 2., -3.])
    self.evaluate(rate.initializer)
    with self.assertRaisesOpError('Argument `rate` must be positive.'):
      d = tfd.ExpGamma(concentration=[5.], rate=rate, validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertsPositiveRateAfterMutation(self):
    rate = tf.Variable([1., 2., 3.])
    self.evaluate(rate.initializer)
    d = tfd.ExpGamma(concentration=[3.], rate=rate, validate_args=True)
    self.evaluate(d.sample(seed=test_util.test_seed()))
    with self.assertRaisesOpError('Argument `rate` must be positive.'):
      with tf.control_dependencies([rate.assign([1., 2., -3.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))


@test_util.test_graph_and_eager_modes
class GammaSamplingTest(test_util.TestCase):

  def testSample(self):
    if not tf.test.is_gpu_available():
      self.skipTest('no GPU')
    with tf.device('GPU'):
      self.evaluate(exp_gamma_lib.random_exp_gamma(
          shape=tf.constant([], dtype=tf.int32),
          concentration=tf.constant(1.),
          seed=test_util.test_seed()))

  def testSampleXLA(self):
    self.skip_if_no_xla()
    if not tf.executing_eagerly(): return  # jit_compile is eager-only.
    concentration = np.exp(np.random.rand(4, 3).astype(np.float32))
    rate = np.exp(np.random.rand(4, 3).astype(np.float32))
    dist = tfd.ExpGamma(
        concentration=concentration, rate=rate, validate_args=True)
    # Verify the compile succeeds going all the way through the distribution.
    self.evaluate(
        tf.function(lambda: dist.sample(5, seed=test_util.test_seed()),
                    jit_compile=True)())

  def testSampleLowConcentration(self):
    concentration = np.linspace(0.1, 1., 10)
    rate = np.float64(1.)
    num_samples = int(1e5)
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    d = tfd.ExpGamma(
        concentration=concentration, rate=rate, validate_args=True)
    samples = d.sample(num_samples, seed=test_util.test_seed())
    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(
            samples,
            d.cdf,
            false_fail_rate=1e-9))

    self.assertAllMeansClose(
        self.evaluate(samples),
        d.mean(),
        axis=0,
        rtol=0.03)
    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        d.variance(),
        rtol=0.05)

  def testSampleHighConcentration(self):
    concentration = np.linspace(10., 20., 10)
    rate = np.float64(1.)
    num_samples = int(1e5)
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        num_samples)

    d = tfd.ExpGamma(
        concentration=concentration, rate=rate, validate_args=True)
    samples = d.sample(num_samples, seed=test_util.test_seed())
    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm(samples, d.cdf, false_fail_rate=1e-9))

    self.assertAllMeansClose(
        self.evaluate(samples),
        d.mean(),
        axis=0,
        rtol=0.01)
    self.assertAllClose(
        self.evaluate(tf.math.reduce_variance(samples, axis=0)),
        d.variance(),
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
      return tf.math.exp(tfd.ExpGamma(concentration, rate).sample(
          num_samples, seed=test_util.test_seed()))

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


if __name__ == '__main__':
  test_util.main()
