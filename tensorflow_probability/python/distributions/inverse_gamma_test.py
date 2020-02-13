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

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class InverseGammaTest(test_util.TestCase):

  def testInverseGammaShape(self):
    alpha = tf.constant([3.0] * 5)
    beta = tf.constant(11.0)
    inv_gamma = tfd.InverseGamma(
        concentration=alpha, scale=beta, validate_args=True)

    self.assertEqual(self.evaluate(inv_gamma.batch_shape_tensor()), (5,))
    self.assertEqual(inv_gamma.batch_shape, tf.TensorShape([5]))
    self.assertAllEqual(self.evaluate(inv_gamma.event_shape_tensor()), [])
    self.assertEqual(inv_gamma.event_shape, tf.TensorShape([]))

  def testInverseGammaLogPDF(self):
    batch_size = 6
    alpha = tf.constant([2.0] * batch_size)
    beta = tf.constant([3.0] * batch_size)
    alpha_v = 2.0
    beta_v = 3.0
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
    inv_gamma = tfd.InverseGamma(
        concentration=alpha, scale=beta, validate_args=True)
    expected_log_pdf = stats.invgamma.logpdf(x, alpha_v, scale=beta_v)
    log_pdf = inv_gamma.log_prob(x)
    self.assertEqual(log_pdf.shape, (6,))
    self.assertAllClose(self.evaluate(log_pdf), expected_log_pdf)

    pdf = inv_gamma.prob(x)
    self.assertEqual(pdf.shape, (6,))
    self.assertAllClose(self.evaluate(pdf), np.exp(expected_log_pdf))

    # TODO(b/144948687): Avoid `nan` at boundary. Ideally we'd do this test:
    # log_pdf_at_boundary = self.evaluate(inv_gamma.log_prob(0.))
    # self.assertTrue(np.isinf(log_pdf_at_boundary).all())
    # pdf_at_boundary = self.evaluate(inv_gamma.prob(0.))
    # self.assertAllEqual(pdf_at_boundary, np.zeros_like(pdf_at_boundary))

  def testInverseGammaLogPDFMultidimensional(self):
    batch_size = 6
    alpha = tf.constant([[2.0, 4.0]] * batch_size)
    beta = tf.constant([[3.0, 4.0]] * batch_size)
    alpha_v = np.array([2.0, 4.0])
    beta_v = np.array([3.0, 4.0])
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    inv_gamma = tfd.InverseGamma(
        concentration=alpha, scale=beta, validate_args=True)
    expected_log_pdf = stats.invgamma.logpdf(x, alpha_v, scale=beta_v)
    log_pdf = inv_gamma.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    self.assertAllClose(log_pdf_values, expected_log_pdf)

    pdf = inv_gamma.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))
    self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testInverseGammaLogPDFMultidimensionalBroadcasting(self):
    batch_size = 6
    alpha = tf.constant([[2.0, 4.0]] * batch_size)
    beta = tf.constant(3.0)
    alpha_v = np.array([2.0, 4.0])
    beta_v = 3.0
    x = np.array([[2.5, 2.5, 4.0, 0.1, 1.0, 2.0]], dtype=np.float32).T
    inv_gamma = tfd.InverseGamma(
        concentration=alpha, scale=beta, validate_args=True)
    expected_log_pdf = stats.invgamma.logpdf(x, alpha_v, scale=beta_v)
    log_pdf = inv_gamma.log_prob(x)
    log_pdf_values = self.evaluate(log_pdf)
    self.assertEqual(log_pdf.shape, (6, 2))
    self.assertAllClose(log_pdf_values, expected_log_pdf)

    pdf = inv_gamma.prob(x)
    pdf_values = self.evaluate(pdf)
    self.assertEqual(pdf.shape, (6, 2))
    self.assertAllClose(pdf_values, np.exp(expected_log_pdf))

  def testInverseGammaCDF(self):
    batch_size = 6
    alpha_v = 2.0
    beta_v = 3.0
    alpha = tf.constant([alpha_v] * batch_size)
    beta = tf.constant([beta_v] * batch_size)
    x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

    inv_gamma = tfd.InverseGamma(
        concentration=alpha, scale=beta, validate_args=True)
    expected_cdf = stats.invgamma.cdf(x, alpha_v, scale=beta_v)

    cdf = inv_gamma.cdf(x)
    self.assertEqual(cdf.shape, (batch_size,))
    self.assertAllClose(self.evaluate(cdf), expected_cdf)

  def testInverseGammaMode(self):
    alpha_v = np.array([5.5, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    inv_gamma = tfd.InverseGamma(
        concentration=alpha_v, scale=beta_v, validate_args=True)
    expected_modes = beta_v / (alpha_v + 1)
    self.assertEqual(inv_gamma.mode().shape, (3,))
    self.assertAllClose(self.evaluate(inv_gamma.mode()), expected_modes)

  def testInverseGammaMeanAllDefined(self):
    alpha_v = np.array([5.5, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    inv_gamma = tfd.InverseGamma(
        concentration=alpha_v, scale=beta_v, validate_args=True)
    expected_means = stats.invgamma.mean(alpha_v, scale=beta_v)
    self.assertEqual(inv_gamma.mean().shape, (3,))
    self.assertAllClose(self.evaluate(inv_gamma.mean()), expected_means)

  def testInverseGammaMeanAllowNanStats(self):
    # Mean will not be defined for the first entry.
    alpha_v = np.array([1.0, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    inv_gamma = tfd.InverseGamma(
        concentration=alpha_v,
        scale=beta_v,
        allow_nan_stats=False,
        validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(inv_gamma.mean())

  def testInverseGammaMeanNanStats(self):
    # Mode will not be defined for the first two entries.
    alpha_v = np.array([0.5, 1.0, 3.0, 2.5])
    beta_v = np.array([1.0, 2.0, 4.0, 5.0])
    inv_gamma = tfd.InverseGamma(
        concentration=alpha_v,
        scale=beta_v,
        allow_nan_stats=True,
        validate_args=True)
    expected_means = beta_v / (alpha_v - 1)
    expected_means[0] = np.nan
    expected_means[1] = np.nan
    self.assertEqual(inv_gamma.mean().shape, (4,))
    self.assertAllClose(self.evaluate(inv_gamma.mean()), expected_means)

  def testInverseGammaVarianceAllDefined(self):
    alpha_v = np.array([7.0, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    inv_gamma = tfd.InverseGamma(
        concentration=alpha_v, scale=beta_v, validate_args=True)
    expected_variances = stats.invgamma.var(alpha_v, scale=beta_v)
    self.assertEqual(inv_gamma.variance().shape, (3,))
    self.assertAllClose(self.evaluate(inv_gamma.variance()), expected_variances)

  def testInverseGammaVarianceAllowNanStats(self):
    alpha_v = np.array([1.5, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    inv_gamma = tfd.InverseGamma(
        concentration=alpha_v,
        scale=beta_v,
        allow_nan_stats=False,
        validate_args=True)
    with self.assertRaisesOpError('x < y'):
      self.evaluate(inv_gamma.variance())

  def testInverseGammaVarianceNanStats(self):
    alpha_v = np.array([1.5, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    inv_gamma = tfd.InverseGamma(
        concentration=alpha_v,
        scale=beta_v,
        allow_nan_stats=True,
        validate_args=True)
    expected_variances = stats.invgamma.var(alpha_v, scale=beta_v)
    expected_variances[0] = np.nan
    self.assertEqual(inv_gamma.variance().shape, (3,))
    self.assertAllClose(self.evaluate(inv_gamma.variance()), expected_variances)

  def testInverseGammaEntropy(self):
    alpha_v = np.array([1.0, 3.0, 2.5])
    beta_v = np.array([1.0, 4.0, 5.0])
    expected_entropy = stats.invgamma.entropy(alpha_v, scale=beta_v)
    inv_gamma = tfd.InverseGamma(
        concentration=alpha_v, scale=beta_v, validate_args=True)
    self.assertEqual(inv_gamma.entropy().shape, (3,))
    self.assertAllClose(self.evaluate(inv_gamma.entropy()), expected_entropy)

  def testInverseGammaSample(self):
    alpha_v = 4.0
    beta_v = 3.0
    alpha = tf.constant(alpha_v)
    beta = tf.constant(beta_v)
    n = int(5e5)
    inv_gamma = tfd.InverseGamma(
        concentration=alpha, scale=beta, validate_args=True)
    samples = inv_gamma.sample(
        n, seed=test_util.test_seed(hardcoded_seed=137))
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n,))
    self.assertEqual(sample_values.shape, (n,))
    self.assertAllClose(
        sample_values.mean(),
        stats.invgamma.mean(
            alpha_v, scale=beta_v),
        atol=.0040)
    self.assertAllClose(
        sample_values.var(),
        stats.invgamma.var(alpha_v, scale=beta_v),
        atol=.15)
    self.assertTrue(self._kstest(alpha_v, beta_v, sample_values))

  def testInverseGammaFullyReparameterized(self):
    alpha = tf.constant(4.0)
    beta = tf.constant(3.0)
    def inverse_gamma_sampler(alpha, beta):
      inv_gamma = tfd.InverseGamma(
          concentration=alpha, scale=beta, validate_args=True)
      return inv_gamma.sample(10, seed=test_util.test_seed())
    _, [grad_alpha, grad_beta] = tfp.math.value_and_gradient(
        inverse_gamma_sampler, [alpha, beta])
    self.assertIsNotNone(grad_alpha)
    self.assertIsNotNone(grad_beta)

  def testInverseGammaSampleMultiDimensional(self):
    alpha_v = np.array([np.arange(5, 105, dtype=np.float32)])  # 1 x 100
    beta_v = np.array([np.arange(1, 11, dtype=np.float32)]).T  # 10 x 1
    inv_gamma = tfd.InverseGamma(
        concentration=alpha_v, scale=beta_v, validate_args=True)
    n = int(1e5)
    samples = inv_gamma.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)
    self.assertEqual(samples.shape, (n, 10, 100))
    self.assertEqual(sample_values.shape, (n, 10, 100))
    zeros = np.zeros_like(alpha_v + beta_v)  # 10 x 100
    alpha_bc = alpha_v + zeros
    beta_bc = beta_v + zeros
    self.assertAllClose(
        sample_values.mean(axis=0),
        stats.invgamma.mean(
            alpha_bc, scale=beta_bc),
        atol=.25)
    self.assertAllClose(
        sample_values.var(axis=0),
        stats.invgamma.var(alpha_bc, scale=beta_bc),
        atol=4.5)
    fails = 0
    trials = 0
    for ai, a in enumerate(np.reshape(alpha_v, [-1])):
      for bi, b in enumerate(np.reshape(beta_v, [-1])):
        s = sample_values[:, bi, ai]
        trials += 1
        fails += 0 if self._kstest(a, b, s) else 1
    self.assertLess(fails, trials * 0.03)

  def _kstest(self, alpha, beta, samples):
    # Uses the Kolmogorov-Smirnov test for goodness of fit.
    ks, _ = stats.kstest(samples, stats.invgamma(alpha, scale=beta).cdf)
    # Return True when the test passes.
    return ks < 0.02

  def testInverseGammaPdfOfSampleMultiDims(self):
    inv_gamma = tfd.InverseGamma(
        concentration=[7., 11.], scale=[[5.], [6.]], validate_args=True)
    num = 50000
    samples = inv_gamma.sample(num, seed=test_util.test_seed())
    pdfs = inv_gamma.prob(samples)
    sample_vals, pdf_vals = self.evaluate([samples, pdfs])
    self.assertEqual(samples.shape, (num, 2, 2))
    self.assertEqual(pdfs.shape, (num, 2, 2))
    self.assertAllClose(
        stats.invgamma.mean(
            [[7., 11.], [7., 11.]], scale=np.array([[5., 5.], [6., 6.]])),
        sample_vals.mean(axis=0),
        atol=.1)
    self.assertAllClose(
        stats.invgamma.var([[7., 11.], [7., 11.]],
                           scale=np.array([[5., 5.], [6., 6.]])),
        sample_vals.var(axis=0),
        atol=.1)
    self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.02)
    self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.02)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1e-3):
    s_p = zip(sample_vals, pdf_vals)
    prev = (0, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testAssertsValidSample(self):
    inv_gamma = tfd.InverseGamma(
        concentration=[3., 5., 2.], scale=2., validate_args=True)
    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(inv_gamma.cdf([4., -1.7, 2.]))

  def testInverseGammaNonPositiveInitializationParamsRaises(self):
    alpha_v = tf.constant(0.0, name='alpha')
    beta_v = tf.constant(1.0, name='beta')
    with self.assertRaisesOpError('`concentration` must be positive'):
      inv_gamma = tfd.InverseGamma(
          concentration=alpha_v, scale=beta_v, validate_args=True)
      self.evaluate(inv_gamma.mean())
    alpha_v = tf.constant(1.0, name='alpha')
    beta_v = tf.constant(0.0, name='beta')
    with self.assertRaisesOpError('`scale` must be positive'):
      inv_gamma = tfd.InverseGamma(
          concentration=alpha_v, scale=beta_v, validate_args=True)
      self.evaluate(inv_gamma.mean())

  def testModifiedVariableAssertion(self):
    c = tf.Variable(0.9)
    s = tf.Variable(2.)
    self.evaluate([c.initializer, s.initializer])
    inv_gamma = tfd.InverseGamma(concentration=c, scale=s, validate_args=True)
    with self.assertRaisesOpError('`concentration` must be positive'):
      with tf.control_dependencies([c.assign(-2.)]):
        self.evaluate(inv_gamma.mean())

    with self.assertRaisesOpError('`scale` must be positive'):
      with tf.control_dependencies([c.assign(0.9), s.assign(-2.)]):
        self.evaluate(inv_gamma.mean())

  def testSupportBijectorOutsideRange(self):
    dist = tfd.InverseGamma(
        concentration=[7., 2., 5.],
        scale=2.,
        validate_args=True)
    x = np.array([-7.2, -1e-6, -1.3])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

if __name__ == '__main__':
  tf.test.main()
