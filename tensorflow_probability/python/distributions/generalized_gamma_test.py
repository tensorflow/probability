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
"""Tests for GeneralizedGamma distribution."""
# Dependency imports
import numpy as np
from scipy import stats

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class _GeneralizedGammaTest(object):

  def testGeneralizedGammaShape(self):
    concentration = np.array([1.] * 5, dtype=self.dtype)
    scale = np.array([2.] * 5, dtype=self.dtype)
    exponent = np.array([1.] * 5, dtype=self.dtype)
    generalizedgamma = tfd.GeneralizedGamma(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)
    if self.use_static_shape:
      self.assertEqual((5,), self.evaluate(generalizedgamma.batch_shape_tensor()))
      self.assertEqual(tf.TensorShape([5]), generalizedgamma.batch_shape)
      self.assertAllEqual([], self.evaluate(generalizedgamma.event_shape_tensor()))
      self.assertEqual(tf.TensorShape([]), generalizedgamma.event_shape)

  def testInvalidScale(self):
    scale = self.make_input(np.array([-0.01, 0., 2.], dtype=self.dtype))
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      generalizedgamma = tfd.GeneralizedGamma(concentration=1., scale=scale, exponent=1., validate_args=True)
      self.evaluate(generalizedgamma.mean())

    scale = tf.Variable([0.01])
    self.evaluate(scale.initializer)
    generalizedgamma = tfd.GeneralizedGamma(concentration=1., scale=scale, exponent=1., validate_args=True)
    self.assertIs(scale, generalizedgamma.scale)
    self.evaluate(generalizedgamma.mean())
    with tf.control_dependencies([scale.assign([-0.01])]):
      with self.assertRaisesOpError('Argument `scale` must be positive.'):
        self.evaluate(generalizedgamma.mean())

  def testInvalidConcentration(self):
    concentration = [-0.01, 0., 2.]
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      generalizedgamma = tfd.GeneralizedGamma(
          concentration=concentration, scale=1., exponent=1., validate_args=True)
      self.evaluate(generalizedgamma.mean())

    concentration = tf.Variable([0.01])
    self.evaluate(concentration.initializer)
    generalizedgamma = tfd.GeneralizedGamma(
        concentration=concentration, scale=1., exponent=1., validate_args=True)
    self.assertIs(concentration, generalizedgamma.concentration)
    self.evaluate(generalizedgamma.mean())
    with tf.control_dependencies([concentration.assign([-0.01])]):
      with self.assertRaisesOpError(
          'Argument `concentration` must be positive.'):
        self.evaluate(generalizedgamma.mean())

  def testGeneralizedGammaEntropy(self):
    """ Don't have a reference GGamma implementation so test against Weibull"""
    concentration = np.array([7.8], dtype=self.dtype)
    scale = np.array([1.1], dtype=self.dtype)
    exponent = np.array([7.8], dtype=self.dtype)
    generalizedgamma = tfd.GeneralizedGamma(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    entropy = generalizedgamma.entropy()
    expected_entropy = (
        np.euler_gamma * (1. - 1. / concentration) + np.log(scale) -
        np.log(concentration) + 1.)

    self.assertAllClose(
        self.evaluate(entropy), expected_entropy, atol=1e-5,
        rtol=1e-5)  # relaxed tol for fp32 in JAX
    self.assertEqual(self.evaluate(generalizedgamma.batch_shape_tensor()), entropy.shape)

  def testGeneralizedGammaLogPdf(self):
    batch_size = 6
    concentration = np.array([3.] * batch_size, dtype=self.dtype)
    scale = np.array([2.] * batch_size, dtype=self.dtype)
    exponent = np.array([3.] * batch_size, dtype=self.dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self.dtype)
    generalizedgamma = tfd.GeneralizedGamma(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    log_pdf = generalizedgamma.log_prob(self.make_input(x))
    self.assertAllClose(
        stats.weibull_min.logpdf(x, c=concentration, scale=scale),
        self.evaluate(log_pdf))
    self.assertEqual(self.evaluate(generalizedgamma.batch_shape_tensor()), log_pdf.shape)

    pdf = generalizedgamma.prob(x)
    self.assertAllClose(
        stats.weibull_min.pdf(x, c=concentration, scale=scale),
        self.evaluate(pdf))
    self.assertEqual(self.evaluate(generalizedgamma.batch_shape_tensor()), pdf.shape)

  def testGeneralizedGammaCDF(self):
    batch_size = 6
    concentration = np.array([3.] * batch_size, dtype=self.dtype)
    scale = np.array([2.] * batch_size, dtype=self.dtype)
    exponent = np.array([3.] * batch_size, dtype=self.dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self.dtype)

    generalizedgamma = tfd.GeneralizedGamma(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    log_cdf = generalizedgamma.log_cdf(self.make_input(x))
    self.assertAllClose(
        self.evaluate(log_cdf),
        stats.weibull_min.logcdf(x, c=concentration, scale=scale))
    self.assertEqual(self.evaluate(generalizedgamma.batch_shape_tensor()), log_cdf.shape)

    cdf = generalizedgamma.cdf(self.make_input(x))
    self.assertAllClose(
        self.evaluate(cdf),
        stats.weibull_min.cdf(x, c=concentration, scale=scale))
    self.assertEqual(self.evaluate(generalizedgamma.batch_shape_tensor()), cdf.shape)

  def testGeneralizedGammaMean(self):
    batch_size = 3
    concentration = np.array([9.] * batch_size, dtype=self.dtype)
    scale = np.array([1.5] * batch_size, dtype=self.dtype)
    exponent = np.array([9.] * batch_size, dtype=self.dtype)
    generalizedgamma = tfd.GeneralizedGamma(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    mean = generalizedgamma.mean()
    self.assertAllClose(
        self.evaluate(mean),
        stats.weibull_min.mean(c=concentration, scale=scale, loc=0))
    self.assertEqual(self.evaluate(generalizedgamma.batch_shape_tensor()), mean.shape)

  def testGeneralizedGammaVariance(self):
    scale = np.array([2.], dtype=self.dtype)
    concentration = np.array([3.], dtype=self.dtype)
    exponent = np.array([3.], dtype=self.dtype)
    generalizedgamma = tfd.GeneralizedGamma(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    variance = generalizedgamma.variance()
    self.assertAllClose(
        self.evaluate(variance),
        stats.weibull_min.var(c=concentration, scale=scale, loc=0),
        atol=1e-5,
        rtol=1e-5)  # relaxed tol for fp32 in JAX
    self.assertEqual(
        self.evaluate(generalizedgamma.batch_shape_tensor()), variance.shape)

  def testGeneralizedGammaStd(self):
    concentration = np.array([1.3], dtype=self.dtype)
    scale = np.array([2.7], dtype=self.dtype)
    exponent = np.array([1.3], dtype=self.dtype)
    generalizedgamma = tfd.GeneralizedGamma(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    stddev = generalizedgamma.stddev()
    self.assertAllClose(
        self.evaluate(stddev),
        stats.weibull_min.std(c=concentration, scale=scale),
        atol=1e-5,
        rtol=1e-5)  # relaxed tol for fp32 in JAX
    self.assertEqual(self.evaluate(generalizedgamma.batch_shape_tensor()), stddev.shape)

  def testGeneralizedGammaMode(self):
    batch_size = 25
    concentration = np.array([1.] * batch_size, dtype=self.dtype)
    scale = np.array([33.] * batch_size, dtype=self.dtype)
    exponent = np.array([1.] * batch_size, dtype=self.dtype)
    expected_mode = ((
        (concentration - 1.) / concentration)**(1. / concentration)) * scale

    generalizedgamma = tfd.GeneralizedGamma(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    mode = generalizedgamma.mode()
    self.assertAllClose(self.evaluate(mode), expected_mode)
    self.assertEqual(self.evaluate(generalizedgamma.batch_shape_tensor()), mode.shape)

  def testGeneralizedGammaSample(self):
    concentration = self.dtype(4.)
    scale = self.dtype(1.)
    exponent = self.dtype(4.)
    n = int(100e3)

    generalizedgamma = tfd.GeneralizedGamma(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        exponent=self.make_input(exponent),
        validate_args=True)

    samples = generalizedgamma.sample(n, seed=test_util.test_seed())
    sample_values = self.evaluate(samples)

    low = self.dtype(0.)
    high = self.dtype(np.inf)

    self.assertAllClose(
        stats.weibull_min.mean(c=concentration, scale=scale, loc=0),
        sample_values.mean(),
        rtol=0.01)

    self.evaluate(
        st.assert_true_mean_equal_by_dkwm(
            samples,
            low=low,
            high=high,
            expected=generalizedgamma.mean(),
            false_fail_rate=self.dtype(1e-6)))

    self.assertAllClose(
        stats.weibull_min.var(c=concentration, scale=scale, loc=0),
        sample_values.var(),
        rtol=0.01)

  def testSampleLikeArgsGetDistDType(self):
    if self.dtype is np.float32:
      # Raw Python literals should always be interpreted as fp32.
      dist = tfd.GeneralizedGamma(1., 2., 1.)
    elif self.dtype is np.float64:
      # The make_input function will cast them to self.dtype
      dist = tfd.GeneralizedGamma(self.make_input(1.), self.make_input(2.), self.make_input(1.))
    self.assertEqual(self.dtype, dist.dtype)
    for method in ('log_prob', 'prob', 'log_cdf', 'cdf'):
      self.assertEqual(self.dtype, getattr(dist, method)(1.).dtype)
    for method in ('entropy', 'mean', 'variance', 'stddev', 'mode'):
      self.assertEqual(self.dtype, getattr(dist, method)().dtype)

  def testSupportBijectorOutsideRange(self):
    concentration = np.array([2., 4., 5.], dtype=self.dtype)
    scale = np.array([2., 4., 5.], dtype=self.dtype)
    exponent = np.array([2., 4., 5.], dtype=self.dtype)
    dist = tfd.GeneralizedGamma(
        concentration=concentration, scale=scale, exponent=exponent, validate_args=True)
    x = np.array([-8.3, -0.4, -1e-6])
    bijector_inverse_x = dist.experimental_default_event_space_bijector(
    ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

  def testGeneralizedGammaGeneralizedGammaKL(self):
    a_concentration = np.array([3.])
    a_scale = np.array([2.])
    a_exponent = np.array([3.])
    b_concentration = np.array([6.])
    b_scale = np.array([4.])
    b_exponent = np.array([6.])
    
    a = tfd.GeneralizedGamma(
        concentration=a_concentration, scale=a_scale, exponent=a_exponent, validate_args=True)
    b = tfd.GeneralizedGamma(
        concentration=b_concentration, scale=b_scale, exponent=b_exponent, validate_args=True)

    kl = tfd.kl_divergence(a, b)
    expected_kl = (
        np.log(a_concentration / a_scale**a_concentration) -
        np.log(b_concentration / b_scale**b_concentration) +
        ((a_concentration - b_concentration) *
         (np.log(a_scale) - np.euler_gamma / a_concentration)) +
        ((a_scale / b_scale)**b_concentration *
         np.exp(np.math.lgamma(b_concentration / a_concentration + 1.))) - 1.)

    x = a.sample(int(1e5), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(a.log_prob(x) - b.log_prob(x), axis=0)
    kl_sample_val = self.evaluate(kl_sample)

    self.assertAllClose(expected_kl, kl_sample_val, atol=0.0, rtol=1e-2)
    self.assertAllClose(expected_kl, self.evaluate(kl))

  def testGeneralizedGammaGammaKL(self):
    a_concentration = np.array([3.])
    a_scale = np.array([2.])
    a_exponent = np.array([3.])
    b_concentration = np.array([6.])
    b_rate = np.array([0.25])

    a = tfd.GeneralizedGamma(
        concentration=a_concentration, scale=a_scale, exponent=a_exponent, validate_args=True)
    b = tfd.Gamma(
        concentration=b_concentration, rate=b_rate, validate_args=True)

    kl = tfd.kl_divergence(a, b)

    x = a.sample(int(1e5), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(a.log_prob(x) - b.log_prob(x), axis=0)
    kl_sample_val = self.evaluate(kl_sample)

    self.assertAllClose(kl_sample_val, self.evaluate(kl), atol=0.0, rtol=1e-2)

  def testGeneralizedGammaGammaKLAgreeGeneralizedGammaGeneralizedGamma(self):
    a_concentration = np.array([3.])
    a_scale = np.array([2.])
    a_exponent = np.array([3.])
    b_concentration = np.array([1.])
    b_rate = np.array([0.25])
    b_exponent = np.array([1.])

    a = tfd.GeneralizedGamma(
        concentration=a_concentration, scale=a_scale, exponent=a_exponent, validate_args=True)
    b = tfd.Gamma(
        concentration=b_concentration, rate=b_rate, validate_args=True)
    c = tfd.GeneralizedGamma(
        concentration=b_concentration, scale=1 / b_rate, exponent=b_exponent, validate_args=True)

    kl_generalizedgamma_generalizedgamma = tfd.kl_divergence(a, c)
    kl_generalizedgamma_gamma = tfd.kl_divergence(a, b)

    self.assertAllClose(
        self.evaluate(kl_generalizedgamma_gamma),
        self.evaluate(kl_generalizedgamma_generalizedgamma),
        atol=0.0,
        rtol=1e-6)


@test_util.test_all_tf_execution_regimes
class GeneralizedGammaTestStaticShapeFloat32(test_util.TestCase, _GeneralizedGammaTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class GeneralizedGammaTestStaticShapeFloat64(test_util.TestCase, _GeneralizedGammaTest):
  dtype = np.float64
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class GeneralizedGammaTestDynamicShapeFloat32(test_util.TestCase, _GeneralizedGammaTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class GeneralizedGammaTestDynamicShapeFloat64(test_util.TestCase, _GeneralizedGammaTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == '__main__':
  test_util.main()
