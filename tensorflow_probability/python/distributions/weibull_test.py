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
"""Tests for Weibull distribution."""
import math

# Dependency imports

import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import weibull
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class _WeibullTest(object):

  def testWeibullShape(self):
    concentration = np.array([1.] * 5, dtype=self.dtype)
    scale = np.array([2.] * 5, dtype=self.dtype)
    dist = weibull.Weibull(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        validate_args=True)
    if self.use_static_shape:
      self.assertEqual((5,), self.evaluate(dist.batch_shape_tensor()))
      self.assertEqual(tf.TensorShape([5]), dist.batch_shape)
      self.assertAllEqual([], self.evaluate(dist.event_shape_tensor()))
      self.assertEqual(tf.TensorShape([]), dist.event_shape)

  def testInvalidScale(self):
    scale = self.make_input(np.array([-0.01, 0., 2.], dtype=self.dtype))
    with self.assertRaisesOpError('Argument `scale` must be positive.'):
      dist = weibull.Weibull(concentration=1., scale=scale, validate_args=True)
      self.evaluate(dist.mean())

    scale = tf.Variable([0.01])
    self.evaluate(scale.initializer)
    dist = weibull.Weibull(concentration=1., scale=scale, validate_args=True)
    self.assertIs(scale, dist.scale)
    self.evaluate(dist.mean())
    with tf.control_dependencies([scale.assign([-0.01])]):
      with self.assertRaisesOpError('Argument `scale` must be positive.'):
        self.evaluate(dist.mean())

  def testInvalidConcentration(self):
    concentration = [-0.01, 0., 2.]
    with self.assertRaisesOpError('Argument `concentration` must be positive.'):
      dist = weibull.Weibull(
          concentration=concentration, scale=1., validate_args=True)
      self.evaluate(dist.mean())

    concentration = tf.Variable([0.01])
    self.evaluate(concentration.initializer)
    dist = weibull.Weibull(
        concentration=concentration, scale=1., validate_args=True)
    self.assertIs(concentration, dist.concentration)
    self.evaluate(dist.mean())
    with tf.control_dependencies([concentration.assign([-0.01])]):
      with self.assertRaisesOpError(
          'Argument `concentration` must be positive.'):
        self.evaluate(dist.mean())

  def testWeibullEntropy(self):
    concentration = np.array([7.8], dtype=self.dtype)
    scale = np.array([1.1], dtype=self.dtype)

    dist = weibull.Weibull(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        validate_args=True)

    entropy = dist.entropy()
    expected_entropy = (
        np.euler_gamma * (1. - 1. / concentration) + np.log(scale) -
        np.log(concentration) + 1.)

    self.assertAllClose(
        self.evaluate(entropy), expected_entropy, atol=1e-5,
        rtol=1e-5)  # relaxed tol for fp32 in JAX
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), entropy.shape)

  def testWeibullLogPdf(self):
    batch_size = 6
    concentration = np.array([3.] * batch_size, dtype=self.dtype)
    scale = np.array([2.] * batch_size, dtype=self.dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self.dtype)
    dist = weibull.Weibull(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        validate_args=True)

    log_pdf = dist.log_prob(self.make_input(x))
    self.assertAllClose(
        stats.weibull_min.logpdf(x, c=concentration, scale=scale),
        self.evaluate(log_pdf))
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), log_pdf.shape)

    pdf = dist.prob(x)
    self.assertAllClose(
        stats.weibull_min.pdf(x, c=concentration, scale=scale),
        self.evaluate(pdf))
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), pdf.shape)

  def testWeibullCDF(self):
    batch_size = 6
    concentration = np.array([3.] * batch_size, dtype=self.dtype)
    scale = np.array([2.] * batch_size, dtype=self.dtype)
    x = np.array([2., 3., 4., 5., 6., 7.], dtype=self.dtype)

    dist = weibull.Weibull(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        validate_args=True)

    log_cdf = dist.log_cdf(self.make_input(x))
    self.assertAllClose(
        self.evaluate(log_cdf),
        stats.weibull_min.logcdf(x, c=concentration, scale=scale))
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), log_cdf.shape)

    cdf = dist.cdf(self.make_input(x))
    self.assertAllClose(
        self.evaluate(cdf),
        stats.weibull_min.cdf(x, c=concentration, scale=scale))
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), cdf.shape)

  def testWeibullMean(self):
    batch_size = 3
    concentration = np.array([9.] * batch_size, dtype=self.dtype)
    scale = np.array([1.5] * batch_size, dtype=self.dtype)

    dist = weibull.Weibull(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        validate_args=True)

    mean = dist.mean()
    self.assertAllClose(
        self.evaluate(mean),
        stats.weibull_min.mean(c=concentration, scale=scale, loc=0))
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), mean.shape)

  def testWeibullVariance(self):
    scale = np.array([2.], dtype=self.dtype)
    concentration = np.array([3.], dtype=self.dtype)

    dist = weibull.Weibull(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        validate_args=True)

    variance = dist.variance()
    self.assertAllClose(
        self.evaluate(variance),
        stats.weibull_min.var(c=concentration, scale=scale, loc=0),
        atol=1e-5,
        rtol=1e-5)  # relaxed tol for fp32 in JAX
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), variance.shape)

  def testWeibullStd(self):
    concentration = np.array([1.3], dtype=self.dtype)
    scale = np.array([2.7], dtype=self.dtype)

    dist = weibull.Weibull(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        validate_args=True)

    stddev = dist.stddev()
    self.assertAllClose(
        self.evaluate(stddev),
        stats.weibull_min.std(c=concentration, scale=scale),
        atol=1e-5,
        rtol=1e-5)  # relaxed tol for fp32 in JAX
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), stddev.shape)

  def testWeibullMode(self):
    batch_size = 25
    concentration = np.array([1.] * batch_size, dtype=self.dtype)
    scale = np.array([33.] * batch_size, dtype=self.dtype)
    expected_mode = ((
        (concentration - 1.) / concentration)**(1. / concentration)) * scale

    dist = weibull.Weibull(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        validate_args=True)

    mode = dist.mode()
    self.assertAllClose(self.evaluate(mode), expected_mode)
    self.assertEqual(self.evaluate(dist.batch_shape_tensor()), mode.shape)

  def testWeibullSample(self):
    concentration = self.dtype(4.)
    scale = self.dtype(1.)
    n = int(100e3)

    dist = weibull.Weibull(
        concentration=self.make_input(concentration),
        scale=self.make_input(scale),
        validate_args=True)

    samples = dist.sample(n, seed=test_util.test_seed())
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
            expected=dist.mean(),
            false_fail_rate=self.dtype(1e-6)))

    self.assertAllClose(
        stats.weibull_min.var(c=concentration, scale=scale, loc=0),
        sample_values.var(),
        rtol=0.01)

  def testSampleLikeArgsGetDistDType(self):
    if self.dtype is np.float32:
      # Raw Python literals should always be interpreted as fp32.
      dist = weibull.Weibull(1., 2.)
    elif self.dtype is np.float64:
      # The make_input function will cast them to self.dtype
      dist = weibull.Weibull(self.make_input(1.), self.make_input(2.))
    self.assertEqual(self.dtype, dist.dtype)
    for method in ('log_prob', 'prob', 'log_cdf', 'cdf'):
      self.assertEqual(self.dtype, getattr(dist, method)(1.).dtype)
    for method in ('entropy', 'mean', 'variance', 'stddev', 'mode'):
      self.assertEqual(self.dtype, getattr(dist, method)().dtype)

  def testSupportBijectorOutsideRange(self):
    concentration = np.array([2., 4., 5.], dtype=self.dtype)
    scale = np.array([2., 4., 5.], dtype=self.dtype)
    dist = weibull.Weibull(
        concentration=concentration, scale=scale, validate_args=True)
    x = np.array([-8.3, -0.4, -1e-6])
    bijector_inverse_x = dist.experimental_default_event_space_bijector(
    ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

  def testWeibullWeibullKL(self):
    a_concentration = np.array([3.])
    a_scale = np.array([2.])
    b_concentration = np.array([6.])
    b_scale = np.array([4.])

    a = weibull.Weibull(
        concentration=a_concentration, scale=a_scale, validate_args=True)
    b = weibull.Weibull(
        concentration=b_concentration, scale=b_scale, validate_args=True)

    kl = kullback_leibler.kl_divergence(a, b)
    expected_kl = (
        np.log(a_concentration / a_scale**a_concentration)
        - np.log(b_concentration / b_scale**b_concentration)
        + (
            (a_concentration - b_concentration)
            * (np.log(a_scale) - np.euler_gamma / a_concentration)
        )
        + (
            (a_scale / b_scale) ** b_concentration
            * np.exp(math.lgamma(b_concentration / a_concentration + 1.0))
        )
        - 1.0
    )

    x = a.sample(int(1e5), seed=test_util.test_seed())
    kl_samples = a.log_prob(x) - b.log_prob(x)
    kl_samples_ = self.evaluate(kl_samples)

    self.assertAllMeansClose(
        kl_samples_, expected_kl, axis=0, atol=0.0, rtol=1e-2)
    self.assertAllClose(self.evaluate(kl), expected_kl)

  def testWeibullGammaKL(self):
    a_concentration = np.array([3.])
    a_scale = np.array([2.])
    b_concentration = np.array([6.])
    b_rate = np.array([0.25])

    a = weibull.Weibull(
        concentration=a_concentration, scale=a_scale, validate_args=True)
    b = gamma.Gamma(
        concentration=b_concentration, rate=b_rate, validate_args=True)

    kl = kullback_leibler.kl_divergence(a, b)

    x = a.sample(int(1e5), seed=test_util.test_seed())
    kl_samples = a.log_prob(x) - b.log_prob(x)
    kl_samples_ = self.evaluate(kl_samples)

    self.assertAllMeansClose(
        kl_samples_, self.evaluate(kl), axis=0, atol=0.0, rtol=1e-2)

  def testWeibullGammaKLAgreeWeibullWeibull(self):
    a_concentration = np.array([3.])
    a_scale = np.array([2.])
    b_concentration = np.array([1.])
    b_rate = np.array([0.25])

    a = weibull.Weibull(
        concentration=a_concentration, scale=a_scale, validate_args=True)
    b = gamma.Gamma(
        concentration=b_concentration, rate=b_rate, validate_args=True)
    c = weibull.Weibull(
        concentration=b_concentration, scale=1 / b_rate, validate_args=True)

    kl_weibull_weibull = kullback_leibler.kl_divergence(a, c)
    kl_weibull_gamma = kullback_leibler.kl_divergence(a, b)

    self.assertAllClose(
        self.evaluate(kl_weibull_gamma),
        self.evaluate(kl_weibull_weibull),
        atol=0.0,
        rtol=1e-6)


@test_util.test_all_tf_execution_regimes
class WeibullTestStaticShapeFloat32(test_util.TestCase, _WeibullTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class WeibullTestStaticShapeFloat64(test_util.TestCase, _WeibullTest):
  dtype = np.float64
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class WeibullTestDynamicShapeFloat32(test_util.TestCase, _WeibullTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class WeibullTestDynamicShapeFloat64(test_util.TestCase, _WeibullTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == '__main__':
  test_util.main()
