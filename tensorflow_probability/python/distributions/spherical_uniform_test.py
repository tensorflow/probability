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
"""Tests for the spherical uniform distribution."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import test_util


class _SphericalUniformTest(object):

  def VerifySampleAndPdfConsistency(self, uniform):
    """Verifies samples are consistent with the PDF using importance sampling.

    In particular, we verify an estimate the surface area of the n-dimensional
    hypersphere, and the surface areas of the spherical caps demarcated by
    a handful of survival rates.

    Args:
      uniform: A `SphericalUniform` distribution instance.
    """
    dim = tf.compat.dimension_value(uniform.event_shape[-1])
    nsamples = int(6e4)
    self.assertLess(
        self.evaluate(
            st.min_num_samples_for_dkwm_cdf_test(
                discrepancy=0.04, false_fail_rate=1e-9, false_pass_rate=1e-9)),
        nsamples)
    samples = uniform.sample(
        sample_shape=[nsamples], seed=test_util.test_seed())
    samples = tf.debugging.check_numerics(samples, 'samples')
    log_prob = uniform.log_prob(samples)
    log_prob = self.evaluate(
        tf.debugging.check_numerics(log_prob, 'log_prob'))
    true_sphere_surface_area = 2 * (np.pi)**(dim / 2) * self.evaluate(
        tf.exp(-tf.math.lgamma(dim / 2)))
    true_sphere_surface_area += np.zeros_like(log_prob)
    # Check the log prob is a constant and is the reciprocal of the surface
    # area.
    self.assertAllClose(np.exp(log_prob), 1. / true_sphere_surface_area)

    # For sampling, let's check the marginals. x_i**2 ~ Beta(0.5, d - 1 / 2)
    beta_dist = tfp.distributions.Beta(
        self.dtype(0.5),
        self.dtype((dim - 1.) / 2.))
    for i in range(dim):
      self.evaluate(
          st.assert_true_cdf_equal_by_dkwm(
              samples[..., i] ** 2, cdf=beta_dist.cdf, false_fail_rate=1e-9))

  def testSampleAndPdfConsistency2d(self):
    uniform = tfp.distributions.SphericalUniform(
        dimension=2,
        batch_shape=[2],
        dtype=self.dtype,
        validate_args=True, allow_nan_stats=False)
    self.VerifySampleAndPdfConsistency(uniform)

  def testSampleAndPdfConsistency3d(self):
    uniform = tfp.distributions.SphericalUniform(
        dimension=3,
        batch_shape=[2],
        dtype=self.dtype,
        validate_args=True, allow_nan_stats=False)
    self.VerifySampleAndPdfConsistency(uniform)

  def testSampleAndPdfConsistency4d(self):
    uniform = tfp.distributions.SphericalUniform(
        dimension=4,
        batch_shape=[2],
        dtype=self.dtype,
        validate_args=True, allow_nan_stats=False)
    self.VerifySampleAndPdfConsistency(uniform)

  def testSampleAndPdfConsistency5d(self):
    uniform = tfp.distributions.SphericalUniform(
        dimension=5,
        batch_shape=[2],
        dtype=self.dtype,
        validate_args=True, allow_nan_stats=False)
    self.VerifySampleAndPdfConsistency(uniform)

  def VerifyMean(self, dim):
    num_samples = int(7e4)
    uniform = tfp.distributions.SphericalUniform(
        batch_shape=[2, 1],
        dimension=dim,
        dtype=self.dtype,
        validate_args=True,
        allow_nan_stats=False)
    samples = uniform.sample(num_samples, seed=test_util.test_seed())
    sample_mean = tf.reduce_mean(samples, axis=0)
    true_mean, sample_mean = self.evaluate([
        uniform.mean(), sample_mean])
    check1 = st.assert_true_mean_equal_by_dkwm(
        samples=samples, low=-(1. + 1e-7), high=1. + 1e-7,
        expected=true_mean, false_fail_rate=1e-6)
    check2 = assert_util.assert_less(
        st.min_discrepancy_of_true_means_detectable_by_dkwm(
            num_samples,
            low=-1.,
            high=1.,
            # Smaller false fail rate because of different batch sizes between
            # these two checks.
            false_fail_rate=1e-7,
            false_pass_rate=1e-6),
        # 4% relative error
        0.08)
    self.evaluate([check1, check2])

  def testMeanDim2(self):
    self.VerifyMean(dim=2)

  def testMeanDim3(self):
    self.VerifyMean(dim=3)

  def testMeanDim5(self):
    self.VerifyMean(dim=5)

  def testMeanDim10(self):
    self.VerifyMean(dim=10)

  def VerifyCovariance(self, dim):
    num_samples = int(6e4)
    uniform = tfp.distributions.SphericalUniform(
        batch_shape=[2, 1],
        dimension=dim,
        dtype=self.dtype,
        validate_args=True,
        allow_nan_stats=False)
    samples = uniform.sample(num_samples, seed=test_util.test_seed())
    sample_cov = tfp.stats.covariance(samples, sample_axis=0)
    sample_variance = tfp.stats.variance(samples, sample_axis=0)
    true_cov, sample_cov, sample_variance = self.evaluate([
        uniform.covariance(), sample_cov, sample_variance])
    self.assertAllClose(
        np.ones_like(sample_variance) / dim,
        sample_variance, rtol=2e-2)
    # The off-diagonal entries should be close to zero.
    self.assertAllClose(true_cov, sample_cov, rtol=1e-2, atol=4e-3)

  def testCovarianceDim2(self):
    self.VerifyCovariance(dim=2)

  def testCovarianceDim5(self):
    self.VerifyCovariance(dim=5)

  def testCovarianceDim10(self):
    self.VerifyCovariance(dim=10)

  def VerifyEntropy(self, dim):
    uniform = tfp.distributions.SphericalUniform(
        dimension=dim,
        batch_shape=tuple(),
        dtype=self.dtype,
        validate_args=True,
        allow_nan_stats=False)
    samples = uniform.sample(int(1e3), seed=test_util.test_seed())
    sample_entropy = -tf.reduce_mean(uniform.log_prob(samples), axis=0)
    true_entropy, sample_entropy = self.evaluate([
        uniform.entropy(), sample_entropy])
    self.assertAllClose(sample_entropy, true_entropy, rtol=1e-5)

  def testEntropyDim1(self):
    self.VerifyEntropy(dim=1)

  def testEntropyDim2(self):
    self.VerifyEntropy(dim=2)

  def testEntropyDim5(self):
    self.VerifyEntropy(dim=5)

  def testEntropyDim10(self):
    self.VerifyEntropy(dim=10)

  def testAssertValidSample(self):
    uniform = tfp.distributions.SphericalUniform(
        dimension=4,
        batch_shape=tuple(),
        dtype=self.dtype,
        validate_args=True,
        allow_nan_stats=False)

    x = tf.constant([0.1, 0.1, 0.5, 0.5], dtype=self.dtype)
    with self.assertRaisesOpError('Samples must be unit length.'):
      self.evaluate(uniform.prob(x))

  def testBatchShape(self):
    uniform = tfp.distributions.SphericalUniform(
        dimension=4,
        batch_shape=[2, 1],
        dtype=self.dtype,
        validate_args=True,
        allow_nan_stats=False)
    samples = uniform.sample(5, seed=test_util.test_seed())
    self.assertEqual([5, 2, 1, 4], samples.shape)
    test_sample = np.array([[1., 0., 0., 0.]] * 5, dtype=self.dtype)
    log_prob = uniform.log_prob(test_sample)
    self.assertEqual([2, 5], log_prob.shape)

  def testSupportBijectorOutsideRange(self):
    dist = tfp.distributions.SphericalUniform(
        dimension=3,
        batch_shape=tuple(),
        dtype=self.dtype,
        validate_args=True)

    x = np.array([1., 0.2, 0.3], dtype=self.dtype)
    with self.assertRaisesOpError('must sum to `1`'):
      self.evaluate(
          dist.experimental_default_event_space_bijector().inverse(x))


@test_util.test_all_tf_execution_regimes
class SphericalUniformFloat32Test(
    test_util.VectorDistributionTestHelpers,
    test_util.TestCase,
    _SphericalUniformTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class SphericalUniformFloat64(
    test_util.VectorDistributionTestHelpers,
    test_util.TestCase,
    _SphericalUniformTest):
  dtype = np.float64


if __name__ == '__main__':
  test_util.main()
