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
"""Tests for multivariate Power Spherical distribution."""

# Dependency imports
import numpy as np
from scipy import special as sp_special

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import power_spherical
from tensorflow_probability.python.distributions import spherical_uniform
from tensorflow_probability.python.distributions import von_mises_fisher

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.stats import sample_stats


class _PowerSphericalTest(object):

  def testReproducibleGraph(self):
    pspherical = power_spherical.PowerSpherical(
        mean_direction=tf.math.l2_normalize(
            np.array([1., 2.], dtype=self.dtype)),
        concentration=self.dtype(1.2))
    seed = test_util.test_seed()
    s1 = self.evaluate(pspherical.sample(50, seed=seed))
    if tf.executing_eagerly():
      tf.random.set_seed(seed)
    s2 = self.evaluate(pspherical.sample(50, seed=seed))
    self.assertAllEqual(s1, s2)

  def VerifySampleMean(self, mean_dirs, concentration, batch_shape):
    pspherical = power_spherical.PowerSpherical(
        mean_direction=mean_dirs,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    self.assertEqual([mean_dirs.shape[-1]],
                     tensorshape_util.as_list(pspherical.event_shape))
    self.assertEqual(
        batch_shape,
        tensorshape_util.as_list(pspherical.batch_shape))
    nsamples = int(5e4)
    samples = pspherical.sample(nsamples, seed=test_util.test_seed())
    self.assertEqual([nsamples] + batch_shape + [mean_dirs.shape[-1]],
                     tensorshape_util.as_list(samples.shape))
    sample_mean = self.evaluate(samples).mean(axis=0)
    sample_dir = (
        sample_mean / np.linalg.norm(sample_mean, axis=-1, keepdims=True))
    # Assert that positive-concentration distributions have samples with
    # the expected mean direction.
    inner_product = self.evaluate(
        tf.reduce_sum(sample_dir * pspherical.mean_direction, axis=-1))
    # Inner products should be roughly ascending by concentration.
    self.assertAllClose(np.round(np.sort(inner_product, axis=0), decimals=3),
                        np.round(inner_product, decimals=3),
                        rtol=0.5, atol=0.05)
    means = self.evaluate(pspherical.mean())
    # Mean vector for 0-concentration is precisely (0, 0).
    self.assertAllEqual(np.zeros_like(means[0]), means[0])
    mean_lengths = np.linalg.norm(means, axis=-1)
    # Length of the mean vector is strictly ascending with concentration.
    self.assertAllEqual(mean_lengths, np.sort(mean_lengths, axis=0))
    self.assertAllClose(np.linalg.norm(sample_mean, axis=-1), mean_lengths,
                        rtol=0.5, atol=0.05)

  def testSampleMeanDir2d(self):
    mean_dirs = tf.math.l2_normalize(
        np.array([[1., 1], [-2, 1], [0, -1]], dtype=self.dtype), axis=-1)
    concentration = np.array(
        [[0], [0.1], [2], [40], [1000]], dtype=self.dtype)
    self.VerifySampleMean(mean_dirs, concentration, [5, 3])

  def testSampleMeanDir3d(self):
    mean_dirs = tf.math.l2_normalize(
        np.array([[1., 2, 3], [-2, -3, -1]], dtype=self.dtype), axis=-1)
    concentration = np.array(
        [[0], [0.1], [2], [40], [1000]], dtype=self.dtype)
    self.VerifySampleMean(mean_dirs, concentration, [5, 2])

  def testSampleMeanDir5d(self):
    mean_dirs = tf.math.l2_normalize(
        np.array([[1., 2, 3, -1., 5.]], dtype=self.dtype), axis=-1)
    concentration = np.array(
        [[0], [0.1], [2], [40], [1000]], dtype=self.dtype)
    self.VerifySampleMean(mean_dirs, concentration, [5, 1])

  def VerifyPdfWithNumpy(self, pspherical, atol=1e-4):
    """Verifies log_prob evaluations with numpy/scipy.

    Both uniform random points and sampled points are evaluated.

    Args:
      pspherical: A `tfp.distributions.PowerSpherical` instance.
      atol: Absolute difference tolerable.
    """
    dim = tf.compat.dimension_value(pspherical.event_shape[-1])
    nsamples = 10
    # Sample some random points uniformly over the hypersphere using numpy.
    sample_shape = [nsamples] + tensorshape_util.as_list(
        pspherical.batch_shape) + [dim]
    uniforms = np.random.randn(*sample_shape)
    uniforms /= np.linalg.norm(uniforms, axis=-1, keepdims=True)
    uniforms = uniforms.astype(dtype_util.as_numpy_dtype(pspherical.dtype))
    # Concatenate in some sampled points from the distribution under test.
    pspherical_samples = pspherical.sample(
        sample_shape=[nsamples], seed=test_util.test_seed())
    samples = tf.concat([uniforms, pspherical_samples], axis=0)
    samples = tf.debugging.check_numerics(samples, 'samples')
    samples = self.evaluate(samples)
    log_prob = pspherical.log_prob(samples)
    log_prob = self.evaluate(log_prob)

    # Check that the log_prob is not nan or +inf. It can be -inf since
    # if we sample a direction diametrically opposite to the mean direction,
    # we'll get an inner product of -1.
    self.assertFalse(np.any(np.isnan(log_prob)))
    self.assertFalse(np.any(np.isposinf(log_prob)))
    conc = self.evaluate(pspherical.concentration)
    mean_dir = self.evaluate(pspherical.mean_direction)
    alpha = (dim - 1.) / 2. + conc
    beta = (dim - 1.) / 2.

    expected = (
        sp_special.xlog1py(conc, np.sum(samples * mean_dir, axis=-1)) -
        (alpha + beta) * np.log(2.) - beta * np.log(np.pi) -
        sp_special.gammaln(alpha) + sp_special.gammaln(alpha + beta))
    self.assertAllClose(expected, log_prob, atol=atol)

  def VerifySampleAndPdfConsistency(self, pspherical, rtol=0.075):
    """Verifies samples are consistent with the PDF using importance sampling.

    In particular, we verify an estimate the surface area of the n-dimensional
    hypersphere, and the surface areas of the spherical caps demarcated by
    a handful of survival rates.

    Args:
      pspherical: A `PowerSpherical` distribution instance.
      rtol: Relative difference tolerable.
    """
    dim = tf.compat.dimension_value(pspherical.event_shape[-1])
    nsamples = int(1e5)
    samples = pspherical.sample(
        sample_shape=[nsamples], seed=test_util.test_seed())
    samples = tf.debugging.check_numerics(samples, 'samples')
    log_prob = pspherical.log_prob(samples)
    log_prob, samples = self.evaluate([log_prob, samples])
    # Check that the log_prob is not nan or +inf. It can be -inf since
    # if we sample a direction diametrically opposite to the mean direction,
    # we'll get an inner product of -1.
    self.assertFalse(np.any(np.isnan(log_prob)))
    self.assertFalse(np.any(np.isposinf(log_prob)))
    log_importance = -log_prob
    sphere_surface_area_estimate, importance = self.evaluate([
        tf.reduce_mean(tf.math.exp(log_importance), axis=0),
        tf.exp(log_importance)])
    true_sphere_surface_area = 2 * (np.pi)**(dim / 2) * self.evaluate(
        tf.exp(-tf.math.lgamma(dim / 2)))
    # Broadcast to correct size
    true_sphere_surface_area += np.zeros_like(sphere_surface_area_estimate)
    # Highly concentrated distributions do not get enough coverage to provide
    # a reasonable full-sphere surface area estimate. These are covered below
    # by CDF-based hypersphere cap surface area estimates.
    # Because the PowerSpherical distribution has zero mass at
    # -`mean_direction` (and points close to -`mean_direction` due to floating
    # point), we only compute this at concentration = 0, which has guaranteed
    # mass everywhere.
    self.assertAllClose(
        true_sphere_surface_area[0],
        sphere_surface_area_estimate[0], rtol=rtol)

    # Assert surface area of hyperspherical cap For some CDFs in [.05,.45],
    # (h must be greater than 0 for the hypersphere cap surface area
    # calculation to hold).
    for survival_rate in 0.95, .9, .75, .6:
      cdf = (1 - survival_rate)
      mean_dir = self.evaluate(pspherical.mean_direction)
      dotprods = np.sum(samples * mean_dir, -1)
      # Empirical estimate of the effective dot-product of the threshold that
      # selects for a given CDF level, that is the cosine of the largest
      # passable angle, or the minimum cosine for a within-CDF sample.
      dotprod_thresh = np.percentile(
          dotprods, 100 * survival_rate, axis=0, keepdims=True)
      # We mask this sum because it is possible for the log_prob to be -inf when
      # the mean_direction is -mean_dir.
      importance_masked = np.ma.array(
          importance, mask=dotprods <= dotprod_thresh)
      sphere_cap_surface_area_ests = (
          cdf * (importance_masked).sum(0) /
          (dotprods > dotprod_thresh).sum(0))
      h = (1 - dotprod_thresh)
      self.assertGreaterEqual(h.min(), 0)  # h must be >= 0 for the eqn below
      true_sphere_cap_surface_area = (
          0.5 * true_sphere_surface_area *
          self.evaluate(tf.math.betainc((dim - 1) / 2, 0.5, 2 * h - h**2)))
      if dim == 3:  # For 3-d we have a simpler form we can double-check.
        self.assertAllClose(2 * np.pi * h, true_sphere_cap_surface_area)

      self.assertAllClose(
          true_sphere_cap_surface_area,
          sphere_cap_surface_area_ests +
          np.zeros_like(true_sphere_cap_surface_area),
          rtol=rtol)

  def testSampleAndPdfConsistency2d(self):
    mean_dir = tf.math.l2_normalize([[1., 2], [-2, -3]], axis=-1)
    concentration = [[0], [1e-5], [0.1], [1], [4]]
    pspherical = power_spherical.PowerSpherical(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    self.VerifySampleAndPdfConsistency(pspherical)
    self.VerifyPdfWithNumpy(pspherical)

  def testSampleAndPdfConsistency3d(self):
    mean_dir = tf.math.l2_normalize([[1., 2, 3], [-2, -3, -1]], axis=-1)
    concentration = [[0], [1e-5], [0.1], [1], [4]]
    pspherical = power_spherical.PowerSpherical(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    self.VerifySampleAndPdfConsistency(pspherical)
    self.VerifyPdfWithNumpy(pspherical, atol=.002)

  def testSampleAndPdfConsistency4d(self):
    mean_dir = tf.math.l2_normalize([[1., 2, 3, 4], [-2, -3, -1, 0]], axis=-1)
    concentration = [[0], [1e-4], [0.1], [1], [4]]
    pspherical = power_spherical.PowerSpherical(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    self.VerifySampleAndPdfConsistency(pspherical)
    self.VerifyPdfWithNumpy(pspherical)

  def testSampleAndPdfConsistency5d(self):
    mean_dir = tf.math.l2_normalize(
        [[1., 2, 3, 4, 5], [-2, -3, -1, 0, 1]], axis=-1)
    concentration = [[0], [5e-2], [0.1], [1], [4]]
    pspherical = power_spherical.PowerSpherical(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    self.VerifySampleAndPdfConsistency(pspherical)
    self.VerifyPdfWithNumpy(pspherical, atol=2e-4)

  def testSampleAndPdfForMeanDirNorthPole(self):
    mean_dir = np.array([1., 0., 0., 0., 0.], dtype=self.dtype)
    concentration = [[0], [5e-2], [0.1], [1], [4]]
    pspherical = power_spherical.PowerSpherical(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    self.VerifySampleAndPdfConsistency(pspherical)
    self.VerifyPdfWithNumpy(pspherical, atol=2e-4)
    self.VerifySampleMean(mean_dir, concentration, [5, 1])

    # Verify the covariance.
    samples = pspherical.sample(int(7e4), seed=test_util.test_seed())
    sample_cov = sample_stats.covariance(samples, sample_axis=0)
    true_cov, sample_cov = self.evaluate([
        pspherical.covariance(), sample_cov])
    self.assertAllClose(true_cov, sample_cov, rtol=0.15, atol=1.5e-2)

  def VerifyCovariance(self, dim):
    seed_stream = test_util.test_seed_stream()
    num_samples = int(5e4)
    mean_direction = tf.random.uniform(
        shape=[5, dim],
        minval=self.dtype(1.),
        maxval=self.dtype(2.),
        dtype=self.dtype,
        seed=seed_stream())
    mean_direction = tf.nn.l2_normalize(mean_direction, axis=-1)
    concentration = tf.math.log(
        tf.random.uniform(
            shape=[2, 1],
            minval=self.dtype(1.),
            maxval=self.dtype(100.),
            dtype=self.dtype,
            seed=seed_stream()))

    ps = power_spherical.PowerSpherical(
        mean_direction=mean_direction,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    samples = ps.sample(num_samples, seed=test_util.test_seed())
    sample_cov = sample_stats.covariance(samples, sample_axis=0)
    true_cov, sample_cov = self.evaluate([
        ps.covariance(), sample_cov])
    self.assertAllClose(true_cov, sample_cov, rtol=0.15, atol=1.5e-2)

  def testCovarianceDim2(self):
    self.VerifyCovariance(dim=2)

  def testCovarianceDim5(self):
    self.VerifyCovariance(dim=5)

  def testCovarianceDim10(self):
    self.VerifyCovariance(dim=10)

  def VerifyEntropy(self, dim):
    seed_stream = test_util.test_seed_stream()
    mean_direction = tf.random.uniform(
        shape=[5, dim],
        minval=self.dtype(1.),
        maxval=self.dtype(2.),
        dtype=self.dtype,
        seed=seed_stream())
    mean_direction = tf.nn.l2_normalize(mean_direction, axis=-1)
    concentration = tf.math.log(
        tf.random.uniform(
            shape=[2, 1],
            minval=self.dtype(1.),
            maxval=self.dtype(100.),
            dtype=self.dtype,
            seed=seed_stream()))
    ps = power_spherical.PowerSpherical(
        mean_direction=mean_direction,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    samples = ps.sample(int(3e4), seed=test_util.test_seed())
    entropy_samples = -ps.log_prob(samples)
    true_entropy, entropy_samples = self.evaluate([
        ps.entropy(), entropy_samples])
    self.assertAllMeansClose(entropy_samples, true_entropy, axis=0, rtol=3e-2)

  def testEntropyDim2(self):
    self.VerifyEntropy(dim=2)

  def testEntropyDim3(self):
    self.VerifyEntropy(dim=3)

  def testEntropyDim5(self):
    self.VerifyEntropy(dim=5)

  def testEntropyDim10(self):
    self.VerifyEntropy(dim=10)

  def testAssertsValidImmutableParams(self):
    with self.assertRaisesOpError('`concentration` must be non-negative'):
      pspherical = power_spherical.PowerSpherical(
          mean_direction=tf.math.l2_normalize([1., 2, 3], axis=-1),
          concentration=-1.,
          validate_args=True,
          allow_nan_stats=False)
      self.evaluate(pspherical.mean())

    with self.assertRaisesOpError(
        '`mean_direction` must be a vector of at least size 2'):
      pspherical = power_spherical.PowerSpherical(
          mean_direction=[1.],
          concentration=0.,
          validate_args=True,
          allow_nan_stats=False)
      self.evaluate(pspherical.mean())

    with self.assertRaisesOpError('`mean_direction` must be unit-length'):
      pspherical = power_spherical.PowerSpherical(
          mean_direction=tf.convert_to_tensor([1., 2, 3]),
          concentration=1.,
          validate_args=True,
          allow_nan_stats=False)
      self.evaluate(pspherical.mean())

  def testAssertsValidMutableParams(self):
    mean_direction = tf.Variable(tf.math.l2_normalize([1., 2, 3], axis=-1))
    concentration = tf.Variable(1.)
    pspherical = power_spherical.PowerSpherical(
        mean_direction=mean_direction,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)

    self.evaluate([mean_direction.initializer, concentration.initializer])
    self.evaluate(concentration.assign(-1.))
    with self.assertRaisesOpError('`concentration` must be non-negative'):
      self.evaluate(pspherical.mean())

    self.evaluate((concentration.assign(1.),
                   mean_direction.assign([1., 2., 3.])))
    with self.assertRaisesOpError('`mean_direction` must be unit-length'):
      self.evaluate(pspherical.mean())

    mean_direction = tf.Variable([1.])
    with self.assertRaisesOpError(
        '`mean_direction` must be a vector of at least size 2'):
      pspherical = power_spherical.PowerSpherical(
          mean_direction=mean_direction,
          concentration=concentration,
          validate_args=True,
          allow_nan_stats=False)
      self.evaluate(mean_direction.initializer)
      self.evaluate(pspherical.mean())

  def testAssertValidSample(self):
    mean_dir = tf.math.l2_normalize([[1., 2, 3], [-2, -3, -1]], axis=-1)
    concentration = [[0.], [2.]]
    pspherical = power_spherical.PowerSpherical(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)

    with self.assertRaisesOpError('Samples must be unit length.'):
      self.evaluate(pspherical.prob([0.5, 0.5, 0.5]))

    msg = 'must have innermost dimension matching'
    static_shape_assertion = self.assertRaisesRegex(ValueError, msg)
    dynamic_shape_assertion = self.assertRaisesOpError(msg)

    x = [[1., 0., 0., 0.]]
    with static_shape_assertion:
      self.evaluate(pspherical.log_prob(x))

    x_var = tf.Variable(x, shape=tf.TensorShape(None))
    shape_assertion = (static_shape_assertion if tf.executing_eagerly()
                       else dynamic_shape_assertion)
    self.evaluate(x_var.initializer)
    with shape_assertion:
      self.evaluate(pspherical.log_prob(x_var))

  def testSupportBijectorOutsideRange(self):
    mean_dir = np.array([[1., 2., 3.], [-2., -3., -1.]]).astype(np.float32)
    mean_dir /= np.linalg.norm(mean_dir, axis=-1)[:, np.newaxis]
    concentration = [[0], [0.1], [2], [40], [1000]]
    dist = power_spherical.PowerSpherical(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True)

    x = mean_dir
    x[0][0] += 0.01
    with self.assertRaisesOpError('must sum to `1`'):
      self.evaluate(
          dist.experimental_default_event_space_bijector().inverse(x[0]))

    with self.assertRaisesOpError('must be non-negative'):
      self.evaluate(
          dist.experimental_default_event_space_bijector().inverse(x[1]))

  def VerifyPowerSphericaUniformZeroKL(self, dim):
    seed_stream = test_util.test_seed_stream()
    mean_direction = tf.random.uniform(
        shape=[5, dim],
        minval=self.dtype(1.),
        maxval=self.dtype(2.),
        dtype=self.dtype,
        seed=seed_stream())
    mean_direction = tf.nn.l2_normalize(mean_direction, axis=-1)
    # Zero concentration is the same as a uniform distribution on the sphere.
    # Check that the log_probs agree and the KL divergence is zero.
    concentration = self.dtype(0.)

    ps = power_spherical.PowerSpherical(
        mean_direction=mean_direction, concentration=concentration)
    su = spherical_uniform.SphericalUniform(dimension=dim, dtype=self.dtype)

    x = ps.sample(int(5e4), seed=test_util.test_seed())

    ps_lp = ps.log_prob(x)
    su_lp = su.log_prob(x)
    ps_lp_, su_lp_ = self.evaluate([ps_lp, su_lp])
    self.assertAllClose(ps_lp_, su_lp_, rtol=1e-6)
    true_kl = kullback_leibler.kl_divergence(ps, su)
    true_kl_ = self.evaluate([true_kl])
    self.assertAllClose(true_kl_, np.zeros_like(true_kl_), atol=1e-4)

  def VerifyPowerSphericaUniformKL(self, dim):
    seed_stream = test_util.test_seed_stream()
    mean_direction = tf.random.uniform(
        shape=[5, dim],
        minval=self.dtype(1.),
        maxval=self.dtype(2.),
        dtype=self.dtype,
        seed=seed_stream())
    mean_direction = tf.nn.l2_normalize(mean_direction, axis=-1)
    concentration = tf.math.log(
        tf.random.uniform(
            shape=[2, 1],
            minval=self.dtype(1.),
            maxval=self.dtype(100.),
            dtype=self.dtype,
            seed=seed_stream()))

    ps = power_spherical.PowerSpherical(
        mean_direction=mean_direction, concentration=concentration)
    su = spherical_uniform.SphericalUniform(dimension=dim, dtype=self.dtype)

    x = ps.sample(int(5e4), seed=test_util.test_seed())

    kl_samples = ps.log_prob(x) - su.log_prob(x)
    true_kl = kullback_leibler.kl_divergence(ps, su)
    true_kl_, kl_samples_ = self.evaluate([true_kl, kl_samples])
    self.assertAllMeansClose(kl_samples_, true_kl_, axis=0, atol=0.0, rtol=7e-1)

  def testKLPowerSphericalSphericalUniformDim2(self):
    self.VerifyPowerSphericaUniformZeroKL(dim=2)
    self.VerifyPowerSphericaUniformKL(dim=2)

  def testKLPowerSphericalSphericalUniformDim3(self):
    self.VerifyPowerSphericaUniformZeroKL(dim=3)
    self.VerifyPowerSphericaUniformKL(dim=3)

  def testKLPowerSphericalSphericalUniformDim5(self):
    self.VerifyPowerSphericaUniformZeroKL(dim=5)
    self.VerifyPowerSphericaUniformKL(dim=5)

  def testKLPowerSphericalSphericalUniformDim10(self):
    self.VerifyPowerSphericaUniformZeroKL(dim=10)
    self.VerifyPowerSphericaUniformKL(dim=10)

  def VerifyPowerSphericalVonMisesFisherZeroKL(self, dim):
    seed_stream = test_util.test_seed_stream()
    mean_direction = tf.random.uniform(
        shape=[5, dim],
        minval=self.dtype(1.),
        maxval=self.dtype(2.),
        dtype=self.dtype,
        seed=seed_stream())
    mean_direction = tf.nn.l2_normalize(mean_direction, axis=-1)
    # Zero concentration is the same as a uniform distribution on the sphere.
    # Check that the KL divergence is zero.
    concentration = self.dtype(0.)

    ps = power_spherical.PowerSpherical(
        mean_direction=mean_direction, concentration=concentration)
    vmf = von_mises_fisher.VonMisesFisher(
        mean_direction=mean_direction, concentration=concentration)
    true_kl = kullback_leibler.kl_divergence(ps, vmf)
    true_kl_ = self.evaluate(true_kl)
    self.assertAllClose(true_kl_, np.zeros_like(true_kl_), atol=1e-4)

  def testInvalidPowerSphericalvMFKl(self):
    seed_stream = test_util.test_seed_stream()
    mean_direction1 = tf.random.uniform(
        shape=[5, 3],
        minval=self.dtype(1.),
        maxval=self.dtype(2.),
        dtype=self.dtype,
        seed=seed_stream())
    mean_direction1 = tf.nn.l2_normalize(mean_direction1, axis=-1)

    mean_direction2 = tf.random.uniform(
        shape=[5, 4],
        minval=self.dtype(1.),
        maxval=self.dtype(2.),
        dtype=self.dtype,
        seed=seed_stream())
    mean_direction2 = tf.nn.l2_normalize(mean_direction2, axis=-1)

    concentration = self.dtype(0.)

    ps = power_spherical.PowerSpherical(
        mean_direction=mean_direction1, concentration=concentration)
    vmf = von_mises_fisher.VonMisesFisher(
        mean_direction=mean_direction2, concentration=concentration)
    with self.assertRaisesRegex(ValueError, 'Can not compute the KL'):
      kullback_leibler.kl_divergence(ps, vmf)

  def VerifyPowerSphericalVonMisesFisherKL(self, dim):
    seed_stream = test_util.test_seed_stream()
    mean_direction1 = tf.random.uniform(
        shape=[5, dim],
        minval=self.dtype(1.),
        maxval=self.dtype(2.),
        dtype=self.dtype,
        seed=seed_stream())
    mean_direction2 = tf.random.uniform(
        shape=[5, dim],
        minval=self.dtype(1.),
        maxval=self.dtype(2.),
        dtype=self.dtype,
        seed=seed_stream())

    mean_direction1 = tf.nn.l2_normalize(mean_direction1, axis=-1)
    mean_direction2 = tf.nn.l2_normalize(mean_direction2, axis=-1)
    concentration1 = tf.math.log(
        tf.random.uniform(
            shape=[2, 1],
            minval=self.dtype(1.),
            maxval=self.dtype(100.),
            dtype=self.dtype,
            seed=seed_stream()))
    concentration2 = tf.math.log(
        tf.random.uniform(
            shape=[2, 1],
            minval=self.dtype(1.),
            maxval=self.dtype(100.),
            dtype=self.dtype,
            seed=seed_stream()))

    ps = power_spherical.PowerSpherical(
        mean_direction=mean_direction1, concentration=concentration1)
    vmf = von_mises_fisher.VonMisesFisher(
        mean_direction=mean_direction2, concentration=concentration2)
    x = ps.sample(int(6e4), seed=test_util.test_seed())

    kl_samples = ps.log_prob(x) - vmf.log_prob(x)
    true_kl = kullback_leibler.kl_divergence(ps, vmf)
    true_kl_, kl_samples_ = self.evaluate([true_kl, kl_samples])
    self.assertAllMeansClose(kl_samples_, true_kl_, axis=0, atol=0.0, rtol=7e-1)

  def testKLPowerSphericalVonMisesFisherDim2(self):
    self.VerifyPowerSphericalVonMisesFisherZeroKL(dim=2)
    self.VerifyPowerSphericalVonMisesFisherKL(dim=2)

  def testKLPowerSphericalVonMisesFisherDim3(self):
    self.VerifyPowerSphericalVonMisesFisherZeroKL(dim=3)
    self.VerifyPowerSphericalVonMisesFisherKL(dim=3)


@test_util.test_all_tf_execution_regimes
class PowerSphericalTestFloat32(
    test_util.VectorDistributionTestHelpers,
    test_util.TestCase,
    _PowerSphericalTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class PowerSphericalTestFloat64(
    test_util.VectorDistributionTestHelpers,
    test_util.TestCase,
    _PowerSphericalTest):
  dtype = np.float64


if __name__ == '__main__':
  test_util.main()
