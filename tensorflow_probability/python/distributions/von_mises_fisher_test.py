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
"""Tests for multivariate von Mises-Fisher distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import special as sp_special

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions.von_mises_fisher import _bessel_ive
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class VonMisesFisherTest(test_util.VectorDistributionTestHelpers,
                         test_util.TestCase):

  def testBesselIve(self):
    with self.assertRaisesRegexp(ValueError, r'imprecise for large v'):
      _bessel_ive(2.0, 1.0)
    # Zero is not a supported value for z.
    with self.assertRaisesOpError(r'NaN'):
      self.evaluate(_bessel_ive(1.5, 0.0))
    z = np.logspace(-6, 2, 20).astype(np.float64)
    for v in np.float64([-0.5, 0, 0.5, 1, 1.5]):
      self.assertAllClose(sp_special.ive(v, z), _bessel_ive(v, z))

  def testReproducibleGraph(self):
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=tf.math.l2_normalize([1., 2.]),
        concentration=1.2)
    seed = test_util.test_seed()
    s1 = self.evaluate(vmf.sample(50, seed=seed))
    if tf.executing_eagerly():
      tf.random.set_seed(seed)
    s2 = self.evaluate(vmf.sample(50, seed=seed))
    self.assertAllEqual(s1, s2)

  def testSampleMeanDir2d(self):
    mean_dirs = tf.math.l2_normalize([[1., 1], [-2, 1], [0, -1]], axis=-1)
    concentration = [[0], [0.1], [2], [40], [1000]]
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_dirs,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    self.assertEqual([5, 3], tensorshape_util.as_list(vmf.batch_shape))
    self.assertEqual([2], tensorshape_util.as_list(vmf.event_shape))
    nsamples = int(2e4)
    samples = vmf.sample(nsamples, seed=test_util.test_seed())
    self.assertEqual([nsamples, 5, 3, 2],
                     tensorshape_util.as_list(samples.shape))
    sample_mean = self.evaluate(samples).mean(axis=0)
    # Assert that positive-concentration distributions have samples with
    # the expected mean direction.
    sample_dir = (
        sample_mean / np.linalg.norm(sample_mean, axis=-1, keepdims=True))
    inner_product = self.evaluate(
        tf.reduce_sum(sample_dir * vmf.mean_direction, axis=-1))
    # All except the 0-concentration distribution should have >0 inner product
    # with the mean direction of the distribution.
    self.assertAllGreater(inner_product[1:], 0.1)
    # Pick out >1 concentration distributions to assert ~1 inner product with
    # mean direction.
    self.assertAllClose(np.ones_like(inner_product)[2:], inner_product[2:],
                        atol=1e-3)
    # Inner products should be roughly ascending by concentration.
    self.assertAllClose(np.round(np.sort(inner_product, axis=0), decimals=3),
                        np.round(inner_product, decimals=3),
                        atol=.005)
    means = self.evaluate(vmf.mean())
    # Mean vector for 0-concentration is precisely (0, 0).
    self.assertAllEqual(np.zeros_like(means[0]), means[0])
    mean_lengths = np.linalg.norm(means, axis=-1)
    # Length of the mean vector is strictly ascending with concentration.
    self.assertAllEqual(mean_lengths, np.sort(mean_lengths, axis=0))
    self.assertAllClose(np.linalg.norm(sample_mean, axis=-1), mean_lengths,
                        atol=0.03)

  def testSampleMeanDir3d(self):
    mean_dir = tf.math.l2_normalize([[1., 2, 3], [-2, -3, -1]], axis=-1)
    concentration = [[0], [0.1], [2], [40], [1000]]
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    self.assertEqual([5, 2], tensorshape_util.as_list(vmf.batch_shape))
    self.assertEqual([3], tensorshape_util.as_list(vmf.event_shape))
    nsamples = int(2e4)
    samples = vmf.sample(
        sample_shape=[nsamples], seed=test_util.test_seed())
    self.assertEqual([nsamples, 5, 2, 3],
                     tensorshape_util.as_list(samples.shape))
    sample_mean = self.evaluate(samples).mean(axis=0)
    # Assert that positive-concentration distributions have samples with
    # the expected mean direction.
    sample_dir = (
        sample_mean / np.linalg.norm(sample_mean, axis=-1, keepdims=True))
    inner_product = self.evaluate(
        tf.reduce_sum(sample_dir * vmf.mean_direction, axis=-1))
    # All except the 0-concentration distribution should have >0 inner product
    # with the mean direction of the distribution.
    self.assertAllGreater(inner_product[1:], 0.1)
    # Pick out >1 concentration distributions to assert ~1 inner product with
    # mean direction.
    self.assertAllClose(np.ones_like(inner_product)[2:], inner_product[2:],
                        atol=1e-3)
    # Inner products should be roughly ascending by concentration.
    self.assertAllEqual(np.round(np.sort(inner_product, axis=0), decimals=3),
                        np.round(inner_product, decimals=3))
    means = self.evaluate(vmf.mean())
    # Mean vector for 0-concentration is precisely (0, 0, 0).
    self.assertAllEqual(np.zeros_like(means[0]), means[0])
    mean_lengths = np.linalg.norm(means, axis=-1)
    # Length of the mean vector is strictly ascending with concentration.
    self.assertAllEqual(mean_lengths, np.sort(mean_lengths, axis=0))
    self.assertAllClose(np.linalg.norm(sample_mean, axis=-1), mean_lengths,
                        atol=0.03)

  def _verifyPdfWithNumpy(self, vmf, atol=1e-4):
    """Verifies log_prob evaluations with numpy/scipy.

    Both uniform random points and sampled points are evaluated.

    Args:
      vmf: A `tfp.distributions.VonMisesFisher` instance.
      atol: Absolute difference tolerable.
    """
    dim = tf.compat.dimension_value(vmf.event_shape[-1])
    nsamples = 10
    # Sample some random points uniformly over the hypersphere using numpy.
    sample_shape = [nsamples] + tensorshape_util.as_list(
        vmf.batch_shape) + [dim]
    uniforms = np.random.randn(*sample_shape)
    uniforms /= np.linalg.norm(uniforms, axis=-1, keepdims=True)
    uniforms = uniforms.astype(dtype_util.as_numpy_dtype(vmf.dtype))
    # Concatenate in some sampled points from the distribution under test.
    vmf_samples = vmf.sample(
        sample_shape=[nsamples], seed=test_util.test_seed())
    samples = tf.concat([uniforms, vmf_samples], axis=0)
    samples = tf.debugging.check_numerics(samples, 'samples')
    samples = self.evaluate(samples)
    log_prob = vmf.log_prob(samples)
    log_prob = tf.debugging.check_numerics(log_prob, 'log_prob')
    conc = self.evaluate(vmf.concentration)
    mean_dir = self.evaluate(vmf.mean_direction)
    log_true_sphere_surface_area = (
        np.log(2) + (dim / 2) * np.log(np.pi) - sp_special.gammaln(dim / 2))
    expected = (
        conc * np.sum(samples * mean_dir, axis=-1) +
        np.where(conc > 0,
                 (dim / 2 - 1) * np.log(conc) -
                 (dim / 2) * np.log(2 * np.pi) -
                 np.log(sp_special.ive(dim / 2 - 1, conc)) -
                 np.abs(conc),
                 -log_true_sphere_surface_area))
    self.assertAllClose(expected, self.evaluate(log_prob),
                        atol=atol)

  def _verifySampleAndPdfConsistency(self, vmf, rtol=0.075):
    """Verifies samples are consistent with the PDF using importance sampling.

    In particular, we verify an estimate the surface area of the n-dimensional
    hypersphere, and the surface areas of the spherical caps demarcated by
    a handful of survival rates.

    Args:
      vmf: A `VonMisesFisher` distribution instance.
      rtol: Relative difference tolerable.
    """
    dim = tf.compat.dimension_value(vmf.event_shape[-1])
    nsamples = int(6e4)
    samples = vmf.sample(
        sample_shape=[nsamples], seed=test_util.test_seed())
    samples = tf.debugging.check_numerics(samples, 'samples')
    log_prob = vmf.log_prob(samples)
    log_prob = tf.debugging.check_numerics(log_prob, 'log_prob')
    log_importance = -log_prob
    sphere_surface_area_estimate, samples, importance, conc = self.evaluate([
        tf.exp(
            tf.reduce_logsumexp(log_importance, axis=0) -
            tf.math.log(tf.cast(nsamples, dtype=tf.float32))), samples,
        tf.exp(log_importance), vmf.concentration
    ])
    true_sphere_surface_area = 2 * (np.pi)**(dim / 2) * self.evaluate(
        tf.exp(-tf.math.lgamma(dim / 2)))
    # Broadcast to correct size
    true_sphere_surface_area += np.zeros_like(sphere_surface_area_estimate)
    # Highly concentrated distributions do not get enough coverage to provide
    # a reasonable full-sphere surface area estimate. These are covered below
    # by CDF-based hypersphere cap surface area estimates.
    self.assertAllClose(
        true_sphere_surface_area[np.where(conc < 3)],
        sphere_surface_area_estimate[np.where(conc < 3)],
        rtol=rtol)

    # Assert surface area of hyperspherical cap For some CDFs in [.05,.45],
    # (h must be greater than 0 for the hypersphere cap surface area
    # calculation to hold).
    for survival_rate in 0.95, .9, .75, .6:
      cdf = (1 - survival_rate)
      mean_dir = self.evaluate(vmf.mean_direction)
      dotprods = np.sum(samples * mean_dir, -1)
      # Empirical estimate of the effective dot-product of the threshold that
      # selects for a given CDF level, that is the cosine of the largest
      # passable angle, or the minimum cosine for a within-CDF sample.
      dotprod_thresh = np.percentile(
          dotprods, 100 * survival_rate, axis=0, keepdims=True)
      dotprod_above_thresh = np.float32(dotprods > dotprod_thresh)
      sphere_cap_surface_area_ests = (
          cdf * (importance * dotprod_above_thresh).sum(0) /
          dotprod_above_thresh.sum(0))
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

  def _verifyCovariance(self, vmf):
    dim = tf.compat.dimension_value(vmf.event_shape[-1])
    nsamples = 10000
    samples = vmf.sample(nsamples, seed=test_util.test_seed())
    samples = tf.debugging.check_numerics(samples, 'samples')
    cov = vmf.covariance()
    samples, cov = self.evaluate([samples, cov])
    batched_samples = np.reshape(samples, [nsamples, -1, dim])
    batch_size = batched_samples.shape[1]
    est_cov = np.zeros([batch_size, dim, dim], dtype=cov.dtype)
    for bi in range(batched_samples.shape[1]):
      est_cov[bi] = np.cov(batched_samples[:, bi], rowvar=False)
    self.assertAllClose(
        np.reshape(est_cov, cov.shape),
        cov,
        atol=0.015)

  def testSampleAndPdfConsistency2d(self):
    mean_dir = tf.math.l2_normalize([[1., 2], [-2, -3]], axis=-1)
    concentration = [[0], [1e-5], [0.1], [1], [10]]
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_dir, concentration=concentration,
        validate_args=True, allow_nan_stats=False)
    self._verifySampleAndPdfConsistency(vmf)
    self._verifyCovariance(vmf)
    self._verifyPdfWithNumpy(vmf)

  def testSampleAndPdfConsistency3d(self):
    mean_dir = tf.math.l2_normalize([[1., 2, 3], [-2, -3, -1]], axis=-1)
    concentration = [[0], [1e-5], [0.1], [1], [10]]
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_dir, concentration=concentration,
        validate_args=True, allow_nan_stats=False)
    self._verifySampleAndPdfConsistency(vmf)
    # TODO(bjp): Enable self._verifyCovariance(vmf)
    self._verifyPdfWithNumpy(vmf, atol=.002)

  def testSampleAndPdfConsistency4d(self):
    mean_dir = tf.math.l2_normalize([[1., 2, 3, 4], [-2, -3, -1, 0]], axis=-1)
    concentration = [[0], [1e-4], [0.1], [1], [10]]
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_dir, concentration=concentration,
        validate_args=True, allow_nan_stats=False)
    self._verifySampleAndPdfConsistency(vmf)
    # TODO(bjp): Enable self._verifyCovariance(vmf)
    self._verifyPdfWithNumpy(vmf)

  def testSampleAndPdfConsistency5d(self):
    mean_dir = tf.math.l2_normalize(
        [[1., 2, 3, 4, 5], [-2, -3, -1, 0, 1]], axis=-1)
    # TODO(bjp): Numerical instability 0 < k < 1e-2 concentrations.
    # Should resolve by eliminating the bessel_i recurrence in favor of
    # a more stable algorithm, e.g. cephes.
    concentration = [[0], [5e-2], [0.1], [1], [10]]
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_dir, concentration=concentration,
        validate_args=True, allow_nan_stats=False)
    self._verifySampleAndPdfConsistency(vmf)
    # TODO(bjp): Enable self._verifyCovariance(vmf)
    self._verifyPdfWithNumpy(vmf, atol=2e-4)

  def testInternalShapeInference(self):
    # Regression test for the effect of b/139013403 on vMF sampling.
    sample_shape = tf.constant([2])
    # There needs to be a 1 dimension in the batch shape to trigger the bug
    mean_dir = tf.math.l2_normalize([1., 2, 3, 4], axis=-1)
    concentration = [0]
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_dir, concentration=concentration,
        validate_args=True, allow_nan_stats=False)
    self.evaluate(vmf.sample(sample_shape, seed=test_util.test_seed()))

  def testAssertsValidImmutableParams(self):
    with self.assertRaisesOpError('`concentration` must be non-negative'):
      vmf = tfp.distributions.VonMisesFisher(
          mean_direction=tf.math.l2_normalize([1., 2, 3], axis=-1),
          concentration=-1.,
          validate_args=True,
          allow_nan_stats=False)
      self.evaluate(vmf.mean())

    with self.assertRaisesOpError(
        '`mean_direction` may not have scalar event shape'):
      vmf = tfp.distributions.VonMisesFisher(
          mean_direction=[1.],
          concentration=0.,
          validate_args=True,
          allow_nan_stats=False)
      self.evaluate(vmf.mean())

    with self.assertRaisesOpError('`mean_direction` must be unit-length'):
      vmf = tfp.distributions.VonMisesFisher(
          mean_direction=tf.convert_to_tensor([1., 2, 3]),
          concentration=1.,
          validate_args=True,
          allow_nan_stats=False)
      self.evaluate(vmf.mean())

  def testAssertsValidMutableParams(self):
    mean_direction = tf.Variable(tf.math.l2_normalize([1., 2, 3], axis=-1))
    concentration = tf.Variable(1.)
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_direction,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)

    self.evaluate([mean_direction.initializer, concentration.initializer])
    self.evaluate(concentration.assign(-1.))
    with self.assertRaisesOpError('`concentration` must be non-negative'):
      self.evaluate(vmf.mean())

    self.evaluate((concentration.assign(1.),
                   mean_direction.assign([1., 2., 3.])))
    with self.assertRaisesOpError('`mean_direction` must be unit-length'):
      self.evaluate(vmf.mean())

    mean_direction = tf.Variable([1.])
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_direction,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)
    self.evaluate(mean_direction.initializer)
    with self.assertRaisesOpError(
        '`mean_direction` may not have scalar event shape'):
      self.evaluate(vmf.mean())

  def testAssertValidSample(self):
    mean_dir = tf.math.l2_normalize([[1., 2, 3], [-2, -3, -1]], axis=-1)
    concentration = [[0.], [2.]]
    vmf = tfp.distributions.VonMisesFisher(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True,
        allow_nan_stats=False)

    with self.assertRaisesOpError('Samples must be unit length.'):
      self.evaluate(vmf.prob([0.5, 0.5, 0.5]))

    msg = 'must have innermost dimension matching'
    static_shape_assertion = self.assertRaisesRegexp(ValueError, msg)
    dynamic_shape_assertion = self.assertRaisesOpError(msg)

    x = [[1., 0., 0., 0.]]
    with static_shape_assertion:
      self.evaluate(vmf.log_prob(x))

    x_var = tf.Variable(x, shape=tf.TensorShape(None))
    shape_assertion = (static_shape_assertion if tf.executing_eagerly()
                       else dynamic_shape_assertion)
    self.evaluate(x_var.initializer)
    with shape_assertion:
      self.evaluate(vmf.log_prob(x_var))

  def testSupportBijectorOutsideRange(self):
    mean_dir = np.array([[1., 2., 3.], [-2., -3., -1.]]).astype(np.float32)
    mean_dir /= np.linalg.norm(mean_dir, axis=-1)[:, np.newaxis]
    concentration = [[0], [0.1], [2], [40], [1000]]
    dist = tfp.distributions.VonMisesFisher(
        mean_direction=mean_dir,
        concentration=concentration,
        validate_args=True)

    x = mean_dir
    x[0][0] += 0.01
    with self.assertRaisesOpError('must sum to `1`'):
      self.evaluate(
          dist._experimental_default_event_space_bijector().inverse(x[0]))

    with self.assertRaisesOpError('must be non-negative'):
      self.evaluate(
          dist._experimental_default_event_space_bijector().inverse(x[1]))

if __name__ == '__main__':
  tf.test.main()
