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
"""Tests for the von Mises distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import special as sp_special
from scipy import stats as sp_stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class _VonMisesTest(object):

  def make_tensor(self, x):
    x = tf.cast(x, self.dtype)
    return tf1.placeholder_with_default(
        x, shape=x.shape if self.use_static_shape else None)

  def testVonMisesShape(self):
    loc = self.make_tensor([.1] * 5)
    concentration = self.make_tensor([.2] * 5)
    von_mises = tfd.VonMises(
        loc=loc, concentration=concentration, validate_args=True)

    self.assertEqual([
        5,
    ], self.evaluate(von_mises.batch_shape_tensor()))
    self.assertAllEqual([], self.evaluate(von_mises.event_shape_tensor()))
    if self.use_static_shape:
      self.assertEqual(tf.TensorShape([5]), von_mises.batch_shape)
    self.assertEqual(tf.TensorShape([]), von_mises.event_shape)

  def testInvalidConcentration(self):
    with self.assertRaisesOpError(
        'Argument `concentration` must be non-negative'):
      loc = self.make_tensor(0.)
      concentration = self.make_tensor(-.01)
      von_mises = tfd.VonMises(loc, concentration, validate_args=True)
      self.evaluate(von_mises.entropy())

  def testVonMisesLogPdf(self):
    locs_v = .1
    concentrations_v = .2
    x = np.array([2., 3., 4., 5., 6., 7.])
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v),
        self.make_tensor(concentrations_v),
        validate_args=True)
    expected_log_prob = sp_stats.vonmises.logpdf(
        x, concentrations_v, loc=locs_v)
    log_prob = von_mises.log_prob(self.make_tensor(x))
    self.assertAllClose(expected_log_prob, self.evaluate(log_prob))

  def testVonMisesLogPdfUniform(self):
    x = np.array([2., 3., 4., 5., 6., 7.])
    von_mises = tfd.VonMises(
        self.make_tensor(.1), self.make_tensor(0.), validate_args=True)
    log_prob = von_mises.log_prob(self.make_tensor(x))
    expected_log_prob = np.array([-np.log(2. * np.pi)] * 6)
    self.assertAllClose(expected_log_prob, self.evaluate(log_prob))

  def testVonMisesPdf(self):
    locs_v = .1
    concentrations_v = .2
    x = np.array([2., 3., 4., 5., 6., 7.])
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v),
        self.make_tensor(concentrations_v),
        validate_args=True)
    prob = von_mises.prob(self.make_tensor(x))
    expected_prob = sp_stats.vonmises.pdf(x, concentrations_v, loc=locs_v)
    self.assertAllClose(expected_prob, self.evaluate(prob))

  def testVonMisesPdfUniform(self):
    x = np.array([2., 3., 4., 5., 6., 7.])
    von_mises = tfd.VonMises(
        self.make_tensor(1.), self.make_tensor(0.), validate_args=True)
    prob = von_mises.prob(self.make_tensor(x))
    expected_prob = np.array([1. / (2. * np.pi)] * 6)
    self.assertAllClose(expected_prob, self.evaluate(prob))

  def testVonMisesCdf(self):
    locs_v = np.reshape(np.linspace(-10., 10., 20), [-1, 1, 1])
    concentrations_v = np.reshape(np.logspace(-3., 3., 20), [1, -1, 1])
    x = np.reshape(np.linspace(-10., 10., 20), [1, 1, -1])
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v),
        self.make_tensor(concentrations_v),
        validate_args=True)
    cdf = von_mises.cdf(self.make_tensor(x))
    expected_cdf = sp_stats.vonmises.cdf(x, concentrations_v, loc=locs_v)
    self.assertAllClose(expected_cdf, self.evaluate(cdf), atol=1e-4, rtol=1e-4)

  def testVonMisesCdfUniform(self):
    x = np.linspace(-np.pi, np.pi, 20)
    von_mises = tfd.VonMises(
        self.make_tensor(0.), self.make_tensor(0.), validate_args=True)
    cdf = von_mises.cdf(self.make_tensor(x))
    expected_cdf = (x + np.pi) / (2. * np.pi)
    self.assertAllClose(expected_cdf, self.evaluate(cdf))

  def testVonMisesCdfGradient(self):
    # The CDF is implemented manually, with custom gradients.
    # This test checks that the gradients are correct.
    # The gradient checker only works in graph mode and with static shapes.
    if tf.executing_eagerly() or not self.use_static_shape:
      return

    with self.cached_session():
      n = 10
      locs = tf.cast(tf.constant([1.] * n), self.dtype)
      concentrations = tf.cast(tf.constant(np.logspace(-3, 3, n)), self.dtype)
      von_mises = tfd.VonMises(locs, concentrations)
      x = tf.constant(self.evaluate(
          von_mises.sample(seed=test_util.test_seed())))
      cdf = von_mises.cdf(x)

      self.assertLess(
          tf1.test.compute_gradient_error(x, x.shape, cdf, cdf.shape), 1e-3)
      self.assertLess(
          tf1.test.compute_gradient_error(locs, locs.shape, cdf, cdf.shape),
          1e-3)
      self.assertLess(
          tf1.test.compute_gradient_error(concentrations, concentrations.shape,
                                          cdf, cdf.shape), 1e-3)

  def testVonMisesCdfGradientSimple(self):
    # This is a simple finite difference test that also works in the Eager mode.
    loc = self.make_tensor(0.5)
    concentration = self.make_tensor(0.7)
    x = self.make_tensor(0.6)

    _, [dcdf_dloc, dcdf_dconcentration, dcdf_dx] = self.evaluate(
        tfp.math.value_and_gradient(lambda l, c, x: tfd.VonMises(l, c).cdf(x),
                                    [loc, concentration, x]))

    eps = 1e-3
    dcdf_dloc_diff = self.evaluate(
        (tfd.VonMises(loc + eps, concentration, validate_args=True).cdf(x) -
         tfd.VonMises(loc - eps, concentration, validate_args=True).cdf(x)) /
        (2 * eps))
    dcdf_dconcentration_diff = self.evaluate(
        (tfd.VonMises(loc, concentration + eps, validate_args=True).cdf(x) -
         tfd.VonMises(loc, concentration - eps, validate_args=True).cdf(x)) /
        (2 * eps))
    dcdf_dx_diff = self.evaluate(
        (tfd.VonMises(loc, concentration, validate_args=True).cdf(x + eps) -
         tfd.VonMises(loc, concentration, validate_args=True).cdf(x - eps)) /
        (2 * eps))

    self.assertAlmostEqual(dcdf_dloc, dcdf_dloc_diff, places=3)
    self.assertAlmostEqual(
        dcdf_dconcentration, dcdf_dconcentration_diff, places=3)
    self.assertAlmostEqual(dcdf_dx, dcdf_dx_diff, places=3)

  def testVonMisesEntropy(self):
    locs_v = np.array([-2., -1., 0.3, 3.2]).reshape([-1, 1])
    concentrations_v = np.array([0.01, 0.01, 1., 10.]).reshape([1, -1])
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v),
        self.make_tensor(concentrations_v),
        validate_args=True)
    expected_entropy = sp_stats.vonmises.entropy(concentrations_v, loc=locs_v)
    self.assertAllClose(expected_entropy, self.evaluate(von_mises.entropy()))

  def testVonMisesEntropyUniform(self):
    von_mises = tfd.VonMises(-3., 0., validate_args=True)
    expected_entropy = np.log(2. * np.pi)
    self.assertAllClose(expected_entropy, self.evaluate(von_mises.entropy()))

  def testVonMisesMean(self):
    locs_v = np.array([-3., -2., -1., 0.3, 2.3])
    concentrations_v = np.array([0., 0.1, 1., 2., 10.])
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v),
        self.make_tensor(concentrations_v),
        validate_args=True)
    self.assertAllClose(locs_v, self.evaluate(von_mises.mean()))

  def testVonMisesVariance(self):
    locs_v = np.array([-3., -2., -1., 0.3, 2.3])
    concentrations_v = np.array([0., 0.1, 1., 2., 10.])
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v),
        self.make_tensor(concentrations_v),
        validate_args=True)
    expected_vars = 1. - sp_special.i1(concentrations_v) / sp_special.i0(
        concentrations_v)
    self.assertAllClose(expected_vars, self.evaluate(von_mises.variance()))

  def testVonMisesStddev(self):
    locs_v = np.array([-3., -2., -1., 0.3, 2.3]).reshape([1, -1])
    concentrations_v = np.array([0., 0.1, 1., 2., 10.]).reshape([-1, 1])
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v),
        self.make_tensor(concentrations_v),
        validate_args=True)
    expected_stddevs = (np.sqrt(1. - sp_special.i1(concentrations_v)
                                / sp_special.i0(concentrations_v))
                        + np.zeros_like(locs_v))
    self.assertAllClose(expected_stddevs, self.evaluate(von_mises.stddev()))

  def testVonMisesMode(self):
    locs_v = np.array([-3., -2., -1., 0.3, 2.3])
    concentrations_v = np.array([0., 0.1, 1., 2., 10.])
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v),
        self.make_tensor(concentrations_v),
        validate_args=True)
    expected_modes = locs_v
    self.assertAllClose(expected_modes, self.evaluate(von_mises.mode()))

  def testVonMisesVonMisesKL(self):
    d1 = tfd.VonMises(
        loc=self.make_tensor(np.array([[0.05, 0.1, 0.2]])),
        concentration=self.make_tensor(np.array([[0., 0.3, 0.4]])),
        validate_args=True)
    d2 = tfd.VonMises(
        loc=self.make_tensor(np.array([[0.7, 0.5, 0.3], [0.1, 0.3, 0.5]])),
        concentration=self.make_tensor(np.array([[0.8, 0., 0.5]])),
        validate_args=True)

    kl_actual = tfd.kl_divergence(d1, d2)
    x = d1.sample(int(1e5), seed=test_util.test_seed(hardcoded_seed=0))
    kl_sample = tf.reduce_mean(
        d1.log_prob(x) - d2.log_prob(x), axis=0)
    kl_same = tfd.kl_divergence(d1, d1)

    [kl_actual_val, kl_sample_val,
     kl_same_val] = self.evaluate([kl_actual, kl_sample, kl_same])

    # Computed by reference code.
    kl_expected = np.array([[0.15402061, 0.02212654, 0.00282222],
                            [0.15402061, 0.02212654, 0.00671171]])
    self.assertAllClose(kl_actual_val, kl_expected)
    self.assertAllClose(kl_actual_val, kl_sample_val, atol=0., rtol=1e-1)
    self.assertAllClose(kl_same_val, np.zeros((1, 3)))

  def testVonMisesSampleMoments(self):
    locs_v = np.array([-1., 0.3, 2.3])
    concentrations_v = np.array([1., 2., 10.])
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v),
        self.make_tensor(concentrations_v),
        validate_args=True)

    n = 10000
    seed = test_util.test_seed()
    samples = von_mises.sample(n, seed=seed)

    expected_mean = von_mises.mean()
    actual_mean = tf.atan2(
        tf.reduce_mean(tf.sin(samples), axis=0),
        tf.reduce_mean(tf.cos(samples), axis=0))

    expected_variance = von_mises.variance()
    standardized_samples = samples - tf.expand_dims(von_mises.mean(), 0)
    actual_variance = 1. - tf.reduce_mean(
        tf.cos(standardized_samples), axis=0)

    [
        expected_mean_val, expected_variance_val, actual_mean_val,
        actual_variance_val
    ] = self.evaluate(
        [expected_mean, expected_variance, actual_mean, actual_variance])

    self.assertAllClose(expected_mean_val, actual_mean_val, rtol=0.1)
    self.assertAllClose(expected_variance_val, actual_variance_val, rtol=0.1)

  def testVonMisesSampleVarianceUniform(self):
    von_mises = tfd.VonMises(
        self.make_tensor(1.), self.make_tensor(0.), validate_args=True)

    n = 10000
    samples = von_mises.sample(n, seed=test_util.test_seed())

    # For circular uniform distribution, the mean is not well-defined,
    # so only checking the variance.
    expected_variance = 1.
    standardized_samples = samples - tf.expand_dims(von_mises.mean(), 0)
    actual_variance = 1. - tf.reduce_mean(
        tf.cos(standardized_samples), axis=0)

    self.assertAllClose(
        expected_variance, self.evaluate(actual_variance), rtol=0.1)

  def testVonMisesSampleKsTest(self):
    concentrations_v = np.logspace(-3, 3, 50)
    # We are fixing the location to zero. The reason is that for loc != 0,
    # scipy's von Mises distribution CDF becomes shifted, so it's no longer
    # in [0, 1], but is in something like [-0.3, 0.7]. This breaks kstest.
    von_mises = tfd.VonMises(
        self.make_tensor(0.),
        self.make_tensor(concentrations_v),
        validate_args=True)
    n = 10000
    sample_values = self.evaluate(
        von_mises.sample(n, seed=test_util.test_seed()))
    self.assertEqual(sample_values.shape, (n, 50))

    fails = 0
    trials = 0
    for concentrationi, concentration in enumerate(concentrations_v):
      s = sample_values[:, concentrationi]
      trials += 1
      p = sp_stats.kstest(s, sp_stats.vonmises(concentration).cdf)[1]
      if p <= 0.05:
        fails += 1
    self.assertLess(fails, trials * 0.1)

  def testVonMisesSampleUniformKsTest(self):
    locs_v = np.linspace(-10., 10., 50)
    von_mises = tfd.VonMises(
        self.make_tensor(locs_v), self.make_tensor(0.), validate_args=True)
    n = 10000
    sample_values = self.evaluate(
        von_mises.sample(n, seed=test_util.test_seed(hardcoded_seed=137)))
    self.assertEqual(sample_values.shape, (n, 50))

    fails = 0
    trials = 0
    for loci, _ in enumerate(locs_v):
      s = sample_values[:, loci]
      # [-pi, pi] -> [0, 1]
      s = (s + np.pi) / (2. * np.pi)
      trials += 1
      # Compare to the CDF of Uniform(0, 1) random variable.
      p = sp_stats.kstest(s, sp_stats.uniform.cdf)[1]
      if p <= 0.05:
        fails += 1
    self.assertLess(fails, trials * 0.1)

  def testVonMisesSampleAverageGradient(self):
    loc = self.make_tensor([1.] * 7)
    concentration = self.make_tensor(np.logspace(-3, 3, 7))
    grad_ys = np.ones(7, dtype_util.as_numpy_dtype(self.dtype))
    n = 1000

    def loss(loc, concentration):
      von_mises = tfd.VonMises(loc, concentration, validate_args=True)
      samples = von_mises.sample(n, seed=test_util.test_seed())
      return tf.reduce_mean(samples, axis=0)

    _, [grad_loc, grad_concentration] = self.evaluate(
        tfp.math.value_and_gradient(
            loss, [loc, concentration]))

    # dsamples / dloc = 1 => dloss / dloc = dloss / dsamples = grad_ys
    self.assertAllClose(grad_loc, grad_ys, atol=1e-1, rtol=1e-1)
    self.assertAllClose(grad_concentration, [0.] * 7, atol=1e-1, rtol=1e-1)

  def testVonMisesSampleCircularVarianceGradient(self):
    loc = self.make_tensor([1.] * 7)
    concentration = self.make_tensor(np.logspace(-3, 3, 7))
    n = 1000

    def loss(loc, concentration):
      von_mises = tfd.VonMises(loc, concentration, validate_args=True)
      samples = von_mises.sample(n, seed=test_util.test_seed())
      return tf.reduce_mean(1. - tf.cos(samples - loc), axis=0)

    _, [grad_loc, grad_concentration] = self.evaluate(
        tfp.math.value_and_gradient(
            loss, [loc, concentration]))

    def analytical_loss(concentration):
      return 1. - tf.math.bessel_i1e(concentration) / tf.math.bessel_i0e(
          concentration)

    _, expected_grad_concentration, = self.evaluate(
        tfp.math.value_and_gradient(
            analytical_loss, concentration))

    self.assertAllClose(grad_loc, [0.] * 7, atol=1e-2, rtol=1e-2)
    self.assertAllClose(
        grad_concentration, expected_grad_concentration, atol=1e-1, rtol=1e-1)

  def testVonMisesSampleExtremeConcentration(self):
    loc = self.make_tensor([1., np.nan, 1., 1., np.nan])
    min_value = np.finfo(dtype_util.as_numpy_dtype(self.dtype)).min
    max_value = np.finfo(dtype_util.as_numpy_dtype(self.dtype)).max
    concentration = self.make_tensor([min_value, 1., max_value, np.nan, np.nan])
    von_mises = tfd.VonMises(loc, concentration, validate_args=False)

    samples = von_mises.sample(seed=test_util.test_seed())
    # Check that it does not end up in an infinite loop.
    self.assertEqual(self.evaluate(samples).shape, (5,))

  def testAssertsNonNegativeConcentration(self):
    concentration = tf.Variable(1.)
    d = tfd.VonMises(loc=0., concentration=concentration, validate_args=True)
    with self.assertRaisesOpError(
        'Argument `concentration` must be non-negative'):
      self.evaluate([v.initializer for v in d.variables])
      with tf.control_dependencies([concentration.assign(-1.)]):
        _ = self.evaluate(d.entropy())

  def testSupportBijectorOutsideRange(self):
    locs = np.array([-3., -2., -1., 0.3, 2.3])
    concentrations = np.array([0., 0.1, 1., 2., 10.])
    dist = tfd.VonMises(locs, concentration=concentrations, validate_args=True)
    eps = 1e-6
    x = np.array([[-np.pi - eps], [np.pi + eps]])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))


class VonMisesTestStaticShapeFloat32(test_util.TestCase, _VonMisesTest):
  dtype = tf.float32
  use_static_shape = True


class VonMisesTestDynamicShapeFloat64(test_util.TestCase, _VonMisesTest):
  dtype = tf.float64
  use_static_shape = False


if __name__ == '__main__':
  tf.test.main()
