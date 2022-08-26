# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for LogitNormal."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import logitnormal as ln_lib
from tensorflow_probability.python.distributions import normal

from tensorflow_probability.python.internal import test_util


def logit_normal_trapezoid_rule(loc, scale):
  """Brute-force statistics of LogitNormal(loc, scale) by quadrature."""
  # LogitNormal samples as
  #   z ~ Normal(loc, scale)
  #   return sigmoid(z)
  # We find the statistics by integrating f(z) * Normal.pdf(z) over z.
  # The function f is always bounded, and for z outside +-10 * scale,
  # the Normal cdf is small enough to be negligible.  Thus it suffices
  # to integrate from loc - 10 * scale to loc + 10 * scale
  n = 10000
  width = 10.0
  xs = tf.linspace(loc - width*scale, loc + width*scale, n)
  def trapezoid(vals):
    total = tf.reduce_sum(vals, axis=0) - 0.5 * (vals[0] + vals[-1])
    return total * 2 * width * scale / tf.cast((n-1), xs.dtype)
  return xs, trapezoid


def logit_normal_mean_trapezoid(loc, scale):
  """Brute-force the mean of LogitNormal(loc, scale) by quadrature."""
  dist = normal.Normal(loc, scale)
  grid, compute = logit_normal_trapezoid_rule(loc, scale)
  return compute(tf.sigmoid(grid) * dist.prob(grid))


def logit_normal_variance_trapezoid(loc, scale):
  """Brute-force the variance of LogitNormal(loc, scale) by quadrature."""
  dist = normal.Normal(loc, scale)
  grid, compute = logit_normal_trapezoid_rule(loc, scale)
  probs = dist.prob(grid)
  sigmoids = tf.sigmoid(grid)
  mean = compute(sigmoids * probs)
  return compute((sigmoids - mean)**2 * probs)


@test_util.test_all_tf_execution_regimes
class LogitNormalTest(test_util.TestCase):

  def testLogitNormalMeanApprox(self):
    loc, scale = [-1.5, 0., 1.5], 0.4
    dist = ln_lib.LogitNormal(loc=loc, scale=scale, validate_args=True)
    x = dist.sample(int(10e3), seed=test_util.test_seed())
    [x_, mean_approx_] = self.evaluate([x, dist.mean_approx()])
    self.assertAllMeansClose(x_, mean_approx_, axis=0, atol=1e-4, rtol=0.01)

  def testLogitNormalMeanLogProbApprox(self):
    loc, scale = [-1.5, 0., 1.5], 0.4
    dist = ln_lib.LogitNormal(loc=loc, scale=scale, validate_args=True)
    x = dist.sample(int(10e3), seed=test_util.test_seed())
    y = tf.constant([0., 0.1, 0.5, 1.], dist.dtype)[:, tf.newaxis]
    samples = bernoulli.Bernoulli(probs=x).log_prob(y[..., tf.newaxis])
    [samples_, mean_approx_, mean_approx_default_] = self.evaluate([
        samples, dist.mean_log_prob_approx(y), dist.mean_log_prob_approx()])
    self.assertAllMeansClose(
        samples_, mean_approx_, axis=1, atol=1e-4, rtol=0.02)
    self.assertAllMeansClose(
        samples_[-1, :], mean_approx_default_, axis=0, atol=1e-4, rtol=0.02)

  def testLogitNormalVarianceApprox(self):
    seed_stream = test_util.test_seed_stream()
    loc = tf.random.uniform(shape=[30], seed=seed_stream())
    scale = tf.random.uniform(
        minval=0.1, maxval=5., shape=[30], seed=seed_stream())
    dist = ln_lib.LogitNormal(loc=loc, scale=scale, validate_args=True)
    x = dist.sample(int(10e3), seed=test_util.test_seed())
    variance_sample = tf.math.reduce_variance(x, axis=0)
    [variance_sample_, variance_approx_] = self.evaluate([
        variance_sample, dist.variance_approx()])
    self.assertAllClose(
        variance_sample_, variance_approx_, atol=1e-4, rtol=0.03)

  def testLogitNormalMeanGH(self):
    locs, scales = tf.meshgrid(tf.linspace(-10.0, 10.0, 10),
                               tf.exp(tf.linspace(-3.0, 0.0, 10)))
    ghs = ln_lib.logit_normal_mean_gh(locs, scales, deg=50)
    traps = logit_normal_mean_trapezoid(locs, scales)
    self.assertAllClose(traps, ghs, rtol=1e-4)

  def testLogitNormalVarianceGH(self):
    locs, scales = tf.meshgrid(tf.linspace(-10.0, 10.0, 10),
                               tf.exp(tf.linspace(-3.0, 0.0, 10)))
    ghs = ln_lib.logit_normal_variance_gh(locs, scales, deg=50)
    traps = logit_normal_variance_trapezoid(locs, scales)
    self.assertAllClose(traps, ghs, rtol=1e-4)

  def testLogitNormalMeanAndVariance(self):
    locs, scales = tf.meshgrid(tf.linspace(-10.0, 10.0, 10),
                               tf.exp(tf.linspace(-3.0, 3.0, 10)))
    dist = ln_lib.LogitNormal(
        loc=locs,
        scale=scales,
        validate_args=True,
        gauss_hermite_scale_limit=1.,
        num_probit_terms_approx=6)
    means = dist.mean_approx()
    trap_means = logit_normal_mean_trapezoid(locs, scales)
    self.assertAllClose(trap_means, means, rtol=1e-4)
    variances = dist.variance_approx()
    trap_variances = logit_normal_variance_trapezoid(locs, scales)
    self.assertAllClose(trap_variances, variances, rtol=1e-4)

  def testLogitNormalLogitNormalKL(self):
    batch_size = 6
    mu_a = np.array([3.0] * batch_size)
    sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    mu_b = np.array([-3.0] * batch_size)
    sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    ln_a = ln_lib.LogitNormal(loc=mu_a, scale=sigma_a, validate_args=True)
    ln_b = ln_lib.LogitNormal(loc=mu_b, scale=sigma_b, validate_args=True)

    kl = kullback_leibler.kl_divergence(ln_a, ln_b)
    kl_val = self.evaluate(kl)

    normal_a = normal.Normal(loc=mu_a, scale=sigma_a, validate_args=True)
    normal_b = normal.Normal(loc=mu_b, scale=sigma_b, validate_args=True)
    kl_expected_from_normal = kullback_leibler.kl_divergence(normal_a, normal_b)

    kl_expected_from_formula = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
        (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b)))

    x = ln_a.sample(int(1e5), seed=test_util.test_seed())
    kl_samples = ln_a.log_prob(x) - ln_b.log_prob(x)
    kl_samples_ = self.evaluate(kl_samples)

    self.assertEqual(kl.shape, (batch_size,))
    self.assertAllClose(kl_val, kl_expected_from_normal)
    self.assertAllClose(kl_val, kl_expected_from_formula)
    self.assertAllMeansClose(
        kl_samples_, kl_expected_from_formula, axis=0, atol=0.0, rtol=1e-2)

    # TODO(b/144948687) Avoid `nan` at boundary. Ideally we'd do this test:
#   def testPdfAtBoundary(self):
#     dist = ln_lib.LogitNormal(loc=[-5., 3.], scale=[[1., 2.], [3., 2.]],
#                            validate_args=True)
#     pdf_at_boundary = self.evaluate(dist.prob(0.))
#     self.assertAllEqual(pdf_at_boundary, np.zeros_like(pdf_at_boundary))
#
#     log_pdf_at_boundary = self.evaluate(dist.log_prob(0.))
#     self.assertAllNegativeInf(log_pdf_at_boundary)

  def testAssertValidSample(self):
    dist = ln_lib.LogitNormal(loc=0., scale=3., validate_args=True)
    self.evaluate(dist.prob([.1, .3, .6]))
    self.evaluate(dist.prob([.2, .3, .5]))
    # Either condition can trigger.
    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(dist.prob([-1., 0.1, 0.5]))
    with self.assertRaisesOpError('Sample must be less than or equal to `1`.'):
      self.evaluate(dist.prob([.1, .2, 1.2]))

  def testSupportBijectorOutsideRange(self):
    mu = np.array([1., 2., 3.])
    sigma = np.array([2., 4., 1.2])
    dist = ln_lib.LogitNormal(mu, sigma, validate_args=True)
    eps = 1e-6
    x = np.array([-2.3, -eps, 1. + eps, 1.4])
    bijector_inverse_x = dist.experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

if __name__ == '__main__':
  test_util.main()
