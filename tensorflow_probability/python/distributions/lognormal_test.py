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
"""Tests for LogNormal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class LogNormalTest(test_util.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def testLogNormalStats(self):

    loc = np.float32([3., 1.5])
    scale = np.float32([0.4, 1.1])
    dist = tfd.LogNormal(loc=loc, scale=scale, validate_args=True)

    self.assertAllClose(self.evaluate(dist.mean()),
                        np.exp(loc + scale**2 / 2))
    self.assertAllClose(self.evaluate(dist.variance()),
                        (np.exp(scale**2) - 1) * np.exp(2 * loc + scale**2))
    self.assertAllClose(self.evaluate(dist.stddev()),
                        np.sqrt(self.evaluate(dist.variance())))
    self.assertAllClose(self.evaluate(dist.mode()),
                        np.exp(loc - scale**2))
    self.assertAllClose(self.evaluate(dist.entropy()),
                        np.log(scale * np.exp(loc + 0.5) * np.sqrt(2 * np.pi)))

  def testLogNormalSample(self):
    loc, scale = 1.5, 0.4
    dist = tfd.LogNormal(loc=loc, scale=scale, validate_args=True)
    samples = self.evaluate(dist.sample(6000, seed=test_util.test_seed()))
    self.assertAllClose(np.mean(samples),
                        self.evaluate(dist.mean()),
                        atol=0.1)
    self.assertAllClose(np.std(samples),
                        self.evaluate(dist.stddev()),
                        atol=0.1)

  def testLogNormalPDF(self):
    loc, scale = 1.5, 0.4
    dist = tfd.LogNormal(loc=loc, scale=scale, validate_args=True)

    x = np.array([1e-4, 1.0, 2.0], dtype=np.float32)

    log_pdf = dist.log_prob(x)
    analytical_log_pdf = -np.log(x * scale * np.sqrt(2 * np.pi)) - (
        np.log(x) - loc)**2 / (2. * scale**2)

    self.assertAllClose(self.evaluate(log_pdf), analytical_log_pdf)

  def testLogNormalCDF(self):
    loc, scale = 1.5, 0.4
    dist = tfd.LogNormal(loc=loc, scale=scale, validate_args=True)

    x = np.array([1e-4, 1.0, 2.0], dtype=np.float32)

    cdf = dist.cdf(x)
    analytical_cdf = .5 + .5 * tf.math.erf(
        (np.log(x) - loc) / (scale * np.sqrt(2)))
    self.assertAllClose(self.evaluate(cdf),
                        self.evaluate(analytical_cdf))

  def testLogNormalLogNormalKL(self):
    batch_size = 6
    mu_a = np.array([3.0] * batch_size)
    sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    mu_b = np.array([-3.0] * batch_size)
    sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    ln_a = tfd.LogNormal(loc=mu_a, scale=sigma_a, validate_args=True)
    ln_b = tfd.LogNormal(loc=mu_b, scale=sigma_b, validate_args=True)

    kl = tfd.kl_divergence(ln_a, ln_b)
    kl_val = self.evaluate(kl)

    normal_a = tfd.Normal(loc=mu_a, scale=sigma_a, validate_args=True)
    normal_b = tfd.Normal(loc=mu_b, scale=sigma_b, validate_args=True)
    kl_expected_from_normal = tfd.kl_divergence(normal_a, normal_b)

    kl_expected_from_formula = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
        (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b)))

    x = ln_a.sample(int(2e5), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(ln_a.log_prob(x) - ln_b.log_prob(x), axis=0)
    kl_sample_ = self.evaluate(kl_sample)

    self.assertEqual(kl.shape, (batch_size,))
    self.assertAllClose(kl_val, kl_expected_from_normal)
    self.assertAllClose(kl_val, kl_expected_from_formula)
    self.assertAllClose(
        kl_expected_from_formula, kl_sample_, atol=0.0, rtol=1e-2)

  # TODO(b/144948687) Avoid `nan` at boundary. Ideally we'd do this test:
  # def testPdfAtBoundary(self):
  #   dist = tfd.LogNormal(loc=5., scale=2.)
  #   pdf = self.evaluate(dist.prob(0.))
  #   log_pdf = self.evaluate(dist.log_prob(0.))
  #   self.assertEqual(pdf, 0.)
  #   self.assertAllNegativeInf(log_pdf)

  def testAssertValidSample(self):
    dist = tfd.LogNormal(loc=[-3., 1., 4.], scale=2., validate_args=True)
    with self.assertRaisesOpError('Sample must be non-negative.'):
      self.evaluate(dist.cdf([3., -0.2, 1.]))

  def testSupportBijectorOutsideRange(self):
    dist = tfd.LogNormal(loc=1., scale=2., validate_args=True)
    with self.assertRaisesOpError('must be greater than or equal to 0'):
      dist._experimental_default_event_space_bijector().inverse(
          [-4.2, -1e-6, -1.3])

if __name__ == '__main__':
  tf.test.main()
