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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class LogitNormalTest(test_util.TestCase):

  def testLogitNormalMeanApprox(self):
    loc, scale = [0., 1.5], 0.4
    dist = tfd.LogitNormal(loc=loc, scale=scale, validate_args=True)
    x = dist.sample(6000, seed=test_util.test_seed())
    mean_sample = tf.reduce_mean(x, axis=0)
    [mean_sample_, mean_approx_] = self.evaluate([
        mean_sample, dist.mean_approx()])
    self.assertAllClose(mean_sample_, mean_approx_, atol=0.02, rtol=0.02)

  def testLogitNormalLogitNormalKL(self):
    batch_size = 6
    mu_a = np.array([3.0] * batch_size)
    sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    mu_b = np.array([-3.0] * batch_size)
    sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    ln_a = tfd.LogitNormal(loc=mu_a, scale=sigma_a, validate_args=True)
    ln_b = tfd.LogitNormal(loc=mu_b, scale=sigma_b, validate_args=True)

    kl = tfd.kl_divergence(ln_a, ln_b)
    kl_val = self.evaluate(kl)

    normal_a = tfd.Normal(loc=mu_a, scale=sigma_a, validate_args=True)
    normal_b = tfd.Normal(loc=mu_b, scale=sigma_b, validate_args=True)
    kl_expected_from_normal = tfd.kl_divergence(normal_a, normal_b)

    kl_expected_from_formula = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
        (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b)))

    x = ln_a.sample(int(1e5), seed=test_util.test_seed())
    kl_sample = tf.reduce_mean(ln_a.log_prob(x) - ln_b.log_prob(x), axis=0)
    kl_sample_ = self.evaluate(kl_sample)

    self.assertEqual(kl.shape, (batch_size,))
    self.assertAllClose(kl_val, kl_expected_from_normal)
    self.assertAllClose(kl_val, kl_expected_from_formula)
    self.assertAllClose(
        kl_expected_from_formula, kl_sample_, atol=0.0, rtol=1e-2)

    # TODO(b/144948687) Avoid `nan` at boundary. Ideally we'd do this test:
#   def testPdfAtBoundary(self):
#     dist = tfd.LogitNormal(loc=[-5., 3.], scale=[[1., 2.], [3., 2.]],
#                            validate_args=True)
#     pdf_at_boundary = self.evaluate(dist.prob(0.))
#     self.assertAllEqual(pdf_at_boundary, np.zeros_like(pdf_at_boundary))
#
#     log_pdf_at_boundary = self.evaluate(dist.log_prob(0.))
#     self.assertAllNegativeInf(log_pdf_at_boundary)

  def testAssertValidSample(self):
    dist = tfd.LogitNormal(loc=0., scale=3., validate_args=True)
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
    dist = tfd.LogitNormal(mu, sigma, validate_args=True)
    eps = 1e-6
    x = np.array([-2.3, -eps, 1. + eps, 1.4])
    bijector_inverse_x = dist._experimental_default_event_space_bijector(
        ).inverse(x)
    self.assertAllNan(self.evaluate(bijector_inverse_x))

if __name__ == '__main__':
  tf.test.main()
