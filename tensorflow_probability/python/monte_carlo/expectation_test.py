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
"""Tests for Monte Carlo Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow_probability.python.internal.monte_carlo import _get_samples


class GetSamplesTest(tfp_test_util.TestCase):
  """Test the private method 'get_samples'."""

  def test_raises_if_both_z_and_n_are_none(self):
    dist = tfd.Normal(loc=0., scale=1.)
    z = None
    n = None
    seed = None
    with self.assertRaisesRegexp(ValueError, 'exactly one'):
      _get_samples(dist, z, n, seed)

  def test_raises_if_both_z_and_n_are_not_none(self):
    dist = tfd.Normal(loc=0., scale=1.)
    z = dist.sample(seed=42)
    n = 1
    seed = None
    with self.assertRaisesRegexp(ValueError, 'exactly one'):
      _get_samples(dist, z, n, seed)

  def test_returns_n_samples_if_n_provided(self):
    dist = tfd.Normal(loc=0., scale=1.)
    z = None
    n = 10
    seed = None
    z = _get_samples(dist, z, n, seed)
    self.assertEqual((10,), z.shape)

  def test_returns_z_if_z_provided(self):
    dist = tfd.Normal(loc=0., scale=1.)
    z = dist.sample(10, seed=42)
    n = None
    seed = None
    z = _get_samples(dist, z, n, seed)
    self.assertEqual((10,), z.shape)


class ExpectationTest(tfp_test_util.TestCase):

  def test_works_correctly(self):
    x = tf.constant([-1e6, -100, -10, -1, 1, 10, 100, 1e6])

    # We use the prefex "efx" to mean "E_p[f(X)]".
    f = lambda u: u
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      p = tfd.Normal(loc=x, scale=1.)
      efx_true = x
      samples = p.sample(int(1e5), seed=1)
      efx_reparam = tfp.monte_carlo.expectation(f, samples, p.log_prob)
      efx_score = tfp.monte_carlo.expectation(f, samples, p.log_prob,
                                              use_reparameterization=False)

    [
        efx_true_,
        efx_reparam_,
        efx_score_,
        efx_true_grad_,
        efx_reparam_grad_,
        efx_score_grad_,
    ] = self.evaluate([
        efx_true,
        efx_reparam,
        efx_score,
        tape.gradient(efx_true, x),
        tape.gradient(efx_reparam, x),
        tape.gradient(efx_score, x),
    ])

    self.assertAllEqual(np.ones_like(efx_true_grad_), efx_true_grad_)

    self.assertAllClose(efx_true_, efx_reparam_, rtol=0.005, atol=0.)
    self.assertAllClose(efx_true_, efx_score_, rtol=0.005, atol=0.)

    self.assertAllEqual(np.ones_like(efx_true_grad_, dtype=np.bool),
                        np.isfinite(efx_reparam_grad_))
    self.assertAllEqual(np.ones_like(efx_true_grad_, dtype=np.bool),
                        np.isfinite(efx_score_grad_))

    self.assertAllClose(efx_true_grad_, efx_reparam_grad_,
                        rtol=0.03, atol=0.)
    # Variance is too high to be meaningful, so we'll only check those which
    # converge.
    self.assertAllClose(efx_true_grad_[2:-2],
                        efx_score_grad_[2:-2],
                        rtol=0.05, atol=0.)

  def test_docstring_example_normal(self):
    num_draws = int(1e5)
    mu_p = tf.constant(0.)
    mu_q = tf.constant(1.)
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(mu_p)
      tape.watch(mu_q)
      p = tfd.Normal(loc=mu_p, scale=1.)
      q = tfd.Normal(loc=mu_q, scale=2.)
      exact_kl_normal_normal = tfd.kl_divergence(p, q)
      approx_kl_normal_normal = tfp.monte_carlo.expectation(
          f=lambda x: p.log_prob(x) - q.log_prob(x),
          samples=p.sample(num_draws, seed=42),
          log_prob=p.log_prob,
          use_reparameterization=(p.reparameterization_type ==
                                  tfd.FULLY_REPARAMETERIZED))
    [exact_kl_normal_normal_, approx_kl_normal_normal_] = self.evaluate([
        exact_kl_normal_normal, approx_kl_normal_normal])
    self.assertEqual(
        True,
        p.reparameterization_type == tfd.FULLY_REPARAMETERIZED)
    self.assertAllClose(exact_kl_normal_normal_, approx_kl_normal_normal_,
                        rtol=0.01, atol=0.)

    # Compare gradients. (Not present in `docstring`.)
    gradp = lambda fp: tape.gradient(fp, mu_p)
    gradq = lambda fq: tape.gradient(fq, mu_q)
    [
        gradp_exact_kl_normal_normal_,
        gradq_exact_kl_normal_normal_,
        gradp_approx_kl_normal_normal_,
        gradq_approx_kl_normal_normal_,
    ] = self.evaluate([
        gradp(exact_kl_normal_normal),
        gradq(exact_kl_normal_normal),
        gradp(approx_kl_normal_normal),
        gradq(approx_kl_normal_normal),
    ])
    self.assertAllClose(gradp_exact_kl_normal_normal_,
                        gradp_approx_kl_normal_normal_,
                        rtol=0.01, atol=0.)
    self.assertAllClose(gradq_exact_kl_normal_normal_,
                        gradq_approx_kl_normal_normal_,
                        rtol=0.01, atol=0.)

  def test_docstring_example_bernoulli(self):
    num_draws = int(1e5)
    probs_p = tf.constant(0.4)
    probs_q = tf.constant(0.7)
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(probs_p)
      tape.watch(probs_q)
      p = tfd.Bernoulli(probs=probs_p)
      q = tfd.Bernoulli(probs=probs_q)
      exact_kl_bernoulli_bernoulli = tfp.monte_carlo.expectation(
          f=lambda x: p.log_prob(x) - q.log_prob(x),
          samples=p.sample(num_draws, seed=42),
          log_prob=p.log_prob,
          use_reparameterization=(
              p.reparameterization_type == tfd.FULLY_REPARAMETERIZED))
      approx_kl_bernoulli_bernoulli = tfd.kl_divergence(p, q)
    [
        exact_kl_bernoulli_bernoulli_,
        approx_kl_bernoulli_bernoulli_,
    ] = self.evaluate([
        exact_kl_bernoulli_bernoulli,
        approx_kl_bernoulli_bernoulli,
    ])
    self.assertEqual(False,
                     p.reparameterization_type == tfd.FULLY_REPARAMETERIZED)
    self.assertAllClose(
        exact_kl_bernoulli_bernoulli_,
        approx_kl_bernoulli_bernoulli_,
        rtol=0.01,
        atol=0.)
    print(exact_kl_bernoulli_bernoulli_, approx_kl_bernoulli_bernoulli_)

    # Compare gradients. (Not present in `docstring`.)
    gradp = lambda fp: tape.gradient(fp, probs_p)
    gradq = lambda fq: tape.gradient(fq, probs_q)
    [
        gradp_exact_kl_bernoulli_bernoulli_,
        gradq_exact_kl_bernoulli_bernoulli_,
        gradp_approx_kl_bernoulli_bernoulli_,
        gradq_approx_kl_bernoulli_bernoulli_,
    ] = self.evaluate([
        gradp(exact_kl_bernoulli_bernoulli),
        gradq(exact_kl_bernoulli_bernoulli),
        gradp(approx_kl_bernoulli_bernoulli),
        gradq(approx_kl_bernoulli_bernoulli),
    ])
    # Notice that variance (i.e., `rtol`) is higher when using score-trick.
    self.assertAllClose(
        gradp_exact_kl_bernoulli_bernoulli_,
        gradp_approx_kl_bernoulli_bernoulli_,
        rtol=0.05,
        atol=0.)
    self.assertAllClose(
        gradq_exact_kl_bernoulli_bernoulli_,
        gradq_approx_kl_bernoulli_bernoulli_,
        rtol=0.03,
        atol=0.)

  def test_works_with_structured_samples(self):

    # Check that we don't accidentally destroy the structure of `samples` when
    # it's a dict or other non-Tensor object from a joint distribution.
    p = tfd.JointDistributionNamed({
        'x': tfd.Normal(0., 1.),
        'y': tfd.Normal(0., 1.)})

    total_variance_with_reparam = tfp.monte_carlo.expectation(
        f=lambda d: d['x']**2 + d['y']**2,
        samples=p.sample(1000, seed=42),
        log_prob=p.log_prob,
        use_reparameterization=True)
    total_variance_without_reparam = tfp.monte_carlo.expectation(
        f=lambda d: d['x']**2 + d['y']**2,
        samples=p.sample(1000, seed=42),
        log_prob=p.log_prob,
        use_reparameterization=False)
    [
        total_variance_with_reparam_,
        total_variance_without_reparam_
    ] = self.evaluate([
        total_variance_with_reparam,
        total_variance_without_reparam])
    self.assertAllClose(total_variance_with_reparam_, 2., atol=0.2)
    self.assertAllClose(total_variance_without_reparam_, 2., atol=0.2)

if __name__ == '__main__':
  tf.test.main()
