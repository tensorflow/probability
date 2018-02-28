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

from tensorflow_probability.python import monte_carlo as monte_carlo_lib
from tensorflow_probability.python.monte_carlo import _get_samples
from tensorflow.contrib import layers as layers_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.distributions import distribution as distribution_lib
from tensorflow.python.ops.distributions import gamma as gamma_lib
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test

layers = layers_lib
mc = monte_carlo_lib


class GetSamplesTest(test.TestCase):
  """Test the private method 'get_samples'."""

  def test_raises_if_both_z_and_n_are_none(self):
    with self.test_session():
      dist = normal_lib.Normal(loc=0., scale=1.)
      z = None
      n = None
      seed = None
      with self.assertRaisesRegexp(ValueError, 'exactly one'):
        _get_samples(dist, z, n, seed)

  def test_raises_if_both_z_and_n_are_not_none(self):
    with self.test_session():
      dist = normal_lib.Normal(loc=0., scale=1.)
      z = dist.sample(seed=42)
      n = 1
      seed = None
      with self.assertRaisesRegexp(ValueError, 'exactly one'):
        _get_samples(dist, z, n, seed)

  def test_returns_n_samples_if_n_provided(self):
    with self.test_session():
      dist = normal_lib.Normal(loc=0., scale=1.)
      z = None
      n = 10
      seed = None
      z = _get_samples(dist, z, n, seed)
      self.assertEqual((10,), z.get_shape())

  def test_returns_z_if_z_provided(self):
    with self.test_session():
      dist = normal_lib.Normal(loc=0., scale=1.)
      z = dist.sample(10, seed=42)
      n = None
      seed = None
      z = _get_samples(dist, z, n, seed)
      self.assertEqual((10,), z.get_shape())


class ExpectationTest(test.TestCase):

  def test_works_correctly(self):
    with self.test_session() as sess:
      x = constant_op.constant([-1e6, -100, -10, -1, 1, 10, 100, 1e6])
      p = normal_lib.Normal(loc=x, scale=1.)

      # We use the prefex "efx" to mean "E_p[f(X)]".
      f = lambda u: u
      efx_true = x
      samples = p.sample(int(1e5), seed=1)
      efx_reparam = mc.expectation(f, samples, p.log_prob)
      efx_score = mc.expectation(f, samples, p.log_prob,
                                 use_reparametrization=False)

      [
          efx_true_,
          efx_reparam_,
          efx_score_,
          efx_true_grad_,
          efx_reparam_grad_,
          efx_score_grad_,
      ] = sess.run([
          efx_true,
          efx_reparam,
          efx_score,
          gradients_impl.gradients(efx_true, x)[0],
          gradients_impl.gradients(efx_reparam, x)[0],
          gradients_impl.gradients(efx_score, x)[0],
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
    with self.test_session() as sess:
      num_draws = int(1e5)
      mu_p = constant_op.constant(0.)
      mu_q = constant_op.constant(1.)
      p = normal_lib.Normal(loc=mu_p, scale=1.)
      q = normal_lib.Normal(loc=mu_q, scale=2.)
      exact_kl_normal_normal = kullback_leibler.kl_divergence(p, q)
      approx_kl_normal_normal = monte_carlo_lib.expectation(
          f=lambda x: p.log_prob(x) - q.log_prob(x),
          samples=p.sample(num_draws, seed=42),
          log_prob=p.log_prob,
          use_reparametrization=(p.reparameterization_type
                                 == distribution_lib.FULLY_REPARAMETERIZED))
      [exact_kl_normal_normal_, approx_kl_normal_normal_] = sess.run([
          exact_kl_normal_normal, approx_kl_normal_normal])
      self.assertEqual(
          True,
          p.reparameterization_type == distribution_lib.FULLY_REPARAMETERIZED)
      self.assertAllClose(exact_kl_normal_normal_, approx_kl_normal_normal_,
                          rtol=0.01, atol=0.)

      # Compare gradients. (Not present in `docstring`.)
      gradp = lambda fp: gradients_impl.gradients(fp, mu_p)[0]
      gradq = lambda fq: gradients_impl.gradients(fq, mu_q)[0]
      [
          gradp_exact_kl_normal_normal_,
          gradq_exact_kl_normal_normal_,
          gradp_approx_kl_normal_normal_,
          gradq_approx_kl_normal_normal_,
      ] = sess.run([
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

  def test_docstring_example_gamma(self):
    with self.test_session() as sess:
      num_draws = int(1e5)
      concentration_p = constant_op.constant(1.)
      concentration_q = constant_op.constant(2.)
      p = gamma_lib.Gamma(concentration=concentration_p, rate=1.)
      q = gamma_lib.Gamma(concentration=concentration_q, rate=3.)
      approx_kl_gamma_gamma = monte_carlo_lib.expectation(
          f=lambda x: p.log_prob(x) - q.log_prob(x),
          samples=p.sample(num_draws, seed=42),
          log_prob=p.log_prob,
          use_reparametrization=(p.reparameterization_type
                                 == distribution_lib.FULLY_REPARAMETERIZED))
      exact_kl_gamma_gamma = kullback_leibler.kl_divergence(p, q)
      [exact_kl_gamma_gamma_, approx_kl_gamma_gamma_] = sess.run([
          exact_kl_gamma_gamma, approx_kl_gamma_gamma])
      self.assertEqual(
          False,
          p.reparameterization_type == distribution_lib.FULLY_REPARAMETERIZED)
      self.assertAllClose(exact_kl_gamma_gamma_, approx_kl_gamma_gamma_,
                          rtol=0.01, atol=0.)

      # Compare gradients. (Not present in `docstring`.)
      gradp = lambda fp: gradients_impl.gradients(fp, concentration_p)[0]
      gradq = lambda fq: gradients_impl.gradients(fq, concentration_q)[0]
      [
          gradp_exact_kl_gamma_gamma_,
          gradq_exact_kl_gamma_gamma_,
          gradp_approx_kl_gamma_gamma_,
          gradq_approx_kl_gamma_gamma_,
      ] = sess.run([
          gradp(exact_kl_gamma_gamma),
          gradq(exact_kl_gamma_gamma),
          gradp(approx_kl_gamma_gamma),
          gradq(approx_kl_gamma_gamma),
      ])
      # Notice that variance (i.e., `rtol`) is higher when using score-trick.
      self.assertAllClose(gradp_exact_kl_gamma_gamma_,
                          gradp_approx_kl_gamma_gamma_,
                          rtol=0.05, atol=0.)
      self.assertAllClose(gradq_exact_kl_gamma_gamma_,
                          gradq_approx_kl_gamma_gamma_,
                          rtol=0.03, atol=0.)


if __name__ == '__main__':
  test.main()
