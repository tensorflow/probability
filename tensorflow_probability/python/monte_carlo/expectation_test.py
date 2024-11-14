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

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal.monte_carlo import _get_samples
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.monte_carlo import expectation


class GetSamplesTest(test_util.TestCase):
  """Test the private method 'get_samples'."""

  def test_raises_if_both_z_and_n_are_none(self):
    dist = normal.Normal(loc=0., scale=1.)
    z = None
    n = None
    seed = test_util.test_seed()
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      _get_samples(dist, z, n, seed)

  def test_raises_if_both_z_and_n_are_not_none(self):
    dist = normal.Normal(loc=0., scale=1.)
    z = dist.sample(seed=test_util.test_seed())
    n = 1
    seed = test_util.test_seed()
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      _get_samples(dist, z, n, seed)

  def test_returns_n_samples_if_n_provided(self):
    dist = normal.Normal(loc=0., scale=1.)
    z = None
    n = 10
    seed = test_util.test_seed()
    z = _get_samples(dist, z, n, seed)
    self.assertEqual((10,), z.shape)

  def test_returns_z_if_z_provided(self):
    dist = normal.Normal(loc=0., scale=1.)
    seed = test_util.test_seed()
    z = dist.sample(10, seed=seed)
    n = None
    z = _get_samples(dist, z, n, seed)
    self.assertEqual((10,), z.shape)


class ExpectationTest(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def test_works_correctly(self):
    x = tf.constant([-1e6, -100, -10, -1, 1, 10, 100, 1e6])

    # We use the prefex "efx" to mean "E_p[f(X)]".
    f = lambda u: u

    efx_true = x

    def e_fx_reparam(x):
      p = normal.Normal(loc=x, scale=1.)
      samples = p.sample(int(1e5), seed=test_util.test_seed())
      return expectation.expectation(f, samples, p.log_prob)

    def e_fx_score(x):
      p = normal.Normal(loc=x, scale=1.)
      samples = p.sample(int(1e5), seed=test_util.test_seed())
      return expectation.expectation(
          f, samples, p.log_prob, use_reparameterization=False)

    efx_true, efx_true_grad = gradient.value_and_gradient(lambda x: x, x)
    efx_reparam, efx_reparam_grad = gradient.value_and_gradient(e_fx_reparam, x)
    efx_score, efx_score_grad = gradient.value_and_gradient(e_fx_score, x)

    self.assertAllEqual(tf.ones_like(efx_true_grad), efx_true_grad)

    self.assertAllClose(efx_true, efx_reparam, rtol=0.01, atol=0.)
    self.assertAllClose(efx_true, efx_score, rtol=0.01, atol=0.)

    self.assertAllEqual(tf.ones_like(efx_true_grad, dtype=tf.bool),
                        tf.math.is_finite(efx_reparam_grad))
    self.assertAllEqual(tf.ones_like(efx_true_grad, dtype=tf.bool),
                        tf.math.is_finite(efx_score_grad))

    self.assertAllClose(efx_true_grad, efx_reparam_grad,
                        rtol=0.03, atol=0.)
    # Variance is too high to be meaningful, so we'll only check those which
    # converge.
    self.assertAllClose(efx_true_grad[2:-2],
                        efx_score_grad[2:-2],
                        rtol=0.05, atol=0.)

  @test_util.numpy_disable_gradient_test
  def test_docstring_example_normal(self):
    num_draws = int(1e5)
    mu_p = tf.constant(0.)
    mu_q = tf.constant(1.)

    def exact_kl_normal_normal(mu_p, mu_q):
      p = normal.Normal(loc=mu_p, scale=1.)
      q = normal.Normal(loc=mu_q, scale=2.)
      return kullback_leibler.kl_divergence(p, q)

    def approximate_kl_normal_normal(mu_p, mu_q):
      p = normal.Normal(loc=mu_p, scale=1.)
      q = normal.Normal(loc=mu_q, scale=2.)
      return expectation.expectation(
          f=lambda x: p.log_prob(x) - q.log_prob(x),
          samples=p.sample(num_draws, seed=test_util.test_seed()),
          log_prob=p.log_prob,
          use_reparameterization=(p.reparameterization_type ==
                                  reparameterization.FULLY_REPARAMETERIZED))

    approx_kl_, approx_kl_grad = gradient.value_and_gradient(
        approximate_kl_normal_normal, mu_p, mu_q)
    exact_kl_, exact_kl_grad = gradient.value_and_gradient(
        exact_kl_normal_normal, mu_p, mu_q)

    self.assertAllClose(exact_kl_, approx_kl_, rtol=0.01, atol=0.)

    # Compare gradients. (Not present in `docstring`.)
    self.assertAllCloseNested(exact_kl_grad, approx_kl_grad,
                              rtol=0.01, atol=0.)

  @test_util.numpy_disable_gradient_test
  def test_docstring_example_bernoulli(self):
    num_draws = int(1e5)
    probs_p = tf.constant(0.4)
    probs_q = tf.constant(0.7)

    def exact_kl_bernoulli_bernoulli(probs_p, probs_q):
      p = bernoulli.Bernoulli(probs=probs_p)
      q = bernoulli.Bernoulli(probs=probs_q)
      return kullback_leibler.kl_divergence(p, q)

    def approx_kl_bernoulli_bernoulli(probs_p, probs_q):
      p = bernoulli.Bernoulli(probs=probs_p)
      q = bernoulli.Bernoulli(probs=probs_q)
      return expectation.expectation(
          f=lambda x: p.log_prob(x) - q.log_prob(x),
          samples=p.sample(num_draws, seed=test_util.test_seed()),
          log_prob=p.log_prob,
          use_reparameterization=(p.reparameterization_type ==
                                  reparameterization.FULLY_REPARAMETERIZED))

    approx_kl_, approx_kl_grad = gradient.value_and_gradient(
        approx_kl_bernoulli_bernoulli, probs_p, probs_q)
    exact_kl_, exact_kl_grad = gradient.value_and_gradient(
        exact_kl_bernoulli_bernoulli, probs_p, probs_q)

    self.assertAllClose(exact_kl_, approx_kl_, rtol=0.02, atol=0.)

    # Compare gradients. (Not present in `docstring`.)
    # Notice that variance (i.e., `rtol`) is higher when using score-trick.
    self.assertAllCloseNested(exact_kl_grad, approx_kl_grad, rtol=0.05, atol=0.)

  def test_works_with_structured_samples(self):

    # Check that we don't accidentally destroy the structure of `samples` when
    # it's a dict or other non-Tensor object from a joint distribution.
    p = jdn.JointDistributionNamed({
        'x': normal.Normal(0., 1.),
        'y': normal.Normal(0., 1.)
    })

    seed = test_util.test_seed()
    total_variance_with_reparam = expectation.expectation(
        f=lambda d: d['x']**2 + d['y']**2,
        samples=p.sample(1000, seed=seed),
        log_prob=p.log_prob,
        use_reparameterization=True)
    total_variance_without_reparam = expectation.expectation(
        f=lambda d: d['x']**2 + d['y']**2,
        samples=p.sample(1000, seed=seed),
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
  test_util.main()
