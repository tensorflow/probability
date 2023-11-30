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
"""Tests for Csiszar divergences."""

import functools

# Dependency imports
from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.stats import leave_one_out
from tensorflow_probability.python.vi import csiszar_divergence as cd


def tridiag(d, diag_value, offdiag_value):
  """d x d matrix with given value on diag, and one super/sub diag."""
  diag_mat = tf.eye(d) * (diag_value - offdiag_value)
  three_bands = tf.linalg.band_part(tf.fill([d, d], offdiag_value), 1, 1)
  return diag_mat + three_bands


@test_util.test_all_tf_execution_regimes
class AmariAlphaTest(test_util.TestCase):

  def setUp(self):
    super(AmariAlphaTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for alpha in [-1., 0., 1., 2.]:
      for normalized in [True, False]:
        self.assertAllClose(
            self.evaluate(
                cd.amari_alpha(0., alpha=alpha, self_normalized=normalized)),
            0.)

  def test_correct_when_alpha0(self):
    self.assertAllClose(
        self.evaluate(cd.amari_alpha(self._logu, alpha=0.)), -self._logu)

    self.assertAllClose(
        self.evaluate(
            cd.amari_alpha(self._logu, alpha=0., self_normalized=True)),
        -self._logu + (self._u - 1.))

  def test_correct_when_alpha1(self):
    self.assertAllClose(
        self.evaluate(cd.amari_alpha(self._logu, alpha=1.)),
        self._u * self._logu)

    self.assertAllClose(
        self.evaluate(
            cd.amari_alpha(self._logu, alpha=1., self_normalized=True)),
        self._u * self._logu - (self._u - 1.))

  def test_correct_when_alpha_not_01(self):
    for alpha in [-2, -1., -0.5, 0.5, 2.]:
      self.assertAllClose(
          self.evaluate(
              cd.amari_alpha(self._logu, alpha=alpha, self_normalized=False)),
          ((self._u**alpha - 1)) / (alpha * (alpha - 1.)))

      self.assertAllClose(
          self.evaluate(
              cd.amari_alpha(self._logu, alpha=alpha, self_normalized=True)),
          ((self._u**alpha - 1.) - alpha * (self._u - 1)) / (alpha *
                                                             (alpha - 1.)))


@test_util.test_all_tf_execution_regimes
class KLReverseTest(test_util.TestCase):

  def setUp(self):
    super(KLReverseTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for normalized in [True, False]:
      self.assertAllClose(
          self.evaluate(cd.kl_reverse(0., self_normalized=normalized)), 0.)

  def test_correct(self):
    self.assertAllClose(self.evaluate(cd.kl_reverse(self._logu)), -self._logu)

    self.assertAllClose(
        self.evaluate(cd.kl_reverse(self._logu, self_normalized=True)),
        -self._logu + (self._u - 1.))


@test_util.test_all_tf_execution_regimes
class KLForwardTest(test_util.TestCase):

  def setUp(self):
    super(KLForwardTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for normalized in [True, False]:
      self.assertAllClose(
          self.evaluate(cd.kl_forward(0., self_normalized=normalized)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.kl_forward(self._logu)), self._u * self._logu)

    self.assertAllClose(
        self.evaluate(cd.kl_forward(self._logu, self_normalized=True)),
        self._u * self._logu - (self._u - 1.))


@test_util.test_all_tf_execution_regimes
class JensenShannonTest(test_util.TestCase):

  def setUp(self):
    super(JensenShannonTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.jensen_shannon(0.)), np.log(0.25))

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(cd.jensen_shannon(self._logu)),
        self.evaluate(
            cd.symmetrized_csiszar_function(self._logu, cd.jensen_shannon)))

    self.assertAllClose(
        self.evaluate(cd.jensen_shannon(self._logu, self_normalized=True)),
        self.evaluate(
            cd.symmetrized_csiszar_function(
                self._logu,
                lambda x: cd.jensen_shannon(x, self_normalized=True))))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.jensen_shannon(self._logu)),
        (self._u * self._logu - (1 + self._u) * np.log1p(self._u)))

    self.assertAllClose(
        self.evaluate(cd.jensen_shannon(self._logu, self_normalized=True)),
        (self._u * self._logu - (1 + self._u) * np.log((1 + self._u) / 2)))


@test_util.test_all_tf_execution_regimes
class ArithmeticGeometricMeanTest(test_util.TestCase):

  def setUp(self):
    super(ArithmeticGeometricMeanTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.arithmetic_geometric(0.)), np.log(4))
    self.assertAllClose(
        self.evaluate(cd.arithmetic_geometric(0., self_normalized=True)), 0.)

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(cd.arithmetic_geometric(self._logu)),
        self.evaluate(
            cd.symmetrized_csiszar_function(self._logu,
                                            cd.arithmetic_geometric)))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.arithmetic_geometric(self._logu)),
        (1. + self._u) * np.log((1. + self._u) / np.sqrt(self._u)))

    self.assertAllClose(
        self.evaluate(
            cd.arithmetic_geometric(self._logu, self_normalized=True)),
        (1. + self._u) * np.log(0.5 * (1. + self._u) / np.sqrt(self._u)))


@test_util.test_all_tf_execution_regimes
class TotalVariationTest(test_util.TestCase):

  def setUp(self):
    super(TotalVariationTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.total_variation(0.)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.total_variation(self._logu)),
        0.5 * np.abs(self._u - 1))


@test_util.test_all_tf_execution_regimes
class PearsonTest(test_util.TestCase):

  def setUp(self):
    super(PearsonTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.pearson(0.)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.pearson(self._logu)), np.square(self._u - 1))


@test_util.test_all_tf_execution_regimes
class SquaredHellingerTest(test_util.TestCase):

  def setUp(self):
    super(SquaredHellingerTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.squared_hellinger(0.)), 0.)

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(cd.squared_hellinger(self._logu)),
        self.evaluate(
            cd.symmetrized_csiszar_function(self._logu, cd.squared_hellinger)))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.squared_hellinger(self._logu)),
        np.square(np.sqrt(self._u) - 1))


@test_util.test_all_tf_execution_regimes
class TriangularTest(test_util.TestCase):

  def setUp(self):
    super(TriangularTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.triangular(0.)), 0.)

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(cd.triangular(self._logu)),
        self.evaluate(
            cd.symmetrized_csiszar_function(self._logu, cd.triangular)))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.triangular(self._logu)),
        np.square(self._u - 1) / (1 + self._u))


@test_util.test_all_tf_execution_regimes
class TPowerTest(test_util.TestCase):

  def setUp(self):
    super(TPowerTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.t_power(0., t=-0.1)), 0.)
    self.assertAllClose(self.evaluate(cd.t_power(0., t=0.5)), 0.)
    self.assertAllClose(self.evaluate(cd.t_power(0., t=1.1)), 0.)
    self.assertAllClose(
        self.evaluate(cd.t_power(0., t=-0.1, self_normalized=True)), 0.)
    self.assertAllClose(
        self.evaluate(cd.t_power(0., t=0.5, self_normalized=True)), 0.)
    self.assertAllClose(
        self.evaluate(cd.t_power(0., t=1.1, self_normalized=True)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.t_power(self._logu, t=np.float64(-0.1))),
        self._u**-0.1 - 1.)
    self.assertAllClose(
        self.evaluate(cd.t_power(self._logu, t=np.float64(0.5))),
        -self._u**0.5 + 1.)
    self.assertAllClose(
        self.evaluate(cd.t_power(self._logu, t=np.float64(1.1))),
        self._u**1.1 - 1.)

  def test_correct_self_normalized(self):
    self.assertAllClose(
        self.evaluate(
            cd.t_power(self._logu, t=np.float64(-0.1), self_normalized=True)),
        self._u**-0.1 - 1. + 0.1 * (self._u - 1.))
    self.assertAllClose(
        self.evaluate(
            cd.t_power(self._logu, t=np.float64(0.5), self_normalized=True)),
        -self._u**0.5 + 1. + 0.5 * (self._u - 1.))
    self.assertAllClose(
        self.evaluate(
            cd.t_power(self._logu, t=np.float64(1.1), self_normalized=True)),
        self._u**1.1 - 1. - 1.1 * (self._u - 1.))


@test_util.test_all_tf_execution_regimes
class Log1pAbsTest(test_util.TestCase):

  def setUp(self):
    super(Log1pAbsTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.log1p_abs(0.)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.log1p_abs(self._logu)),
        self._u**(np.sign(self._u - 1)) - 1)


@test_util.test_all_tf_execution_regimes
class JeffreysTest(test_util.TestCase):

  def setUp(self):
    super(JeffreysTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.jeffreys(0.)), 0.)

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(cd.jeffreys(self._logu)),
        self.evaluate(cd.symmetrized_csiszar_function(self._logu, cd.jeffreys)))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.jeffreys(self._logu)),
        0.5 * (self._u * self._logu - self._logu))


@test_util.test_all_tf_execution_regimes
class ChiSquareTest(test_util.TestCase):

  def setUp(self):
    super(ChiSquareTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.chi_square(0.)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.chi_square(self._logu)), self._u**2 - 1)


@test_util.test_all_tf_execution_regimes
class ModifiedGanTest(test_util.TestCase):

  def setUp(self):
    super(ModifiedGanTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(cd.modified_gan(0.)), np.log(2))
    self.assertAllClose(
        self.evaluate(cd.modified_gan(0., self_normalized=True)), np.log(2))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(cd.modified_gan(self._logu)),
        np.log1p(self._u) - self._logu)

    self.assertAllClose(
        self.evaluate(cd.modified_gan(self._logu, self_normalized=True)),
        np.log1p(self._u) - self._logu + 0.5 * (self._u - 1))


@test_util.test_all_tf_execution_regimes
class SymmetrizedCsiszarFunctionTest(test_util.TestCase):

  def setUp(self):
    super(SymmetrizedCsiszarFunctionTest, self).setUp()
    self._logu = np.linspace(-10., 10., 100)
    self._u = np.exp(self._logu)

  def test_jensen_shannon(self):
    # The following functions come from the claim made in the
    # symmetrized_csiszar_function docstring.
    def js1(logu):
      return (-logu
              - (1. + tf.exp(logu)) * (
                  tf.nn.softplus(logu)))

    def js2(logu):
      return 2. * (tf.exp(logu) * (
          logu - tf.nn.softplus(logu)))

    self.assertAllClose(
        self.evaluate(cd.symmetrized_csiszar_function(self._logu, js1)),
        self.evaluate(cd.jensen_shannon(self._logu)))

    self.assertAllClose(
        self.evaluate(cd.symmetrized_csiszar_function(self._logu, js2)),
        self.evaluate(cd.jensen_shannon(self._logu)))

  def test_jeffreys(self):
    self.assertAllClose(
        self.evaluate(
            cd.symmetrized_csiszar_function(self._logu, cd.kl_reverse)),
        self.evaluate(cd.jeffreys(self._logu)))

    self.assertAllClose(
        self.evaluate(
            cd.symmetrized_csiszar_function(self._logu, cd.kl_forward)),
        self.evaluate(cd.jeffreys(self._logu)))


@test_util.test_all_tf_execution_regimes
class DualCsiszarFunctionTest(test_util.TestCase):

  def setUp(self):
    super(DualCsiszarFunctionTest, self).setUp()
    self._logu = np.linspace(-10., 10., 100)
    self._u = np.exp(self._logu)

  def test_kl_forward(self):
    self.assertAllClose(
        self.evaluate(cd.dual_csiszar_function(self._logu, cd.kl_forward)),
        self.evaluate(cd.kl_reverse(self._logu)))

  def test_kl_reverse(self):
    self.assertAllClose(
        self.evaluate(cd.dual_csiszar_function(self._logu, cd.kl_reverse)),
        self.evaluate(cd.kl_forward(self._logu)))


@test_util.test_all_tf_execution_regimes
class MonteCarloVariationalLossTest(test_util.TestCase):

  def test_kl_forward(self):
    q = normal.Normal(
        loc=np.ones(6), scale=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))

    p = normal.Normal(loc=q.loc + 0.1, scale=q.scale - 0.2)

    seed = test_util.test_seed()

    approx_kl = cd.monte_carlo_variational_loss(
        discrepancy_fn=cd.kl_forward,
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        sample_size=int(4e5),
        seed=seed)

    approx_kl_self_normalized = cd.monte_carlo_variational_loss(
        discrepancy_fn=(lambda logu: cd.kl_forward(logu, self_normalized=True)),
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        sample_size=int(4e5),
        seed=seed)

    exact_kl = kullback_leibler.kl_divergence(p, q)

    [approx_kl_, approx_kl_self_normalized_, exact_kl_] = self.evaluate([
        approx_kl, approx_kl_self_normalized, exact_kl])

    self.assertAllClose(approx_kl_, exact_kl_,
                        rtol=0.10, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                        rtol=0.06, atol=0.)

  def test_kl_reverse(self):
    q = normal.Normal(
        loc=np.ones(6), scale=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))

    p = normal.Normal(loc=q.loc + 0.1, scale=q.scale - 0.2)

    seed = test_util.test_seed()

    approx_kl = cd.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=cd.kl_reverse,
        sample_size=int(4.5e5),
        seed=seed)

    approx_kl_self_normalized = cd.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=(lambda logu: cd.kl_reverse(logu, self_normalized=True)),
        sample_size=int(4.5e5),
        seed=seed)

    exact_kl = kullback_leibler.kl_divergence(q, p)

    [approx_kl_, approx_kl_self_normalized_, exact_kl_] = self.evaluate([
        approx_kl, approx_kl_self_normalized, exact_kl])

    self.assertAllClose(approx_kl_, exact_kl_,
                        rtol=0.13, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                        rtol=0.07, atol=0.)

  def test_kl_forward_multidim(self):
    d = 5  # Dimension

    p = mvn_tril.MultivariateNormalTriL(
        scale_tril=tf.linalg.cholesky(
            tridiag(d, diag_value=1, offdiag_value=0.5)))

    # Variance is very high when approximating Forward KL, so we make
    # scale_diag large. This ensures q
    # "covers" p and thus Var_q[p/q] is smaller.
    q = mvn_diag.MultivariateNormalDiag(scale_diag=[1.] * d)

    seed = test_util.test_seed()

    approx_kl = cd.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=cd.kl_forward,
        sample_size=int(6e5),
        seed=seed)

    approx_kl_self_normalized = cd.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=(lambda logu: cd.kl_forward(logu, self_normalized=True)),
        sample_size=int(6e5),
        seed=seed)

    exact_kl = kullback_leibler.kl_divergence(p, q)

    [approx_kl_, approx_kl_self_normalized_, exact_kl_] = self.evaluate([
        approx_kl, approx_kl_self_normalized, exact_kl])

    self.assertAllClose(approx_kl_, exact_kl_,
                        rtol=0.14, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                        rtol=0.14, atol=0.)

  def test_kl_reverse_multidim(self):
    d = 5  # Dimension

    p = mvn_tril.MultivariateNormalTriL(
        scale_tril=tf.linalg.cholesky(
            tridiag(d, diag_value=1, offdiag_value=0.5)))

    # Variance is very high when approximating Reverse KL with self
    # normalization, because we pick up a term E_q[p / q]. So we make
    # scale_diag large. This ensures q "covers" p and thus Var_q[p/q] is
    # smaller.
    q = mvn_diag.MultivariateNormalDiag(scale_diag=[1.] * d)

    seed = test_util.test_seed()

    approx_kl = cd.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=cd.kl_reverse,
        sample_size=int(6e5),
        seed=seed)

    approx_kl_self_normalized = cd.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=(lambda logu: cd.kl_reverse(logu, self_normalized=True)),
        sample_size=int(6e5),
        seed=seed)

    exact_kl = kullback_leibler.kl_divergence(q, p)

    [approx_kl_, approx_kl_self_normalized_, exact_kl_] = self.evaluate([
        approx_kl, approx_kl_self_normalized, exact_kl])

    self.assertAllClose(approx_kl_, exact_kl_,
                        rtol=0.02, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                        rtol=0.14, atol=0.)

  def test_kl_with_joint_q(self):

    # Target distribution: equiv to MVNFullCovariance(cov=[[1., 1.], [1., 2.]])
    def target_log_prob_fn(z, x):
      return normal.Normal(0., 1.).log_prob(z) + normal.Normal(z,
                                                               1.).log_prob(x)

    # Factored q distribution: equiv to MVNDiag(scale_diag=[1., sqrt(2)])
    q_sequential = jds.JointDistributionSequential([  # Should pass as *args.
        normal.Normal(0., 1.),
        normal.Normal(0., tf.sqrt(2.))
    ])
    q_named = jdn.JointDistributionNamed({  # Should pass as **kwargs.
        'x': normal.Normal(0., tf.sqrt(2.)),
        'z': normal.Normal(0., 1.)
    })

    seed = test_util.test_seed()

    reverse_kl_sequential = cd.monte_carlo_variational_loss(
        target_log_prob_fn=target_log_prob_fn,
        surrogate_posterior=q_sequential,
        discrepancy_fn=cd.kl_reverse,
        sample_size=int(3e5),
        seed=seed)

    reverse_kl_named = cd.monte_carlo_variational_loss(
        target_log_prob_fn=target_log_prob_fn,
        surrogate_posterior=q_named,
        discrepancy_fn=cd.kl_reverse,
        sample_size=int(3e5),
        seed=seed)

    reverse_kl_sequential_, reverse_kl_named_, = self.evaluate(
        [reverse_kl_sequential, reverse_kl_named])

    # Compare to analytic MVN.kl[q|p] == 0.6534264.
    self.assertAllClose(reverse_kl_sequential_, 0.6534264, rtol=0.07, atol=0.)
    self.assertAllClose(reverse_kl_named_, 0.6534264, rtol=0.07, atol=0.)

  def test_importance_weighted_objective(self):
    seed = test_util.test_seed(sampler_type='stateless')

    # Use a normalized target, so the true normalizing constant (lowest possible
    # loss) is zero.
    target = normal.Normal(loc=0., scale=1.)
    proposal = student_t.StudentT(2, loc=3., scale=2.)

    elbo_loss = cd.monte_carlo_variational_loss(
        target_log_prob_fn=target.log_prob,
        surrogate_posterior=proposal,
        discrepancy_fn=cd.kl_reverse,
        sample_size=int(3e4),
        importance_sample_size=1,
        seed=seed)
    self.assertAllGreater(elbo_loss, 0.)

    # Check that importance sampling reduces the loss towards zero.
    iwae_10_loss = cd.monte_carlo_variational_loss(
        target_log_prob_fn=target.log_prob,
        surrogate_posterior=proposal,
        discrepancy_fn=cd.kl_reverse,
        sample_size=int(3e4),
        importance_sample_size=10,
        seed=seed)
    self.assertAllGreater(elbo_loss, iwae_10_loss)
    self.assertAllGreater(iwae_10_loss, 0)

    iwae_100_loss = cd.monte_carlo_variational_loss(
        target_log_prob_fn=target.log_prob,
        surrogate_posterior=proposal,
        discrepancy_fn=cd.kl_reverse,
        sample_size=int(3e4),
        importance_sample_size=100,
        seed=seed)
    self.assertAllGreater(iwae_10_loss, iwae_100_loss)
    self.assertAllClose(iwae_100_loss, 0, atol=0.1)

    # Check reproducibility
    elbo_loss_again = cd.monte_carlo_variational_loss(
        target_log_prob_fn=target.log_prob,
        surrogate_posterior=proposal,
        discrepancy_fn=cd.kl_reverse,
        sample_size=int(3e4),
        importance_sample_size=1,
        seed=seed)
    self.assertAllClose(elbo_loss_again, elbo_loss)

    iwae_10_loss_again = cd.monte_carlo_variational_loss(
        target_log_prob_fn=target.log_prob,
        surrogate_posterior=proposal,
        discrepancy_fn=cd.kl_reverse,
        sample_size=int(3e4),
        importance_sample_size=10,
        seed=seed)
    self.assertAllClose(iwae_10_loss_again, iwae_10_loss)

  @test_util.numpy_disable_gradient_test
  def test_score_trick(self):
    d = 5  # Dimension
    sample_size = int(4.5e5)
    seed = test_util.test_seed()

    # Variance is very high when approximating Forward KL, so we make
    # scale_diag large. This ensures q "covers" p and thus Var_q[p/q] is
    # smaller.
    s = tf.constant(1.)

    def construct_monte_carlo_csiszar_f_divergence(func, gradient_estimator):
      def _fn(s):
        p = mvn_tril.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(
                tridiag(d, diag_value=1, offdiag_value=0.5)))
        q = mvn_diag.MultivariateNormalDiag(scale_diag=tf.tile([s], [d]))
        return cd.monte_carlo_variational_loss(
            target_log_prob_fn=p.log_prob,
            surrogate_posterior=q,
            discrepancy_fn=func,
            sample_size=sample_size,
            gradient_estimator=gradient_estimator,
            seed=seed)
      return _fn

    approx_kl = construct_monte_carlo_csiszar_f_divergence(
        cd.kl_reverse,
        gradient_estimator=cd.GradientEstimators.REPARAMETERIZATION)

    approx_kl_self_normalized = construct_monte_carlo_csiszar_f_divergence(
        lambda logu: cd.kl_reverse(logu, self_normalized=True),
        gradient_estimator=cd.GradientEstimators.REPARAMETERIZATION)

    approx_kl_score_trick = construct_monte_carlo_csiszar_f_divergence(
        cd.kl_reverse, gradient_estimator=cd.GradientEstimators.SCORE_FUNCTION)

    approx_kl_self_normalized_score_trick = (
        construct_monte_carlo_csiszar_f_divergence(
            lambda logu: cd.kl_reverse(logu, self_normalized=True),
            gradient_estimator=cd.GradientEstimators.SCORE_FUNCTION))

    def exact_kl(s):
      p = mvn_tril.MultivariateNormalTriL(
          scale_tril=tf.linalg.cholesky(
              tridiag(d, diag_value=1, offdiag_value=0.5)))
      q = mvn_diag.MultivariateNormalDiag(scale_diag=tf.tile([s], [d]))
      return kullback_leibler.kl_divergence(q, p)

    [
        approx_kl_,
        approx_kl_grad_,
        approx_kl_self_normalized_,
        approx_kl_self_normalized_grad_,
        approx_kl_score_trick_,
        approx_kl_score_trick_grad_,
        approx_kl_self_normalized_score_trick_,
        approx_kl_self_normalized_score_trick_grad_,
        exact_kl_,
        exact_kl_grad_,
    ] = self.evaluate(
        list(gradient.value_and_gradient(approx_kl, s)) +
        list(gradient.value_and_gradient(approx_kl_self_normalized, s)) +
        list(gradient.value_and_gradient(approx_kl_score_trick, s)) + list(
            gradient.value_and_gradient(approx_kl_self_normalized_score_trick,
                                        s)) +
        list(gradient.value_and_gradient(exact_kl, s)))

    # Test average divergence.
    self.assertAllClose(approx_kl_, exact_kl_,
                        rtol=0.04, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                        rtol=0.08, atol=0.)

    self.assertAllClose(approx_kl_score_trick_, exact_kl_,
                        rtol=0.04, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_score_trick_, exact_kl_,
                        rtol=0.08, atol=0.)

    # Test average gradient-divergence.
    self.assertAllClose(approx_kl_grad_, exact_kl_grad_,
                        rtol=0.04, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_grad_, exact_kl_grad_,
                        rtol=0.04, atol=0.)

    self.assertAllClose(approx_kl_score_trick_grad_, exact_kl_grad_,
                        rtol=0.05, atol=0.)

    self.assertAllClose(
        approx_kl_self_normalized_score_trick_grad_, exact_kl_grad_,
        rtol=0.04, atol=0.)

  @test_util.numpy_disable_gradient_test
  def test_sticking_the_landing_gradient_is_zero_at_optimum(self):
    target_dist = normal.Normal(loc=2., scale=3.)

    def apply_fn(loc, raw_scale):
      return normal.Normal(loc=loc, scale=tf.nn.softplus(raw_scale))

    optimal_params = (target_dist.mean(),
                      softplus.Softplus().inverse(target_dist.stddev()))

    def loss(params, gradient_estimator):
      return cd.monte_carlo_variational_loss(
          target_dist.log_prob,
          surrogate_posterior=apply_fn(*params),
          stopped_surrogate_posterior=apply_fn(
              *[tf.stop_gradient(p) for p in params]),
          gradient_estimator=gradient_estimator,
          seed=test_util.test_seed(sampler_type='stateless'))

    elbo_loss, _ = gradient.value_and_gradient(
        functools.partial(
            loss, gradient_estimator=cd.GradientEstimators.REPARAMETERIZATION),
        [optimal_params])
    stl_loss, stl_grad = gradient.value_and_gradient(
        functools.partial(
            loss,
            gradient_estimator=(cd.GradientEstimators.DOUBLY_REPARAMETERIZED)),
        [optimal_params])
    self.assertAllClose(elbo_loss, stl_loss)
    for g in stl_grad[0]:
      self.assertAllClose(g, tf.zeros_like(g))

  @test_util.numpy_disable_gradient_test
  def test_doubly_reparameterized_reduces_iwae_gradient_variance(self):

    target_dist = normal.Normal(loc=2., scale=3.)

    def apply_fn(loc, raw_scale):
      return normal.Normal(loc=loc, scale=tf.nn.softplus(raw_scale))

    initial_params = (-3., softplus.Softplus().inverse(1.))

    def loss(params, gradient_estimator, seed):
      return cd.monte_carlo_variational_loss(
          target_dist.log_prob,
          surrogate_posterior=apply_fn(*params),
          stopped_surrogate_posterior=apply_fn(
              *[tf.stop_gradient(p) for p in params]),
          gradient_estimator=gradient_estimator,
          sample_size=10,
          importance_sample_size=10,
          seed=seed)

    seeds = samplers.split_seed(test_util.test_seed(sampler_type='stateless'),
                                n=30)
    iwae_grads = []
    dreg_grads = []
    for seed in seeds:
      iwae_loss, iwae_grad = gradient.value_and_gradient(
          functools.partial(
              loss,
              gradient_estimator=cd.GradientEstimators.REPARAMETERIZATION,
              seed=seed), [initial_params])
      dreg_loss, dreg_grad = gradient.value_and_gradient(
          functools.partial(
              loss,
              gradient_estimator=(cd.GradientEstimators.DOUBLY_REPARAMETERIZED),
              seed=seed), [initial_params])
      self.assertAllClose(iwae_loss, dreg_loss)
      iwae_grads.append(tf.convert_to_tensor(iwae_grad))
      dreg_grads.append(tf.convert_to_tensor(dreg_grad))

    self.assertAllClose(tf.reduce_mean(iwae_grads, axis=0),
                        tf.reduce_mean(dreg_grads, axis=0), atol=0.1)

    self.assertAllGreater(
        (tf.math.reduce_std(dreg_grads, axis=0) -
         tf.math.reduce_std(iwae_grads, axis=0)),
        0.)

  @parameterized.named_parameters(
      (
          '_score_function',
          # TODO(b/213378570): Support score function gradients for
          # importance-weighted bounds.
          cd.GradientEstimators.SCORE_FUNCTION,
          1),
      ('_reparameterization', cd.GradientEstimators.REPARAMETERIZATION, 5),
      ('_doubly_reparameterized', cd.GradientEstimators.DOUBLY_REPARAMETERIZED,
       5),
      ('_vimco', cd.GradientEstimators.VIMCO, 5))
  def test_gradient_estimators_do_not_modify_loss(self,
                                                  gradient_estimator,
                                                  importance_sample_size):

    def target_log_prob_fn(x):
      return normal.Normal(4., scale=1.).log_prob(x)

    seed = test_util.test_seed(sampler_type='stateless')
    sample_size = 10000

    surrogate_posterior = normal.Normal(loc=7., scale=2.)

    # Manually estimate the expected multi-sample / IWAE loss.
    zs, q_lp = surrogate_posterior.experimental_sample_and_log_prob(
        [sample_size, importance_sample_size],
        # Brittle hack to ensure that the q samples match those
        # drawn in `monte_carlo_variational_loss`.
        seed=samplers.split_seed(seed, 2)[0])
    log_weights = target_log_prob_fn(zs) - q_lp
    iwae_loss = -tf.reduce_mean(
        tf.math.reduce_logsumexp(log_weights, axis=1) - tf.math.log(
            tf.cast(importance_sample_size, dtype=log_weights.dtype)),
        axis=0)

    loss = cd.monte_carlo_variational_loss(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        gradient_estimator=gradient_estimator,
        stopped_surrogate_posterior=surrogate_posterior,  # Gradients unused.
        importance_sample_size=importance_sample_size,
        sample_size=sample_size,
        seed=seed)
    self.assertAllClose(iwae_loss, loss, atol=0.03)


@test_util.test_all_tf_execution_regimes
class CsiszarVIMCOTest(test_util.TestCase):

  def _csiszar_vimco_helper(self, logu):
    """Numpy implementation of `csiszar_vimco_helper`."""

    # Since this is a naive/intuitive implementation, we compensate by using the
    # highest precision we can.
    logu = np.float128(logu)
    n = logu.shape[0]
    u = np.exp(logu)
    loogeoavg_u = []  # Leave-one-out geometric-average of exp(logu).
    for j in range(n):
      loogeoavg_u.append(np.exp(np.mean(
          [logu[i, ...] for i in range(n) if i != j],
          axis=0)))
    loogeoavg_u = np.array(loogeoavg_u)

    loosum_u = []  # Leave-one-out sum of exp(logu).
    for j in range(n):
      loosum_u.append(np.sum(
          [u[i, ...] for i in range(n) if i != j],
          axis=0))
    loosum_u = np.array(loosum_u)

    # Natural log of the average u except each is swapped-out for its
    # leave-`i`-th-out Geometric average.
    log_sooavg_u = np.log(loosum_u + loogeoavg_u) - np.log(n)

    log_avg_u = np.log(np.mean(u, axis=0))
    return log_avg_u, log_sooavg_u

  @test_util.numpy_disable_gradient_test
  def test_vimco_and_gradient(self):
    dims = 5  # Dimension
    num_draws = int(1e3)
    num_batch_draws = int(3)
    seed = test_util.test_seed(sampler_type='stateless')

    f = lambda logu: cd.kl_reverse(logu, self_normalized=False)
    np_f = lambda logu: -logu
    p = mvn_tril.MultivariateNormalTriL(
        scale_tril=tf.linalg.cholesky(
            tridiag(dims, diag_value=1, offdiag_value=0.5)))
    # Variance is very high when approximating Forward KL, so we make
    # scale_diag large. This ensures q "covers" p and thus Var_q[p/q] is
    # smaller.
    build_q = (
        lambda s: mvn_diag.MultivariateNormalDiag(  # pylint:disable=g-long-lambda
            scale_diag=tf.tile([s], [dims])))

    def vimco_loss(s):
      return cd.monte_carlo_variational_loss(
          p.log_prob,
          surrogate_posterior=build_q(s),
          importance_sample_size=num_draws,
          sample_size=num_batch_draws,
          gradient_estimator=cd.GradientEstimators.VIMCO,
          discrepancy_fn=f,
          seed=seed)

    def logu(s):
      q = build_q(s)
      x = q.sample(sample_shape=[num_draws, num_batch_draws],
                   # Brittle hack to ensure that the q samples match those
                   # drawn in `monte_carlo_variational_loss`.
                   seed=samplers.split_seed(seed, 2)[0])
      x = tf.stop_gradient(x)
      return p.log_prob(x) - q.log_prob(x)

    def f_log_sum_u(s):
      return f(leave_one_out.log_soomean_exp(logu(s), axis=0)[::-1][0])

    def q_log_prob_x(s):
      q = build_q(s)
      x = q.sample(sample_shape=[num_draws, num_batch_draws],
                   # Brittle hack to ensure that the q samples match those
                   # drawn in `monte_carlo_variational_loss`.
                   seed=samplers.split_seed(seed, 2)[0])
      x = tf.stop_gradient(x)
      return q.log_prob(x)

    s = tf.constant(1.)
    logu_ = self.evaluate(logu(s))
    vimco_, grad_vimco_ = self.evaluate(
        gradient.value_and_gradient(vimco_loss, s))
    f_log_sum_u_, grad_mean_f_log_sum_u_ = self.evaluate(
        gradient.value_and_gradient(f_log_sum_u, s))
    grad_mean_f_log_sum_u_ /= num_batch_draws
    jacobian_logqx_ = self.evaluate(
        # Compute `jacobian(q_log_prob_x, s)` using `batch_jacobian` and messy
        # indexing.
        gradient.batch_jacobian(
            lambda s: q_log_prob_x(s[0, 0, ...])[None, ...],
            s[tf.newaxis, tf.newaxis, ...])[0, ..., 0])
    np_log_avg_u, np_log_sooavg_u = self._csiszar_vimco_helper(logu_)

    # Test VIMCO loss is correct.
    self.assertAllClose(np_f(np_log_avg_u).mean(axis=0), vimco_,
                        rtol=1e-4, atol=1e-5)

    # Test gradient of VIMCO loss is correct.
    #
    # To make this computation we'll inject two gradients from TF:
    # - grad[mean(f(log(sum(p(x)/q(x)))))]
    # - jacobian[log(q(x))].
    #
    # We now justify why using these (and only these) TF values for
    # ground-truth does not undermine the completeness of this test.
    #
    # Regarding `grad_mean_f_log_sum_u_`, note that we validate the
    # correctness of the zero-th order derivative (for each batch member).
    # Since `cd.csiszar_vimco_helper` itself does not manipulate any
    # gradient information, we can safely rely on TF.
    self.assertAllClose(np_f(np_log_avg_u), f_log_sum_u_, rtol=1e-4, atol=1e-5)
    #
    # Regarding `jacobian_logqx_`, note that testing the gradient of
    # `q.log_prob` is outside the scope of this unit-test thus we may safely
    # use TF to find it.

    # The `mean` is across batches and the `sum` is across iid samples.
    np_grad_vimco = (
        grad_mean_f_log_sum_u_
        + np.mean(
            np.sum(
                jacobian_logqx_ * (np_f(np_log_avg_u)
                                   - np_f(np_log_sooavg_u)),
                axis=0),
            axis=0))

    self.assertAllClose(np_grad_vimco, grad_vimco_, rtol=0.03, atol=1e-3)

  def test_vimco_with_joint_q(self):

    # Target distribution: equiv to MVNFullCovariance(cov=[[1., 1.], [1., 2.]])
    def p_log_prob(z, x):
      return normal.Normal(0., 1.).log_prob(z) + normal.Normal(z,
                                                               1.).log_prob(x)

    # Factored q distribution: equiv to MVNDiag(scale_diag=[1., sqrt(2)])
    q_sequential = jds.JointDistributionSequential([  # Should pass as *args.
        normal.Normal(0., 1.),
        normal.Normal(0., tf.sqrt(2.))
    ])
    q_named = jdn.JointDistributionNamed({  # Should pass as **kwargs.
        'x': normal.Normal(0., tf.sqrt(2.)),
        'z': normal.Normal(0., 1.)
    })

    seed = test_util.test_seed()

    reverse_kl_sequential = cd.monte_carlo_variational_loss(
        p_log_prob,
        surrogate_posterior=q_sequential,
        importance_sample_size=int(3e5),
        gradient_estimator=cd.GradientEstimators.VIMCO,
        seed=seed)

    reverse_kl_named = cd.monte_carlo_variational_loss(
        p_log_prob,
        surrogate_posterior=q_named,
        importance_sample_size=int(3e5),
        gradient_estimator=cd.GradientEstimators.VIMCO,
        seed=seed)

    [reverse_kl_sequential_, reverse_kl_named_
    ] = self.evaluate([reverse_kl_sequential, reverse_kl_named])

    self.assertAllClose(reverse_kl_sequential_, reverse_kl_named_, atol=0.02)

  def test_vimco_reproducibility(self):

    # Target distribution: equiv to MVNFullCovariance(cov=[[1., 1.], [1., 2.]])
    def p_log_prob(z, x):
      return normal.Normal(0., 1.).log_prob(z) + normal.Normal(z,
                                                               1.).log_prob(x)

    # Factored q distribution: equiv to MVNDiag(scale_diag=[1., sqrt(2)])
    q = jdn.JointDistributionNamed({  # Should pass as **kwargs.
        'x': normal.Normal(0., tf.sqrt(2.)),
        'z': normal.Normal(0., 1.)
    })

    seed = test_util.test_seed(sampler_type='stateless')

    reverse_kl = cd.monte_carlo_variational_loss(
        p_log_prob,
        surrogate_posterior=q,
        importance_sample_size=10,
        gradient_estimator=cd.GradientEstimators.VIMCO,
        seed=seed)

    reverse_kl_again = cd.monte_carlo_variational_loss(
        p_log_prob,
        surrogate_posterior=q,
        importance_sample_size=10,
        gradient_estimator=cd.GradientEstimators.VIMCO,
        seed=seed)

    self.assertAllClose(reverse_kl_again, reverse_kl)


if __name__ == '__main__':
  test_util.main()
