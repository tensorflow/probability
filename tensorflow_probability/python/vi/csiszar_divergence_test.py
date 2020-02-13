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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


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
                tfp.vi.amari_alpha(
                    0., alpha=alpha, self_normalized=normalized)),
            0.)

  def test_correct_when_alpha0(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.amari_alpha(self._logu, alpha=0.)),
        -self._logu)

    self.assertAllClose(
        self.evaluate(
            tfp.vi.amari_alpha(self._logu, alpha=0., self_normalized=True)),
        -self._logu + (self._u - 1.))

  def test_correct_when_alpha1(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.amari_alpha(self._logu, alpha=1.)),
        self._u * self._logu)

    self.assertAllClose(
        self.evaluate(
            tfp.vi.amari_alpha(self._logu, alpha=1., self_normalized=True)),
        self._u * self._logu - (self._u - 1.))

  def test_correct_when_alpha_not_01(self):
    for alpha in [-2, -1., -0.5, 0.5, 2.]:
      self.assertAllClose(
          self.evaluate(
              tfp.vi.amari_alpha(self._logu,
                                 alpha=alpha,
                                 self_normalized=False)),
          ((self._u**alpha - 1)) / (alpha * (alpha - 1.)))

      self.assertAllClose(
          self.evaluate(
              tfp.vi.amari_alpha(self._logu,
                                 alpha=alpha,
                                 self_normalized=True)),
          ((self._u**alpha - 1.)
           - alpha * (self._u - 1)) / (alpha * (alpha - 1.)))


@test_util.test_all_tf_execution_regimes
class KLReverseTest(test_util.TestCase):

  def setUp(self):
    super(KLReverseTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for normalized in [True, False]:
      self.assertAllClose(
          self.evaluate(tfp.vi.kl_reverse(0., self_normalized=normalized)),
          0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.kl_reverse(self._logu)),
        -self._logu)

    self.assertAllClose(
        self.evaluate(tfp.vi.kl_reverse(self._logu, self_normalized=True)),
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
          self.evaluate(tfp.vi.kl_forward(0., self_normalized=normalized)),
          0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.kl_forward(self._logu)),
        self._u * self._logu)

    self.assertAllClose(
        self.evaluate(tfp.vi.kl_forward(self._logu, self_normalized=True)),
        self._u * self._logu - (self._u - 1.))


@test_util.test_all_tf_execution_regimes
class JensenShannonTest(test_util.TestCase):

  def setUp(self):
    super(JensenShannonTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.jensen_shannon(0.)), np.log(0.25))

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.jensen_shannon(self._logu)),
        self.evaluate(tfp.vi.symmetrized_csiszar_function(
            self._logu, tfp.vi.jensen_shannon)))

    self.assertAllClose(
        self.evaluate(
            tfp.vi.jensen_shannon(self._logu, self_normalized=True)),
        self.evaluate(tfp.vi.symmetrized_csiszar_function(
            self._logu,
            lambda x: tfp.vi.jensen_shannon(x, self_normalized=True))))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.jensen_shannon(self._logu)),
        (self._u * self._logu
         - (1 + self._u) * np.log1p(self._u)))

    self.assertAllClose(
        self.evaluate(
            tfp.vi.jensen_shannon(self._logu, self_normalized=True)),
        (self._u * self._logu
         - (1 + self._u) * np.log((1 + self._u) / 2)))


@test_util.test_all_tf_execution_regimes
class ArithmeticGeometricMeanTest(test_util.TestCase):

  def setUp(self):
    super(ArithmeticGeometricMeanTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.arithmetic_geometric(0.)), np.log(4))
    self.assertAllClose(
        self.evaluate(
            tfp.vi.arithmetic_geometric(0., self_normalized=True)), 0.)

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.arithmetic_geometric(self._logu)),
        self.evaluate(tfp.vi.symmetrized_csiszar_function(
            self._logu, tfp.vi.arithmetic_geometric)))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.arithmetic_geometric(self._logu)),
        (1. + self._u) * np.log((1. + self._u) / np.sqrt(self._u)))

    self.assertAllClose(
        self.evaluate(
            tfp.vi.arithmetic_geometric(self._logu, self_normalized=True)),
        (1. + self._u) * np.log(0.5 * (1. + self._u) / np.sqrt(self._u)))


@test_util.test_all_tf_execution_regimes
class TotalVariationTest(test_util.TestCase):

  def setUp(self):
    super(TotalVariationTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(tfp.vi.total_variation(0.)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.total_variation(self._logu)),
        0.5 * np.abs(self._u - 1))


@test_util.test_all_tf_execution_regimes
class PearsonTest(test_util.TestCase):

  def setUp(self):
    super(PearsonTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(tfp.vi.pearson(0.)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.pearson(self._logu)),
        np.square(self._u - 1))


@test_util.test_all_tf_execution_regimes
class SquaredHellingerTest(test_util.TestCase):

  def setUp(self):
    super(SquaredHellingerTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(tfp.vi.squared_hellinger(0.)), 0.)

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.squared_hellinger(self._logu)),
        self.evaluate(tfp.vi.symmetrized_csiszar_function(
            self._logu, tfp.vi.squared_hellinger)))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.squared_hellinger(self._logu)),
        np.square(np.sqrt(self._u) - 1))


@test_util.test_all_tf_execution_regimes
class TriangularTest(test_util.TestCase):

  def setUp(self):
    super(TriangularTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(tfp.vi.triangular(0.)), 0.)

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.triangular(self._logu)),
        self.evaluate(tfp.vi.symmetrized_csiszar_function(
            self._logu, tfp.vi.triangular)))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.triangular(self._logu)),
        np.square(self._u - 1) / (1 + self._u))


@test_util.test_all_tf_execution_regimes
class TPowerTest(test_util.TestCase):

  def setUp(self):
    super(TPowerTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(tfp.vi.t_power(0., t=-0.1)), 0.)
    self.assertAllClose(self.evaluate(tfp.vi.t_power(0., t=0.5)), 0.)
    self.assertAllClose(self.evaluate(tfp.vi.t_power(0., t=1.1)), 0.)
    self.assertAllClose(
        self.evaluate(tfp.vi.t_power(0., t=-0.1, self_normalized=True)), 0.)
    self.assertAllClose(
        self.evaluate(tfp.vi.t_power(0., t=0.5, self_normalized=True)), 0.)
    self.assertAllClose(
        self.evaluate(tfp.vi.t_power(0., t=1.1, self_normalized=True)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.t_power(self._logu, t=np.float64(-0.1))),
        self._u ** -0.1 - 1.)
    self.assertAllClose(
        self.evaluate(tfp.vi.t_power(self._logu, t=np.float64(0.5))),
        -self._u ** 0.5 + 1.)
    self.assertAllClose(
        self.evaluate(tfp.vi.t_power(self._logu, t=np.float64(1.1))),
        self._u ** 1.1 - 1.)

  def test_correct_self_normalized(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.t_power(self._logu, t=np.float64(-0.1),
                                     self_normalized=True)),
        self._u ** -0.1 - 1. + 0.1 * (self._u - 1.))
    self.assertAllClose(
        self.evaluate(tfp.vi.t_power(self._logu, t=np.float64(0.5),
                                     self_normalized=True)),
        -self._u ** 0.5 + 1. + 0.5 * (self._u - 1.))
    self.assertAllClose(
        self.evaluate(tfp.vi.t_power(self._logu, t=np.float64(1.1),
                                     self_normalized=True)),
        self._u ** 1.1 - 1. - 1.1 * (self._u - 1.))


@test_util.test_all_tf_execution_regimes
class Log1pAbsTest(test_util.TestCase):

  def setUp(self):
    super(Log1pAbsTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(tfp.vi.log1p_abs(0.)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.log1p_abs(self._logu)),
        self._u**(np.sign(self._u - 1)) - 1)


@test_util.test_all_tf_execution_regimes
class JeffreysTest(test_util.TestCase):

  def setUp(self):
    super(JeffreysTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(tfp.vi.jeffreys(0.)), 0.)

  def test_symmetric(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.jeffreys(self._logu)),
        self.evaluate(tfp.vi.symmetrized_csiszar_function(
            self._logu, tfp.vi.jeffreys)))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.jeffreys(self._logu)),
        0.5 * (self._u * self._logu - self._logu))


@test_util.test_all_tf_execution_regimes
class ChiSquareTest(test_util.TestCase):

  def setUp(self):
    super(ChiSquareTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(self.evaluate(tfp.vi.chi_square(0.)), 0.)

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.chi_square(self._logu)),
        self._u**2 - 1)


@test_util.test_all_tf_execution_regimes
class ModifiedGanTest(test_util.TestCase):

  def setUp(self):
    super(ModifiedGanTest, self).setUp()
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.modified_gan(0.)), np.log(2))
    self.assertAllClose(
        self.evaluate(
            tfp.vi.modified_gan(0., self_normalized=True)), np.log(2))

  def test_correct(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.modified_gan(self._logu)),
        np.log1p(self._u) - self._logu)

    self.assertAllClose(
        self.evaluate(tfp.vi.modified_gan(self._logu, self_normalized=True)),
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
        self.evaluate(tfp.vi.symmetrized_csiszar_function(self._logu, js1)),
        self.evaluate(tfp.vi.jensen_shannon(self._logu)))

    self.assertAllClose(
        self.evaluate(tfp.vi.symmetrized_csiszar_function(self._logu, js2)),
        self.evaluate(tfp.vi.jensen_shannon(self._logu)))

  def test_jeffreys(self):
    self.assertAllClose(
        self.evaluate(tfp.vi.symmetrized_csiszar_function(
            self._logu, tfp.vi.kl_reverse)),
        self.evaluate(tfp.vi.jeffreys(self._logu)))

    self.assertAllClose(
        self.evaluate(tfp.vi.symmetrized_csiszar_function(
            self._logu, tfp.vi.kl_forward)),
        self.evaluate(tfp.vi.jeffreys(self._logu)))


@test_util.test_all_tf_execution_regimes
class DualCsiszarFunctionTest(test_util.TestCase):

  def setUp(self):
    super(DualCsiszarFunctionTest, self).setUp()
    self._logu = np.linspace(-10., 10., 100)
    self._u = np.exp(self._logu)

  def test_kl_forward(self):
    self.assertAllClose(
        self.evaluate(
            tfp.vi.dual_csiszar_function(self._logu, tfp.vi.kl_forward)),
        self.evaluate(tfp.vi.kl_reverse(self._logu)))

  def test_kl_reverse(self):
    self.assertAllClose(
        self.evaluate(
            tfp.vi.dual_csiszar_function(self._logu, tfp.vi.kl_reverse)),
        self.evaluate(tfp.vi.kl_forward(self._logu)))


@test_util.test_all_tf_execution_regimes
class MonteCarloVariationalLossTest(test_util.TestCase):

  def test_kl_forward(self):
    q = tfd.Normal(
        loc=np.ones(6),
        scale=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))

    p = tfd.Normal(loc=q.loc + 0.1, scale=q.scale - 0.2)

    seed = test_util.test_seed()

    approx_kl = tfp.vi.monte_carlo_variational_loss(
        discrepancy_fn=tfp.vi.kl_forward,
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        sample_size=int(4e5),
        seed=seed)

    approx_kl_self_normalized = tfp.vi.monte_carlo_variational_loss(
        discrepancy_fn=(
            lambda logu: tfp.vi.kl_forward(logu, self_normalized=True)),
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        sample_size=int(4e5),
        seed=seed)

    exact_kl = tfd.kl_divergence(p, q)

    [approx_kl_, approx_kl_self_normalized_, exact_kl_] = self.evaluate([
        approx_kl, approx_kl_self_normalized, exact_kl])

    self.assertAllClose(approx_kl_, exact_kl_,
                        rtol=0.10, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                        rtol=0.06, atol=0.)

  def test_kl_reverse(self):
    q = tfd.Normal(
        loc=np.ones(6),
        scale=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))

    p = tfd.Normal(loc=q.loc + 0.1, scale=q.scale - 0.2)

    seed = test_util.test_seed()

    approx_kl = tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=tfp.vi.kl_reverse,
        sample_size=int(4.5e5),
        seed=seed)

    approx_kl_self_normalized = tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=(
            lambda logu: tfp.vi.kl_reverse(logu, self_normalized=True)),
        sample_size=int(4.5e5),
        seed=seed)

    exact_kl = tfd.kl_divergence(q, p)

    [approx_kl_, approx_kl_self_normalized_, exact_kl_] = self.evaluate([
        approx_kl, approx_kl_self_normalized, exact_kl])

    self.assertAllClose(approx_kl_, exact_kl_,
                        rtol=0.13, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                        rtol=0.07, atol=0.)

  def test_kl_forward_multidim(self):
    d = 5  # Dimension

    p = tfd.MultivariateNormalFullCovariance(
        covariance_matrix=tridiag(d, diag_value=1, offdiag_value=0.5))

    # Variance is very high when approximating Forward KL, so we make
    # scale_diag large. This ensures q
    # "covers" p and thus Var_q[p/q] is smaller.
    q = tfd.MultivariateNormalDiag(scale_diag=[1.]*d)

    seed = test_util.test_seed()

    approx_kl = tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=tfp.vi.kl_forward,
        sample_size=int(6e5),
        seed=seed)

    approx_kl_self_normalized = tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=(
            lambda logu: tfp.vi.kl_forward(logu, self_normalized=True)),
        sample_size=int(6e5),
        seed=seed)

    exact_kl = tfd.kl_divergence(p, q)

    [approx_kl_, approx_kl_self_normalized_, exact_kl_] = self.evaluate([
        approx_kl, approx_kl_self_normalized, exact_kl])

    self.assertAllClose(approx_kl_, exact_kl_,
                        rtol=0.14, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                        rtol=0.14, atol=0.)

  def test_kl_reverse_multidim(self):
    d = 5  # Dimension

    p = tfd.MultivariateNormalFullCovariance(
        covariance_matrix=tridiag(d, diag_value=1, offdiag_value=0.5))

    # Variance is very high when approximating Reverse KL with self
    # normalization, because we pick up a term E_q[p / q]. So we make
    # scale_diag large. This ensures q "covers" p and thus Var_q[p/q] is
    # smaller.
    q = tfd.MultivariateNormalDiag(scale_diag=[1.]*d)

    seed = test_util.test_seed()

    approx_kl = tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=tfp.vi.kl_reverse,
        sample_size=int(6e5),
        seed=seed)

    approx_kl_self_normalized = tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=p.log_prob,
        surrogate_posterior=q,
        discrepancy_fn=(
            lambda logu: tfp.vi.kl_reverse(logu, self_normalized=True)),
        sample_size=int(6e5),
        seed=seed)

    exact_kl = tfd.kl_divergence(q, p)

    [approx_kl_, approx_kl_self_normalized_, exact_kl_] = self.evaluate([
        approx_kl, approx_kl_self_normalized, exact_kl])

    self.assertAllClose(approx_kl_, exact_kl_,
                        rtol=0.02, atol=0.)

    self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                        rtol=0.14, atol=0.)

  def test_kl_with_joint_q(self):

    # Target distribution: equiv to MVNFullCovariance(cov=[[1., 1.], [1., 2.]])
    def target_log_prob_fn(z, x):
      return tfd.Normal(0., 1.).log_prob(z) + tfd.Normal(z, 1.).log_prob(x)

    # Factored q distribution: equiv to MVNDiag(scale_diag=[1., sqrt(2)])
    q_sequential = tfd.JointDistributionSequential([  # Should pass as *args.
        tfd.Normal(0., 1.),
        tfd.Normal(0., tf.sqrt(2.))
    ])
    q_named = tfd.JointDistributionNamed({  # Should pass as **kwargs.
        'x': tfd.Normal(0., tf.sqrt(2.)),
        'z': tfd.Normal(0., 1.)
    })

    seed = test_util.test_seed()

    reverse_kl_sequential = tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=target_log_prob_fn,
        surrogate_posterior=q_sequential,
        discrepancy_fn=tfp.vi.kl_reverse,
        sample_size=int(3e5),
        seed=seed)

    reverse_kl_named = tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=target_log_prob_fn,
        surrogate_posterior=q_named,
        discrepancy_fn=tfp.vi.kl_reverse,
        sample_size=int(3e5),
        seed=seed)

    reverse_kl_sequential_, reverse_kl_named_, = self.evaluate(
        [reverse_kl_sequential, reverse_kl_named])

    # Compare to analytic MVN.kl[q|p] == 0.6534264.
    self.assertAllClose(reverse_kl_sequential_, 0.6534264, rtol=0.07, atol=0.)
    self.assertAllClose(reverse_kl_named_, 0.6534264, rtol=0.07, atol=0.)

  def test_score_trick(self):
    d = 5  # Dimension
    sample_size = int(4.5e5)
    seed = test_util.test_seed()

    # Variance is very high when approximating Forward KL, so we make
    # scale_diag large. This ensures q "covers" p and thus Var_q[p/q] is
    # smaller.
    s = tf.constant(1.)

    def construct_monte_carlo_csiszar_f_divergence(
        func, use_reparameterization=True):
      def _fn(s):
        p = tfd.MultivariateNormalFullCovariance(
            covariance_matrix=tridiag(d, diag_value=1, offdiag_value=0.5))
        q = tfd.MultivariateNormalDiag(scale_diag=tf.tile([s], [d]))
        return tfp.vi.monte_carlo_variational_loss(
            target_log_prob_fn=p.log_prob,
            surrogate_posterior=q,
            discrepancy_fn=func,
            sample_size=sample_size,
            use_reparameterization=use_reparameterization,
            seed=seed)
      return _fn

    approx_kl = construct_monte_carlo_csiszar_f_divergence(
        tfp.vi.kl_reverse)

    approx_kl_self_normalized = construct_monte_carlo_csiszar_f_divergence(
        lambda logu: tfp.vi.kl_reverse(logu, self_normalized=True))

    approx_kl_score_trick = construct_monte_carlo_csiszar_f_divergence(
        tfp.vi.kl_reverse, use_reparameterization=False)

    approx_kl_self_normalized_score_trick = (
        construct_monte_carlo_csiszar_f_divergence(
            lambda logu: tfp.vi.kl_reverse(logu, self_normalized=True),
            use_reparameterization=False))

    def exact_kl(s):
      p = tfd.MultivariateNormalFullCovariance(
          covariance_matrix=tridiag(d, diag_value=1, offdiag_value=0.5))
      q = tfd.MultivariateNormalDiag(scale_diag=tf.tile([s], [d]))
      return tfd.kl_divergence(q, p)

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
        list(tfp.math.value_and_gradient(approx_kl, s)) +
        list(tfp.math.value_and_gradient(approx_kl_self_normalized, s)) +
        list(tfp.math.value_and_gradient(approx_kl_score_trick, s)) +
        list(tfp.math.value_and_gradient(
            approx_kl_self_normalized_score_trick, s)) +
        list(tfp.math.value_and_gradient(exact_kl, s)))

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

  def test_vimco_and_gradient(self):
    dims = 5  # Dimension
    num_draws = int(1e3)
    num_batch_draws = int(3)
    seed = test_util.test_seed()

    with tf.GradientTape(persistent=True) as tape:
      f = lambda logu: tfp.vi.kl_reverse(logu, self_normalized=False)
      np_f = lambda logu: -logu

      s = tf.constant(1.)
      tape.watch(s)
      p = tfd.MultivariateNormalFullCovariance(
          covariance_matrix=tridiag(dims, diag_value=1, offdiag_value=0.5))

      # Variance is very high when approximating Forward KL, so we make
      # scale_diag large. This ensures q "covers" p and thus Var_q[p/q] is
      # smaller.
      q = tfd.MultivariateNormalDiag(
          scale_diag=tf.tile([s], [dims]))

      vimco = tfp.vi.csiszar_vimco(
          f=f,
          p_log_prob=p.log_prob,
          q=q,
          num_draws=num_draws,
          num_batch_draws=num_batch_draws,
          seed=seed)

      # We want the seed to be the same since we will use computations
      # with the same underlying sample to show correctness of vimco.
      if tf.executing_eagerly():
        tf.random.set_seed(seed)
      x = q.sample(sample_shape=[num_draws, num_batch_draws], seed=seed)
      x = tf.stop_gradient(x)
      logu = p.log_prob(x) - q.log_prob(x)
      f_log_sum_u = f(tfp.stats.log_soomean_exp(logu, axis=0)[::-1][0])
      q_log_prob_x = q.log_prob(x)

    grad_vimco = tape.gradient(vimco, s)
    grad_mean_f_log_sum_u = tape.gradient(f_log_sum_u, s) / num_batch_draws
    jacobian_logqx = tape.jacobian(q_log_prob_x, s)

    [
        logu_,
        jacobian_logqx_,
        vimco_,
        grad_vimco_,
        f_log_sum_u_,
        grad_mean_f_log_sum_u_,
    ] = self.evaluate([
        logu,
        jacobian_logqx,
        vimco,
        grad_vimco,
        f_log_sum_u,
        grad_mean_f_log_sum_u,
    ])

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
    # Since `tfp.vi.csiszar_vimco_helper` itself does not manipulate any
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
      return tfd.Normal(0., 1.).log_prob(z) + tfd.Normal(z, 1.).log_prob(x)

    # Factored q distribution: equiv to MVNDiag(scale_diag=[1., sqrt(2)])
    q_sequential = tfd.JointDistributionSequential([  # Should pass as *args.
        tfd.Normal(0., 1.),
        tfd.Normal(0., tf.sqrt(2.))
    ])
    q_named = tfd.JointDistributionNamed({  # Should pass as **kwargs.
        'x': tfd.Normal(0., tf.sqrt(2.)),
        'z': tfd.Normal(0., 1.)
    })

    seed = test_util.test_seed()

    reverse_kl_sequential = tfp.vi.csiszar_vimco(
        f=tfp.vi.kl_reverse,
        p_log_prob=p_log_prob,
        q=q_sequential,
        num_draws=int(3e5),
        seed=seed)

    reverse_kl_named = tfp.vi.csiszar_vimco(
        f=tfp.vi.kl_reverse,
        p_log_prob=p_log_prob,
        q=q_named,
        num_draws=int(3e5),
        seed=seed)

    [reverse_kl_sequential_, reverse_kl_named_
    ] = self.evaluate([reverse_kl_sequential, reverse_kl_named])

    self.assertAllClose(reverse_kl_sequential_, reverse_kl_named_, atol=0.02)


if __name__ == '__main__':
  tf.test.main()
