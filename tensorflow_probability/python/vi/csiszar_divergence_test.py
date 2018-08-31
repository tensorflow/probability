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

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.platform import test

tfd = tfp.distributions


def tridiag(d, diag_value, offdiag_value):
  """d x d matrix with given value on diag, and one super/sub diag."""
  diag_mat = tf.eye(d) * (diag_value - offdiag_value)
  three_bands = tf.matrix_band_part(
      tf.fill([d, d], offdiag_value), 1, 1)
  return diag_mat + three_bands


class AmariAlphaTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for alpha in [-1., 0., 1., 2.]:
      for normalized in [True, False]:
        with self.test_session(graph=tf.Graph()):
          self.assertAllClose(
              self.evaluate(
                  tfp.vi.amari_alpha(
                      0., alpha=alpha, self_normalized=normalized)),
              0.)

  def test_correct_when_alpha0(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.amari_alpha(self._logu, alpha=0.)),
          -self._logu)

      self.assertAllClose(
          self.evaluate(
              tfp.vi.amari_alpha(self._logu, alpha=0., self_normalized=True)),
          -self._logu + (self._u - 1.))

  def test_correct_when_alpha1(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.amari_alpha(self._logu, alpha=1.)),
          self._u * self._logu)

      self.assertAllClose(
          self.evaluate(
              tfp.vi.amari_alpha(self._logu, alpha=1., self_normalized=True)),
          self._u * self._logu - (self._u - 1.))

  def test_correct_when_alpha_not_01(self):
    for alpha in [-2, -1., -0.5, 0.5, 2.]:
      with self.test_session(graph=tf.Graph()):
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


class KLReverseTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for normalized in [True, False]:
      with self.test_session(graph=tf.Graph()):
        self.assertAllClose(
            self.evaluate(tfp.vi.kl_reverse(0., self_normalized=normalized)),
            0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.kl_reverse(self._logu)),
          -self._logu)

      self.assertAllClose(
          self.evaluate(tfp.vi.kl_reverse(self._logu, self_normalized=True)),
          -self._logu + (self._u - 1.))


class KLForwardTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    for normalized in [True, False]:
      with self.test_session(graph=tf.Graph()):
        self.assertAllClose(
            self.evaluate(tfp.vi.kl_forward(0., self_normalized=normalized)),
            0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.kl_forward(self._logu)),
          self._u * self._logu)

      self.assertAllClose(
          self.evaluate(tfp.vi.kl_forward(self._logu, self_normalized=True)),
          self._u * self._logu - (self._u - 1.))


class JensenShannonTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.jensen_shannon(0.)), np.log(0.25))

  def test_symmetric(self):
    with self.test_session():
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
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.jensen_shannon(self._logu)),
          (self._u * self._logu
           - (1 + self._u) * np.log1p(self._u)))

      self.assertAllClose(
          self.evaluate(
              tfp.vi.jensen_shannon(self._logu, self_normalized=True)),
          (self._u * self._logu
           - (1 + self._u) * np.log((1 + self._u) / 2)))


class ArithmeticGeometricMeanTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.arithmetic_geometric(0.)), np.log(4))
      self.assertAllClose(
          self.evaluate(
              tfp.vi.arithmetic_geometric(0., self_normalized=True)), 0.)

  def test_symmetric(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.arithmetic_geometric(self._logu)),
          self.evaluate(tfp.vi.symmetrized_csiszar_function(
              self._logu, tfp.vi.arithmetic_geometric)))

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.arithmetic_geometric(self._logu)),
          (1. + self._u) * np.log((1. + self._u) / np.sqrt(self._u)))

      self.assertAllClose(
          self.evaluate(
              tfp.vi.arithmetic_geometric(self._logu, self_normalized=True)),
          (1. + self._u) * np.log(0.5 * (1. + self._u) / np.sqrt(self._u)))


class TotalVariationTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(self.evaluate(tfp.vi.total_variation(0.)), 0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.total_variation(self._logu)),
          0.5 * np.abs(self._u - 1))


class PearsonTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(self.evaluate(tfp.vi.pearson(0.)), 0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.pearson(self._logu)),
          np.square(self._u - 1))


class SquaredHellingerTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(self.evaluate(tfp.vi.squared_hellinger(0.)), 0.)

  def test_symmetric(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.squared_hellinger(self._logu)),
          self.evaluate(tfp.vi.symmetrized_csiszar_function(
              self._logu, tfp.vi.squared_hellinger)))

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.squared_hellinger(self._logu)),
          np.square(np.sqrt(self._u) - 1))


class TriangularTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(self.evaluate(tfp.vi.triangular(0.)), 0.)

  def test_symmetric(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.triangular(self._logu)),
          self.evaluate(tfp.vi.symmetrized_csiszar_function(
              self._logu, tfp.vi.triangular)))

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.triangular(self._logu)),
          np.square(self._u - 1) / (1 + self._u))


class TPowerTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
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
    with self.test_session():
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
    with self.test_session():
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


class Log1pAbsTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(self.evaluate(tfp.vi.log1p_abs(0.)), 0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.log1p_abs(self._logu)),
          self._u**(np.sign(self._u - 1)) - 1)


class JeffreysTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(self.evaluate(tfp.vi.jeffreys(0.)), 0.)

  def test_symmetric(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.jeffreys(self._logu)),
          self.evaluate(tfp.vi.symmetrized_csiszar_function(
              self._logu, tfp.vi.jeffreys)))

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.jeffreys(self._logu)),
          0.5 * (self._u * self._logu - self._logu))


class ChiSquareTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(self.evaluate(tfp.vi.chi_square(0.)), 0.)

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.chi_square(self._logu)),
          self._u**2 - 1)


class ModifiedGanTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10, 100)
    self._u = np.exp(self._logu)

  def test_at_zero(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.modified_gan(0.)), np.log(2))
      self.assertAllClose(
          self.evaluate(
              tfp.vi.modified_gan(0., self_normalized=True)), np.log(2))

  def test_correct(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.modified_gan(self._logu)),
          np.log1p(self._u) - self._logu)

      self.assertAllClose(
          self.evaluate(tfp.vi.modified_gan(self._logu, self_normalized=True)),
          np.log1p(self._u) - self._logu + 0.5 * (self._u - 1))


class SymmetrizedCsiszarFunctionTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10., 100)
    self._u = np.exp(self._logu)

  def test_jensen_shannon(self):
    with self.test_session():

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
    with self.test_session():
      self.assertAllClose(
          self.evaluate(tfp.vi.symmetrized_csiszar_function(
              self._logu, tfp.vi.kl_reverse)),
          self.evaluate(tfp.vi.jeffreys(self._logu)))

      self.assertAllClose(
          self.evaluate(tfp.vi.symmetrized_csiszar_function(
              self._logu, tfp.vi.kl_forward)),
          self.evaluate(tfp.vi.jeffreys(self._logu)))


class DualCsiszarFunctionTest(test.TestCase):

  def setUp(self):
    self._logu = np.linspace(-10., 10., 100)
    self._u = np.exp(self._logu)

  def test_kl_forward(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(
              tfp.vi.dual_csiszar_function(self._logu, tfp.vi.kl_forward)),
          self.evaluate(tfp.vi.kl_reverse(self._logu)))

  def test_kl_reverse(self):
    with self.test_session():
      self.assertAllClose(
          self.evaluate(
              tfp.vi.dual_csiszar_function(self._logu, tfp.vi.kl_reverse)),
          self.evaluate(tfp.vi.kl_forward(self._logu)))


class MonteCarloCsiszarFDivergenceTest(test.TestCase):

  def test_kl_forward(self):
    with self.test_session() as sess:
      q = tfd.Normal(
          loc=np.ones(6),
          scale=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))

      p = tfd.Normal(loc=q.loc + 0.1, scale=q.scale - 0.2)

      approx_kl = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=tfp.vi.kl_forward,
          p_log_prob=p.log_prob,
          q=q,
          num_draws=int(1e5),
          seed=1)

      approx_kl_self_normalized = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=lambda logu: tfp.vi.kl_forward(logu, self_normalized=True),
          p_log_prob=p.log_prob,
          q=q,
          num_draws=int(1e5),
          seed=1)

      exact_kl = tfd.kl_divergence(p, q)

      [approx_kl_, approx_kl_self_normalized_, exact_kl_] = sess.run([
          approx_kl, approx_kl_self_normalized, exact_kl])

      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.08, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.02, atol=0.)

  def test_kl_reverse(self):
    with self.test_session() as sess:

      q = tfd.Normal(
          loc=np.ones(6),
          scale=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))

      p = tfd.Normal(loc=q.loc + 0.1, scale=q.scale - 0.2)

      approx_kl = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=tfp.vi.kl_reverse,
          p_log_prob=p.log_prob,
          q=q,
          num_draws=int(1e5),
          seed=1)

      approx_kl_self_normalized = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=lambda logu: tfp.vi.kl_reverse(logu, self_normalized=True),
          p_log_prob=p.log_prob,
          q=q,
          num_draws=int(1e5),
          seed=1)

      exact_kl = tfd.kl_divergence(q, p)

      [approx_kl_, approx_kl_self_normalized_, exact_kl_] = sess.run([
          approx_kl, approx_kl_self_normalized, exact_kl])

      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.07, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.02, atol=0.)

  def test_kl_reverse_multidim(self):

    with self.test_session() as sess:
      d = 5  # Dimension

      p = tfd.MultivariateNormalFullCovariance(
          covariance_matrix=tridiag(d, diag_value=1, offdiag_value=0.5))

      q = tfd.MultivariateNormalDiag(scale_diag=[0.5]*d)

      approx_kl = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=tfp.vi.kl_reverse,
          p_log_prob=p.log_prob,
          q=q,
          num_draws=int(1e5),
          seed=1)

      approx_kl_self_normalized = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=lambda logu: tfp.vi.kl_reverse(logu, self_normalized=True),
          p_log_prob=p.log_prob,
          q=q,
          num_draws=int(1e5),
          seed=1)

      exact_kl = tfd.kl_divergence(q, p)

      [approx_kl_, approx_kl_self_normalized_, exact_kl_] = sess.run([
          approx_kl, approx_kl_self_normalized, exact_kl])

      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.02, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.08, atol=0.)

  def test_kl_forward_multidim(self):

    with self.test_session() as sess:
      d = 5  # Dimension

      p = tfd.MultivariateNormalFullCovariance(
          covariance_matrix=tridiag(d, diag_value=1, offdiag_value=0.5))

      # Variance is very high when approximating Forward KL, so we make
      # scale_diag larger than in test_kl_reverse_multidim. This ensures q
      # "covers" p and thus Var_q[p/q] is smaller.
      q = tfd.MultivariateNormalDiag(scale_diag=[1.]*d)

      approx_kl = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=tfp.vi.kl_forward,
          p_log_prob=p.log_prob,
          q=q,
          num_draws=int(1e5),
          seed=1)

      approx_kl_self_normalized = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=lambda logu: tfp.vi.kl_forward(logu, self_normalized=True),
          p_log_prob=p.log_prob,
          q=q,
          num_draws=int(1e5),
          seed=1)

      exact_kl = tfd.kl_divergence(p, q)

      [approx_kl_, approx_kl_self_normalized_, exact_kl_] = sess.run([
          approx_kl, approx_kl_self_normalized, exact_kl])

      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.06, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.05, atol=0.)

  def test_score_trick(self):

    with self.test_session() as sess:
      d = 5  # Dimension
      num_draws = int(1e5)
      seed = 1

      p = tfd.MultivariateNormalFullCovariance(
          covariance_matrix=tridiag(d, diag_value=1, offdiag_value=0.5))

      # Variance is very high when approximating Forward KL, so we make
      # scale_diag larger than in test_kl_reverse_multidim. This ensures q
      # "covers" p and thus Var_q[p/q] is smaller.
      s = tf.constant(1.)
      q = tfd.MultivariateNormalDiag(
          scale_diag=tf.tile([s], [d]))

      approx_kl = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=tfp.vi.kl_reverse,
          p_log_prob=p.log_prob,
          q=q,
          num_draws=num_draws,
          seed=seed)

      approx_kl_self_normalized = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=lambda logu: tfp.vi.kl_reverse(logu, self_normalized=True),
          p_log_prob=p.log_prob,
          q=q,
          num_draws=num_draws,
          seed=seed)

      approx_kl_score_trick = tfp.vi.monte_carlo_csiszar_f_divergence(
          f=tfp.vi.kl_reverse,
          p_log_prob=p.log_prob,
          q=q,
          num_draws=num_draws,
          use_reparametrization=False,
          seed=seed)

      approx_kl_self_normalized_score_trick = (
          tfp.vi.monte_carlo_csiszar_f_divergence(
              f=lambda logu: tfp.vi.kl_reverse(logu, self_normalized=True),
              p_log_prob=p.log_prob,
              q=q,
              num_draws=num_draws,
              use_reparametrization=False,
              seed=seed))

      exact_kl = tfd.kl_divergence(q, p)

      grad_sum = lambda fs: tf.gradients(fs, s)[0]

      [
          approx_kl_grad_,
          approx_kl_self_normalized_grad_,
          approx_kl_score_trick_grad_,
          approx_kl_self_normalized_score_trick_grad_,
          exact_kl_grad_,
          approx_kl_,
          approx_kl_self_normalized_,
          approx_kl_score_trick_,
          approx_kl_self_normalized_score_trick_,
          exact_kl_,
      ] = sess.run([
          grad_sum(approx_kl),
          grad_sum(approx_kl_self_normalized),
          grad_sum(approx_kl_score_trick),
          grad_sum(approx_kl_self_normalized_score_trick),
          grad_sum(exact_kl),
          approx_kl,
          approx_kl_self_normalized,
          approx_kl_score_trick,
          approx_kl_self_normalized_score_trick,
          exact_kl,
      ])

      # Test average divergence.
      self.assertAllClose(approx_kl_, exact_kl_,
                          rtol=0.02, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_, exact_kl_,
                          rtol=0.08, atol=0.)

      self.assertAllClose(approx_kl_score_trick_, exact_kl_,
                          rtol=0.02, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_score_trick_, exact_kl_,
                          rtol=0.08, atol=0.)

      # Test average gradient-divergence.
      self.assertAllClose(approx_kl_grad_, exact_kl_grad_,
                          rtol=0.007, atol=0.)

      self.assertAllClose(approx_kl_self_normalized_grad_, exact_kl_grad_,
                          rtol=0.011, atol=0.)

      self.assertAllClose(approx_kl_score_trick_grad_, exact_kl_grad_,
                          rtol=0.018, atol=0.)

      self.assertAllClose(
          approx_kl_self_normalized_score_trick_grad_, exact_kl_grad_,
          rtol=0.017, atol=0.)


class CsiszarVIMCOTest(test.TestCase):

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

  def _csiszar_vimco_helper_grad(self, logu, delta):
    """Finite difference approximation of `grad(csiszar_vimco_helper, logu)`."""

    # This code actually estimates the sum of the Jacobiab because that's what
    # TF's `gradients` does.
    np_log_avg_u1, np_log_sooavg_u1 = self._csiszar_vimco_helper(
        logu[..., None] + np.diag([delta]*len(logu)))
    np_log_avg_u, np_log_sooavg_u = self._csiszar_vimco_helper(
        logu[..., None])
    return [
        (np_log_avg_u1 - np_log_avg_u) / delta,
        np.sum(np_log_sooavg_u1 - np_log_sooavg_u, axis=0) / delta,
    ]

  def test_vimco_helper_1(self):
    """Tests that function calculation correctly handles batches."""

    logu = np.linspace(-100., 100., 100).reshape([10, 2, 5])
    with self.test_session() as sess:
      np_log_avg_u, np_log_sooavg_u = self._csiszar_vimco_helper(logu)
      [log_avg_u, log_sooavg_u] = sess.run(tfp.vi.csiszar_vimco_helper(logu))
      self.assertAllClose(np_log_avg_u, log_avg_u,
                          rtol=1e-8, atol=0.)
      self.assertAllClose(np_log_sooavg_u, log_sooavg_u,
                          rtol=1e-8, atol=0.)

  def test_vimco_helper_2(self):
    """Tests that function calculation correctly handles overflow."""

    # Using 700 (rather than 1e3) since naive numpy version can't handle higher.
    logu = np.float32([0., 700, -1, 1])
    with self.test_session() as sess:
      np_log_avg_u, np_log_sooavg_u = self._csiszar_vimco_helper(logu)
      [log_avg_u, log_sooavg_u] = sess.run(tfp.vi.csiszar_vimco_helper(logu))
      self.assertAllClose(np_log_avg_u, log_avg_u,
                          rtol=1e-6, atol=0.)
      self.assertAllClose(np_log_sooavg_u, log_sooavg_u,
                          rtol=1e-5, atol=0.)

  def test_vimco_helper_3(self):
    """Tests that function calculation correctly handles underlow."""

    logu = np.float32([0., -1000, -1, 1])
    with self.test_session() as sess:
      np_log_avg_u, np_log_sooavg_u = self._csiszar_vimco_helper(logu)
      [log_avg_u, log_sooavg_u] = sess.run(tfp.vi.csiszar_vimco_helper(logu))
      self.assertAllClose(np_log_avg_u, log_avg_u,
                          rtol=1e-5, atol=0.)
      self.assertAllClose(np_log_sooavg_u, log_sooavg_u,
                          rtol=1e-4, atol=1e-15)

  def test_vimco_helper_gradient_using_finite_difference_1(self):
    """Tests that gradient calculation correctly handles batches."""

    logu_ = np.linspace(-100., 100., 100).reshape([10, 2, 5])
    with self.test_session() as sess:
      logu = tf.constant(logu_)

      grad = lambda flogu: tf.gradients(flogu, logu)[0]
      log_avg_u, log_sooavg_u = tfp.vi.csiszar_vimco_helper(logu)

      [
          grad_log_avg_u,
          grad_log_sooavg_u,
      ] = sess.run([grad(log_avg_u), grad(log_sooavg_u)])

      # We skip checking against finite-difference approximation since it
      # doesn't support batches.

      # Verify claim in docstring.
      self.assertAllClose(
          np.ones_like(grad_log_avg_u.sum(axis=0)),
          grad_log_avg_u.sum(axis=0))
      self.assertAllClose(
          np.ones_like(grad_log_sooavg_u.mean(axis=0)),
          grad_log_sooavg_u.mean(axis=0))

  def test_vimco_helper_gradient_using_finite_difference_2(self):
    """Tests that gradient calculation correctly handles overflow."""

    delta = 1e-3
    logu_ = np.float32([0., 1000, -1, 1])
    with self.test_session() as sess:
      logu = tf.constant(logu_)

      [
          np_grad_log_avg_u,
          np_grad_log_sooavg_u,
      ] = self._csiszar_vimco_helper_grad(logu_, delta)

      grad = lambda flogu: tf.gradients(flogu, logu)[0]
      log_avg_u, log_sooavg_u = tfp.vi.csiszar_vimco_helper(logu)

      [
          grad_log_avg_u,
          grad_log_sooavg_u,
      ] = sess.run([grad(log_avg_u), grad(log_sooavg_u)])

      self.assertAllClose(np_grad_log_avg_u, grad_log_avg_u,
                          rtol=delta, atol=0.)
      self.assertAllClose(np_grad_log_sooavg_u, grad_log_sooavg_u,
                          rtol=delta, atol=0.)
      # Verify claim in docstring.
      self.assertAllClose(
          np.ones_like(grad_log_avg_u.sum(axis=0)),
          grad_log_avg_u.sum(axis=0))
      self.assertAllClose(
          np.ones_like(grad_log_sooavg_u.mean(axis=0)),
          grad_log_sooavg_u.mean(axis=0))

  def test_vimco_helper_gradient_using_finite_difference_3(self):
    """Tests that gradient calculation correctly handles underlow."""

    delta = 1e-3
    logu_ = np.float32([0., -1000, -1, 1])
    with self.test_session() as sess:
      logu = tf.constant(logu_)

      [
          np_grad_log_avg_u,
          np_grad_log_sooavg_u,
      ] = self._csiszar_vimco_helper_grad(logu_, delta)

      grad = lambda flogu: tf.gradients(flogu, logu)[0]
      log_avg_u, log_sooavg_u = tfp.vi.csiszar_vimco_helper(logu)

      [
          grad_log_avg_u,
          grad_log_sooavg_u,
      ] = sess.run([grad(log_avg_u), grad(log_sooavg_u)])

      self.assertAllClose(np_grad_log_avg_u, grad_log_avg_u,
                          rtol=delta, atol=0.)
      self.assertAllClose(np_grad_log_sooavg_u, grad_log_sooavg_u,
                          rtol=delta, atol=0.)
      # Verify claim in docstring.
      self.assertAllClose(
          np.ones_like(grad_log_avg_u.sum(axis=0)),
          grad_log_avg_u.sum(axis=0))
      self.assertAllClose(
          np.ones_like(grad_log_sooavg_u.mean(axis=0)),
          grad_log_sooavg_u.mean(axis=0))

  def test_vimco_and_gradient(self):

    with self.test_session() as sess:
      dims = 5  # Dimension
      num_draws = int(20)
      num_batch_draws = int(3)
      seed = 1

      f = lambda logu: tfp.vi.kl_reverse(logu, self_normalized=False)
      np_f = lambda logu: -logu

      p = tfd.MultivariateNormalFullCovariance(
          covariance_matrix=tridiag(dims, diag_value=1, offdiag_value=0.5))

      # Variance is very high when approximating Forward KL, so we make
      # scale_diag larger than in test_kl_reverse_multidim. This ensures q
      # "covers" p and thus Var_q[p/q] is smaller.
      s = tf.constant(1.)
      q = tfd.MultivariateNormalDiag(
          scale_diag=tf.tile([s], [dims]))

      vimco = tfp.vi.csiszar_vimco(
          f=f,
          p_log_prob=p.log_prob,
          q=q,
          num_draws=num_draws,
          num_batch_draws=num_batch_draws,
          seed=seed)

      x = q.sample(sample_shape=[num_draws, num_batch_draws],
                   seed=seed)
      x = tf.stop_gradient(x)
      logu = p.log_prob(x) - q.log_prob(x)
      f_log_sum_u = f(tfp.vi.csiszar_vimco_helper(logu)[0])

      grad_sum = lambda fs: tf.gradients(fs, s)[0]

      def jacobian(x):
        # Warning: this function is slow and may not even finish if prod(shape)
        # is larger than, say, 100.
        shape = x.shape.as_list()
        assert all(s is not None for s in shape)
        x = tf.reshape(x, shape=[-1])
        r = [grad_sum(x[i]) for i in range(np.prod(shape))]
        return tf.reshape(tf.stack(r), shape=shape)

      [
          logu_,
          jacobian_logqx_,
          vimco_,
          grad_vimco_,
          f_log_sum_u_,
          grad_mean_f_log_sum_u_,
      ] = sess.run([
          logu,
          jacobian(q.log_prob(x)),
          vimco,
          grad_sum(vimco),
          f_log_sum_u,
          grad_sum(f_log_sum_u) / num_batch_draws,
      ])

      np_log_avg_u, np_log_sooavg_u = self._csiszar_vimco_helper(logu_)

      # Test VIMCO loss is correct.
      self.assertAllClose(np_f(np_log_avg_u).mean(axis=0), vimco_,
                          rtol=1e-5, atol=0.)

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
      self.assertAllClose(np_f(np_log_avg_u), f_log_sum_u_, rtol=1e-4, atol=0.)
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

      self.assertAllClose(np_grad_vimco, grad_vimco_,
                          rtol=1e-5, atol=0.)


if __name__ == "__main__":
  test.main()
