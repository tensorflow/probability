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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class NormalTest(test_util.TestCase):

  def testNormalConjugateKnownSigmaPosterior(self):
    with tf1.Session():
      mu0 = tf.constant([3.0])
      sigma0 = tf.constant([math.sqrt(10.0)])
      sigma = tf.constant([math.sqrt(2.0)])
      x = tf.constant([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0])
      s = tf.reduce_sum(x)
      n = tf.size(x)
      prior = tfd.Normal(loc=mu0, scale=sigma0)
      posterior = tfd.normal_conjugates_known_scale_posterior(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertIsInstance(posterior, tfd.Normal)
      posterior_log_pdf = self.evaluate(posterior.log_prob(x))
      self.assertEqual(posterior_log_pdf.shape, (6,))

  def testNormalConjugateKnownSigmaPosteriorND(self):
    with tf1.Session():
      batch_size = 6
      mu0 = tf.constant([[3.0, -3.0]] * batch_size)
      sigma0 = tf.constant([[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
      sigma = tf.constant([[math.sqrt(2.0)]] * batch_size)
      x = tf.transpose(
          a=tf.constant([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=tf.float32))
      s = tf.reduce_sum(x)
      n = tf.size(x)
      prior = tfd.Normal(loc=mu0, scale=sigma0)
      posterior = tfd.normal_conjugates_known_scale_posterior(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertIsInstance(posterior, tfd.Normal)
      posterior_log_pdf = self.evaluate(posterior.log_prob(x))
      self.assertEqual(posterior_log_pdf.shape, (6, 2))

  def testNormalConjugateKnownSigmaNDPosteriorND(self):
    with tf1.Session():
      batch_size = 6
      mu0 = tf.constant([[3.0, -3.0]] * batch_size)
      sigma0 = tf.constant([[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
      sigma = tf.constant([[math.sqrt(2.0), math.sqrt(4.0)]] * batch_size)
      x = tf.constant(
          [[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], [2.5, -2.5, -4.0, 0.0, 1.0, -2.0]],
          dtype=tf.float32)
      s = tf.reduce_sum(x, axis=[1])
      x = tf.transpose(a=x)  # Reshape to shape (6, 2)
      n = tf.constant([6] * 2)
      prior = tfd.Normal(loc=mu0, scale=sigma0)
      posterior = tfd.normal_conjugates_known_scale_posterior(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertIsInstance(posterior, tfd.Normal)

      # Calculate log_pdf under the 2 models
      posterior_log_pdf = posterior.log_prob(x)
      self.assertEqual(posterior_log_pdf.shape, (6, 2))
      self.assertEqual(self.evaluate(posterior_log_pdf).shape, (6, 2))

  def testNormalConjugateKnownSigmaPredictive(self):
    with tf1.Session():
      batch_size = 6
      mu0 = tf.constant([3.0] * batch_size)
      sigma0 = tf.constant([math.sqrt(10.0)] * batch_size)
      sigma = tf.constant([math.sqrt(2.0)] * batch_size)
      x = tf.constant([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0])
      s = tf.reduce_sum(x)
      n = tf.size(x)
      prior = tfd.Normal(loc=mu0, scale=sigma0)
      predictive = tfd.normal_conjugates_known_scale_predictive(
          prior=prior, scale=sigma, s=s, n=n)

      # Smoke test
      self.assertIsInstance(predictive, tfd.Normal)
      predictive_log_pdf = self.evaluate(predictive.log_prob(x))
      self.assertEqual(predictive_log_pdf.shape, (6,))

  def _mvn_linear_update_test_helper(self,
                                     prior_mean,
                                     prior_scale,
                                     linear_transformation,
                                     likelihood_scale,
                                     observation,
                                     candidate_posterior_mean,
                                     candidate_posterior_prec,
                                     atol=1e-5,
                                     rtol=1e-7):
    """Checks an MVN linear update against the naive dense computation."""

    # Do the test computation in float64, to ensure numerical stability.
    (prior_mean, prior_scale, linear_transformation,
     likelihood_scale, observation) = [tf.cast(t, tf.float64) for t in (
         prior_mean, prior_scale, linear_transformation,
         likelihood_scale, observation)]

    # Convert scale matrices to precision matrices.
    prior_cov = tf.matmul(prior_scale, prior_scale, adjoint_b=True)
    prior_prec = tf.linalg.inv(prior_cov)
    likelihood_cov = tf.matmul(likelihood_scale, likelihood_scale,
                               adjoint_b=True)

    # Run the regression.
    posterior_prec = prior_prec + tf.matmul(linear_transformation,
                                            tf.linalg.solve(
                                                likelihood_cov,
                                                linear_transformation),
                                            adjoint_a=True)
    posterior_mean = tf.linalg.solve(
        posterior_prec,
        (tf.linalg.matmul(
            linear_transformation,
            tf.linalg.solve(likelihood_cov,
                            observation[..., tf.newaxis]),
            adjoint_a=True) +
         tf.linalg.matvec(prior_prec, prior_mean)[..., tf.newaxis])
        )[..., 0]

    (candidate_posterior_mean_, candidate_posterior_prec_,
     posterior_mean_, posterior_prec_) = self.evaluate(
         (candidate_posterior_mean, candidate_posterior_prec,
          tf.cast(posterior_mean, tf.float32),
          tf.cast(posterior_prec, tf.float32)))

    self.assertAllClose(
        candidate_posterior_mean_, posterior_mean_, atol=atol, rtol=rtol)
    self.assertAllClose(
        candidate_posterior_prec_, posterior_prec_, atol=atol, rtol=rtol)

  @test_util.jax_disable_test_missing_functionality(
      "JAX uses Gaussian elimination which leads to numerical instability.")
  def testMVNConjugateLinearUpdateSupportsBatchShape(self):
    strm = test_util.test_seed_stream()
    num_latents = 2
    num_outputs = 4
    batch_shape = [3, 1]

    prior_mean = tf.ones([num_latents])
    prior_scale = tf.eye(num_latents) * 5.
    likelihood_scale = tf.linalg.LinearOperatorLowerTriangular(
        tfb.FillScaleTriL().forward(
            tf.random.normal(
                shape=batch_shape + [int(num_outputs * (num_outputs + 1) / 2)],
                seed=strm())))
    linear_transformation = tf.random.normal(
        batch_shape + [num_outputs, num_latents], seed=strm()) * 5.
    true_latent = tf.random.normal(batch_shape + [num_latents], seed=strm())
    observation = tf.linalg.matvec(linear_transformation, true_latent)
    posterior_mean, posterior_prec = (
        tfd.mvn_conjugate_linear_update(
            prior_mean=prior_mean,
            prior_scale=prior_scale,
            linear_transformation=linear_transformation,
            likelihood_scale=likelihood_scale,
            observation=observation))

    self._mvn_linear_update_test_helper(
        prior_mean=prior_mean,
        prior_scale=prior_scale,
        linear_transformation=linear_transformation,
        likelihood_scale=likelihood_scale.to_dense(),
        observation=observation,
        candidate_posterior_mean=posterior_mean,
        candidate_posterior_prec=posterior_prec.to_dense(),
        rtol=1e-5)

  def testMVNConjugateLinearUpdatePreservesStructuredLinops(self):
    strm = test_util.test_seed_stream()
    num_outputs = 4

    prior_scale = tf.linalg.LinearOperatorScaledIdentity(num_outputs, 4.)
    likelihood_scale = tf.linalg.LinearOperatorScaledIdentity(num_outputs, 0.2)
    linear_transformation = tf.linalg.LinearOperatorIdentity(num_outputs)
    observation = tf.random.normal([num_outputs], seed=strm())
    posterior_mean, posterior_prec = (
        tfd.mvn_conjugate_linear_update(
            prior_scale=prior_scale,
            linear_transformation=linear_transformation,
            likelihood_scale=likelihood_scale,
            observation=observation))
    # TODO(davmre): enable next line once internal CI is updated to recent TF.
    # self.assertIsInstance(posterior_prec,
    #                       tf.linalg.LinearOperatorScaledIdentity)

    self._mvn_linear_update_test_helper(
        prior_mean=tf.zeros([num_outputs]),
        prior_scale=prior_scale.to_dense(),
        linear_transformation=linear_transformation.to_dense(),
        likelihood_scale=likelihood_scale.to_dense(),
        observation=observation,
        candidate_posterior_mean=posterior_mean,
        candidate_posterior_prec=posterior_prec.to_dense())

    # Also check the result against the scalar calculation.
    scalar_posterior_dist = tfd.normal_conjugates_known_scale_posterior(
        prior=tfd.Normal(loc=0., scale=prior_scale.diag_part()),
        scale=likelihood_scale.diag_part(),
        s=observation, n=1)
    (posterior_mean_, posterior_prec_,
     scalar_posterior_mean_, scalar_posterior_prec_) = self.evaluate(
         (posterior_mean, posterior_prec.to_dense(),
          scalar_posterior_dist.mean(),
          tf.linalg.diag(1./scalar_posterior_dist.variance())))
    self.assertAllClose(posterior_mean_, scalar_posterior_mean_)
    self.assertAllClose(posterior_prec_, scalar_posterior_prec_)


if __name__ == "__main__":
  tf.test.main()
