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
"""Tests for variational optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class OptimizationTests(test_util.TestCase):

  def test_variational_em(self):

    seed = test_util.test_seed()

    num_samples = 10000
    mu, sigma = 3., 5.
    np.random.seed(seed)
    x = np.random.randn(num_samples) * sigma + mu

    # Test that the tape automatically picks up any trainable variables in
    # the model, even though it's just a function with no explicit
    # `.trainable_variables`
    likelihood_scale = tfp.util.TransformedVariable(
        1., tfb.Softplus(), name='scale')
    def trainable_log_prob(z):
      lp = tfd.Normal(0., 1.).log_prob(z)
      lp += tf.reduce_sum(tfd.Normal(
          z[..., tf.newaxis], likelihood_scale).log_prob(x), axis=-1)
      return lp

    # For this simple normal-normal model, the true posterior is also normal.
    z_posterior_precision = (1./sigma**2 * num_samples + 1.**2)
    z_posterior_stddev = np.sqrt(1./z_posterior_precision)
    z_posterior_mean = (1./sigma**2 * num_samples * mu) / z_posterior_precision

    q_loc = tf.Variable(0., name='mu')
    q_scale = tfp.util.TransformedVariable(1., tfb.Softplus(), name='q_scale')
    q = tfd.Normal(q_loc, q_scale)
    loss_curve = tfp.vi.fit_surrogate_posterior(
        trainable_log_prob, q,
        num_steps=1000,
        sample_size=10,
        optimizer=tf.optimizers.Adam(0.1),
        seed=seed)
    self.evaluate(tf1.global_variables_initializer())
    with tf.control_dependencies([loss_curve]):
      final_q_loc = tf.identity(q.mean())
      final_q_scale = tf.identity(q.stddev())
      final_likelihood_scale = tf.identity(likelihood_scale)

    # We expect to recover the true posterior because the variational family
    # includes the true posterior, and the true parameters because we observed
    # a large number of sampled points.
    final_likelihood_scale_, final_q_loc_, final_q_scale_ = self.evaluate((
        final_likelihood_scale, final_q_loc, final_q_scale))
    self.assertAllClose(final_likelihood_scale_, sigma, atol=0.2)
    self.assertAllClose(final_q_loc_, z_posterior_mean, atol=0.2)
    self.assertAllClose(final_q_scale_, z_posterior_stddev, atol=0.1)

  def test_fit_posterior_with_joint_q(self):

    # Target distribution: equiv to MVNFullCovariance(cov=[[1., 1.], [1., 2.]])
    def p_log_prob(z, x):
      return tfd.Normal(0., 1.).log_prob(z) + tfd.Normal(z, 1.).log_prob(x)

    # The Q family is a joint distribution that can express any 2D MVN.
    b = tf.Variable([0., 0.])
    l = tfp.util.TransformedVariable(tf.eye(2), tfb.FillScaleTriL())
    def trainable_q_fn():
      z = yield tfd.JointDistributionCoroutine.Root(
          tfd.Normal(b[0], l[0, 0], name='z'))
      _ = yield tfd.Normal(b[1] + l[1, 0] * z, l[1, 1], name='x')
    q = tfd.JointDistributionCoroutine(trainable_q_fn)

    seed = test_util.test_seed()
    loss_curve = tfp.vi.fit_surrogate_posterior(
        p_log_prob, q, num_steps=1000, sample_size=100,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        seed=seed)
    self.evaluate(tf1.global_variables_initializer())
    loss_curve_ = self.evaluate((loss_curve))

    # Since the Q family includes the true distribution, the optimized
    # loss should be (approximately) zero.
    self.assertAllClose(loss_curve_[-1], 0., atol=0.1)

  def test_imhogeneous_poisson_process_example(self):
    # Toy 1D data.
    index_points = np.array([-10., -7.2, -4., -0.1, 0.1, 4., 6.2, 9.]).reshape(
        [-1, 1]).astype(np.float32)
    observed_counts = np.array(
        [100, 90, 60, 13, 18, 37, 55, 42]).astype(np.float32)

    # Trainable GP hyperparameters.
    kernel_log_amplitude = tf.Variable(0., name='kernel_log_amplitude')
    kernel_log_lengthscale = tf.Variable(0., name='kernel_log_lengthscale')
    observation_noise_log_scale = tf.Variable(
        0., name='observation_noise_log_scale')

    # Generative model.
    def model_fn():
      kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
          amplitude=tf.exp(kernel_log_amplitude),
          length_scale=tf.exp(kernel_log_lengthscale))
      latent_log_rates = yield tfd.JointDistributionCoroutine.Root(
          tfd.GaussianProcess(
              kernel,
              index_points=index_points,
              observation_noise_variance=tf.exp(observation_noise_log_scale),
              name='latent_log_rates'))
      yield tfd.Independent(
          tfd.Poisson(log_rate=latent_log_rates),
          reinterpreted_batch_ndims=1, name='y')
    model = tfd.JointDistributionCoroutine(model_fn, name='model')

    # Variational model.
    logit_locs = tf.Variable(tf.zeros(observed_counts.shape))
    logit_softplus_scales = tf.Variable(tf.ones(observed_counts.shape) * -1)
    def variational_model_fn():
      _ = yield tfd.JointDistributionCoroutine.Root(tfd.Independent(
          tfd.Normal(loc=logit_locs,
                     scale=tf.nn.softplus(logit_softplus_scales)),
          reinterpreted_batch_ndims=1))
      _ = yield tfd.VectorDeterministic(observed_counts)
    q = tfd.JointDistributionCoroutine(variational_model_fn,
                                       name='variational_model')

    losses, sample_path = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=lambda *args: model.log_prob(args),
        surrogate_posterior=q,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=100,
        seed=test_util.test_seed(),
        sample_size=1,
        trace_fn=lambda loss, grads, variables: (loss, q.sample(seed=42)[0]))

    self.evaluate(tf1.global_variables_initializer())
    losses_, sample_path_ = self.evaluate((losses, sample_path))
    self.assertLess(losses_[-1], 80.)  # Optimal loss is roughly 40.
    # Optimal latent logits are approximately the log observed counts.
    self.assertAllClose(sample_path_[-1], np.log(observed_counts), atol=1.0)

if __name__ == '__main__':
  tf.test.main()
