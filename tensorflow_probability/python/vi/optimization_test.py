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

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import fill_scale_tril
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import gaussian_process
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.experimental.util import trainable
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.util import deferred_tensor
from tensorflow_probability.python.vi import optimization


JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class OptimizationTests(test_util.TestCase):

  @test_util.jax_disable_variable_test
  def test_variational_em(self):

    seed = test_util.test_seed()

    num_samples = 10000
    mu, sigma = 3., 5.
    x = test_util.test_np_rng().randn(num_samples) * sigma + mu

    # Test that the tape automatically picks up any trainable variables in
    # the model, even though it's just a function with no explicit
    # `.trainable_variables`
    likelihood_scale = deferred_tensor.TransformedVariable(
        1., softplus.Softplus(), name='scale')
    def trainable_log_prob(z):
      lp = normal.Normal(0., 1.).log_prob(z)
      lp += tf.reduce_sum(
          normal.Normal(z[..., tf.newaxis], likelihood_scale).log_prob(x),
          axis=-1)
      return lp

    # For this simple normal-normal model, the true posterior is also normal.
    z_posterior_precision = (1./sigma**2 * num_samples + 1.**2)
    z_posterior_stddev = np.sqrt(1./z_posterior_precision)
    z_posterior_mean = (1./sigma**2 * num_samples * mu) / z_posterior_precision

    q_loc = tf.Variable(0., name='mu')
    q_scale = deferred_tensor.TransformedVariable(
        1., softplus.Softplus(), name='q_scale')
    q = normal.Normal(q_loc, q_scale)
    loss_curve = optimization.fit_surrogate_posterior(
        trainable_log_prob,
        q,
        num_steps=1000,
        sample_size=10,
        optimizer=tf.optimizers.Adam(0.1),
        seed=seed)
    self.evaluate(tf1.global_variables_initializer())
    with tf.control_dependencies([loss_curve]):
      final_q_loc = tf.identity(q.mean())
      final_q_scale = tf.identity(q.stddev())
      final_likelihood_scale = tf.convert_to_tensor(likelihood_scale)

    # We expect to recover the true posterior because the variational family
    # includes the true posterior, and the true parameters because we observed
    # a large number of sampled points.
    final_likelihood_scale_, final_q_loc_, final_q_scale_ = self.evaluate((
        final_likelihood_scale, final_q_loc, final_q_scale))
    self.assertAllClose(final_likelihood_scale_, sigma, atol=0.2)
    self.assertAllClose(final_q_loc_, z_posterior_mean, atol=0.2)
    self.assertAllClose(final_q_scale_, z_posterior_stddev, atol=0.1)

  @test_util.jax_disable_variable_test
  def test_importance_sampling_example(self):
    init_seed, opt_seed, eval_seed = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'), n=3)

    def log_prob(z, x):
      return normal.Normal(0., 1.).log_prob(z) + normal.Normal(z,
                                                               1.).log_prob(x)
    conditioned_log_prob = lambda z: log_prob(z, x=5.)

    q_z = trainable.make_trainable(normal.Normal, seed=init_seed)
    # Fit `q` with an importance-weighted variational loss.
    loss_curve = optimization.fit_surrogate_posterior(
        conditioned_log_prob,
        surrogate_posterior=q_z,
        importance_sample_size=10,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=100,
        seed=opt_seed)
    self.evaluate(tf1.global_variables_initializer())
    loss_curve = self.evaluate(loss_curve)

    # Estimate posterior statistics with importance sampling.
    zs, q_log_prob = self.evaluate(q_z.experimental_sample_and_log_prob(
        1000, seed=eval_seed))
    self_normalized_log_weights = tf.nn.log_softmax(
        conditioned_log_prob(zs) - q_log_prob)
    posterior_mean = tf.reduce_sum(
        tf.exp(self_normalized_log_weights) * zs,
        axis=0)
    self.assertAllClose(posterior_mean, 2.5, atol=1e-1)

    posterior_variance = tf.reduce_sum(
        tf.exp(self_normalized_log_weights) * (zs - posterior_mean)**2,
        axis=0)
    self.assertAllClose(posterior_variance, 0.5, atol=1e-1)

    # Test reproducibility
    q_z_again = trainable.make_trainable(normal.Normal, seed=init_seed)
    # Fit `q` with an importance-weighted variational loss.
    loss_curve_again = optimization.fit_surrogate_posterior(
        conditioned_log_prob,
        surrogate_posterior=q_z_again,
        importance_sample_size=10,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=100,
        seed=opt_seed)
    self.evaluate(tf1.global_variables_initializer())
    loss_curve_again = self.evaluate(loss_curve_again)
    self.assertAllClose(loss_curve_again, loss_curve)

  @test_util.jax_disable_variable_test
  def test_fit_posterior_with_joint_q(self):

    # Target distribution: equiv to MVNFullCovariance(cov=[[1., 1.], [1., 2.]])
    def p_log_prob(z, x):
      return normal.Normal(0., 1.).log_prob(z) + normal.Normal(z,
                                                               1.).log_prob(x)

    # The Q family is a joint distribution that can express any 2D MVN.
    b = tf.Variable([0., 0.])
    l = deferred_tensor.TransformedVariable(
        tf.eye(2), fill_scale_tril.FillScaleTriL())
    def trainable_q_fn():
      z = yield jdc.JointDistributionCoroutine.Root(
          normal.Normal(b[0], l[0, 0], name='z'))
      _ = yield normal.Normal(b[1] + l[1, 0] * z, l[1, 1], name='x')

    q = jdc.JointDistributionCoroutine(trainable_q_fn)

    seed = test_util.test_seed()
    loss_curve = optimization.fit_surrogate_posterior(
        p_log_prob,
        q,
        num_steps=1000,
        sample_size=100,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        seed=seed)
    self.evaluate(tf1.global_variables_initializer())
    loss_curve_ = self.evaluate((loss_curve))

    # Since the Q family includes the true distribution, the optimized
    # loss should be (approximately) zero.
    self.assertAllClose(loss_curve_[-1], 0., atol=0.1)

  @test_util.jax_disable_variable_test
  def test_inhomogeneous_poisson_process_example(self):
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
      kernel = exponentiated_quadratic.ExponentiatedQuadratic(
          amplitude=tf.exp(kernel_log_amplitude),
          length_scale=tf.exp(kernel_log_lengthscale))
      latent_log_rates = yield jdc.JointDistributionCoroutine.Root(
          gaussian_process.GaussianProcess(
              kernel,
              index_points=index_points,
              observation_noise_variance=tf.exp(observation_noise_log_scale),
              name='latent_log_rates'))
      yield independent.Independent(
          poisson.Poisson(log_rate=latent_log_rates),
          reinterpreted_batch_ndims=1,
          name='y')

    model = jdc.JointDistributionCoroutine(model_fn, name='model')

    # Variational model.
    logit_locs = tf.Variable(tf.zeros(observed_counts.shape))
    logit_softplus_scales = tf.Variable(tf.ones(observed_counts.shape) * -1)
    def variational_model_fn():
      _ = yield jdc.JointDistributionCoroutine.Root(
          independent.Independent(
              normal.Normal(
                  loc=logit_locs, scale=tf.nn.softplus(logit_softplus_scales)),
              reinterpreted_batch_ndims=1))
      _ = yield deterministic.VectorDeterministic(observed_counts)

    q = jdc.JointDistributionCoroutine(
        variational_model_fn, name='variational_model')

    losses, sample_path = optimization.fit_surrogate_posterior(
        target_log_prob_fn=lambda *args: model.log_prob(args),
        surrogate_posterior=q,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=100,
        seed=test_util.test_seed(),
        sample_size=1,
        trace_fn=lambda t: (t.loss, q.sample(seed=42)[0]))

    self.evaluate(tf1.global_variables_initializer())
    losses_, sample_path_ = self.evaluate((losses, sample_path))
    self.assertLess(losses_[-1], 80.)  # Optimal loss is roughly 40.
    # Optimal latent logits are approximately the log observed counts.
    self.assertAllClose(sample_path_[-1], np.log(observed_counts), atol=1.0)


@test_util.test_all_tf_execution_regimes
class StatelessOptimizationTests(test_util.TestCase):

  def test_importance_sampling_example(self):
    if not JAX_MODE:
      self.skipTest('Requires optax.')
    import optax  # pylint: disable=g-import-not-at-top

    init_seed, opt_seed, eval_seed = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'), n=3)

    def log_prob(z, x):
      return normal.Normal(0., 1.).log_prob(z) + normal.Normal(z,
                                                               1.).log_prob(x)
    conditioned_log_prob = lambda z: log_prob(z, x=5.)

    init_normal, build_normal = trainable.make_trainable_stateless(
        normal.Normal)
    # Fit `q` with an importance-weighted variational loss.
    optimized_parameters, _ = optimization.fit_surrogate_posterior_stateless(
        conditioned_log_prob,
        build_surrogate_posterior_fn=build_normal,
        initial_parameters=init_normal(seed=init_seed),
        importance_sample_size=10,
        optimizer=optax.adam(0.1),
        num_steps=200,
        seed=opt_seed)
    q_z = build_normal(*optimized_parameters)

    # Estimate posterior statistics with importance sampling.
    zs, q_log_prob = self.evaluate(q_z.experimental_sample_and_log_prob(
        1000, seed=eval_seed))
    self_normalized_log_weights = tf.nn.log_softmax(
        conditioned_log_prob(zs) - q_log_prob)
    posterior_mean = tf.reduce_sum(
        tf.exp(self_normalized_log_weights) * zs,
        axis=0)
    self.assertAllClose(posterior_mean, 2.5, atol=1e-1)

    posterior_variance = tf.reduce_sum(
        tf.exp(self_normalized_log_weights) * (zs - posterior_mean)**2,
        axis=0)
    self.assertAllClose(posterior_variance, 0.5, atol=1e-1)

  def test_inhomogeneous_poisson_process_example(self):
    opt_seed, eval_seed = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'), n=2)

    # Toy 1D data.
    index_points = np.array([-10., -7.2, -4., -0.1, 0.1, 4., 6.2, 9.]).reshape(
        [-1, 1]).astype(np.float32)
    observed_counts = np.array(
        [100, 90, 60, 13, 18, 37, 55, 42]).astype(np.float32)

    # Generative model.
    def model_fn():
      kernel_amplitude = yield lognormal.LogNormal(
          loc=0., scale=1., name='kernel_amplitude')
      kernel_lengthscale = yield lognormal.LogNormal(
          loc=0., scale=1., name='kernel_lengthscale')
      observation_noise_scale = yield lognormal.LogNormal(
          loc=0., scale=1., name='observation_noise_scale')
      kernel = exponentiated_quadratic.ExponentiatedQuadratic(
          amplitude=kernel_amplitude, length_scale=kernel_lengthscale)
      latent_log_rates = yield gaussian_process.GaussianProcess(
          kernel,
          index_points=index_points,
          observation_noise_variance=observation_noise_scale,
          name='latent_log_rates')
      yield independent.Independent(
          poisson.Poisson(log_rate=latent_log_rates),
          reinterpreted_batch_ndims=1,
          name='y')

    model = jdab.JointDistributionCoroutineAutoBatched(model_fn)
    pinned = model.experimental_pin(y=observed_counts)

    initial_parameters = (0., 0., 0.,  # Raw kernel parameters.
                          tf.zeros_like(observed_counts),  # `logit_locs`
                          tf.zeros_like(observed_counts))  # `logit_raw_scales`

    def build_surrogate_posterior_fn(
        raw_kernel_amplitude, raw_kernel_lengthscale,
        raw_observation_noise_scale,
        logit_locs, logit_raw_scales):

      def variational_model_fn():
        # Fit the kernel parameters as point masses.
        yield deterministic.Deterministic(
            tf.nn.softplus(raw_kernel_amplitude), name='kernel_amplitude')
        yield deterministic.Deterministic(
            tf.nn.softplus(raw_kernel_lengthscale), name='kernel_lengthscale')
        yield deterministic.Deterministic(
            tf.nn.softplus(raw_observation_noise_scale),
            name='kernel_observation_noise_scale')
        # Factored normal posterior over the GP logits.
        yield independent.Independent(
            normal.Normal(
                loc=logit_locs, scale=tf.nn.softplus(logit_raw_scales)),
            reinterpreted_batch_ndims=1,
            name='latent_log_rates')

      return jdab.JointDistributionCoroutineAutoBatched(variational_model_fn)

    if not JAX_MODE:
      return
    import optax  # pylint: disable=g-import-not-at-top

    def seeded_target_log_prob_fn(*xs, seed=None):
      # Add a tiny amount of noise to the target log-prob to see if it works.
      ret = pinned.unnormalized_log_prob(xs)
      return ret + samplers.normal(ret.shape, stddev=0.01, seed=seed)

    [optimized_parameters,
     (losses, _, sample_path)] = optimization.fit_surrogate_posterior_stateless(
         target_log_prob_fn=seeded_target_log_prob_fn,
         build_surrogate_posterior_fn=build_surrogate_posterior_fn,
         initial_parameters=initial_parameters,
         optimizer=optax.adam(learning_rate=0.1),
         sample_size=1,
         num_steps=500,
         trace_fn=lambda traceable_quantities: (  # pylint: disable=g-long-lambda
             traceable_quantities.loss,
             tf.nn.softplus(traceable_quantities.parameters[0]),
             build_surrogate_posterior_fn(*traceable_quantities.parameters).
             sample(seed=eval_seed)[-1]),
         seed=opt_seed)
    surrogate_posterior = build_surrogate_posterior_fn(*optimized_parameters)
    surrogate_posterior.sample(seed=eval_seed)

    losses_, sample_path_ = self.evaluate((losses, sample_path))
    self.assertLess(losses_[-1], 80.)  # Optimal loss is roughly 40.
    # Optimal latent logits are approximately the log observed counts.
    self.assertAllClose(sample_path_[-1], np.log(observed_counts), atol=1.0)


if __name__ == '__main__':
  test_util.main()
