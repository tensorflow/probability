# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for spike and slab sampler."""

from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import normal_conjugate_posteriors as ncp
from tensorflow_probability.python.experimental.sts_gibbs import dynamic_spike_and_slab
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


def _compute_conditional_weights_mean(nonzeros, weights_posterior_precision,
                                      x_transpose_y):
  indices = tf.where(nonzeros)[:, 0]
  conditional_posterior_precision_chol = tf.linalg.cholesky(
      tf.gather(
          tf.gather(weights_posterior_precision, indices),
          indices, axis=1))
  return tf.linalg.cholesky_solve(
      conditional_posterior_precision_chol,
      tf.gather(x_transpose_y, indices)[..., tf.newaxis])[..., 0]


class SpikeAndSlabTest(test_util.TestCase):

  def _random_regression_task(self, num_outputs, num_features,
                              weights=None, observation_noise_scale=0.1,
                              seed=None):
    design_seed, weights_seed, noise_seed = samplers.split_seed(seed, n=3)

    design_matrix = samplers.uniform([num_outputs, num_features],
                                     seed=design_seed)
    if weights is None:
      weights = samplers.normal([num_features], seed=weights_seed)
    targets = (tf.linalg.matvec(design_matrix, weights) +
               observation_noise_scale * samplers.normal(
                   [num_outputs], seed=noise_seed))
    return design_matrix, weights, targets

  def test_sampler_respects_pseudo_observations(self):
    design_matrix = self.evaluate(
        samplers.uniform([20, 5], seed=test_util.test_seed()))
    first_obs = 2.
    second_obs = 10.
    first_sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler(
        design_matrix,
        default_pseudo_observations=first_obs)
    second_sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler(
        design_matrix,
        default_pseudo_observations=second_obs)

    self.assertNotAllClose(
        first_sampler.weights_prior_precision,
        second_sampler.weights_prior_precision)
    self.assertAllClose(
        first_sampler.weights_prior_precision / first_obs,
        second_sampler.weights_prior_precision / second_obs)

  @parameterized.named_parameters(
      ('default_precision', 1.),
      ('ten_pseudo_obs', 10.))
  def test_posterior_on_nonzero_subset_matches_bayesian_regression(
      self, default_pseudo_observations):
    # Generate a synthetic regression task.
    design_matrix, _, targets = self.evaluate(
        self._random_regression_task(
            num_features=5, num_outputs=20,
            seed=test_util.test_seed()))

    # Utilities to extract values for nonzero-weight features.
    nonzeros = np.array([True, False, True, False, True])
    nonzero_subvector = lambda x: x[..., nonzeros]
    nonzero_submatrix = (
        lambda x: self.evaluate(x)[..., nonzeros][..., nonzeros, :])

    # Compute the weight posterior mean and precision for these nonzeros.
    sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler(
        design_matrix,
        default_pseudo_observations=default_pseudo_observations)
    initial_state = sampler._initialize_sampler_state(
        targets=targets, nonzeros=nonzeros)

    # Compute the analytic posterior for the regression problem restricted to
    # only the selected features. Note that by slicing a submatrix of the
    # prior precision we are implicitly *conditioning* on having observed the
    # other weights to be zero (which is sensible in this case), versus slicing
    # into the covariance which would give the marginal (unconditional) prior
    # on the selected weights.
    (restricted_weights_posterior_mean, _) = ncp.mvn_conjugate_linear_update(
        prior_scale=tf.linalg.cholesky(
            tf.linalg.inv(nonzero_submatrix(sampler.weights_prior_precision))),
        linear_transformation=nonzero_subvector(design_matrix),
        likelihood_scale=tf.eye(20),
        observation=targets)

    # The sampler's posterior should match the posterior from the restricted
    # problem.
    weights_posterior_precision = (sampler.x_transpose_x +
                                   sampler.weights_prior_precision)
    conditional_weights_mean = _compute_conditional_weights_mean(
        initial_state.nonzeros,
        weights_posterior_precision,
        initial_state.x_transpose_y)
    self.assertAllClose(
        self.evaluate(
            conditional_weights_mean),
        restricted_weights_posterior_mean)

  def test_noise_variance_posterior_matches_expected(self):
    # Generate a synthetic regression task.
    num_features = 5
    num_outputs = 20
    design_matrix, _, targets = self.evaluate(
        self._random_regression_task(
            num_features=num_features, num_outputs=num_outputs,
            seed=test_util.test_seed()))

    observation_noise_variance_prior_concentration = 0.03
    observation_noise_variance_prior_scale = 0.015
    # Posterior on noise variance if all weights are zero.
    naive_posterior = inverse_gamma.InverseGamma(
        concentration=(observation_noise_variance_prior_concentration +
                       num_outputs / 2.),
        scale=(observation_noise_variance_prior_scale +
               tf.reduce_sum(tf.square(targets), axis=-1) / 2.))

    # Compare to sampler with weights constrained to near-zero.
    # We can do this by reducing the width of the slab (here),
    # or by reducing the probability of the slab (below). Both should give
    # equivalent noise posteriors.
    tight_slab_sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler(
        design_matrix,
        weights_prior_precision=tf.eye(num_features) * 1e6,
        observation_noise_variance_prior_concentration=(
            observation_noise_variance_prior_concentration),
        observation_noise_variance_prior_scale=(
            observation_noise_variance_prior_scale))
    self.assertAllClose(
        tight_slab_sampler.observation_noise_variance_posterior_concentration,
        naive_posterior.concentration)
    self.assertAllClose(
        tight_slab_sampler._initialize_sampler_state(
            targets=targets, nonzeros=tf.ones([num_features], dtype=tf.bool)
        ).observation_noise_variance_posterior_scale,
        naive_posterior.scale,
        atol=1e-2)

    downweighted_slab_sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler(
        design_matrix,
        observation_noise_variance_prior_concentration=(
            observation_noise_variance_prior_concentration),
        observation_noise_variance_prior_scale=(
            observation_noise_variance_prior_scale))
    self.assertAllClose(
        (downweighted_slab_sampler.
         observation_noise_variance_posterior_concentration),
        naive_posterior.concentration)
    self.assertAllClose(
        downweighted_slab_sampler._initialize_sampler_state(
            targets=targets, nonzeros=tf.zeros([num_features], dtype=tf.bool)
        ).observation_noise_variance_posterior_scale,
        naive_posterior.scale)

  @parameterized.parameters(
      (2, 3, 1),
      (2, 3, 1),
      (100, 20, 10),
      (100, 20, 10),
      (40, 20, 12))
  def test_updated_state_matches_initial_computation(
      self, num_outputs, num_features, num_flips):

    rng = test_util.test_np_rng()
    initial_nonzeros = rng.randint(
        low=0, high=2, size=[num_features]).astype(bool)
    flip_idxs = rng.choice(
        num_features, size=num_flips, replace=False).astype(np.int32)
    should_flip = np.array([True] * num_flips)

    nonzeros = initial_nonzeros.copy()
    for i in range(num_flips):
      nonzeros[..., flip_idxs[i]] = (
          nonzeros[..., flip_idxs[i]] != should_flip[i])

    design_matrix, _, targets = self._random_regression_task(
        num_outputs=num_outputs, num_features=num_features,
        seed=test_util.test_seed())
    sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler(
        design_matrix=design_matrix, nonzero_prior_prob=0.3)

    @tf.function(autograph=False, jit_compile=False)
    def _do_flips():
      state = sampler._initialize_sampler_state(
          targets=targets,
          nonzeros=initial_nonzeros)
      def _do_flip(state, i):
        new_state = sampler._flip_feature(state, tf.gather(flip_idxs, i))
        return mcmc_util.choose(tf.gather(should_flip, i), new_state, state)
      return tf.foldl(_do_flip, elems=tf.range(num_flips), initializer=state)

    self.assertAllCloseNested(
        sampler._initialize_sampler_state(targets, nonzeros),
        _do_flips(),
        atol=1e-6, rtol=1e-6)

  def test_sanity_check_sweep_over_features(self):
    num_outputs = 100
    num_features = 3
    design_matrix, true_weights, targets = self.evaluate(
        self._random_regression_task(
            num_outputs=num_outputs,
            num_features=num_features,
            # Specify weights with a clear sparsity pattern.
            weights=tf.convert_to_tensor([10., 0., -10.]),
            seed=test_util.test_seed()))

    sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler(
        design_matrix,
        # Ensure the probability of keeping an irrelevant feature is tiny.
        nonzero_prior_prob=1e-6)
    initial_state = sampler._initialize_sampler_state(
        targets=targets, nonzeros=tf.convert_to_tensor([True, True, True]))
    final_state = self.evaluate(
        sampler._resample_all_features(
            initial_state, seed=test_util.test_seed()))

    # Check that we recovered the true sparsity pattern and approximate weights.
    weights_posterior_precision = (sampler.x_transpose_x +
                                   sampler.weights_prior_precision)
    conditional_weights_mean = _compute_conditional_weights_mean(
        final_state.nonzeros,
        weights_posterior_precision,
        final_state.x_transpose_y)
    self.assertAllEqual(final_state.nonzeros, [True, False, True])
    indices = tf.where(final_state.nonzeros)
    conditional_weights_mean = tf.scatter_nd(
        indices, conditional_weights_mean, true_weights.shape)
    self.assertAllClose(conditional_weights_mean,
                        true_weights, rtol=0.05, atol=0.15)

    posterior = sampler._get_conditional_posterior(final_state)
    posterior_variances, posterior_weights = self.evaluate(
        posterior.sample(seed=test_util.test_seed()))
    self.assertAllFinite(posterior_variances)
    self.assertAllFinite(posterior_weights)

  def test_samples_from_weights_prior(self):
    nonzero_prior_prob = 0.7
    num_outputs, num_features = 200, 4

    # Setting the design matrix to zero, the targets provide no information
    # about weights, so the sampler should sample from the prior.
    design_matrix = tf.zeros([num_outputs, num_features])
    targets = 0.42 * samplers.normal([num_outputs], seed=test_util.test_seed())
    sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler(
        design_matrix=design_matrix,
        weights_prior_precision=tf.eye(num_features),
        nonzero_prior_prob=nonzero_prior_prob)

    # Draw 100 posterior samples. Since all state needed for the
    # internal feature sweep is a function of the sparsity pattern, it's
    # sufficient to pass the sparsity pattern (by way of the weights) as
    # the outer-loop state.
    @tf.function(autograph=False)
    def loop_body(var_weights_seed, _):
      _, weights, seed = var_weights_seed
      seed, next_seed = samplers.split_seed(seed, n=2)
      variance, weights = sampler.sample_noise_variance_and_weights(
          initial_nonzeros=tf.not_equal(weights, 0.),
          targets=targets,
          seed=seed)
      return variance, weights, next_seed

    init_seed = test_util.test_seed(sampler_type='stateless')
    variance_samples, weight_samples, _ = tf.scan(
        fn=loop_body,
        initializer=(1., tf.ones([num_features]), init_seed),
        elems=tf.range(100))

    # With the default (relatively uninformative) prior, the noise variance
    # posterior mean should be close to the most-likely value.
    self.assertAllClose(tf.reduce_mean(variance_samples),
                        tf.math.reduce_std(targets)**2,
                        atol=0.03)
    # Since there is no evidence for the weights, the sparsity of our samples
    # should match the prior.
    nonzero_weight_samples = tf.cast(tf.not_equal(weight_samples, 0.),
                                     tf.float32)
    self.assertAllClose(nonzero_prior_prob,
                        tf.reduce_mean(nonzero_weight_samples),
                        atol=0.03)

  def test_deterministic_given_seed(self):
    design_matrix, _, targets = self.evaluate(
        self._random_regression_task(
            num_outputs=3, num_features=4,
            seed=test_util.test_seed()))

    sampler = dynamic_spike_and_slab.DynamicSpikeSlabSampler(design_matrix)

    initial_nonzeros = tf.convert_to_tensor([True, False, False, True])
    seed = test_util.test_seed(sampler_type='stateless')

    @tf.function(autograph=False, jit_compile=False)
    def do_sample(seed):
      return sampler.sample_noise_variance_and_weights(
          targets, initial_nonzeros, seed=seed)
    variance1, weights1 = self.evaluate(do_sample(seed))
    variance2, weights2 = self.evaluate(do_sample(seed))
    self.assertAllFinite(variance1)
    self.assertAllClose(variance1, variance2)
    self.assertAllFinite(weights1)
    self.assertAllClose(weights1, weights2)


if __name__ == '__main__':
  test_util.main()
