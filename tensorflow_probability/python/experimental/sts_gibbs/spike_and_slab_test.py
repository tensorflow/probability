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

from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.experimental.sts_gibbs import spike_and_slab

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


def _naive_symmetric_increment(m, idx, increment):
  m = m.copy()
  m[..., idx, :] += increment
  m[..., :, idx] += increment
  m[..., idx, idx] -= increment[..., idx]
  return m


class SpikeAndSlabTest(test_util.TestCase):

  def test_chol_symmetric_increment_batch(self):
    matrix = samplers.normal([7, 5, 5], seed=test_util.test_seed())
    psd_matrix = self.evaluate(
        tf.matmul(matrix, matrix, adjoint_b=True) + tf.eye(5))
    chol = tf.linalg.cholesky(psd_matrix)
    increment = self.evaluate(
        0.1 * samplers.normal([7, 5], seed=test_util.test_seed()))
    for idx in range(5):
      expected_result = _naive_symmetric_increment(psd_matrix, idx, increment)
      chol_result = spike_and_slab._symmetric_increment_chol(
          chol, idx, increment)
      result = tf.matmul(chol_result, chol_result, adjoint_b=True)
      self.assertAllClose(expected_result, result)

  @parameterized.named_parameters(
      ('_positive_diag', np.float32([0.5, 3.75, -2.5])),
      ('_zero_diag', np.float32([0., 3.75, -2.5])),
      ('_negative_diag', np.float32([-0.5, 3.75, -2.5])))
  def test_chol_symmetric_increment_cases(self, increment):
    symmetric_matrix = np.float32(
        [[1.5, -3.75, 2.5],
         [-3.75, 15., -8.75],
         [2.5, -8.75, 7.5]])
    chol = np.linalg.cholesky(symmetric_matrix)
    expected_result = _naive_symmetric_increment(symmetric_matrix, 0, increment)
    chol_result = spike_and_slab._symmetric_increment_chol(
        chol, 0, increment)
    result = tf.matmul(chol_result, chol_result, adjoint_b=True)
    self.assertAllClose(expected_result, result)

  def _random_regression_task(self, num_outputs, num_features, batch_shape=(),
                              weights=None, observation_noise_scale=0.1,
                              seed=None):
    design_seed, weights_seed, noise_seed = samplers.split_seed(seed, n=3)
    batch_shape = list(batch_shape)

    design_matrix = samplers.uniform(batch_shape + [num_outputs, num_features],
                                     seed=design_seed)
    if weights is None:
      weights = samplers.normal(batch_shape + [num_features], seed=weights_seed)
    targets = (tf.linalg.matvec(design_matrix, weights) +
               observation_noise_scale * samplers.normal(
                   batch_shape + [num_outputs], seed=noise_seed))
    return design_matrix, weights, targets

  def test_posterior_on_nonzero_subset_matches_bayesian_regression(self):
    # Generate a synthetic regression task.
    design_matrix, _, targets = self.evaluate(
        self._random_regression_task(
            num_features=5, num_outputs=20, batch_shape=[2],
            seed=test_util.test_seed()))

    # Utilities to extract values for nonzero-weight features.
    nonzeros = tf.convert_to_tensor([True, False, True, False, True])
    nonzero_subvector = (
        lambda x: tf.boolean_mask(x, nonzeros, axis=ps.rank(x) - 1))
    nonzero_submatrix = lambda x: tf.boolean_mask(  # pylint: disable=g-long-lambda
        tf.boolean_mask(x, nonzeros, axis=ps.rank(x) - 2),
        nonzeros,
        axis=ps.rank(x) - 1)

    # Compute the weight posterior mean and precision for these nonzeros.
    sampler = spike_and_slab.SpikeSlabSampler(design_matrix)
    initial_state = sampler._initialize_sampler_state(
        targets=targets, nonzeros=nonzeros)

    # Compute the analytic posterior for the regression problem restricted to
    # only the selected features. Note that by slicing a submatrix of the
    # prior precision we are implicitly *conditioning* on having observed the
    # other weights to be zero (which is sensible in this case), versus slicing
    # into the covariance which would give the marginal (unconditional) prior
    # on the selected weights.
    (restricted_weights_posterior_mean,
     restricted_weights_posterior_prec) = tfd.mvn_conjugate_linear_update(
         prior_scale=tf.linalg.cholesky(
             tf.linalg.inv(nonzero_submatrix(sampler.weights_prior_precision))),
         linear_transformation=nonzero_subvector(design_matrix),
         likelihood_scale=tf.eye(20),
         observation=targets)

    # The sampler's posterior should match the posterior from the restricted
    # problem.
    self.assertAllClose(
        nonzero_subvector(initial_state.conditional_weights_mean),
        restricted_weights_posterior_mean)
    self.assertAllClose(
        nonzero_submatrix(initial_state.conditional_posterior_precision_chol),
        tf.linalg.cholesky(restricted_weights_posterior_prec.to_dense()))

  def test_noise_variance_posterior_matches_expected(self):
    # Generate a synthetic regression task.
    num_features = 5
    num_outputs = 20
    design_matrix, _, targets = self.evaluate(
        self._random_regression_task(
            num_features=num_features, num_outputs=num_outputs, batch_shape=[2],
            seed=test_util.test_seed()))

    observation_noise_variance_prior_concentration = 0.03
    observation_noise_variance_prior_scale = 0.015
    # Posterior on noise variance if all weights are zero.
    naive_posterior = tfd.InverseGamma(
        concentration=(observation_noise_variance_prior_concentration +
                       num_outputs / 2.),
        scale=(observation_noise_variance_prior_scale + tf.reduce_sum(
            tf.square(targets), axis=-1) / 2.))

    # Compare to sampler with weights constrained to near-zero.
    # We can do this by reducing the width of the slab (here),
    # or by reducing the probability of the slab (below). Both should give
    # equivalent noise posteriors.
    tight_slab_sampler = spike_and_slab.SpikeSlabSampler(
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
            targets=targets,
            nonzeros=tf.ones([num_features], dtype=tf.bool)
            ).observation_noise_variance_posterior_scale,
        naive_posterior.scale,
        atol=1e-2)

    downweighted_slab_sampler = spike_and_slab.SpikeSlabSampler(
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
            targets=targets,
            nonzeros=tf.zeros([num_features], dtype=tf.bool)
            ).observation_noise_variance_posterior_scale,
        naive_posterior.scale)

  def test_updated_state_matches_initial_computation(self):
    design_matrix, _, targets = self._random_regression_task(
        num_outputs=2, num_features=3, batch_shape=[],
        seed=test_util.test_seed())

    sampler = spike_and_slab.SpikeSlabSampler(design_matrix=design_matrix,
                                              nonzero_prior_prob=0.3)

    all_nonzero_sampler_state = sampler._initialize_sampler_state(
        targets=targets, nonzeros=tf.convert_to_tensor([True, True, True]))

    # Flipping a weight from nonzero to zero (slab to spike) should result in
    # the same state as if we'd initialized with that sparsity pattern.
    flipped_state_from_update = sampler._flip_feature(
        all_nonzero_sampler_state, idx=0)
    flipped_state_from_scratch = sampler._initialize_sampler_state(
        targets=targets, nonzeros=tf.convert_to_tensor([False, True, True]))
    self.assertAllCloseNested(flipped_state_from_update,
                              flipped_state_from_scratch)

    # Reverse direction (spike to slab).
    double_flipped_state_from_update = sampler._flip_feature(
        flipped_state_from_update, idx=0)
    self.assertAllCloseNested(double_flipped_state_from_update,
                              all_nonzero_sampler_state, atol=1e-4)

  def test_sanity_check_sweep_over_features(self):
    num_outputs = 100
    num_features = 3
    batch_shape = [2]
    design_matrix, true_weights, targets = self.evaluate(
        self._random_regression_task(
            num_outputs=num_outputs,
            num_features=num_features,
            batch_shape=batch_shape,
            # Specify weights with a clear sparsity pattern.
            weights=tf.convert_to_tensor([[10., 0., -10.],
                                          [0., 0., 0.5]]),
            seed=test_util.test_seed()))

    sampler = spike_and_slab.SpikeSlabSampler(design_matrix)
    initial_state = sampler._initialize_sampler_state(
        targets=targets, nonzeros=tf.convert_to_tensor([True, True, True]))
    final_state = self.evaluate(
        sampler._resample_all_features(
            initial_state, seed=test_util.test_seed()))

    # Check that we recovered the true sparsity pattern and approximate weights.
    self.assertAllEqual(final_state.nonzeros, [[True, False, True],
                                               [False, False, True]])
    self.assertAllClose(final_state.conditional_weights_mean,
                        true_weights, rtol=0.05, atol=0.15)
    # Check shapes of other components.
    self.assertAllEqual(final_state.conditional_prior_precision_chol.shape,
                        batch_shape + [num_features, num_features])
    self.assertAllEqual(final_state.conditional_posterior_precision_chol.shape,
                        batch_shape + [num_features, num_features])
    self.assertAllEqual(
        final_state.observation_noise_variance_posterior_scale.shape,
        batch_shape)

    posterior = sampler._get_conditional_posterior(final_state)
    posterior_variances, posterior_weights = self.evaluate(
        posterior.sample(10, seed=test_util.test_seed()))
    self.assertAllFinite(posterior_variances)
    self.assertAllFinite(posterior_weights)

  def test_samples_from_weights_prior(self):
    nonzero_prior_prob = 0.7
    num_outputs, num_features = 200, 4

    # Setting the design matrix to zero, the targets provide no information
    # about weights, so the sampler should sample from the prior.
    design_matrix = tf.zeros([num_outputs, num_features])
    targets = 0.42 * samplers.normal([num_outputs], seed=test_util.test_seed())
    sampler = spike_and_slab.SpikeSlabSampler(
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
            num_outputs=3, num_features=4, batch_shape=[],
            seed=test_util.test_seed()))

    sampler = spike_and_slab.SpikeSlabSampler(design_matrix)

    initial_nonzeros = tf.convert_to_tensor([True, False, False, True])
    seed = test_util.test_seed(sampler_type='stateless')
    variance1, weights1 = self.evaluate(
        sampler.sample_noise_variance_and_weights(
            targets, initial_nonzeros, seed=seed))
    variance2, weights2 = self.evaluate(
        sampler.sample_noise_variance_and_weights(
            targets, initial_nonzeros, seed=seed))
    self.assertAllClose(variance1, variance2)
    self.assertAllClose(weights1, weights2)


if __name__ == '__main__':
  test_util.main()

