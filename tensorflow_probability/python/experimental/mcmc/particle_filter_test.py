# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the _License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for particle filtering."""

import functools

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import linear_gaussian_ssm as lgssm
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental.mcmc import particle_filter
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


@test_util.test_all_tf_execution_regimes
class _ParticleFilterTest(test_util.TestCase):

  def test_batch_of_filters_particles_dim_1(self):

    batch_shape = [3, 2]
    num_particles = 1000
    num_timesteps = 40

    # Batch of priors on object 1D positions and velocities.
    initial_state_prior = jdn.JointDistributionNamed({
        'position': normal.Normal(loc=0., scale=tf.ones(batch_shape)),
        'velocity': normal.Normal(loc=0., scale=tf.ones(batch_shape) * 0.1)
    })

    def transition_fn(_, previous_state):
      return jdn.JointDistributionNamed({
          'position':
              normal.Normal(
                  loc=previous_state['position'] + previous_state['velocity'],
                  scale=0.1),
          'velocity':
              normal.Normal(loc=previous_state['velocity'], scale=0.01)
      })

    def observation_fn(_, state):
      return normal.Normal(loc=state['position'], scale=0.1)

    # Batch of synthetic observations
    true_initial_positions = np.random.randn()

    true_velocities = 0.1 * np.random.randn()
    observed_positions = (
            true_velocities * np.arange(num_timesteps).astype(self.dtype) + true_initial_positions)

    (particles, log_weights, parent_indices,
     incremental_log_marginal_likelihoods) = self.evaluate(
         particle_filter.particle_filter(
             observations=observed_positions,
             initial_state_prior=initial_state_prior,
             transition_fn=transition_fn,
             observation_fn=observation_fn,
             num_particles=num_particles,
             seed=test_util.test_seed(),
             particles_dim=1))

    self.assertAllEqual(particles['position'].shape,
                        [num_timesteps,
                         batch_shape[0],
                         num_particles,
                         batch_shape[1]])
    self.assertAllEqual(particles['velocity'].shape,
                        [num_timesteps,
                         batch_shape[0],
                         num_particles,
                         batch_shape[1]])
    self.assertAllEqual(parent_indices.shape,
                        [num_timesteps,
                         batch_shape[0],
                         num_particles,
                         batch_shape[1]])
    self.assertAllEqual(log_weights.shape,
                        [num_timesteps,
                         batch_shape[0],
                         num_particles,
                         batch_shape[1]])
    self.assertAllEqual(incremental_log_marginal_likelihoods.shape,
                        [num_timesteps] + batch_shape)

    # Uncertainty in velocity should decrease over time.
    velocity_stddev = self.evaluate(
        tf.math.reduce_std(particles['velocity'], axis=2))
    self.assertAllLess((velocity_stddev[-1] - velocity_stddev[0]), 0.)

    trajectories = self.evaluate(
        particle_filter.reconstruct_trajectories(particles,
                                                 parent_indices,
                                                 particles_dim=1))
    self.assertAllEqual([num_timesteps,
                         batch_shape[0],
                         num_particles,
                         batch_shape[1]],
                        trajectories['position'].shape)
    self.assertAllEqual([num_timesteps,
                         batch_shape[0],
                         num_particles,
                         batch_shape[1]],
                        trajectories['velocity'].shape)

    # Verify that `infer_trajectories` also works on batches.
    trajectories, incremental_log_marginal_likelihoods = self.evaluate(
        particle_filter.infer_trajectories(
            observations=observed_positions,
            initial_state_prior=initial_state_prior,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            num_particles=num_particles,
            particles_dim=1,
            seed=test_util.test_seed()))

    self.assertAllEqual([num_timesteps,
                         batch_shape[0],
                         num_particles,
                         batch_shape[1]],
                        trajectories['position'].shape)
    self.assertAllEqual([num_timesteps,
                         batch_shape[0],
                         num_particles,
                         batch_shape[1]],
                        trajectories['velocity'].shape)
    self.assertAllEqual(incremental_log_marginal_likelihoods.shape,
                        [num_timesteps] + batch_shape)


# TODO(b/186068104): add tests with dynamic shapes.
class ParticleFilterTestFloat32(_ParticleFilterTest):
  dtype = np.float32


del _ParticleFilterTest


if __name__ == '__main__':
  test_util.main()
