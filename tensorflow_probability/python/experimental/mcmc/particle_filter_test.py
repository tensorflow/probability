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

  def test_random_walk(self):
    initial_state_prior = jdn.JointDistributionNamed(
        {'position': deterministic.Deterministic(0.)})

    # Biased random walk.
    def particle_dynamics(_, previous_state):
      state_shape = ps.shape(previous_state['position'])
      return jdn.JointDistributionNamed({
          'position':
              transformed_distribution.TransformedDistribution(
                  bernoulli.Bernoulli(
                      probs=tf.fill(state_shape, 0.75), dtype=self.dtype),
                  shift.Shift(previous_state['position']))
      })

    # Completely uninformative observations allowing a test
    # of the pure dynamics.
    def particle_observations(_, state):
      state_shape = ps.shape(state['position'])
      return uniform.Uniform(
          low=tf.fill(state_shape, -100.), high=tf.fill(state_shape, 100.))

    observations = tf.zeros((9,), dtype=self.dtype)
    trajectories, _ = self.evaluate(
        particle_filter.infer_trajectories(
            observations=observations,
            initial_state_prior=initial_state_prior,
            transition_fn=particle_dynamics,
            observation_fn=particle_observations,
            num_particles=16384,
            seed=test_util.test_seed()))
    position = trajectories['position']

    # The trajectories have the following properties:
    # 1. they lie completely in the range [0, 8]
    self.assertAllInRange(position, 0., 8.)
    # 2. each step lies in the range [0, 1]
    self.assertAllInRange(position[1:] - position[:-1], 0., 1.)
    # 3. the expectation and variance of the final positions are 6 and 1.5.
    self.assertAllClose(tf.reduce_mean(position[-1]), 6., atol=0.1)
    self.assertAllClose(tf.math.reduce_variance(position[-1]), 1.5, atol=0.1)

  def test_batch_of_filters(self):

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

    # Batch of synthetic observations, .
    true_initial_positions = np.random.randn(*batch_shape).astype(self.dtype)
    true_velocities = 0.1 * np.random.randn(
        *batch_shape).astype(self.dtype)
    observed_positions = (
        true_velocities *
        np.arange(num_timesteps).astype(
            self.dtype)[..., tf.newaxis, tf.newaxis] +
        true_initial_positions)

    (particles, log_weights, parent_indices,
     incremental_log_marginal_likelihoods) = self.evaluate(
         particle_filter.particle_filter(
             observations=observed_positions,
             initial_state_prior=initial_state_prior,
             transition_fn=transition_fn,
             observation_fn=observation_fn,
             num_particles=num_particles,
             seed=test_util.test_seed()))

    self.assertAllEqual(particles['position'].shape,
                        [num_timesteps, num_particles] + batch_shape)
    self.assertAllEqual(particles['velocity'].shape,
                        [num_timesteps, num_particles] + batch_shape)
    self.assertAllEqual(parent_indices.shape,
                        [num_timesteps, num_particles] + batch_shape)
    self.assertAllEqual(incremental_log_marginal_likelihoods.shape,
                        [num_timesteps] + batch_shape)

    self.assertAllClose(
        self.evaluate(
            tf.reduce_sum(tf.exp(log_weights) *
                          particles['position'], axis=1)),
        observed_positions,
        atol=0.1)

    velocity_means = tf.reduce_sum(tf.exp(log_weights) *
                                   particles['velocity'], axis=1)
    self.assertAllClose(
        self.evaluate(tf.reduce_mean(velocity_means, axis=0)),
        true_velocities, atol=0.05)

    # Uncertainty in velocity should decrease over time.
    velocity_stddev = self.evaluate(
        tf.math.reduce_std(particles['velocity'], axis=1))
    self.assertAllLess((velocity_stddev[-1] - velocity_stddev[0]), 0.)

    trajectories = self.evaluate(
        particle_filter.reconstruct_trajectories(particles, parent_indices))
    self.assertAllEqual([num_timesteps, num_particles] + batch_shape,
                        trajectories['position'].shape)
    self.assertAllEqual([num_timesteps, num_particles] + batch_shape,
                        trajectories['velocity'].shape)

    # Verify that `infer_trajectories` also works on batches.
    trajectories, incremental_log_marginal_likelihoods = self.evaluate(
        particle_filter.infer_trajectories(
            observations=observed_positions,
            initial_state_prior=initial_state_prior,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            num_particles=num_particles,
            seed=test_util.test_seed()))
    self.assertAllEqual([num_timesteps, num_particles] + batch_shape,
                        trajectories['position'].shape)
    self.assertAllEqual([num_timesteps, num_particles] + batch_shape,
                        trajectories['velocity'].shape)
    self.assertAllEqual(incremental_log_marginal_likelihoods.shape,
                        [num_timesteps] + batch_shape)

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
