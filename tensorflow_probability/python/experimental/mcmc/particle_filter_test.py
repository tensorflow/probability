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
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import hidden_markov_model
from tensorflow_probability.python.experimental.mcmc import particle_filter
from tensorflow_probability.python.experimental.mcmc.particle_filter import sequential_monte_carlo
from tensorflow_probability.python.experimental.mcmc.particle_filter import _particle_filter_initial_weighted_particles
from tensorflow_probability.python.experimental.mcmc.particle_filter import _particle_filter_propose_and_update_log_weights_fn
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


@test_util.test_all_tf_execution_regimes
class _ParticleFilterTest(test_util.TestCase):

  # def test_rejuvenation_fn(self):
  #   # A simple HMM with 10 hidden states
  #   stream = test_util.test_seed_stream()
  #   d = hidden_markov_model.HiddenMarkovModel(
  #       initial_distribution=categorical.Categorical(logits=tf.zeros(10)),
  #       transition_distribution=categorical.Categorical(logits=tf.zeros((10, 10))),
  #       observation_distribution=normal.Normal(loc=tf.range(10.), scale=0.3),
  #       num_steps=10
  #   )
  #   observation = categorical.Categorical(
  #       logits=[0] * 10,
  #       dtype=tf.float32).sample(10, seed=stream())
  #
  #   # A dimension for each particle of the particles filters
  #   observations = tf.reshape(tf.tile(observation, [10]),
  #                             [10, tf.shape(observation)[0]])
  #
  #   def rejuvenation_fn(state, step=-1):
  #     posterior = d.posterior_marginals(observation).sample(seed=stream())
  #     return posterior
  #
  #   def rejuvenation_criterion_fn(_):
  #     return 1
  #
  #   rej_particles, _, _, _, _ =\
  #       particle_filter.particle_filter(
  #           observations=observation,
  #           initial_state_prior=d.initial_distribution,
  #           transition_fn=lambda _, s: categorical.Categorical(logits=tf.zeros(s.shape + tuple([10]))),
  #           observation_fn=lambda _, s: normal.Normal(loc=tf.cast(s, tf.float32), scale=0.3),
  #           rejuvenation_criterion_fn=rejuvenation_criterion_fn,
  #           rejuvenation_fn=rejuvenation_fn,
  #           num_particles=10,
  #           seed=stream()
  #       )
  #
  #   delta_rej = tf.where(observations - tf.cast(rej_particles, tf.float32) != 0, 1, 0)
  #
  #   nonrej_particles, _, _, _, _ =\
  #       particle_filter.particle_filter(
  #           observations=observation,
  #           initial_state_prior=d.initial_distribution,
  #           transition_fn=lambda _, s: categorical.Categorical(logits=tf.zeros(s.shape + tuple([10]))),
  #           observation_fn=lambda _, s: normal.Normal(loc=tf.cast(s, tf.float32), scale=0.3),
  #           num_particles=10,
  #           seed=stream()
  #       )
  #   delta_nonrej = tf.where(observations - tf.cast(nonrej_particles, tf.float32) != 0, 1, 0)
  #
  #   delta = tf.reduce_sum(delta_nonrej - delta_rej)
  #
  #   self.assertAllGreaterEqual(self.evaluate(delta), 0)

  def test_extra(self):
    particles, a, b, lps, extra = self.evaluate(
        particle_filter.particle_filter(
              observations=tf.constant([0., 1.1, 2.0, 2.9, 4.0]),
              initial_state_prior=deterministic.Deterministic(0.),
              transition_fn=lambda _, prev_state: normal.Normal(prev_state + 1, 0.1),
              observation_fn=lambda _, state: normal.Normal(loc=state, scale=0.1),
              num_particles=2,
              seed=test_util.test_seed())
    )



# TODO(b/186068104): add tests with dynamic shapes.
class ParticleFilterTestFloat32(_ParticleFilterTest):
  dtype = np.float32


del _ParticleFilterTest


if __name__ == '__main__':
  test_util.main()
