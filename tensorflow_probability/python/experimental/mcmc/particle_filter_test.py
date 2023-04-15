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
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import hidden_markov_model
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
from tensorflow_probability.python.experimental.mcmc import sequential_monte_carlo_kernel as smc_kernel
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


@test_util.test_all_tf_execution_regimes
class _ParticleFilterTest(test_util.TestCase):
  def test_smc_squared(self):
      results = self.evaluate(
          particle_filter.smc_squared(
              inner_observations=tf.convert_to_tensor([1., 3., 5., 7., 9.]),
              inner_initial_state_prior=lambda _, state: normal.Normal(tf.zeros_like(state), 1.),
              initial_parameter_prior=normal.Normal(0., 1.),
              parameter_proposal_kernel=1,  # TODO
              num_outer_particles=20,
              outer_rejuvenation_criterion_fn=lambda *_: False,
              inner_transition_fn=lambda _, state: normal.Normal(state, 1.),
              inner_observation_fn=lambda _, state: normal.Normal(state, 1.),
              num_inner_particles=10,
              seed=1)
      )

      # results = self.evaluate(
      #     particle_filter.particle_filter(
      #         observations=tf.convert_to_tensor([1., 3., 5., 7., 9.]),
      #         initial_state_prior=normal.Normal(0., 1.),
      #         transition_fn=lambda _, state: normal.Normal(state, 1.),
      #         observation_fn=lambda _, state: normal.Normal(state, 1.),
      #         num_particles=1024,
      #         seed=1)
      # )

# TODO(b/186068104): add tests with dynamic shapes.
class ParticleFilterTestFloat32(_ParticleFilterTest):
  dtype = np.float32


del _ParticleFilterTest


if __name__ == '__main__':
  test_util.main()
