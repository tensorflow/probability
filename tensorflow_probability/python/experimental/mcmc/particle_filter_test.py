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
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental.mcmc import particle_filter
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


@test_util.test_all_tf_execution_regimes
class _ParticleFilterTest(test_util.TestCase):

  def test_smc_squared_no_rejuvenation(self):
      def particle_dynamics(params, _, previous_state):
          reshaped_params = tf.reshape(params, [params.shape[0]] + [1] * (previous_state.shape.rank - 1))
          broadcasted_params = tf.broadcast_to(reshaped_params, previous_state.shape)
          return normal.Normal(previous_state + broadcasted_params + 1, 0.0001)

      def rejuvenation_criterion(state):
          cond = tf.logical_and(
              tf.equal(tf.math.mod(state.extra[0], tf.constant(5)), tf.constant(0)),
              tf.not_equal(state.extra[0], tf.constant(0))
          )
          return tf.cond(cond, lambda: tf.constant(True), lambda: tf.constant(False))

      inner_observations = tf.constant([0., 1., 2., 3., 4., 5., 6., 7., 8.])

      params, inner_lp, lp = particle_filter.smc_squared(
          inner_observations=inner_observations,
          inner_initial_state_prior=lambda _, params: mvn_diag.MultivariateNormalDiag(
              loc=[0., 0.],
              scale_diag=[0.05, 0.05]),
          initial_parameter_prior=normal.Normal(0., 0.03),
          num_outer_particles=4,
          num_inner_particles=3,
          outer_rejuvenation_criterion_fn=lambda _: False,
          inner_transition_fn=lambda params: (lambda _, state: independent.Independent(particle_dynamics(params, _, state), 1)),
          inner_observation_fn=lambda params: (lambda _, state: independent.Independent(normal.Normal(state, 0.1), 1)),
          inner_trace_fn=lambda s, r: (
              s.particles[0],  # Params
              s.particles[4],  # Accumulated_log_marginal_likelihood of inner particles
              r.accumulated_log_marginal_likelihood  # Accumulated_log_marginal_likelihood of outer particles
          ),
          parameter_proposal_kernel=lambda state: normal.Normal(0., 0.01)
      )
      print(params)
      print(inner_lp)

      ###
      # Particle filter with same dynamics
      ###

      # def particle_dynamics_pf(_, previous_state):
      #     return normal.Normal(previous_state + 1, 0.001)
      #
      # particles_pf, log_weights_pf, lp_pf = particle_filter.particle_filter(
      #     observations=inner_observations,
      #     initial_state_prior=independent.Independent(deterministic.Deterministic(
      #         tf.zeros_like([0., 0.])), 1
      #     ),
      #     transition_fn=lambda _, state: independent.Independent(particle_dynamics_pf(_, state), 1),
      #     observation_fn=lambda _, state: independent.Independent(normal.Normal(state, 0.01), 1),
      #     num_particles=3,
      #     trace_fn=lambda s, r: (
      #         s.particles,
      #         s.log_weights,
      #         r.accumulated_log_marginal_likelihood
      #     )
      # )




# TODO(b/186068104): add tests with dynamic shapes.
class ParticleFilterTestFloat32(_ParticleFilterTest):
  dtype = np.float32


del _ParticleFilterTest


if __name__ == '__main__':
  test_util.main()
