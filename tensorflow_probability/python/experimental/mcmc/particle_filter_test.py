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

  def test_smc_squared_rejuvenation_parameters(self):
    def particle_dynamics(params, _, previous_state):
      reshaped_params = tf.reshape(params,
                                   [params.shape[0]] +
                                   [1] * (previous_state.shape.rank - 1))
      broadcasted_params = tf.broadcast_to(reshaped_params,
                                           previous_state.shape)
      reshaped_dist = independent.Independent(
          normal.Normal(previous_state + broadcasted_params + 1, 0.1),
          reinterpreted_batch_ndims=1
      )

      return reshaped_dist

    def rejuvenation_criterion(step, state):
      # Rejuvenation every 2 steps
      cond = tf.logical_and(
          tf.equal(tf.math.mod(step, tf.constant(2)), tf.constant(0)),
          tf.not_equal(state.extra[0], tf.constant(0))
      )
      return tf.cond(cond, lambda: tf.constant(True),
                     lambda: tf.constant(False))

    observations = tf.stack([tf.range(30, dtype=tf.float32),
                             tf.range(30, dtype=tf.float32)], axis=1)

    num_outer_particles = 3
    num_inner_particles = 7

    loc = tf.broadcast_to([0., 0.], [num_outer_particles, 2])
    scale_diag = tf.broadcast_to([0.05, 0.05], [num_outer_particles, 2])

    params, _ = self.evaluate(particle_filter.smc_squared(
        observations=observations,
        inner_initial_state_prior=lambda _, params:
        mvn_diag.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag
        ),
        initial_parameter_prior=normal.Normal(3., 1.),
        num_outer_particles=num_outer_particles,
        num_inner_particles=num_inner_particles,
        outer_rejuvenation_criterion_fn=rejuvenation_criterion,
        inner_transition_fn=lambda params:
            lambda _, state: particle_dynamics(params, _, state),
        inner_observation_fn=lambda params: (
            lambda _, state: independent.Independent(
                normal.Normal(state, 2.), 1)
        ),
        outer_trace_fn=lambda s, r: (
            s.particles[0],
            s.particles[1]
        ),
        parameter_proposal_kernel=lambda params: normal.Normal(params, 3),
        seed=test_util.test_seed()
    )
  )

    abs_params = tf.abs(params)
    differences = abs_params[1:] - abs_params[:-1]
    mask_parameters = tf.reduce_all(tf.less_equal(differences, 0), axis=0)

    self.assertAllTrue(mask_parameters)


# TODO(b/186068104): add tests with dynamic shapes.
class ParticleFilterTestFloat32(_ParticleFilterTest):
  dtype = np.float32


del _ParticleFilterTest


if __name__ == '__main__':
  test_util.main()
