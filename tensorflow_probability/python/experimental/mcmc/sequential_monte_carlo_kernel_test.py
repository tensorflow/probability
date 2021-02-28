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
"""Tests for sequential monte carlo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc import SequentialMonteCarlo
from tensorflow_probability.python.experimental.mcmc import WeightedParticles
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


# TODO(davmre): add additional unit tests specific to the SMC kernel.
# Currently, most coverage is from downstream users such as
# `particle_filter_test`, which effectively serves as integration testing.
@test_util.test_all_tf_execution_regimes
class _SequentialMonteCarloTest(test_util.TestCase):

  def test_steps_are_reproducible(self):

    def propose_and_update_log_weights_fn(_, weighted_particles, seed=None):
      proposed_particles = tfd.Normal(
          loc=weighted_particles.particles, scale=1.).sample(seed=seed)
      return WeightedParticles(
          particles=proposed_particles,
          log_weights=weighted_particles.log_weights + tfd.Normal(
              loc=-2.6, scale=0.1).log_prob(proposed_particles))

    num_particles = 16
    initial_state = self.evaluate(
        WeightedParticles(
            particles=tf.random.normal([num_particles],
                                       seed=test_util.test_seed()),
            log_weights=tf.fill([num_particles],
                                -tf.math.log(float(num_particles)))))

    # Run a couple of steps.
    kernel = SequentialMonteCarlo(
        propose_and_update_log_weights_fn=propose_and_update_log_weights_fn,
        resample_fn=tfp.experimental.mcmc.resample_systematic,
        resample_criterion_fn=tfp.experimental.mcmc.ess_below_threshold)
    seed = test_util.test_seed()
    tf.random.set_seed(seed)
    seed_stream = tfp.util.SeedStream(seed=seed, salt='test')
    state, results = kernel.one_step(
        state=initial_state,
        kernel_results=kernel.bootstrap_results(initial_state),
        seed=seed_stream())
    state, results = kernel.one_step(state=state, kernel_results=results,
                                     seed=seed_stream())
    state, results = self.evaluate(
        (tf.nest.map_structure(tf.convert_to_tensor, state),
         tf.nest.map_structure(tf.convert_to_tensor, results)))

    # Re-initialize and run the same steps with the same seed.
    kernel2 = SequentialMonteCarlo(
        propose_and_update_log_weights_fn=propose_and_update_log_weights_fn,
        resample_fn=tfp.experimental.mcmc.resample_systematic,
        resample_criterion_fn=tfp.experimental.mcmc.ess_below_threshold)
    tf.random.set_seed(seed)
    seed_stream = tfp.util.SeedStream(seed=seed, salt='test')
    state2, results2 = kernel2.one_step(
        state=initial_state,
        kernel_results=kernel2.bootstrap_results(initial_state),
        seed=seed_stream())
    state2, results2 = kernel2.one_step(state=state2, kernel_results=results2,
                                        seed=seed_stream())
    state2, results2 = self.evaluate(
        (tf.nest.map_structure(tf.convert_to_tensor, state2),
         tf.nest.map_structure(tf.convert_to_tensor, results2)))

    # Results should match.
    self.assertAllCloseNested(state, state2)
    self.assertAllCloseNested(results, results2)


class SequentialMonteCarloTestFloat32(_SequentialMonteCarloTest):
  dtype = np.float32


del _SequentialMonteCarloTest

if __name__ == '__main__':
  tf.test.main()
