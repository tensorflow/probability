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

import functools

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.mcmc import sequential_monte_carlo_kernel
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import SequentialMonteCarlo
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import weighted_resampling
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import WeightedParticles
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


# TODO(davmre): add additional unit tests specific to the SMC kernel.
# Currently, most coverage is from downstream users such as
# `particle_filter_test`, which effectively serves as integration testing.
@test_util.test_all_tf_execution_regimes
class _SequentialMonteCarloTest(test_util.TestCase):

  def test_steps_are_reproducible(self):

    def propose_and_update_log_weights_fn(_, weighted_particles, seed=None):
      proposed_particles = normal.Normal(
          loc=weighted_particles.particles, scale=1.).sample(seed=seed)
      return WeightedParticles(
          particles=proposed_particles,
          log_weights=weighted_particles.log_weights +
          normal.Normal(loc=-2.6, scale=0.1).log_prob(proposed_particles))

    num_particles = 16
    initial_state = self.evaluate(
        WeightedParticles(
            particles=tf.random.normal([num_particles],
                                       seed=test_util.test_seed()),
            log_weights=tf.fill([num_particles],
                                -tf.math.log(float(num_particles)))))

    # Run a couple of steps.
    seeds = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'), n=2)
    kernel = SequentialMonteCarlo(
        propose_and_update_log_weights_fn=propose_and_update_log_weights_fn,
        resample_fn=weighted_resampling.resample_systematic,
        resample_criterion_fn=sequential_monte_carlo_kernel.ess_below_threshold)
    state, results = kernel.one_step(
        state=initial_state,
        kernel_results=kernel.bootstrap_results(initial_state),
        seed=seeds[0],
        )
    state, results = kernel.one_step(state=state, kernel_results=results,
                                     seed=seeds[1])
    state, results = self.evaluate(
        (tf.nest.map_structure(tf.convert_to_tensor, state),
         tf.nest.map_structure(tf.convert_to_tensor, results)))

    # Re-initialize and run the same steps with the same seed.
    kernel2 = SequentialMonteCarlo(
        propose_and_update_log_weights_fn=propose_and_update_log_weights_fn,
        resample_fn=weighted_resampling.resample_systematic,
        resample_criterion_fn=sequential_monte_carlo_kernel.ess_below_threshold)
    state2, results2 = kernel2.one_step(
        state=initial_state,
        kernel_results=kernel2.bootstrap_results(initial_state),
        seed=seeds[0])
    state2, results2 = kernel2.one_step(state=state2, kernel_results=results2,
                                        seed=seeds[1])
    state2, results2 = self.evaluate(
        (tf.nest.map_structure(tf.convert_to_tensor, state2),
         tf.nest.map_structure(tf.convert_to_tensor, results2)))

    # Results should match.
    self.assertAllCloseNested(state, state2)
    self.assertAllCloseNested(results, results2)

  @test_util.numpy_disable_variable_test
  def testMarginalLikelihoodGradientIsDefined(self):
    num_particles = 16
    seeds = samplers.split_seed(test_util.test_seed(), n=3)
    initial_state = self.evaluate(
        WeightedParticles(
            particles=samplers.normal([num_particles], seed=seeds[0]),
            log_weights=tf.fill([num_particles],
                                -tf.math.log(float(num_particles)))))

    def propose_and_update_log_weights_fn(_,
                                          weighted_particles,
                                          transition_scale,
                                          seed=None):
      proposal_dist = normal.Normal(loc=weighted_particles.particles, scale=1.)
      transition_dist = normal.Normal(
          loc=weighted_particles.particles, scale=transition_scale)
      proposed_particles = proposal_dist.sample(seed=seed)
      return WeightedParticles(
          particles=proposed_particles,
          log_weights=(weighted_particles.log_weights +
                       transition_dist.log_prob(proposed_particles) -
                       proposal_dist.log_prob(proposed_particles)))

    def marginal_logprob(transition_scale):
      kernel = SequentialMonteCarlo(
          propose_and_update_log_weights_fn=functools.partial(
              propose_and_update_log_weights_fn,
              transition_scale=transition_scale))
      state, results = kernel.one_step(
          state=initial_state,
          kernel_results=kernel.bootstrap_results(initial_state),
          seed=seeds[1])
      state, results = kernel.one_step(state=state, kernel_results=results,
                                       seed=seeds[2])
      return results.accumulated_log_marginal_likelihood

    _, grad_lp = gradient.value_and_gradient(marginal_logprob, 1.5)
    self.assertIsNotNone(grad_lp)
    self.assertNotAllZero(grad_lp)


class SequentialMonteCarloTestFloat32(_SequentialMonteCarloTest):
  dtype = np.float32


del _SequentialMonteCarloTest

if __name__ == '__main__':
  test_util.main()
