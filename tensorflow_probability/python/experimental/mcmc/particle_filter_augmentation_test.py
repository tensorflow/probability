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
"""Tests for particle filtering augmentations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class _ParticleFilterUtilTest(test_util.TestCase):

  def test_model_can_use_state_history(self):

    initial_state_prior = (
        tfp.experimental.mcmc.augment_prior_with_state_history(
            tfd.JointDistributionNamed({'x': tfd.Poisson(1.)}), history_size=2))

    # Deterministic dynamics compute a Fibonacci sequence.
    @tfp.experimental.mcmc.augment_with_state_history
    def fibonacci_transition_fn(step, state_with_history):
      del step
      return tfd.JointDistributionNamed(
          {'x': tfd.Deterministic(
              tf.reduce_sum(state_with_history.state_history['x'][..., -2:],
                            axis=-1))})

    # We'll observe the ratio of the current and previous state.
    def observe_ratio_of_last_two_states_fn(_, state_with_history):
      ratio = tf.ones_like(state_with_history.state['x'])
      if state_with_history.state_history is not None:
        ratio = state_with_history.state['x'] / (
            state_with_history.state_history['x'][..., -2]
            + 1e-6)  # Avoid division by 0.
      return tfd.Normal(loc=ratio, scale=0.1)

    # The ratios between successive terms of a Fibonacci sequence
    # should, in the limit, approach the golden ratio.
    golden_ratio = (1. + np.sqrt(5.)) / 2.
    observed_ratios = np.array([golden_ratio] * 10).astype(self.dtype)

    trajectories_with_history, lps = self.evaluate(
        tfp.experimental.mcmc.infer_trajectories(
            observed_ratios,
            initial_state_prior=initial_state_prior,
            transition_fn=fibonacci_transition_fn,
            observation_fn=observe_ratio_of_last_two_states_fn,
            num_particles=100,
            seed=test_util.test_seed()))
    trajectories = trajectories_with_history.state

    # Verify that we actually produced Fibonnaci sequences.
    self.assertAllClose(
        trajectories['x'][2:],
        trajectories['x'][1:-1] + trajectories['x'][:-2])

    # Ratios should get closer to golden as the series progresses, so
    # likelihoods will increase.
    self.assertAllGreaterEqual(lps[2:] - lps[:-2], 0.0)

    # Any particles that sampled initial values of 0. should have been
    # discarded, since those lead to degenerate series that do not approach
    # the golden ratio.
    self.assertAllGreaterEqual(trajectories['x'][0], 1.)

  def test_docstring_example_stochastic_fibonacci(self):
    initial_state_prior = tfd.Poisson(5.)
    initial_state_with_history_prior = (
        tfp.experimental.mcmc.augment_prior_with_state_history(
            initial_state_prior, history_size=2))

    initial_state_with_history_prior.sample(8)

    @tfp.experimental.mcmc.augment_with_state_history
    def fibonacci_transition_fn(_, state_with_history):
      expected_next_element = tf.reduce_sum(
          state_with_history.state_history[:, -2:], axis=1)
      return tfd.Poisson(rate=expected_next_element)

    def observation_fn(_, state_with_history):
      return tfd.Poisson(rate=state_with_history.state)

    tfp.experimental.mcmc.infer_trajectories(
        observations=tf.convert_to_tensor([4., 11., 16., 23., 40., 69., 100.]),
        initial_state_prior=initial_state_with_history_prior,
        transition_fn=fibonacci_transition_fn,
        observation_fn=observation_fn,
        num_particles=8,
        seed=test_util.test_seed())

  def test_model_can_use_observation_history(self):
    observations = np.array(
        [0.1, 3., -0.7, 1.1, 0., 14., -3., 5.8]).astype(self.dtype)
    weights = np.array([0.1, -0.2, 0.7]).astype(self.dtype)

    # Define an autoregressive model on observations. This ignores the
    # state entirely; it depends only on previous observations.
    initial_state_prior = tfd.JointDistributionNamed(
        {'dummy_state': tfd.Deterministic(0.)})
    def dummy_transition_fn(_, state, **kwargs):
      del kwargs
      return tfd.JointDistributionNamed(
          tf.nest.map_structure(tfd.Deterministic, state))

    @tfp.experimental.mcmc.augment_with_observation_history(
        observations=observations,
        history_size=len(weights))
    def autoregressive_observation_fn(step, _, observation_history=None):
      num_terms = prefer_static.minimum(step, len(weights))
      usable_weights = tf.convert_to_tensor(weights)[len(weights)-num_terms:]
      loc = tf.reduce_sum(usable_weights * observation_history)
      return tfd.Normal(loc, 1.0)

    # Manually compute the conditional log-probs of a series of observations
    # under the autoregressive model.
    expected_locs = []
    for current_step in range(len(observations)):
      start_step = max(0, current_step - len(weights))
      context_length = current_step - start_step
      expected_locs.append(
          np.sum(observations[start_step : current_step] *
                 weights[len(weights)-context_length:]))
    expected_lps = self.evaluate(
        tfd.Normal(expected_locs, scale=1.0).log_prob(observations))

    # Check that the particle filter gives the same log-probs.
    _, _, _, lps = self.evaluate(tfp.experimental.mcmc.particle_filter(
        observations,
        initial_state_prior=initial_state_prior,
        transition_fn=dummy_transition_fn,
        observation_fn=autoregressive_observation_fn,
        num_particles=2,
        seed=test_util.test_seed()))
    self.assertAllClose(expected_lps, lps)


class ParticleFilterUtilTestFloat32(_ParticleFilterUtilTest):
  dtype = np.float32


del _ParticleFilterUtilTest


if __name__ == '__main__':
  test_util.main()
