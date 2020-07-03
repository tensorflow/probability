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
"""Tests for iterated filtering."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_graph_and_eager_modes  # `eager_no_tf_function` is too slow.
class IteratedFilterTest(test_util.TestCase):

  def test_batch_estimation(self):

    # Batch of models, each a normal random walk with unknown scale param.
    batch_shape = [2, 3]
    parameter_prior = tfd.TransformedDistribution(
        tfd.Normal(loc=tf.ones(batch_shape), scale=1.),
        tfb.Softplus())
    parameterized_initial_state_prior_fn = (
        lambda parameters: tfd.Normal(loc=0., scale=parameters))
    parameteried_transition_fn = (
        lambda _, state, parameters: tfd.Normal(loc=state, scale=parameters))
    parameterized_observation_fn = (
        lambda _, state, **kwargs: tfd.Normal(loc=state, scale=1.0))

    # Generate a batch of synthetic observations from the model.
    num_timesteps = 100
    true_scales = self.evaluate(
        parameter_prior.sample(seed=test_util.test_seed()))
    trajectories = tf.math.cumsum(
        tf.random.normal([num_timesteps] + batch_shape) * true_scales, axis=0)
    observations = self.evaluate(
        parameterized_observation_fn(0, trajectories).sample(
            seed=test_util.test_seed()))

    # Estimate the batch of scale parameters.
    iterated_filter = tfp.experimental.sequential.IteratedFilter(
        parameter_prior=parameter_prior,
        parameterized_initial_state_prior_fn=(
            parameterized_initial_state_prior_fn),
        parameterized_transition_fn=parameteried_transition_fn,
        parameterized_observation_fn=parameterized_observation_fn)
    estimated_scales = self.evaluate(
        iterated_filter.estimate_parameters(
            observations=observations,
            num_iterations=20,
            num_particles=1024,
            initial_perturbation_scale=1.0,
            cooling_schedule=(
                tfp.experimental.sequential.geometric_cooling_schedule(
                    0.001, k=20)),
            seed=test_util.test_seed()))
    final_scales = tf.nest.map_structure(lambda x: x[-1], estimated_scales)
    # Note that this inference isn't super precise with the current tuning.
    # Varying the seed, the max absolute error across the batch is typically
    # in the range 0.2 - 0.4.
    self.assertAllClose(np.mean(final_scales, axis=0), true_scales, atol=0.8)

  def test_epidemiological_model_docstring_example(self):
    # Toy SIR model
    # (https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
    population_size = 1000
    parameter_prior = tfd.JointDistributionNamed({
        'infection_rate': tfd.Uniform(low=0., high=3.),
        'recovery_rate': tfd.Uniform(low=0., high=3.),
    })

    initial_state_prior_fn = lambda parameters: tfd.JointDistributionNamed({  # pylint: disable=g-long-lambda
        'new_infections': tfd.Poisson(parameters['infection_rate']),
        'new_recoveries': tfd.Deterministic(
            tf.broadcast_to(0., tf.shape(parameters['recovery_rate']))),
        'susceptible': (lambda new_infections:  # pylint: disable=g-long-lambda
                        tfd.Deterministic(population_size - new_infections)),
        'infected': (lambda new_infections:  # pylint: disable=unnecessary-lambda, g-long-lambda
                     tfd.Deterministic(new_infections))})

    # Dynamics model: new infections and recoveries are given by the SIR
    # model with Poisson noise.
    def parameterized_infection_dynamics(_, previous_state, parameters):
      new_infections = tfd.Poisson(
          parameters['infection_rate'] * previous_state['infected'] *
          previous_state['susceptible'] / population_size)
      new_recoveries = tfd.Poisson(
          previous_state['infected'] * parameters['recovery_rate'])
      return tfd.JointDistributionNamed({
          'new_infections': new_infections,
          'new_recoveries': new_recoveries,
          'susceptible': lambda new_infections: tfd.Deterministic(  # pylint: disable=g-long-lambda
              tf.maximum(0., previous_state['susceptible'] - new_infections)),
          'infected': lambda new_infections, new_recoveries: tfd.Deterministic(  # pylint: disable=g-long-lambda
              tf.maximum(0.,
                         (previous_state['infected'] +
                          new_infections - new_recoveries)))})

    # Observation model: each day we detect new cases and recoveries, noisily.
    def parameterized_infection_observations(_, state, parameters):
      del parameters
      return tfd.JointDistributionNamed({
          'new_infections': tfd.Poisson(state['new_infections'] + 0.1),
          'new_recoveries': tfd.Poisson(state['new_recoveries'] + 0.1)})

    iterated_filter = tfp.experimental.sequential.IteratedFilter(
        parameter_prior=parameter_prior,
        parameterized_initial_state_prior_fn=initial_state_prior_fn,
        parameterized_transition_fn=parameterized_infection_dynamics,
        parameterized_observation_fn=parameterized_infection_observations)

    observed_values = {
        'new_infections': tf.convert_to_tensor([
            2., 7., 14., 24., 45., 93., 160., 228., 252., 158., 17.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        'new_recoveries': tf.convert_to_tensor([
            0., 0., 3., 4., 3., 8., 12., 31., 49., 73., 85., 65., 71.,
            58., 42., 65., 36., 31., 32., 27., 31., 20., 19., 19., 14., 27.])
    }
    parameter_particles = self.evaluate(iterated_filter.estimate_parameters(
        observations=observed_values,
        num_iterations=20,
        num_particles=4096,
        initial_perturbation_scale=1.0,
        cooling_schedule=(
            tfp.experimental.sequential.geometric_cooling_schedule(
                0.001, k=20)),
        seed=test_util.test_seed()))
    parameter_estimates = tf.nest.map_structure(
        lambda x: np.mean(x[-1], axis=0),
        parameter_particles)
    # Check that we recovered the generating parameters.
    self.assertAllClose(parameter_estimates['infection_rate'], 1.2, atol=0.1)
    self.assertAllClose(parameter_estimates['recovery_rate'], 0.1, atol=0.05)

  def test_raises_error_on_inconsistent_prior_shapes(self):
    with self.assertRaisesRegexp(
        ValueError, 'The specified prior does not generate consistent shapes'):
      tfp.experimental.sequential.IteratedFilter(
          parameter_prior=tfd.Normal(0., 1.),
          # Pass an invalid prior fn that ignores parameters and always
          # has batch shape `[]`.
          parameterized_initial_state_prior_fn=lambda _: tfd.Normal(0., 1.),
          parameterized_transition_fn=lambda p: tfd.Normal(p, 1.),
          parameterized_observation_fn=lambda s: tfd.Normal(s, 1.))

if __name__ == '__main__':
  tf.test.main()
