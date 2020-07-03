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
"""Tests for experimental.sequential.extended_kalman_filter."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class ExtendedKalmanFilterTest(test_util.TestCase):

  def test_simple_nonlinear_system(self):
    initial_state_prior = tfd.MultivariateNormalDiag(
        0., scale_diag=[1., 0.3], validate_args=True)
    observation_noise_scale = 0.5

    # x_{0, t+1} = x_{0, t} - 0.1 * x_{1, t}**3; x_{1, t+1} = x_{1, t}
    def transition_fn(x):
      return tfd.MultivariateNormalDiag(
          tf.stack(
              [x[..., 0] - 0.1 * tf.pow(x[..., 1], 3), x[..., 1]], axis=-1),
          scale_diag=[0.5, 0.05], validate_args=True)

    def transition_jacobian_fn(x):
      return tf.reshape(
          tf.stack(
              [1., -0.3 * x[..., 1]**2,
               tf.zeros(x.shape[:-1]), tf.ones(x.shape[:-1])], axis=-1),
          [2, 2])

    def observation_fn(x):
      return tfd.MultivariateNormalDiag(
          x[..., :1],
          scale_diag=[observation_noise_scale],
          validate_args=True)

    observation_jacobian_fn = lambda x: [[1., 0.]]

    x = [np.zeros((2,), dtype=np.float32)]
    num_timesteps = 20
    for _ in range(num_timesteps - 1):
      x.append(
          transition_fn(x[-1]).sample(seed=test_util.test_seed()))
    x = tf.stack(x)
    observations = observation_fn(x).sample(seed=test_util.test_seed())

    results = tfp.experimental.sequential.extended_kalman_filter(
        observations=observations,
        initial_state_prior=initial_state_prior,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        transition_jacobian_fn=transition_jacobian_fn,
        observation_jacobian_fn=observation_jacobian_fn)
    (filtered_mean, filtered_cov, predicted_mean, predicted_cov,
     observation_mean, observation_cov, log_marginal_likelihood,
     timestep) = results
    self.assertAllEqual(filtered_mean.shape, [num_timesteps, 2])
    self.assertAllEqual(filtered_cov.shape, [num_timesteps, 2, 2])
    self.assertAllEqual(predicted_mean.shape, [num_timesteps, 2])
    self.assertAllEqual(predicted_cov.shape, [num_timesteps, 2, 2])
    self.assertAllEqual(observation_mean.shape, [num_timesteps, 1])
    self.assertAllEqual(observation_cov.shape, [num_timesteps, 1, 1])
    self.assertAllEqual(log_marginal_likelihood.shape, [num_timesteps])
    self.assertAllEqual(timestep.shape, [num_timesteps])

    # Check that the estimate of the most observable quantity is close to actual
    # (i.e. MSE is close to the observation noise covariance).
    if tf.executing_eagerly():
      # Skip in graph mode, because running the graph resamples the
      # observations.
      self.assertLess(
          tf.reduce_mean((filtered_mean[..., 0] - x[..., 0])**2), 0.3)

    self.assertAllEqual(timestep, np.arange(num_timesteps))

    # test structured input
    observations_struct = {'a': observations, 'b': (observations, observations)}
    nested_results = self.evaluate(
        tfp.experimental.sequential.extended_kalman_filter(
            observations=observations_struct,
            initial_state_prior=initial_state_prior,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            transition_jacobian_fn=transition_jacobian_fn,
            observation_jacobian_fn=observation_jacobian_fn))

    if tf.executing_eagerly():
      for result, nested_result in zip(results, nested_results):
        self.assertAllEqualNested(
            tf.nest.map_structure(lambda _: result, observations_struct),  # pylint: disable=cell-var-from-loop
            nested_result)

    # test observations with event_size > 1
    def observation_fn_3dim(x):
      loc = tf.stack(
          [x[..., 0], tf.reduce_sum(x, axis=-1), tf.reduce_prod(x, axis=-1)],
          axis=-1)
      return tfd.MultivariateNormalDiag(
          loc, scale_diag=[0.5, 0.5, 0.5], validate_args=True)

    def observation_jacobian_fn_3dim(x):
      return tf.reshape(
          tf.stack([1., 0., 1., 1., x[..., 1], x[..., 0]], axis=-1), [3, 2])

    observations_3dim = observation_fn_3dim(x).sample(
        seed=test_util.test_seed())
    results_3dim = tfp.experimental.sequential.extended_kalman_filter(
        observations=observations_3dim,
        initial_state_prior=initial_state_prior,
        transition_fn=transition_fn,
        observation_fn=observation_fn_3dim,
        transition_jacobian_fn=transition_jacobian_fn,
        observation_jacobian_fn=observation_jacobian_fn_3dim)

    self.assertAllEqual(results_3dim[4].shape, [num_timesteps, 3])
    self.assertAllEqual(results_3dim[5].shape, [num_timesteps, 3, 3])

  def test_epidemiological_model(self):
    # A toy, discrete version of an SIR (Susceptible, Infected, Recovered)
    # model. https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

    infection_rate = 2.3
    recovery_rate = 0.5
    population_size = 1000
    detection_rate = 0.5

    observed_infections = np.array(
        [15., 41., 104., 209., 269., 226., 160., 97., 69.,
         34., 19., 20., 13., 5., 6., 3., 2., 1., 1., 1.])

    # Add batch dimensions and an event dimension.
    observed_infections = np.tile(
        observed_infections, (2, 3, 1)).astype(np.float32)
    observed_infections = np.swapaxes(
        observed_infections, -1, 0)[..., np.newaxis]

    # The state `x` is defined by `[num_susceptible, num_infected]`.
    def transition_fn(x):
      p_s = 1. - tf.math.exp(-infection_rate * x[..., 1] / population_size)
      p_r = 1. - tf.math.exp(-recovery_rate)
      d_s = -x[..., 0] * p_s
      return tfd.MultivariateNormalDiag(
          x + tf.stack([d_s, -d_s - x[..., 1] * p_r], axis=-1),
          scale_diag=tf.stack(
              [x[..., 0] * p_s * (1 - p_s), x[..., 1] * p_r * (1 - p_r)],
              axis=-1),
          validate_args=True)

    def transition_jacobian_fn(x):
      shape_out = prefer_static.concat(
          [prefer_static.shape(x)[:-1], (2, 2)], axis=0)
      return tf.reshape(
          tf.stack([
              1. - infection_rate * x[..., 1] / population_size,
              -infection_rate * x[..., 1] / population_size,
              infection_rate * x[..., 0] / population_size,
              1 + infection_rate * x[..., 0] / population_size - recovery_rate],
                   axis=-1),
          shape=shape_out)

    # We use the Binomial mean and covariance, assuming observed infections are
    # detected with `prob=detection_rate`.
    def observation_fn(x):
      return tfd.MultivariateNormalDiag(
          detection_rate * x[..., 1:],
          scale_diag=x[..., 1:] * detection_rate * (1 - detection_rate),
          validate_args=True)

    observation_jacobian_fn = lambda _: [[0., detection_rate]]

    # Start with an incorrect guess for susceptible and infected.
    x = np.array([population_size - 25, 25]).astype(np.float32)
    initial_state_prior = tfd.MultivariateNormalDiag(
        x, scale_diag=0.1*x, validate_args=True)

    results = self.evaluate(
        tfp.experimental.sequential.extended_kalman_filter(
            observations=observed_infections,
            initial_state_prior=initial_state_prior,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            transition_jacobian_fn=transition_jacobian_fn,
            observation_jacobian_fn=observation_jacobian_fn))
    self.assertAllEqual(results[0].shape, [20, 3, 2, 2])
    self.assertAllEqual(results[1].shape, [20, 3, 2, 2, 2])


if __name__ == '__main__':
  tf.test.main()
