# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The Extended Kalman filter."""

import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions.linear_gaussian_ssm import KalmanFilterState
from tensorflow_probability.python.internal import prefer_static
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'extended_kalman_filter',
    'extended_kalman_filter_one_step'
]


def _initialize_accumulated_quantities(observations, num_timesteps):
  """Initialize arrays passed through the filter loop."""
  initial_arrays = [tf.nest.map_structure(
      lambda x: tf.TensorArray(dtype=x.dtype, size=num_timesteps),
      observations) for _ in range(7)]
  initial_arrays.append(tf.nest.map_structure(
      lambda _: tf.TensorArray(dtype=tf.int32, size=num_timesteps),
      observations))
  return KalmanFilterState(*initial_arrays)


def _write_accumulated_quantities(
    step, accumulated_quantities, updated_estimate):
  """Update arrays passed through the filter loop."""
  new_accumulated_quantities = [
      nest.map_structure_up_to(
          element, lambda x, y: x.write(step, y[i]), element, updated_estimate)  # pylint: disable=cell-var-from-loop
      for i, element in enumerate(accumulated_quantities)]
  return KalmanFilterState(*new_accumulated_quantities)


def _matmul_a_b_at(a, b):
  """Calculate `A @ B @ transpose(A)`."""
  return tf.matmul(a, tf.matmul(b, a, transpose_b=True))


# TODO(emilyaf): Add optional control input.
# TODO(emilyaf): Support automatic Jacobian calculation if *_jacobian_fn==None.
# TODO(emilyaf): Explore integration with tfd.LinearGaussianSSM.
def extended_kalman_filter(
    observations,
    initial_state_prior,
    transition_fn,
    observation_fn,
    transition_jacobian_fn,
    observation_jacobian_fn,
    name=None):
  """Applies an Extended Kalman Filter to observed data.

  The [Extended Kalman Filter](
  https://en.wikipedia.org/wiki/Extended_Kalman_filter) is a nonlinear version
  of the Kalman filter, in which the transition function is linearized by
  first-order Taylor expansion around the current mean and covariance of the
  state estimate.

  Args:
    observations: a (structure of) `Tensor`s, each of shape
      `concat([[num_timesteps, b1, ..., bN], [event_size]])` with scalar
      `event_size` and optional batch dimensions `b1, ..., bN`.
    initial_state_prior: a `tfd.Distribution` instance (typically
      `MultivariateNormal`) with `event_shape` equal to `state_size` and an
      optional `batch_shape` of [`b1, ..., bN`], representing the prior over the
      state.
    transition_fn: a Python `callable` that accepts (batched) vectors of length
      `state_size`, and returns a `tfd.Distribution` instance, typically a
      `MultivariateNormal`, representing the state transition and covariance.
    observation_fn: a Python `callable` that accepts a (batched) vector of
      length `state_size` and returns a `tfd.Distribution` instance, typically
      a `MultivariateNormal` representing the observation model and covariance.
    transition_jacobian_fn: a Python `callable` that accepts a (batched) vector
      of length `state_size` and returns a (batched) matrix of shape
      `[state_size, state_size]`, representing the Jacobian of `transition_fn`.
    observation_jacobian_fn: a Python `callable` that accepts a (batched) vector
      of length `state_size` and returns a (batched) matrix of size
      `[state_size, event_size]`, representing the Jacobian of `observation_fn`.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'extended_kalman_filter'`).
  Returns:
    filtered_mean: a (structure of) `Tensor`(s) of shape
      `concat([[num_timesteps, b1, ..., bN], [state_size]])`. The mean of the
      filtered state estimate.
    filtered_cov: a (structure of) `Tensor`(s) of shape
      `concat([[num_timesteps, b1, ..., bN], [state_size, state_size]])`.
       The covariance of the filtered state estimate.
    predicted_mean: a (structure of) `Tensor`(s) of shape
      `concat([[num_timesteps, b1, ..., bN], [state_size]])`. The prior
      predicted means of the state.
    predicted_cov: a (structure of) `Tensor`(s) of shape
      `concat([[num_timesteps, b1, ..., bN], [state_size, state_size]])`
      The prior predicted covariances of the state estimate.
    observation_mean: a (structure of) `Tensor`(s) of shape
      `concat([[num_timesteps, b1, ..., bN], [event_size]])`. The prior
      predicted mean of observations.
    observation_cov: a (structure of) `Tensor`(s) of shape
      `concat([[num_timesteps, b1, ..., bN], [event_size, event_size]])`. The
      prior predicted covariance of observations.
    log_marginal_likelihood: a (structure of) `Tensor`(s) of shape
      `[num_timesteps, b1, ..., bN]`. Log likelihood of the observations with
      respect to the observation.
    timestep: a (structure of) integer `Tensor`(s) of shape
      `[num_timesteps, b1, ..., bN]` containing time indices.

  #### Examples

  **Estimate a simple nonlinear system**: Let's consider a system defined by
  the transition equation `y_{t+1} = y_t - 0.1 * w_t **3` and `w_{t+1} = w_t`,
  such that the state can be expressed as `[y, w]`. The `transition_fn` and
  `transition_jacobian_fn` can be expressed as:

  ```python
  def transition_fn(x):
    return tfd.MultivariateNormalDiag(
        tf.stack(
            [x[..., 0] - 0.1 * x[..., 1]**3, x[..., 1]], axis=-1),
        scale_diag=[0.7, 0.2])

  def transition_jacobian_fn(x):
    return tf.reshape(
      tf.stack(
          [1. - 0.1 * x[..., 1]**3, -0.3 * x[..., 1]**2,
          tf.zeros(x.shape[:-1]), tf.ones(x.shape[:-1])], axis=-1),
      [2, 2])
  ```

  Assume we take noisy measurements of only the first element of the state.

  ```python
  observation_fn = lambda x: tfd.MultivariateNormalDiag(
      x[..., :1], scale_diag=[1.])
  observation_jacobian_fn = lambda x: [[1., 0.]]
  ```

  We define a prior over the initial state, and use it to synthesize data for
  20 steps of the process.

  ```python
  initial_state_prior = tfd.MultivariateNormalDiag(0., scale_diag=[1., 0.3])

  x = [np.zeros((2,), dtype=np.float32)]
  for t in range(20):
    x.append(transition_fn(x[-1]).sample())
  x = tf.stack(x)

  observations=observation_fn(x).sample()
  ```

  Run the Extended Kalman filter on the synthesized observed data.

  ```python
  results = tfp.experimental.sequential.extended_kalman_filter(
      observations,
      initial_state_prior,
      transition_fn,
      observation_fn,
      transition_jacobian_fn,
      observation_jacobian_fn)
  ```
  """
  with tf.name_scope(name or 'extended_kalman_filter'):
    observations_shape = prefer_static.shape(
        tf.nest.flatten(observations)[0])
    num_timesteps = observations_shape[0]

    # add singleton batch dimensions to initial state estimates, if necessary
    dummy_zeros = tf.zeros(observations_shape[1:-1])
    initial_state = initial_state_prior.mean() + dummy_zeros[..., tf.newaxis]
    initial_covariance = (
        initial_state_prior.covariance() +
        dummy_zeros[..., tf.newaxis, tf.newaxis])
    observation_dist = observation_fn(initial_state)

    # Initialize the state estimate.
    initial_estimate = tf.nest.map_structure(
        lambda _: KalmanFilterState(  # pylint: disable=g-long-lambda
            predicted_mean=initial_state,
            predicted_cov=initial_covariance,
            filtered_mean=initial_state,
            filtered_cov=initial_covariance,
            observation_mean=observation_dist.mean(),
            observation_cov=observation_dist.covariance(),
            log_marginal_likelihood=dummy_zeros,
            timestep=tf.zeros(observations_shape[1:-1], dtype=tf.int32) - 1),
        observations)

    initial_accumulated_quantities = _initialize_accumulated_quantities(
        observations, num_timesteps)

    run_ekf_step = functools.partial(
        extended_kalman_filter_one_step,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        transition_jacobian_fn=transition_jacobian_fn,
        observation_jacobian_fn=observation_jacobian_fn)

    def _loop_body(step, current_estimate, accumulated_quantities):
      """Body of the `while_loop` for running forward filtering."""
      current_observations = tf.nest.map_structure(
          lambda x, step=step: tf.gather(x, step), observations)
      updated_estimate = nest.map_structure_up_to(
          current_observations,
          run_ekf_step,
          current_estimate, current_observations)
      new_accumulated_quantities = _write_accumulated_quantities(
          step, accumulated_quantities, updated_estimate)
      return step + 1, updated_estimate, new_accumulated_quantities

    _, _, loop_results = tf.while_loop(
        cond=lambda step, *_: step < num_timesteps,
        body=_loop_body,
        loop_vars=(
            tf.zeros([], tf.int32),
            initial_estimate, initial_accumulated_quantities))
    return [
        tf.nest.map_structure(lambda ta: ta.stack(), x)
        for x in loop_results]


def extended_kalman_filter_one_step(
    state, observation, transition_fn, observation_fn,
    transition_jacobian_fn, observation_jacobian_fn, name=None):
  """A single step of the EKF.

  Args:
    state: A `Tensor` of shape
      `concat([[num_timesteps, b1, ..., bN], [state_size]])` with scalar
      `event_size` and optional batch dimensions `b1, ..., bN`.
    observation: A `Tensor` of shape
      `concat([[num_timesteps, b1, ..., bN], [event_size]])` with scalar
      `event_size` and optional batch dimensions `b1, ..., bN`.
    transition_fn: a Python `callable` that accepts (batched) vectors of length
      `state_size`, and returns a `tfd.Distribution` instance, typically a
      `MultivariateNormal`, representing the state transition and covariance.
    observation_fn: a Python `callable` that accepts a (batched) vector of
      length `state_size` and returns a `tfd.Distribution` instance, typically
      a `MultivariateNormal` representing the observation model and covariance.
    transition_jacobian_fn: a Python `callable` that accepts a (batched) vector
      of length `state_size` and returns a (batched) matrix of shape
      `[state_size, state_size]`, representing the Jacobian of `transition_fn`.
    observation_jacobian_fn: a Python `callable` that accepts a (batched) vector
      of length `state_size` and returns a (batched) matrix of size
      `[state_size, event_size]`, representing the Jacobian of `observation_fn`.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'extended_kalman_filter_one_step'`).
  Returns:
    updated_state: `KalmanFilterState` object containing the updated state
      estimate.
  """
  with tf.name_scope(name or 'extended_kalman_filter_one_step') as name:

    # If observations are scalar, we can avoid some matrix ops.
    observation_size_is_static_and_scalar = (observation.shape[-1] == 1)

    current_state = state.filtered_mean
    current_covariance = state.filtered_cov
    current_jacobian = transition_jacobian_fn(current_state)
    state_prior = transition_fn(current_state)

    predicted_cov = (tf.matmul(
        current_jacobian,
        tf.matmul(current_covariance, current_jacobian, transpose_b=True)) +
                     state_prior.covariance())
    predicted_mean = state_prior.mean()

    observation_dist = observation_fn(predicted_mean)
    observation_mean = observation_dist.mean()
    observation_cov = observation_dist.covariance()

    predicted_jacobian = observation_jacobian_fn(predicted_mean)
    tmp_obs_cov = tf.matmul(predicted_jacobian, predicted_cov)
    residual_covariance = tf.matmul(
        predicted_jacobian, tmp_obs_cov, transpose_b=True) + observation_cov

    if observation_size_is_static_and_scalar:
      gain_transpose = tmp_obs_cov / residual_covariance
    else:
      chol_residual_cov = tf.linalg.cholesky(residual_covariance)
      gain_transpose = tf.linalg.cholesky_solve(chol_residual_cov, tmp_obs_cov)

    filtered_mean = predicted_mean + tf.matmul(
        gain_transpose,
        (observation - observation_mean)[..., tf.newaxis],
        transpose_a=True)[..., 0]

    tmp_term = -tf.matmul(predicted_jacobian, gain_transpose, transpose_a=True)
    tmp_term = tf.linalg.set_diag(tmp_term, tf.linalg.diag_part(tmp_term) + 1.)
    filtered_cov = (
        tf.matmul(
            tmp_term, tf.matmul(predicted_cov, tmp_term), transpose_a=True) +
        tf.matmul(gain_transpose,
                  tf.matmul(observation_cov, gain_transpose), transpose_a=True))

    if observation_size_is_static_and_scalar:
      # A plain Normal would have event shape `[]`; wrapping with Independent
      # ensures `event_shape=[1]` as required.
      predictive_dist = independent.Independent(
          normal.Normal(loc=observation_mean,
                        scale=tf.sqrt(residual_covariance[..., 0])),
          reinterpreted_batch_ndims=1)

    else:
      predictive_dist = mvn_tril.MultivariateNormalTriL(
          loc=observation_mean,
          scale_tril=chol_residual_cov)

    log_marginal_likelihood = predictive_dist.log_prob(observation)

    return KalmanFilterState(
        filtered_mean=filtered_mean,
        filtered_cov=filtered_cov,
        predicted_mean=predicted_mean,
        predicted_cov=predicted_cov,
        observation_mean=observation_mean,
        observation_cov=observation_cov,
        log_marginal_likelihood=log_marginal_likelihood,
        timestep=state.timestep + 1)
