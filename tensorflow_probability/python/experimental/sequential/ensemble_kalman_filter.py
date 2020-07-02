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
"""Utilities for Ensemble Kalman Filtering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'EnsembleKalmanFilterState',
    'ensemble_kalman_filter_predict',
    'ensemble_kalman_filter_update',
    'inflate_by_scaled_identity_fn',
]


# Sample covariance. Handles differing shapes.
def _covariance(x, y=None):
  """Sample covariance, assuming samples are the leftmost axis."""
  x = tf.convert_to_tensor(x, name='x')
  # Covariance *only* uses the centered versions of x (and y).
  x -= tf.reduce_mean(x, axis=0)

  if y is None:
    y = x
  else:
    y = tf.convert_to_tensor(y, name='y', dtype=x.dtype)
    y -= tf.reduce_mean(y, axis=0)

  return tf.reduce_mean(tf.linalg.matmul(
      x[..., tf.newaxis],
      y[..., tf.newaxis], adjoint_b=True), axis=0)


class EnsembleKalmanFilterState(collections.namedtuple(
    'EnsembleKalmanFilterState', ['step', 'particles', 'extra'])):
  """State for Ensemble Kalman Filter algorithms.

  Contents:
    step: Scalar `Integer` tensor. The timestep associated with this state.
    particles: Structure of Floating-point `Tensor`s of shape
      [N, B1, ... Bn, Ek] where `N` is the number of particles in the ensemble,
      `Bi` are the batch dimensions and `Ek` is the size of each state.
    extra: Structure containing any additional information. Can be used
      for keeping track of diagnostics or propagating side information to
      the Ensemble Kalman Filter.
  """
  pass


def inflate_by_scaled_identity_fn(scaling_factor):
  """Return function that scales up covariance matrix by `scaling_factor**2`."""
  def _inflate_fn(particles):
    particle_means = tf.nest.map_structure(
        lambda x: tf.math.reduce_mean(x, axis=0), particles)
    return tf.nest.map_structure(
        lambda x, m: scaling_factor * (x - m) + m,
        particles,
        particle_means)
  return _inflate_fn


def ensemble_kalman_filter_predict(
    state,
    transition_fn,
    seed=None,
    inflate_fn=None,
    name=None):
  """Ensemble Kalman Filter Prediction.

  The [Ensemble Kalman Filter](
  https://en.wikipedia.org/wiki/Ensemble_Kalman_filter) is a Monte Carlo
  version of the traditional Kalman Filter.

  This method is the 'prediction' equation associated with the Ensemble
  Kalman Filter. This takes in an optional `inflate_fn` to perform covariance
  inflation on the ensemble [2].

  Args:
    state: Instance of `EnsembleKalmanFilterState`.
    transition_fn: callable returning a (joint) distribution over the next
      latent state, and any information in the `extra` state.
      Each component should be an instance of
      `MultivariateNormalLinearOperator`.
    seed: Python `int` seed for random ops.
    inflate_fn: Function that takes in the `particles` and returns a
      new set of `particles`. Used for inflating the covariance of points.
      Note this function should try to preserve the sample mean of the
      particles, and scale up the sample covariance.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'ensemble_kalman_filter_predict'`).
  Returns:
    next_state: `EnsembleKalmanFilterState` representing particles after
      applying `transition_fn`.

  #### References

  [1] Geir Evensen. Sequential data assimilation with a nonlinear
      quasi-geostrophic model using Monte Carlo methods to forecast error
      statistics. Journal of Geophysical Research, 1994.

  [2] Jeffrey L. Anderson and Stephen L. Anderson. A Monte Carlo Implementation
      of the Nonlinear Filtering Problem to Produce Ensemble Assimilations and
      Forecasts. Monthly Weather Review, 1999.

  """
  with tf.name_scope(name or 'ensemble_kalman_filter_predict'):
    if inflate_fn is not None:
      state = EnsembleKalmanFilterState(
          step=state.step,
          particles=inflate_fn(state.particles),
          extra=state.extra)

    new_particles_dist, extra = transition_fn(
        state.step, state.particles, state.extra)

    return EnsembleKalmanFilterState(
        step=state.step, particles=new_particles_dist.sample(
            seed=seed), extra=extra)


def ensemble_kalman_filter_update(
    state,
    observation,
    observation_fn,
    damping=1.,
    seed=None,
    name=None):
  """Ensemble Kalman Filter Update.

  The [Ensemble Kalman Filter](
  https://en.wikipedia.org/wiki/Ensemble_Kalman_filter) is a Monte Carlo
  version of the traditional Kalman Filter.

  This method is the 'update' equation associated with the Ensemble
  Kalman Filter. In expectation, the ensemble covariance will match that
  of the true posterior (under a Linear Gaussian State Space Model).

  Args:
    state: Instance of `EnsembleKalmanFilterState`.
    observation: `Tensor` representing the observation for this timestep.
    observation_fn: callable returning an instance of
      `tfd.MultivariateNormalLinearOperator` along with an extra information
      to be returned in the `EnsembleKalmanFilterState`.
    damping: Floating-point `Tensor` representing how much to damp the
      update by. Used to mitigate filter divergence. Default value: 1.
    seed: Python `int` seed for random ops.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'ensemble_kalman_filter_update'`).
  Returns:
    next_state: `EnsembleKalmanFilterState` representing particles at next
      timestep, after applying Kalman update equations.
  """

  with tf.name_scope(name or 'ensemble_kalman_filter_update'):
    observation_particles_dist, extra = observation_fn(
        state.step, state.particles, state.extra)

    common_dtype = dtype_util.common_dtype(
        [observation_particles_dist, observation], dtype_hint=tf.float32)

    observation = tf.convert_to_tensor(observation, dtype=common_dtype)
    observation_size_is_static_and_scalar = (observation.shape[-1] == 1)

    if not isinstance(observation_particles_dist,
                      distributions.MultivariateNormalLinearOperator):
      raise ValueError('Expected `observation_fn` to return an instance of '
                       '`MultivariateNormalLinearOperator`')

    observation_particles = observation_particles_dist.sample(seed=seed)
    observation_particles_covariance = _covariance(observation_particles)

    covariance_between_state_and_predicted_observations = tf.nest.map_structure(
        lambda x: _covariance(x, observation_particles), state.particles)

    observation_particles_diff = observation - observation_particles

    observation_particles_covariance = (
        observation_particles_covariance +
        observation_particles_dist.covariance())

    # We specialize the univariate case.
    # TODO(srvasude): Refactor linear_gaussian_ssm, normal_conjugate_posteriors
    # and this code so we have a central place for normal conjugacy code.
    if observation_size_is_static_and_scalar:
      # In the univariate observation case, the Kalman gain is given by:
      # K = cov(X, Y) / (var(Y) + var_noise). That is we just divide
      # by the particle covariance plus the observation noise.
      kalman_gain = tf.nest.map_structure(
          lambda x: x / observation_particles_covariance,
          covariance_between_state_and_predicted_observations)
      new_particles = tf.nest.map_structure(
          lambda x, g: x + damping * tf.linalg.matvec(  # pylint:disable=g-long-lambda
              g, observation_particles_diff), state.particles, kalman_gain)
    else:
      # TODO(b/153489530): Handle the case where the dimensionality of the
      # observations is large. We can use the Sherman-Woodbury-Morrison
      # identity in this case.

      observation_particles_cholesky = tf.linalg.cholesky(
          observation_particles_covariance)
      added_term = tf.squeeze(tf.linalg.cholesky_solve(
          observation_particles_cholesky,
          observation_particles_diff[..., tf.newaxis]), axis=-1)

      added_term = tf.nest.map_structure(
          lambda x: tf.linalg.matvec(x, added_term),
          covariance_between_state_and_predicted_observations)
      new_particles = tf.nest.map_structure(
          lambda x, a: x + damping * a, state.particles, added_term)

    return EnsembleKalmanFilterState(
        step=state.step + 1, particles=new_particles, extra=extra)
