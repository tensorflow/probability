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

"""Utilities for Ensemble Adjustment Kalman Filtering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.sequential import ensemble_kalman_filter  # pylint:disable=line-too-long
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'ensemble_adjustment_kalman_filter_update'
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


def ensemble_adjustment_kalman_filter_update(
    state,
    observation,
    observation_fn,
    minimum_observation_prior_variance=None,
    name=None):
  """Ensemble Adjustment Kalman Filter Update.

  The Ensemble Adjustment Kalman Filter (EAKF) [1], is a deterministic variant
  of the [Ensemble Kalman Filter](
  https://en.wikipedia.org/wiki/Ensemble_Kalman_filter) [2].

  Specifically, the Ensemble Kalman Filter update step guarantees that the
  expectation of the ensemble covariance matches that of a usual Kalman filter
  update step. The EAKF modifies the update step so as to guarantee the ensemble
  covariance after updating matches that of the true posterior under a Linear
  Gaussian State Space Model. This update is also deterministic.

  This can reduce variance and yield better estimates compared to the
  Ensemble Kalman Filter. In the univariate observation case, this is about the
  same cost as an Ensemble Kalman Filter update, but the multivariate
  observation case will require 2 SVD computations per update.


  Args:
    state: Instance of `EnsembleKalmanFilterState`.
    observation: `Tensor` representing the observation for this timestep.
    observation_fn: callable returning an instance of
      `tfd.MultivariateNormalLinearOperator` along with an extra information
      to be returned in the `EnsembleKalmanFilterState`.
    minimum_observation_prior_variance: Python `Float`. If set, this will be
      minimum for the observation prior variance.
    name: Python `str` name for ops created by this method.
      Default value: `None`
      (i.e., `'ensemble_adjustment_kalman_filter_update'`).
  Returns:
    next_state: `EnsembleKalmanFilterState` representing particles at next
      timestep, after applying Kalman update equations.

  #### References

  [1] Jeffrey L. Anderson. An Ensemble Adjustment Kalman Filter for Data
      Assimilation. Monthly Weather Review, 2001.

  [2] Geir Evensen. Sequential data assimilation with a nonlinear
      quasi-geostrophic model using Monte Carlo methods to forecast error
      statistics. Journal of Geophysical Research, 1994.

  """
  with tf.name_scope(name or 'ensemble_adjustment_kalman_filter_update'):
    observation_prior_dist, extra = observation_fn(
        state.step, state.particles, state.extra)
    common_dtype = dtype_util.common_dtype(
        [observation_prior_dist, observation], dtype_hint=tf.float32)

    observation = tf.convert_to_tensor(observation, dtype=common_dtype)

    # We specialize the univariate case.
    # TODO(b/153489530): Add the multivariate case after debugging what is going
    # wrong with SVD calculations.
    if observation.shape[-1] != 1:
      raise NotImplementedError(
          'Ensemble Adjustment Kalman Filter Update not implemented '
          'for multi-variate distributions.')

    # Updates in the Ensemble Adjustment Kalman Filter are deterministic.
    observation_prior_particles = observation_prior_dist.mean()
    observation_prior_covariance = tf.squeeze(
        _covariance(observation_prior_particles), axis=-1)
    observation_noise_dist_covariance = tf.squeeze(
        observation_prior_dist.covariance(), axis=-1)
    if minimum_observation_prior_variance is not None:
      observation_prior_covariance = tf.where(
          tf.math.equal(observation_prior_covariance, 0.),
          minimum_observation_prior_variance,
          observation_prior_covariance)

    # For the Ensemble Adjustment Kalman filter, updates are
    # deterministic, so we don't add noise.

    # Compute posterior over observations using Bayes' rule.
    # TODO(b/153489530): Add to normal_conjugate_posteriors this update rule.
    # and reuse here.
    zero = dtype_util.as_numpy_dtype(observation_prior_covariance.dtype)(0.)
    one = dtype_util.as_numpy_dtype(observation_prior_covariance.dtype)(1.)

    observation_prior_mean = tf.reduce_mean(observation_prior_particles, axis=0)
    observation_posterior_covariance = tf.math.reciprocal(
        tf.math.reciprocal(observation_prior_covariance) +
        tf.math.reciprocal(observation_noise_dist_covariance))
    observation_posterior_covariance = tf.where(
        tf.math.equal(observation_prior_covariance, 0.),
        zero, observation_posterior_covariance)

    denominator = (
        observation_noise_dist_covariance + observation_prior_covariance)

    observation_posterior_mean = (
        observation_prior_mean * observation_noise_dist_covariance /
        denominator +
        observation * observation_prior_covariance / denominator)

    variance_ratio = tf.where(
        tf.math.equal(observation_prior_covariance, 0.),
        one, observation_posterior_covariance / observation_prior_covariance)

    observation_posterior_particles = (tf.math.sqrt(variance_ratio) * (
        observation_prior_particles - observation_prior_mean) +
                                       observation_posterior_mean)

    covariance_between_state_and_predicted_observations = tf.nest.map_structure(
        lambda x: tf.squeeze(_covariance(  # pylint:disable=g-long-lambda
            x, observation_prior_particles), axis=-1), state.particles)

    observation_diff = (
        observation_posterior_particles - observation_prior_particles)

    deltas = tf.nest.map_structure(
        lambda g: g / observation_prior_covariance * observation_diff,
        covariance_between_state_and_predicted_observations)
    deltas = tf.nest.map_structure(
        lambda d: tf.where(  # pylint: disable=g-long-lambda
            tf.math.equal(observation_prior_covariance, 0.),
            observation_diff, d), deltas)

    new_particles = tf.nest.map_structure(
        lambda x, d: x + d, state.particles, deltas)

    return ensemble_kalman_filter.EnsembleKalmanFilterState(
        step=state.step, particles=new_particles, extra=extra)
