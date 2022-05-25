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

import collections
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'EnsembleKalmanFilterState',
    'ensemble_kalman_filter_predict',
    'ensemble_kalman_filter_update',
    'ensemble_kalman_filter_log_marginal_likelihood',
    'inflate_by_scaled_identity_fn',
]


# Sample covariance. Handles differing shapes.
def _covariance(x, y=None):
  """Sample covariance, assuming samples are the leftmost axis."""
  x = tf.convert_to_tensor(x, name='x')
  # Covariance *only* uses the centered versions of x (and y).
  x = x - tf.reduce_mean(x, axis=0)

  if y is None:
    y = x
  else:
    y = tf.convert_to_tensor(y, name='y', dtype=x.dtype)
    y = y - tf.reduce_mean(y, axis=0)

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
  """Ensemble Kalman filter prediction step.

  The [Ensemble Kalman Filter](
  https://en.wikipedia.org/wiki/Ensemble_Kalman_filter) is a Monte Carlo
  version of the traditional Kalman Filter. See also [2]. It assumes the model

  ```
  X[t] ~ transition_fn(X[t-1])
  Y[t] ~ observation_fn(X[t])
  ```

  Given the ensemble `state.particles` sampled from `P(X[t-1] | Y[t-1])`, this
  function produces the predicted (a.k.a. forecast or background) ensemble
  sampled from `P(X[t] | Y[t-1])`. This is the predicted next state *before*
  assimilating the observation `Y[t]`.

  Typically, with `F` some deterministic mapping, `transition_fn(X)` returns a
  normal distribution centered at `F(X)`.

  Args:
    state: Instance of `EnsembleKalmanFilterState`.
    transition_fn: callable returning a (joint) distribution over the next
      latent state, and any information in the `extra` state.
      Each component should be an instance of
      `MultivariateNormalLinearOperator`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    inflate_fn: Function that takes in the `particles` and returns a new set of
      `particles`. Used for inflating the covariance of points. Note this
      function should try to preserve the sample mean of the particles, and
      scale up the sample covariance [3].
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'ensemble_kalman_filter_predict'`).

  Returns:
    next_state: `EnsembleKalmanFilterState` representing particles after
      applying `transition_fn`.

  #### References

  [1] Geir Evensen. Sequential data assimilation with a nonlinear
      quasi-geostrophic model using Monte Carlo methods to forecast error
      statistics. Journal of Geophysical Research, 1994.

  [2] Matthias Katzfuss, Jonathan R. Stroud & Christopher K. Wikle
      Understanding the Ensemble Kalman Filter.
      The Americal Statistician, 2016.

  [3] Jeffrey L. Anderson and Stephen L. Anderson. A Monte Carlo Implementation
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
  """Ensemble Kalman filter update step.

  The [Ensemble Kalman Filter](
  https://en.wikipedia.org/wiki/Ensemble_Kalman_filter) is a Monte Carlo
  version of the traditional Kalman Filter. See also [2]. It assumes the model

  ```
  X[t] ~ transition_fn(X[t-1])
  Y[t] ~ observation_fn(X[t])
  ```

  Given the ensemble `state.particles` sampled from `P(X[t] | Y[t-1])`, this
  function assimilates obervation `Y[t]` to produce the updated ensemble sampled
  from `P(X[t] | Y[t])`.

  Typically, with `G` some deterministic observation mapping,
  `observation_fn(X)` returns a normal distribution centered at `G(X)`.

  Args:
    state: Instance of `EnsembleKalmanFilterState`.
    observation: `Tensor` representing the observation for this timestep.
    observation_fn: callable returning an instance of
      `tfd.MultivariateNormalLinearOperator` along with an extra information
      to be returned in the `EnsembleKalmanFilterState`.
    damping: Floating-point `Tensor` representing how much to damp the
      update by. Used to mitigate filter divergence. Default value: 1.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'ensemble_kalman_filter_update'`).

  Returns:
    next_state: `EnsembleKalmanFilterState` representing particles at next
      timestep, after applying Kalman update equations.

  #### References

  [1] Geir Evensen. Sequential data assimilation with a nonlinear
      quasi-geostrophic model using Monte Carlo methods to forecast error
      statistics. Journal of Geophysical Research, 1994.

  [2] Matthias Katzfuss, Jonathan R. Stroud & Christopher K. Wikle
      Understanding the Ensemble Kalman Filter.
      The Americal Statistician, 2016.
  """

  with tf.name_scope(name or 'ensemble_kalman_filter_update'):
    # In the example below, we let
    #  Y be the real observations, so Y.shape = [observation_size]
    #  X be an ensemble particles with X.shape = [ensemble_size, state_size]
    #  G be the observation function,
    #   so G(X).shape = [ensemble_size, state_size].
    # In practice, batch dims may appear between the ensemble and state dims.

    # In the traditional EnKF, observation_particles_dist ~ N(G(X), Γ).
    # However, our API would allow any Gaussian.
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

    # predicted_observation_particles = G(X) + E[η],
    # and is shape [n_ensemble] + [observation_size]
    # Note that .mean() is the distribution mean, and the distribution is
    # centered at the predicted observations. This is *not* the ensemble mean.
    predicted_observation_particles = observation_particles_dist.mean()

    # With μ the ensemble average operator. For V a batch of column vectors,
    # let Vᵀ be a batch of row vectors.
    # Cov(G(X)) = (G(X) - μ(G(X))) (G(X) - μ(G(X)))ᵀ
    observation_particles_covariance = _covariance(
        predicted_observation_particles)

    # covariance_between_state_and_predicted_observations
    # Cov(X, G(X))  = (X - μ(X))(G(X) - μ(G(X)))ᵀ
    covariance_between_state_and_predicted_observations = tf.nest.map_structure(
        lambda x: _covariance(x, predicted_observation_particles),
        state.particles)

    # observation_particles_diff = Y - G(X) - η
    observation_particles_diff = (
        observation - observation_particles_dist.sample(seed=seed))

    # = Cov(G(X)) + Γ
    observation_particles_covariance = (
        observation_particles_covariance +
        # Calling _linop_covariance(...).to_dense() rather than
        # observation_particles_dist.covariance() means the shape is
        # [observation_size, observation_size] rather than
        # [ensemble_size] + [observation_size, observation_size].
        # Both work, since this matrix is used to do mat-vecs with ensembles
        # of vectors...however, doing things this way ensures we do an
        # efficient batch-matmul and (more importantly) don't have to do a
        # separate Cholesky for every ensemble member!
        _linop_covariance(observation_particles_dist).to_dense())

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

      # added_term = [Cov(G(X)) + Γ]⁻¹ [Y - G(X) - η]
      observation_particles_cholesky = tf.linalg.cholesky(
          observation_particles_covariance)
      added_term = tf.squeeze(tf.linalg.cholesky_solve(
          observation_particles_cholesky,
          observation_particles_diff[..., tf.newaxis]), axis=-1)

      # added_term
      #  = covariance_between_state_and_predicted_observations @ added_term
      #  = Cov(X, G(X)) [Cov(G(X)) + Γ]⁻¹ [Y - G(X) - η]
      #  = (X - μ(X))(G(X) - μ(G(X)))ᵀ [Cov(G(X)) + Γ]⁻¹ [Y - G(X) - η]
      added_term = tf.nest.map_structure(
          lambda x: tf.linalg.matvec(x, added_term),
          covariance_between_state_and_predicted_observations)

      # new_particles = X + damping * added_term
      new_particles = tf.nest.map_structure(
          lambda x, a: x + damping * a, state.particles, added_term)

    return EnsembleKalmanFilterState(
        step=state.step + 1, particles=new_particles, extra=extra)


def ensemble_kalman_filter_log_marginal_likelihood(
    state,
    observation,
    observation_fn,
    seed=None,
    name=None):
  """Ensemble Kalman filter log marginal likelihood.

  The [Ensemble Kalman Filter](
  https://en.wikipedia.org/wiki/Ensemble_Kalman_filter) is a Monte Carlo
  version of the traditional Kalman Filter. See also [2]. It assumes the model

  ```
  X[t] ~ transition_fn(X[t-1])
  Y[t] ~ observation_fn(X[t])
  ```

  This method estimates (logarithm of) the marginal likelihood of the
  observation at step `t`, `Y[t]`, given `state`. Typically, `state` is the
  predictive ensemble at time `t`. In that case, this function approximates
   `Log[p(Y[t] | Y[t-1], Y[t-2],...)]`
  The approximation is correct under a Linear Gaussian state space model
  assumption, as ensemble size --> infinity.

  Args:
    state: Instance of `EnsembleKalmanFilterState` at step `k`,
      conditioned on previous observations `Y_{1:k}`. Typically this is the
      output of `ensemble_kalman_filter_predict`.
    observation: `Tensor` representing the observation at step `k`.
    observation_fn: callable returning an instance of
      `tfd.MultivariateNormalLinearOperator` along with an extra information
      to be returned in the `EnsembleKalmanFilterState`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name for ops created by this method.
      Default value: `None`
      (i.e., `'ensemble_kalman_filter_log_marginal_likelihood'`).

  Returns:
    log_marginal_likelihood: `Tensor` with same dtype as `state`.

  #### References

  [1] Geir Evensen. Sequential data assimilation with a nonlinear
      quasi-geostrophic model using Monte Carlo methods to forecast error
      statistics. Journal of Geophysical Research, 1994.

  [2] Matthias Katzfuss, Jonathan R. Stroud & Christopher K. Wikle
      Understanding the Ensemble Kalman Filter.
      The Americal Statistician, 2016.
  """

  with tf.name_scope(name or 'ensemble_kalman_filter_log_marginal_likelihood'):
    observation_particles_dist, unused_extra = observation_fn(
        state.step, state.particles, state.extra)

    common_dtype = dtype_util.common_dtype(
        [observation_particles_dist, observation], dtype_hint=tf.float32)

    observation = tf.convert_to_tensor(observation, dtype=common_dtype)

    if not isinstance(observation_particles_dist,
                      distributions.MultivariateNormalLinearOperator):
      raise ValueError('Expected `observation_fn` to return an instance of '
                       '`MultivariateNormalLinearOperator`')

    observation_particles = observation_particles_dist.sample(seed=seed)
    observation_dist = distributions.MultivariateNormalTriL(
        loc=tf.reduce_mean(observation_particles, axis=0),
        scale_tril=tf.linalg.cholesky(_covariance(observation_particles)))

    return observation_dist.log_prob(observation)


def _linop_covariance(dist):
  """LinearOperator backing Cov(dist), without unnecessary broadcasting."""
  # This helps, even if we immediately call .to_dense(). Why?
  # Simply calling dist.covariance() would broadcast up to the full batch shape.
  # Instead, we want the shape to be that of the linear operator only.
  # This (i) saves memory and (ii) allows operations done with this operator
  # to be more efficient.
  if hasattr(dist, 'cov_operator'):
    cov = dist.cov_operator
  else:
    cov = dist.scale.matmul(dist.scale.H)
  # TODO(b/132466537) composition doesn't preserve SPD so we have to hard-set.
  cov._is_positive_definite = True  # pylint: disable=protected-access
  cov._is_self_adjoint = True  # pylint: disable=protected-access
  return cov
