# Copyright 2018 The TensorFlow Probability Authors.
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
"""Seasonal model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


seasonal_init_args = """
    Args:
      num_timesteps: Scalar `int` `Tensor` number of timesteps to model
        with this distribution.
      num_seasons: Scalar Python `int` number of seasons.
      drift_scale: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the standard deviation of the
        change in effect between consecutive occurrences of a given season.
        This is assumed to be the same for all seasons.
      initial_state_prior: instance of `tfd.MultivariateNormal`
        representing the prior distribution on latent states; must
        have event shape `[num_seasons]`.
      observation_noise_scale: Scalar (any additional dimensions are
        treated as batch dimensions) `float` `Tensor` indicating the standard
        deviation of the observation noise.
        Default value: 0.
      num_steps_per_season: Python `int` number of steps in each
        season. This may be either a scalar (shape `[]`), in which case all
        seasons have the same length, or a NumPy array of shape `[num_seasons]`,
        in which seasons have different length, but remain constant around
        different cycles, or a NumPy array of shape `[num_cycles, num_seasons]`,
        in which num_steps_per_season for each season also varies in different
        cycle (e.g., a 4 years cycle with leap day).
        Default value: 1.
      initial_step: Optional scalar `int` `Tensor` specifying the starting
        timestep.
        Default value: 0.
      validate_args: Python `bool`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
        Default value: `False`.
      allow_nan_stats: Python `bool`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
        Default value: `True`.
      name: Python `str` name prefixed to ops created by this class.
        Default value: "SeasonalStateSpaceModel".

    Raises:
      ValueError: if `num_steps_per_season` has invalid shape (neither
        scalar nor `[num_seasons]`).
    """


class SeasonalStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
  """State space model for a seasonal effect.

    A state space model (SSM) posits a set of latent (unobserved) variables that
    evolve over time with dynamics specified by a probabilistic transition model
    `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
    observation model conditioned on the current state, `p(x[t] | z[t])`. The
    special case where both the transition and observation models are Gaussians
    with mean specified as a linear function of the inputs, is known as a linear
    Gaussian state space model and supports tractable exact probabilistic
    calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
    details.

    A seasonal effect model is a special case of a linear Gaussian SSM. The
    latent states represent an unknown effect from each of several 'seasons';
    these are generally not meteorological seasons, but represent regular
    recurring patterns such as hour-of-day or day-of-week effects. The effect of
    each season drifts from one occurrence to the next, following a Gaussian
    random walk:

    ```python
    effects[season, occurrence[i]] = (
      effects[season, occurrence[i-1]] + Normal(loc=0., scale=drift_scale))
    ```

    The latent state has dimension `num_seasons`, containing one effect for each
    seasonal component. The parameters `drift_scale` and
    `observation_noise_scale` are each (a batch of) scalars. The batch shape of
    this `Distribution` is the broadcast batch shape of these parameters and of
    the `initial_state_prior`.

    Note: there is no requirement that the effects sum to zero.

    #### Mathematical Details

    The seasonal effect model implements a
    `tfp.distributions.LinearGaussianStateSpaceModel` with
    `latent_size = num_seasons` and `observation_size = 1`. The latent state
    is organized so that the *current* seasonal effect is always in the first
    (zeroth) dimension. The transition model rotates the latent state to shift
    to a new effect at the end of each season:

    ```
    transition_matrix[t] = (permutation_matrix([1, 2, ..., num_seasons-1, 0])
                            if season_is_changing(t)
                            else eye(num_seasons)
    transition_noise[t] ~ Normal(loc=0., scale_diag=(
                                 [drift_scale, 0, ..., 0]
                                 if season_is_changing(t)
                                 else [0, 0, ..., 0]))
    ```

    where `season_is_changing(t)` is `True` if ``t `mod`
    sum(num_steps_per_season)`` is in the set of final days for each season,
    given by `cumsum(num_steps_per_season) - 1`. The observation model always
    picks out the effect for the current season, i.e., the first element of
    the latent state:

    ```
    observation_matrix = [[1., 0., ..., 0.]]
    observation_noise ~ Normal(loc=0, scale=observation_noise_scale)
    ```

    #### Examples

    A state-space model with day-of-week seasonality on hourly data:

    ```python
    day_of_week = SeasonalStateSpaceModel(
      num_timesteps=30,
      num_seasons=7,
      drift_scale=0.1,
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([7], dtype=tf.float32),
      num_steps_per_season=24)
    ```

    A model with basic month-of-year seasonality on daily data, demonstrating
    seasons of varying length:

    ```python
    month_of_year = SeasonalStateSpaceModel(
      num_timesteps=2 * 365,  # 2 years
      num_seasons=12,
      drift_scale=0.1,
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([12], dtype=tf.float32)),
      num_steps_per_season=[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
      initial_step=22)
    ```

    Note that we've used `initial_step=22` to denote that the model begins
    on January 23 (steps are zero-indexed). This version works over time periods
    not involving a leap year. A general implementation of month-of-year
    seasonality would require additional logic:

    ```python
    num_days_per_month = np.array(
      [[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
       [31, 29, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],  # year with leap day
       [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
       [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31]])

    month_of_year = SeasonalStateSpaceModel(
      num_timesteps=4 * 365 + 2,  # 8 years with leap days
      num_seasons=12,
      drift_scale=0.1,
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([12], dtype=tf.float32)),
      num_steps_per_season=num_days_per_month,
      initial_step=22)
    ```

  """

  @docstring_util.expand_docstring(seasonal_init_args=seasonal_init_args)
  def __init__(self,
               num_timesteps,
               num_seasons,
               drift_scale,
               initial_state_prior,
               observation_noise_scale=0.,
               num_steps_per_season=1,
               initial_step=0,
               validate_args=False,
               allow_nan_stats=True,
               name=None):  # pylint: disable=g-doc-args
    """Build a seasonal effect state space model.

    {seasonal_init_args}
    """

    with tf.name_scope(name or 'SeasonalStateSpaceModel') as name:
      # The initial state prior determines the dtype of sampled values.
      # Other model parameters must have the same dtype.
      dtype = initial_state_prior.dtype
      drift_scale = tf.convert_to_tensor(
          value=drift_scale, name='drift_scale', dtype=dtype)
      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          name='observation_noise_scale',
          dtype=dtype)

      # Coerce `num_steps_per_season` to a canonical form, an array of
      # `num_seasons` integers.
      num_steps_per_season = np.squeeze(np.asarray(num_steps_per_season))
      if num_steps_per_season.ndim == 0:  # scalar case
        num_steps_per_season = np.tile(num_steps_per_season, num_seasons)
      elif ((num_steps_per_season.ndim <= 2)  # 1D and 2D case
            and (num_steps_per_season.shape[-1] != num_seasons)):
        raise ValueError('num_steps_per_season must either be scalar (shape [])'
                         ' or have the last dimension equal to [num_seasons] = '
                         '[{}] (saw: shape {})'.format(
                             num_seasons, num_steps_per_season.shape))

      is_last_day_of_season = build_is_last_day_of_season(num_steps_per_season)

      seasonal_transition_matrix = build_seasonal_transition_matrix(
          num_seasons=num_seasons,
          is_last_day_of_season=is_last_day_of_season,
          dtype=dtype)

      seasonal_transition_noise = build_seasonal_transition_noise(
          drift_scale, num_seasons, is_last_day_of_season)

      observation_matrix = tf.concat([
          tf.ones([1, 1], dtype=dtype),
          tf.zeros([1, num_seasons-1], dtype=dtype)], axis=-1)

      self._drift_scale = drift_scale
      self._observation_noise_scale = observation_noise_scale
      self._num_seasons = num_seasons
      self._num_steps_per_season = num_steps_per_season

      super(SeasonalStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=seasonal_transition_matrix,
          transition_noise=seasonal_transition_noise,
          observation_matrix=observation_matrix,
          observation_noise=tfd.MultivariateNormalDiag(
              scale_diag=observation_noise_scale[..., tf.newaxis]),
          initial_state_prior=initial_state_prior,
          initial_step=initial_step,
          allow_nan_stats=allow_nan_stats,
          validate_args=validate_args,
          name=name)

  @property
  def drift_scale(self):
    """Standard deviation of the drift in effects between seasonal cycles."""
    return self._drift_scale

  @property
  def observation_noise_scale(self):
    """Standard deviation of the observation noise."""
    return self._observation_noise_scale

  @property
  def num_seasons(self):
    """Number of seasons."""
    return self._num_seasons

  @property
  def num_steps_per_season(self):
    """Number of steps in each season."""
    return self._num_steps_per_season


class ConstrainedSeasonalStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
  """Seasonal state space model with effects constrained to sum to zero.

    See `SeasonalStateSpaceModel` for background.

    #### Mathematical details

    The constrained model implements a reparameterization of the
    naive `SeasonalStateSpaceModel`. Instead of directly representing the
    seasonal effects in the latent space, the latent space of the constrained
    model represents the difference between each effect and the mean effect.
    The following discussion assumes familiarity with the mathematical details
    of `SeasonalStateSpaceModel`.

    *Reparameterization and constraints*: let the seasonal effects at a given
    timestep be `E = [e_1, ..., e_N]`. The difference between each effect `e_i`
    and the mean effect is `z_i = e_i - sum_i(e_i)/N`. By itself, this
    transformation is not invertible because recovering the absolute effects
    requires that we know the mean as well. To fix this, we'll define
    `z_N = sum_i(e_i)/N` as the mean effect. It's easy to see that this is
    invertible: given the mean effect and the differences of the first `N - 1`
    effects from the mean, it's easy to solve for all `N` effects. Formally,
    we've defined the invertible linear reparameterization `Z = R E`, where

    ```
    R = [1 - 1/N, -1/N,    ..., -1/N
         -1/N,    1 - 1/N, ..., -1/N,
         ...
         1/N,     1/N,     ...,  1/N]
    ```

    represents the change of basis from 'effect coordinates' E to
    'residual coordinates' Z. The `Z`s form the latent space of the
    `ConstrainedSeasonalStateSpaceModel`.

    To constrain the mean effect `z_N` to zero, we fix the prior to zero,
    `p(z_N) ~ N(0., 0)`, and after the transition at each timestep we project
    `z_N` back to zero. Note that this projection is linear: to set the Nth
    dimension to zero, we simply multiply by the identity matrix with a missing
    element in the bottom right, i.e., `Z_constrained = P Z`,
    where `P = eye(N) - scatter((N-1, N-1), 1)`.

    *Model*: concretely, suppose a naive seasonal effect model has initial state
    prior `N(m, S)`, transition matrix `F` and noise covariance
    `Q`, and observation matrix `H`. Then the corresponding constrained seasonal
    effect model has initial state prior `N(P R m, P R S R' P')`,
    transition matrix `P R F R^-1` and noise covariance `F R Q R' F'`, and
    observation matrix `H R^-1`, where the change-of-basis matrix `R` and
    constraint projection matrix `P` are as defined above. This follows
    directly from applying the reparameterization `Z = R E`, and then enforcing
    the zero-sum constraint on the prior and transition noise covariances.

    In practice, because the sum of effects `z_N` is constrained to be zero, it
    will never contribute a term to any linear operation on the latent space,
    so we can drop that dimension from the model entirely.
    `ConstrainedSeasonalStateSpaceModel` does this, so that it implements the
    `N - 1` dimension latent space `z_1, ..., z_[N-1]`.

    Note that since we constrained the mean effect to be zero, the latent
    `z_i`'s now recover their interpretation as the *actual* effects,
    `z_i = e_i` for `i = `1, ..., N - 1`, even though they were originally
    defined as residuals. The `N`th effect is represented only implicitly, as
    the nonzero mean of the first `N - 1` effects. Although the computational
    represention is not symmetric across all `N` effects, we derived the
    `ConstrainedSeasonalStateSpaceModel` by starting with a symmetric
    representation and imposing only a symmetric constraint (the zero-sum
    constraint), so the probability model remains symmetric over all `N`
    seasonal effects.

    #### Examples

    A constrained state-space model with day-of-week seasonality on hourly data:

    ```python
    day_of_week = ConstrainedSeasonalStateSpaceModel(
      num_timesteps=30,
      num_seasons=7,
      drift_scale=0.1,
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([7-1], dtype=tf.float32)),
      num_steps_per_season=24)
    ```

    A model with basic month-of-year seasonality on daily data, demonstrating
    seasons of varying length:

    ```python
    month_of_year = ConstrainedSeasonalStateSpaceModel(
      num_timesteps=2 * 365,  # 2 years
      num_seasons=12,
      drift_scale=0.1,
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([12-1], dtype=tf.float32)),
      num_steps_per_season=[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
      initial_step=22)
    ```

    Note that we've used `initial_step=22` to denote that the model begins
    on January 23 (steps are zero-indexed). This version works over time periods
    not involving a leap year. A general implementation of month-of-year
    seasonality would require additional logic:

    ```python
    num_days_per_month = np.array(
      [[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
       [31, 29, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],  # year with leap day
       [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
       [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31]])

    month_of_year = ConstrainedSeasonalStateSpaceModel(
      num_timesteps=4 * 365 + 2,  # 8 years with leap days
      num_seasons=12,
      drift_scale=0.1,
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([12-1], dtype=tf.float32)),
      num_steps_per_season=num_days_per_month,
      initial_step=22)
    ```
  """

  def __init__(self,
               num_timesteps,
               num_seasons,
               drift_scale,
               initial_state_prior,
               observation_noise_scale=1e-4,  # Avoid degeneracy.
               num_steps_per_season=1,
               initial_step=0,
               validate_args=False,
               allow_nan_stats=True,
               name=None):  # pylint: disable=g-doc-args
    """Build a seasonal effect state space model with a zero-sum constraint.

    {seasonal_init_args}
    """

    with tf.name_scope(name or 'ConstrainedSeasonalStateSpaceModel') as name:

      # The initial state prior determines the dtype of sampled values.
      # Other model parameters must have the same dtype.
      dtype = initial_state_prior.dtype
      drift_scale = tf.convert_to_tensor(
          value=drift_scale, name='drift_scale', dtype=dtype)
      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          name='observation_noise_scale',
          dtype=dtype)

      # Coerce `num_steps_per_season` to a canonical form, an array of
      # `num_seasons` integers.
      num_steps_per_season = np.squeeze(np.asarray(num_steps_per_season))
      if num_steps_per_season.ndim == 0:  # scalar case
        num_steps_per_season = np.tile(num_steps_per_season, num_seasons)
      elif ((num_steps_per_season.ndim <= 2)  # 1D and 2D case
            and (num_steps_per_season.shape[-1] != num_seasons)):
        raise ValueError('num_steps_per_season must either be scalar (shape [])'
                         ' or have the last dimension equal to [num_seasons] = '
                         '[{}] (saw: shape {})'.format(
                             num_seasons, num_steps_per_season.shape))

      is_last_day_of_season = build_is_last_day_of_season(num_steps_per_season)

      [
          effects_to_residuals,
          residuals_to_effects
      ] = build_effects_to_residuals_matrix(num_seasons, dtype=dtype)

      seasonal_transition_matrix = build_seasonal_transition_matrix(
          num_seasons=num_seasons,
          is_last_day_of_season=is_last_day_of_season,
          dtype=dtype,
          basis_change_matrix=effects_to_residuals,
          basis_change_matrix_inv=residuals_to_effects)

      seasonal_transition_noise = build_constrained_seasonal_transition_noise(
          drift_scale, num_seasons, is_last_day_of_season)

      observation_matrix = tf.concat(
          [tf.ones([1, 1], dtype=dtype),
           tf.zeros([1, num_seasons-1], dtype=dtype)], axis=-1)
      observation_matrix = tf.matmul(observation_matrix, residuals_to_effects)

      self._drift_scale = drift_scale
      self._observation_noise_scale = observation_noise_scale
      self._num_seasons = num_seasons
      self._num_steps_per_season = num_steps_per_season

      super(ConstrainedSeasonalStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=seasonal_transition_matrix,
          transition_noise=seasonal_transition_noise,
          observation_matrix=observation_matrix,
          observation_noise=tfd.MultivariateNormalDiag(
              scale_diag=observation_noise_scale[..., tf.newaxis]),
          initial_state_prior=initial_state_prior,
          initial_step=initial_step,
          allow_nan_stats=allow_nan_stats,
          validate_args=validate_args,
          name=name)

  @property
  def drift_scale(self):
    """Standard deviation of the drift in effects between seasonal cycles."""
    return self._drift_scale

  @property
  def observation_noise_scale(self):
    """Standard deviation of the observation noise."""
    return self._observation_noise_scale

  @property
  def num_seasons(self):
    """Number of seasons."""
    return self._num_seasons

  @property
  def num_steps_per_season(self):
    """Number of steps in each season."""
    return self._num_steps_per_season


def build_is_last_day_of_season(num_steps_per_season):
  """Build utility method to compute whether the season is changing."""
  num_steps_per_cycle = np.sum(num_steps_per_season)
  changepoints = np.cumsum(np.ravel(num_steps_per_season)) - 1
  def is_last_day_of_season(t):
    t_ = dist_util.maybe_get_static_value(t)
    if t_ is not None:  # static case
      step_in_cycle = t_ % num_steps_per_cycle
      return any(step_in_cycle == changepoints)
    else:
      step_in_cycle = tf.math.floormod(t, num_steps_per_cycle)
      return tf.reduce_any(tf.equal(step_in_cycle, changepoints))
  return is_last_day_of_season


def build_effects_to_residuals_matrix(num_seasons, dtype):
  """Build change-of-basis matrices for constrained seasonal effects.

  This method builds the matrix that transforms seasonal effects into
  effect residuals (differences from the mean effect), and additionally
  projects these residuals onto the subspace where the mean effect is zero.

  See `ConstrainedSeasonalStateSpaceModel` for mathematical details.

  Args:
    num_seasons: scalar `int` number of seasons.
    dtype: TensorFlow `dtype` for the returned values.
  Returns:
    effects_to_residuals: `Tensor` of shape
      `[num_seasons-1, num_seasons]`, such that `differences_from_mean_effect =
      matmul(effects_to_residuals, seasonal_effects)`.  In the
      notation of `ConstrainedSeasonalStateSpaceModel`, this is
      `effects_to_residuals = P * R`.
    residuals_to_effects: the (pseudo)-inverse of the above; a
      `Tensor` of shape `[num_seasons, num_seasons-1]`. In the
      notation of `ConstrainedSeasonalStateSpaceModel`, this is
      `residuals_to_effects = R^{-1} * P'`.
  """

  # Build the matrix that converts effects `e_i` into differences from the mean
  # effect `(e_i - sum(e_i)) / num_seasons`, with the mean effect in the last
  # row so that the transformation is invertible.
  effects_to_residuals_fullrank = np.eye(num_seasons) - 1./num_seasons
  effects_to_residuals_fullrank[-1, :] = 1./num_seasons  # compute mean effect
  residuals_to_effects_fullrank = np.linalg.inv(effects_to_residuals_fullrank)

  # Drop the final dimension, effectively setting the mean effect to zero.
  effects_to_residuals = effects_to_residuals_fullrank[:-1, :]
  residuals_to_effects = residuals_to_effects_fullrank[:, :-1]

  # Return Tensor values of the specified dtype.
  effects_to_residuals = tf.cast(
      effects_to_residuals, dtype=dtype, name='effects_to_residuals')
  residuals_to_effects = tf.cast(
      residuals_to_effects, dtype=dtype, name='residuals_to_effects')

  return effects_to_residuals, residuals_to_effects


def build_seasonal_transition_matrix(
    num_seasons, is_last_day_of_season, dtype,
    basis_change_matrix=None, basis_change_matrix_inv=None):
  """Build a function computing transitions for a seasonal effect model."""

  with tf.name_scope('build_seasonal_transition_matrix'):
    # If the season is changing, the transition matrix permutes the latent
    # state to shift all seasons up by a dimension, and sends the current
    # season's effect to the bottom.
    seasonal_permutation = np.concatenate(
        [np.arange(1, num_seasons), [0]], axis=0)
    seasonal_permutation_matrix = tf.constant(
        np.eye(num_seasons)[seasonal_permutation], dtype=dtype)

    # Optionally transform the transition matrix into a reparameterized space,
    # enforcing the zero-sum constraint for ConstrainedSeasonalStateSpaceModel.
    if basis_change_matrix is not None:
      seasonal_permutation_matrix = tf.matmul(
          basis_change_matrix,
          tf.matmul(seasonal_permutation_matrix, basis_change_matrix_inv))

    identity_matrix = tf.eye(
        tf.shape(seasonal_permutation_matrix)[-1], dtype=dtype)

    def seasonal_transition_matrix(t):
      return tf.linalg.LinearOperatorFullMatrix(
          matrix=dist_util.pick_scalar_condition(
              is_last_day_of_season(t),
              seasonal_permutation_matrix,
              identity_matrix))

  return seasonal_transition_matrix


def build_seasonal_transition_noise(
    drift_scale, num_seasons, is_last_day_of_season):
  """Build the transition noise model for a SeasonalStateSpaceModel."""

  # If the current season has just ended, increase the variance of its effect
  # following drift_scale. (the just-ended seasonal effect will always be the
  # bottom element of the vector). Otherwise, do nothing.
  drift_scale_diag = tf.stack(
      [tf.zeros_like(drift_scale)] * (num_seasons - 1) + [drift_scale],
      axis=-1)
  def seasonal_transition_noise(t):
    noise_scale_diag = dist_util.pick_scalar_condition(
        is_last_day_of_season(t),
        drift_scale_diag,
        tf.zeros_like(drift_scale_diag))
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros(num_seasons, dtype=drift_scale.dtype),
        scale_diag=noise_scale_diag)
  return seasonal_transition_noise


def build_constrained_seasonal_transition_noise(
    drift_scale, num_seasons, is_last_day_of_season):
  """Build transition noise distribution for a ConstrainedSeasonalSSM."""

  # Conceptually, this method takes the noise covariance on effects L @ L'
  # computed by `build_seasonal_transition_noise`, with scale factor
  #       L = [ 0, 0, ..., 0
  #             ...
  #             0, 0, ..., drift_scale],
  # and transforms it to act on the constrained-residual representation.
  #
  # The resulting noise covariance M @ M' is equivalent to
  #    M @ M' = effects_to_residuals @ LL' @ residuals_to_effects
  # where `@` is matrix multiplication. However because this matrix is
  # rank-deficient, we can't take its Cholesky decomposition directly, so we'll
  # construct its lower-triangular scale factor `M` by hand instead.
  #
  # Concretely, let `M = P @ R @ L` be the scale factor in the
  # transformed space, with matrices `R`, `P` applying the reparameterization
  # and zero-mean constraint respectively as defined in the
  # "Mathematical Details" section of `ConstrainedSeasonalStateSpaceModel`. It's
  # easy to see (*) that the implied covariance
  # `M @ M' = P @ R @ L @ L' @ R' @ P'` is just the constant matrix
  #  `M @ M' = [ 1, 1, ..., 1, 0
  #              1, 1, ..., 1, 0
  #              ...
  #              1, 1, ..., 1, 0
  #              0, 0, ..., 0, 0] * (drift_scale / num_seasons)**2`
  # with zeros in the final row and column. So we can directly construct
  # the lower-triangular factor
  #  `Q = [ 1, 0, ...  0
  #         1, 0, ..., 0
  #         ...
  #         1, 0, ..., 0
  #         0, 0, ..., 0 ] * drift_scale/num_seasons`
  # such that Q @ Q' = M @ M'. In practice, we don't reify the final row and
  # column full of zeroes, i.e., we construct
  # `Q[:num_seasons-1, :num_seasons-1]` as the scale-TriL covariance factor.
  #
  # (*) Argument: `L` is zero everywhere but the last column, so `R @ L` will be
  # too. Since the last column of `R` is the constant `-1/num_seasons`, `R @ L`
  # is simply the matrix with constant `-drift_scale/num_seasons` in the final
  # column (except the final row, which is negated) and zero in all other
  # columns, and `M = P @ R @ L` additionally zeroes out the final row. Then
  # M @ M' is just the outer product of that final column with itself (since all
  # other columns are zero), which gives the matrix shown above.

  drift_scale_tril_nonzeros = tf.concat([
      tf.ones([num_seasons - 1, 1], dtype=drift_scale.dtype),
      tf.zeros([num_seasons - 1, num_seasons - 2], dtype=drift_scale.dtype)],
                                        axis=-1)
  drift_scale_tril = (drift_scale_tril_nonzeros *
                      drift_scale[..., tf.newaxis, tf.newaxis] / num_seasons)

  # Inject transition noise iff it is the last day of the season.
  def seasonal_transition_noise(t):
    noise_scale_tril = dist_util.pick_scalar_condition(
        is_last_day_of_season(t),
        drift_scale_tril,
        tf.zeros_like(drift_scale_tril))
    return tfd.MultivariateNormalTriL(
        loc=tf.zeros(num_seasons-1, dtype=drift_scale.dtype),
        scale_tril=noise_scale_tril)
  return seasonal_transition_noise


class Seasonal(StructuralTimeSeries):
  """Formal representation of a seasonal effect model.

  A seasonal effect model posits a fixed set of recurring, discrete 'seasons',
  each of which is active for a fixed number of timesteps and, while active,
  contributes a different effect to the time series. These are generally not
  meteorological seasons, but represent regular recurring patterns such as
  hour-of-day or day-of-week effects. Each season lasts for a fixed number of
  timesteps. The effect of each season drifts from one occurrence to the next
  following a Gaussian random walk:

  ```python
  effects[season, occurrence[i]] = (
    effects[season, occurrence[i-1]] + Normal(loc=0., scale=drift_scale))
  ```

  The `drift_scale` parameter governs the standard deviation of the random walk;
  for example, in a day-of-week model it governs the change in effect from this
  Monday to next Monday.

  #### Examples

  A seasonal effect model representing day-of-week seasonality on hourly data:

  ```python
  day_of_week = tfp.sts.Seasonal(num_seasons=7,
                                 num_steps_per_season=24,
                                 observed_time_series=y,
                                 name='day_of_week')
  ```

  A seasonal effect model representing month-of-year seasonality on daily data,
  with explicit priors:

  ```python
  month_of_year = tfp.sts.Seasonal(
    num_seasons=12,
    num_steps_per_season=[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
    drift_scale_prior=tfd.LogNormal(loc=-1., scale=0.1),
    initial_effect_prior=tfd.Normal(loc=0., scale=5.),
    name='month_of_year')
  ```

  Note that this version works over time periods not involving a leap year. A
  general implementation of month-of-year seasonality would require additional
  logic:

  ```python
  num_days_per_month = np.array(
    [[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
     [31, 29, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],  # year with leap day
     [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
     [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31]])

  month_of_year = tfp.sts.Seasonal(
    num_seasons=12,
    num_steps_per_season=num_days_per_month,
    drift_scale_prior=tfd.LogNormal(loc=-1., scale=0.1),
    initial_effect_prior=tfd.Normal(loc=0., scale=5.),
    name='month_of_year')
  ```

  A model representing both day-of-week and hour-of-day seasonality, on hourly
  data:

  ```
  day_of_week = tfp.sts.Seasonal(num_seasons=7,
                                 num_steps_per_season=24,
                                 observed_time_series=y,
                                 name='day_of_week')
  hour_of_day = tfp.sts.Seasonal(num_seasons=24,
                                 num_steps_per_season=1,
                                 observed_time_series=y,
                                 name='hour_of_day')
  model = tfp.sts.Sum(components=[day_of_week, hour_of_day],
                      observed_time_series=y)
  ```

  """

  def __init__(self,
               num_seasons,
               num_steps_per_season=1,
               allow_drift=True,
               drift_scale_prior=None,
               initial_effect_prior=None,
               constrain_mean_effect_to_zero=True,
               observed_time_series=None,
               name=None):
    """Specify a seasonal effects model.

    Args:
      num_seasons: Scalar Python `int` number of seasons.
      num_steps_per_season: Python `int` number of steps in each
        season. This may be either a scalar (shape `[]`), in which case all
        seasons have the same length, or a NumPy array of shape `[num_seasons]`,
        in which seasons have different length, but remain constant around
        different cycles, or a NumPy array of shape `[num_cycles, num_seasons]`,
        in which num_steps_per_season for each season also varies in different
        cycle (e.g., a 4 years cycle with leap day).
        Default value: 1.
      allow_drift: optional Python `bool` specifying whether the seasonal
        effects can drift over time.  Setting this to `False`
        removes the `drift_scale` parameter from the model. This is
        mathematically equivalent to
        `drift_scale_prior = tfd.Deterministic(0.)`, but removing drift
        directly is preferred because it avoids the use of a degenerate prior.
        Default value: `True`.
      drift_scale_prior: optional `tfd.Distribution` instance specifying a prior
        on the `drift_scale` parameter. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      initial_effect_prior: optional `tfd.Distribution` instance specifying a
        normal prior on the initial effect of each season. This may be either
        a scalar `tfd.Normal` prior, in which case it applies independently to
        every season, or it may be multivariate normal (e.g.,
        `tfd.MultivariateNormalDiag`) with event shape `[num_seasons]`, in
        which case it specifies a joint prior across all seasons. If `None`, a
        heuristic default prior is constructed based on the provided
        `observed_time_series`.
        Default value: `None`.
      constrain_mean_effect_to_zero: if `True`, use a model parameterization
        that constrains the mean effect across all seasons to be zero. This
        constraint is generally helpful in identifying the contributions of
        different model components and can lead to more interpretable
        posterior decompositions. It may be undesirable if you plan to directly
        examine the latent space of the underlying state space model.
        Default value: `True`.
      observed_time_series: optional `float` `Tensor` of shape
        `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
        supported when `T > 1`), specifying an observed time series.
        Any priors not explicitly set will be given default values according to
        the scale of the observed time series (or batch of time series). May
        optionally be an instance of `tfp.sts.MaskedTimeSeries`, which includes
        a mask `Tensor` to specify timesteps with missing observations.
        Default value: `None`.
      name: the name of this model component.
        Default value: 'Seasonal'.
    """

    with tf.name_scope(name or 'Seasonal') as name:

      _, observed_stddev, observed_initial = (
          sts_util.empirical_statistics(observed_time_series)
          if observed_time_series is not None else (0., 1., 0.))

      # Heuristic default priors. Overriding these may dramatically
      # change inference performance and results.
      if drift_scale_prior is None:
        drift_scale_prior = tfd.LogNormal(
            loc=tf.math.log(.01 * observed_stddev), scale=3.)
      if initial_effect_prior is None:
        initial_effect_prior = tfd.Normal(
            loc=observed_initial,
            scale=tf.abs(observed_initial) + observed_stddev)

      dtype = tf.debugging.assert_same_float_dtype(
          [drift_scale_prior, initial_effect_prior])

      if isinstance(initial_effect_prior, tfd.Normal):
        initial_state_prior = tfd.MultivariateNormalDiag(
            loc=tf.stack([initial_effect_prior.mean()] * num_seasons, axis=-1),
            scale_diag=tf.stack([initial_effect_prior.stddev()] * num_seasons,
                                axis=-1))
      else:
        initial_state_prior = initial_effect_prior

      if constrain_mean_effect_to_zero:
        # Transform the prior to the residual parameterization used by
        # `ConstrainedSeasonalStateSpaceModel`, imposing a zero-sum constraint.
        # This doesn't change the marginal prior on individual effects, but
        # does introduce dependence between the effects.
        (effects_to_residuals, _) = build_effects_to_residuals_matrix(
            num_seasons, dtype=dtype)
        effects_to_residuals_linop = tf.linalg.LinearOperatorFullMatrix(
            effects_to_residuals)  # Use linop so that matmul broadcasts.
        initial_state_prior_loc = effects_to_residuals_linop.matvec(
            initial_state_prior.mean())
        initial_state_prior_scale_linop = effects_to_residuals_linop.matmul(
            initial_state_prior.scale)  # returns LinearOperator
        initial_state_prior = tfd.MultivariateNormalFullCovariance(
            loc=initial_state_prior_loc,
            covariance_matrix=initial_state_prior_scale_linop.matmul(
                initial_state_prior_scale_linop.to_dense(), adjoint_arg=True))

      self._constrain_mean_effect_to_zero = constrain_mean_effect_to_zero
      self._initial_state_prior = initial_state_prior
      self._num_seasons = num_seasons
      self._num_steps_per_season = num_steps_per_season

      parameters = []
      if allow_drift:
        parameters.append(Parameter(
            'drift_scale', drift_scale_prior,
            tfb.Chain([tfb.AffineScalar(scale=observed_stddev),
                       tfb.Softplus()])))
      self._allow_drift = allow_drift

      super(Seasonal, self).__init__(
          parameters,
          latent_size=(num_seasons - 1
                       if self.constrain_mean_effect_to_zero else num_seasons),
          name=name)

  @property
  def allow_drift(self):
    """Whether the seasonal effects are allowed to drift over time."""
    return self._allow_drift

  @property
  def constrain_mean_effect_to_zero(self):
    """Whether to constrain the mean effect to zero."""
    return self._constrain_mean_effect_to_zero

  @property
  def num_seasons(self):
    """Number of seasons."""
    return self._num_seasons

  @property
  def num_steps_per_season(self):
    """Number of steps per season."""
    return self._num_steps_per_season

  @property
  def initial_state_prior(self):
    """Prior distribution on the initial latent state (level and scale)."""
    return self._initial_state_prior

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map,
                              initial_state_prior=None,
                              initial_step=0):

    if initial_state_prior is None:
      initial_state_prior = self.initial_state_prior

    if not self.allow_drift:
      param_map['drift_scale'] = 0.

    if self.constrain_mean_effect_to_zero:
      return ConstrainedSeasonalStateSpaceModel(
          num_timesteps=num_timesteps,
          num_seasons=self.num_seasons,
          num_steps_per_season=self.num_steps_per_season,
          initial_state_prior=initial_state_prior,
          initial_step=initial_step,
          **param_map)
    else:
      return SeasonalStateSpaceModel(
          num_timesteps=num_timesteps,
          num_seasons=self.num_seasons,
          num_steps_per_season=self.num_steps_per_season,
          initial_state_prior=initial_state_prior,
          initial_step=initial_step,
          **param_map)
