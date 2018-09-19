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
"""Local Linear Trend model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import distribution_util as dist_util

from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


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
    on January 23 (steps are zero-indexed). A general implementation of
    month-of-year seasonality would require additional logic; this
    version works over time periods not involving a leap year.
    """

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
               name=None):
    """Build a state space model implementing seasonal effects.

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
        seasons have the same length, or a NumPy array of shape `[num_seasons]`.
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

    with tf.name_scope(name, 'SeasonalStateSpaceModel',
                       values=[drift_scale, observation_noise_scale]) as name:

      # The initial state prior determines the dtype of sampled values.
      # Other model parameters must have the same dtype.
      dtype = initial_state_prior.dtype
      drift_scale = tf.convert_to_tensor(
          drift_scale, name='drift_scale', dtype=dtype)
      observation_noise_scale = tf.convert_to_tensor(
          observation_noise_scale, name='observation_noise_scale', dtype=dtype)

      # Coerce `num_steps_per_season` to a canonical form, an array of
      # `num_seasons` integers.
      num_steps_per_season = np.asarray(num_steps_per_season)
      if not num_steps_per_season.shape:
        num_steps_per_season = np.tile(num_steps_per_season, num_seasons)
      elif num_steps_per_season.shape != (num_seasons,):
        raise ValueError('num_steps_per_season must either be scalar (shape [])'
                         ' or have length [num_seasons] = [{}] (saw: shape {})'.
                         format(num_seasons, num_steps_per_season.shape))

      # Utility method to compute whether the season is changing.
      num_steps_per_cycle = np.sum(num_steps_per_season)
      changepoints = np.cumsum(num_steps_per_season) - 1
      def is_last_day_of_season(t):
        t_ = dist_util.maybe_get_static_value(t)
        if t_ is not None:  # static case
          step_in_cycle = t_ % num_steps_per_cycle
          return any(step_in_cycle == changepoints)
        else:
          step_in_cycle = tf.floormod(t, num_steps_per_cycle)
          return tf.reduce_any(tf.equal(step_in_cycle, changepoints))

      # If the season is changing, the transition matrix rotates the latent
      # state to shift all seasons up by a dimension, and sends the current
      # season's effect to the bottom.
      seasonal_permutation = np.concatenate(
          [np.arange(1, num_seasons), [0]], axis=0)
      seasonal_permutation_matrix = np.eye(num_seasons)[seasonal_permutation]
      def seasonal_transition_matrix(t):
        return tf.linalg.LinearOperatorFullMatrix(
            matrix=dist_util.pick_scalar_condition(
                is_last_day_of_season(t),
                tf.constant(seasonal_permutation_matrix, dtype=dtype),
                tf.eye(num_seasons, dtype=dtype)))

      # If the season is changing, the transition noise model adds random drift
      # to the effect of the outgoing season.
      drift_scale_diag = tf.stack(
          [tf.zeros_like(drift_scale)] * (num_seasons - 1) + [drift_scale],
          axis=-1)
      def seasonal_transition_noise(t):
        noise_scale = dist_util.pick_scalar_condition(
            is_last_day_of_season(t),
            drift_scale_diag,
            tf.zeros_like(drift_scale_diag, dtype=dtype))
        return tfd.MultivariateNormalDiag(loc=tf.zeros(num_seasons,
                                                       dtype=dtype),
                                          scale_diag=noise_scale)

      self._drift_scale = drift_scale
      self._observation_noise_scale = observation_noise_scale
      self._num_seasons = num_seasons
      self._num_steps_per_season = num_steps_per_season

      super(SeasonalStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=seasonal_transition_matrix,
          transition_noise=seasonal_transition_noise,
          observation_matrix=tf.concat([tf.ones([1, 1], dtype=dtype),
                                        tf.zeros([1, num_seasons-1],
                                                 dtype=dtype)],
                                       axis=-1),
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
    initial_effect_prior=tf.Normal(loc=0., scale=5.),
    name='month_of_year')
  ```

  Note that a general implementation of month-of-year seasonality would require
  additional logic; this version works over time periods not involving a leap
  year.

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
               drift_scale_prior=None,
               initial_effect_prior=None,
               observed_time_series=None,
               name=None):
    """Specify a seasonal effects model.

    Args:
      num_seasons: Scalar Python `int` number of seasons.
      num_steps_per_season: Python `int` number of steps in each
        season. This may be either a scalar (shape `[]`), in which case all
        seasons have the same length, or a NumPy array of shape `[num_seasons]`.
        Default value: 1.
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
      observed_time_series: optional `float` `Tensor` of shape
        `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
        supported when `T > 1`), specifying an observed time series.
        Any priors not explicitly set will be given default values according to
        the scale of the observed time series (or batch of time series).
        Default value: `None`.
      name: the name of this model component.
        Default value: 'Seasonal'.
    """

    with tf.name_scope(name, 'Seasonal', values=[observed_time_series]) as name:

      observed_stddev, observed_initial = (
          sts_util.empirical_statistics(observed_time_series)
          if observed_time_series is not None else (1., 0.))

      # Heuristic default priors. Overriding these may dramatically
      # change inference performance and results.
      if drift_scale_prior is None:
        drift_scale_prior = tfd.LogNormal(loc=tf.log(.01 * observed_stddev),
                                          scale=3.)
      if initial_effect_prior is None:
        initial_effect_prior = tfd.Normal(loc=observed_initial,
                                          scale=observed_stddev)

      self._num_seasons = num_seasons
      self._num_steps_per_season = num_steps_per_season

      tf.assert_same_float_dtype([drift_scale_prior, initial_effect_prior])

      if isinstance(initial_effect_prior, tfd.Normal):
        self._initial_state_prior = tfd.MultivariateNormalDiag(
            loc=tf.stack([initial_effect_prior.mean()] * num_seasons, axis=-1),
            scale_diag=tf.stack([initial_effect_prior.stddev()] * num_seasons,
                                axis=-1))
      else:
        self._initial_state_prior = initial_effect_prior

      super(Seasonal, self).__init__(
          parameters=[
              Parameter('drift_scale', drift_scale_prior, tfb.Softplus()),
          ],
          latent_size=num_seasons,
          name=name)

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

    return SeasonalStateSpaceModel(
        num_timesteps=num_timesteps,
        num_seasons=self.num_seasons,
        num_steps_per_season=self.num_steps_per_season,
        initial_state_prior=initial_state_prior,
        initial_step=initial_step,
        **param_map)
