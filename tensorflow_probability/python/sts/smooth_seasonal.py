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
"""Smooth Seasonal model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import dtype_util

from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


class SmoothSeasonalStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
  """State space model for a smooth seasonal effect.

  A state space model (SSM) posits a set of latent (unobserved) variables that
  evolve over time with dynamics specified by a probabilistic transition model
  `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
  observation model conditioned on the current state, `p(x[t] | z[t])`. The
  special case where both the transition and observation models are Gaussians
  with mean specified as a linear function of the inputs, is known as a linear
  Gaussian state space model and supports tractable exact probabilistic
  calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
  details.

  A smooth seasonal effect model is a special case of a linear Gaussian SSM. It
  is the sum of a set of "cyclic" components, with one component for each
  frequency:

  ```python
  frequencies[j] = 2. * pi * frequency_multipliers[j] / period
  ```

  Each cyclic component contains two latent states which we denote `effect` and
  `auxiliary`. The two latent states for component `j` drift over time via:

  ```python
  effect[t] = (effect[t - 1] * cos(frequencies[j]) +
               auxiliary[t - 1] * sin(frequencies[j]) +
               Normal(0., drift_scale))

  auxiliary[t] = (-effect[t - 1] * sin(frequencies[j]) +
                  auxiliary[t - 1] * cos(frequencies[j]) +
                  Normal(0., drift_scale))
  ```

  The `auxiliary` latent state only appears as a matter of construction and thus
  its interpretation is not particularly important. The total smooth seasonal
  effect is the sum of the `effect` values from each of the cyclic components.

  The parameters `drift_scale` and `observation_noise_scale` are each (a batch
  of) scalars. The batch shape of this `Distribution` is the broadcast batch
  shape of these parameters and of the `initial_state_prior`.

  #### Mathematical Details

  The smooth seasonal effect model implements a
  `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size = 2 *
  len(frequency_multipliers)` and `observation_size = 1`. The latent state is
  the concatenation of the cyclic latent states which themselves comprise an
  `effect` and an `auxiliary` state. The transition matrix is a block diagonal
  matrix where block `j` is:

  ```python
  transition_matrix[j] =  [[cos(frequencies[j]), sin(frequencies[j])],
                           [-sin(frequencies[j]), cos(frequencies[j])]]
  ```

  The observation model picks out the cyclic `effect` values from the latent
  state:

  ```
  observation_matrix = [[1., 0., 1., 0., ..., 1., 0.]]
  observation_noise ~ Normal(loc=0, scale=observation_noise_scale)
  ```

  For further mathematical details please see [1].

  #### Examples

  A state space model with smooth daily seasonality on hourly data. In other
  words, each day there is a pattern which broadly repeats itself over the
  course of the day and doesn't change too much from one hour to the next. Four
  random samples from such a model can be obtained via:

  ```python
  from matplotlib import pylab as plt

  ssm = SmoothSeasonalStateSpaceModel(
      num_timesteps=100,
      period=24,
      frequency_multipliers=[1, 4],
      drift_scale=0.1,
      initial_state_prior=tfd.MultivariateNormalDiag(
          scale_diag=tf.fill([4], 2.0)),
  )

  fig, axes = plt.subplots(4)

  series = ssm.sample(4)

  for series, ax in zip(series[..., 0], axes):
    ax.set_xticks(tf.range(ssm.num_timesteps, delta=ssm.period))
    ax.grid()
    ax.plot(series)

  plt.show()
  ```

  A comparison of the above with a comparable `Seasonal` component gives an
  example of the difference between these two components:

  ```python
  ssm = SeasonalStateSpaceModel(
      num_timesteps=100,
      num_seasons=24,
      num_steps_per_season=1,
      drift_scale=0.1,
      initial_state_prior=tfd.MultivariateNormalDiag(
          scale_diag=tf.fill([24], 2.0)),
  )
  ```

  #### References

  [1]: Harvey, A. Forecasting, Structural Time Series Models and the Kalman
    Filter. Cambridge: Cambridge University Press, 1990.

  """

  def __init__(self,
               num_timesteps,
               period,
               frequency_multipliers,
               drift_scale,
               initial_state_prior,
               observation_noise_scale=0.,
               name=None,
               **linear_gaussian_ssm_kwargs):
    """Build a smooth seasonal state space model.

    Args:
      num_timesteps: Scalar `int` `Tensor` number of timesteps to model
        with this distribution.
      period: positive scalar `float` `Tensor` giving the number of timesteps
        required for the longest cyclic effect to repeat.
      frequency_multipliers: One-dimensional `float` `Tensor` listing the
        frequencies (cyclic components) included in the model, as multipliers of
        the base/fundamental frequency `2. * pi / period`. Each component is
        specified by the number of times it repeats per period, and adds two
        latent dimensions to the model. A smooth seasonal model that can
        represent any periodic function is given by `frequency_multipliers = [1,
        2, ..., floor(period / 2)]`. However, it is often desirable to enforce a
        smoothness assumption (and reduce the computational burden) by dropping
        some of the higher frequencies.
      drift_scale: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the standard deviation of the
        latent state transitions.
      initial_state_prior: instance of `tfd.MultivariateNormal`
        representing the prior distribution on latent states.  Must have
        event shape `[num_features]`.
      observation_noise_scale: Scalar (any additional dimensions are
        treated as batch dimensions) `float` `Tensor` indicating the standard
        deviation of the observation noise.
        Default value: `0.`.
      name: Python `str` name prefixed to ops created by this class.
        Default value: 'SmoothSeasonalStateSpaceModel'.
      **linear_gaussian_ssm_kwargs: Optional additional keyword arguments to
        to the base `tfd.LinearGaussianStateSpaceModel` constructor.
    """
    parameters = dict(locals())
    parameters.update(linear_gaussian_ssm_kwargs)
    del parameters['linear_gaussian_ssm_kwargs']
    with tf.name_scope(name or 'SmoothSeasonalStateSpaceModel') as name:

      dtype = dtype_util.common_dtype(
          [period, frequency_multipliers, drift_scale, initial_state_prior])

      period = tf.convert_to_tensor(
          value=period, name='period', dtype=dtype)

      frequency_multipliers = tf.convert_to_tensor(
          value=frequency_multipliers,
          name='frequency_multipliers',
          dtype=dtype)

      drift_scale = tf.convert_to_tensor(
          value=drift_scale, name='drift_scale', dtype=dtype)

      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          name='observation_noise_scale',
          dtype=dtype)

      num_frequencies = static_num_frequencies(frequency_multipliers)

      observation_matrix = tf.tile(
          tf.constant([[1., 0.]], dtype=dtype),
          multiples=[1, num_frequencies])

      transition_matrix = build_smooth_seasonal_transition_matrix(
          period=period,
          frequency_multipliers=frequency_multipliers,
          dtype=dtype)

      self._drift_scale = drift_scale
      self._observation_noise_scale = observation_noise_scale
      self._period = period
      self._frequency_multipliers = frequency_multipliers

      super(SmoothSeasonalStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=transition_matrix,
          transition_noise=tfd.MultivariateNormalDiag(
              scale_diag=(drift_scale[..., tf.newaxis] *
                          tf.ones([2 * num_frequencies], dtype=dtype)),
              name='transition_noise'),
          observation_matrix=observation_matrix,
          observation_noise=tfd.MultivariateNormalDiag(
              scale_diag=observation_noise_scale[..., tf.newaxis],
              name='observation_noise'),
          initial_state_prior=initial_state_prior,
          name=name,
          **linear_gaussian_ssm_kwargs)
      self._parameters = parameters

  @property
  def drift_scale(self):
    """Standard deviation of the drift in the cyclic effects."""
    return self._drift_scale

  @property
  def observation_noise_scale(self):
    """Standard deviation of the observation noise."""
    return self._observation_noise_scale

  @property
  def period(self):
    """The seasonal period."""
    return self._period

  @property
  def frequency_multipliers(self):
    """Multipliers of the fundamental frequency."""
    return self._frequency_multipliers


def build_smooth_seasonal_transition_matrix(period,
                                            frequency_multipliers,
                                            dtype):
  """Build the transition matrix for a SmoothSeasonalStateSpaceModel."""

  two_pi = tf.constant(2. * np.pi, dtype=dtype)
  frequencies = two_pi * frequency_multipliers / period
  num_frequencies = static_num_frequencies(frequency_multipliers)

  sin_frequencies = tf.sin(frequencies)
  cos_frequencies = tf.cos(frequencies)

  trigonometric_values = tf.stack(
      [cos_frequencies, sin_frequencies, -sin_frequencies, cos_frequencies],
      axis=-1)

  transition_matrix = tf.linalg.LinearOperatorBlockDiag(
      [tf.linalg.LinearOperatorFullMatrix(
          matrix=tf.reshape(trigonometric_values[i], [2, 2]),
          is_square=True) for i in range(num_frequencies)]
  )

  return transition_matrix


def static_num_frequencies(frequency_multipliers):
  """Statically known number of frequencies. Raises if not possible."""

  frequency_multipliers = tf.convert_to_tensor(
      frequency_multipliers, name='frequency_multipliers')

  num_frequencies = tf.compat.dimension_value(
      dimension=frequency_multipliers.shape[0])

  if num_frequencies is None:
    raise ValueError('The number of frequencies must be statically known. Saw '
                     '`frequency_multipliers` with shape {}'.format(
                         frequency_multipliers.shape))

  return num_frequencies


class SmoothSeasonal(StructuralTimeSeries):
  """Formal representation of a smooth seasonal effect model.

  The smooth seasonal model uses a set of trigonometric terms in order to
  capture a recurring pattern whereby adjacent (in time) effects are
  similar. The model uses `frequencies` calculated via:

  ```python
  frequencies[j] = 2. * pi * frequency_multipliers[j] / period
  ```

  and then posits two latent states for each `frequency`. The two latent states
  associated with frequency `j` drift over time via:

  ```python
  effect[t] = (effect[t - 1] * cos(frequencies[j]) +
               auxiliary[t - 1] * sin(frequencies[j]) +
               Normal(0., drift_scale))

  auxiliary[t] = (-effect[t - 1] * sin(frequencies[j]) +
                  auxiliary[t - 1] * cos(frequencies[j]) +
                  Normal(0., drift_scale))
  ```

  where `effect` is the smooth seasonal effect and `auxiliary` only appears as a
  matter of construction. The interpretation of `auxiliary` is thus not
  particularly important.

  #### Examples

  A smooth seasonal effect model representing smooth weekly seasonality on daily
  data:

  ```python
  component = SmoothSeasonal(
      period=7,
      frequency_multipliers=[1, 2, 3],
      initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=tf.ones([6])),
  )
  ```

  """

  def __init__(self,
               period,
               frequency_multipliers,
               allow_drift=True,
               drift_scale_prior=None,
               initial_state_prior=None,
               observed_time_series=None,
               name=None):
    """Specify a smooth seasonal effects model.

    Args:
      period: positive scalar `float` `Tensor` giving the number of timesteps
        required for the longest cyclic effect to repeat.
      frequency_multipliers: One-dimensional `float` `Tensor` listing the
        frequencies (cyclic components) included in the model, as multipliers of
        the base/fundamental frequency `2. * pi / period`. Each component is
        specified by the number of times it repeats per period, and adds two
        latent dimensions to the model. A smooth seasonal model that can
        represent any periodic function is given by `frequency_multipliers = [1,
        2, ..., floor(period / 2)]`. However, it is often desirable to enforce a
        smoothness assumption (and reduce the computational burden) by dropping
        some of the higher frequencies.
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
      initial_state_prior: instance of `tfd.MultivariateNormal` representing
        the prior distribution on the latent states. Must have event shape
        `[2 * len(frequency_multipliers)]`. If `None`, a heuristic default prior
        is constructed based on the provided `observed_time_series`.
      observed_time_series: optional `float` `Tensor` of shape
        `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
        supported when `T > 1`), specifying an observed time series.
        Any priors not explicitly set will be given default values according to
        the scale of the observed time series (or batch of time series). May
        optionally be an instance of `tfp.sts.MaskedTimeSeries`, which includes
        a mask `Tensor` to specify timesteps with missing observations.
        Default value: `None`.
      name: the name of this model component.
        Default value: 'SmoothSeasonal'.

    """

    with tf.name_scope(name or 'SmoothSeasonal') as name:

      _, observed_stddev, observed_initial = (
          sts_util.empirical_statistics(observed_time_series)
          if observed_time_series is not None else (0., 1., 0.))

      latent_size = 2 * static_num_frequencies(frequency_multipliers)

      # Heuristic default priors. Overriding these may dramatically
      # change inference performance and results.
      if drift_scale_prior is None:
        drift_scale_prior = tfd.LogNormal(
            loc=tf.math.log(.01 * observed_stddev), scale=3.)

      if initial_state_prior is None:
        initial_state_scale = (
            tf.abs(observed_initial) + observed_stddev)[..., tf.newaxis]
        ones = tf.ones([latent_size], dtype=drift_scale_prior.dtype)
        initial_state_prior = tfd.MultivariateNormalDiag(
            scale_diag=initial_state_scale * ones)

      self._initial_state_prior = initial_state_prior
      self._period = period
      self._frequency_multipliers = frequency_multipliers

      parameters = []
      if allow_drift:
        parameters.append(Parameter(
            'drift_scale', drift_scale_prior,
            tfb.Chain([tfb.Scale(scale=observed_stddev),
                       tfb.Softplus()])))
      self._allow_drift = allow_drift

      super(SmoothSeasonal, self).__init__(
          parameters=parameters,
          latent_size=latent_size,
          name=name)

  @property
  def allow_drift(self):
    """Whether the seasonal effects are allowed to drift over time."""
    return self._allow_drift

  @property
  def period(self):
    """The seasonal period."""
    return self._period

  @property
  def frequency_multipliers(self):
    """Multipliers of the fundamental frequency."""
    return self._frequency_multipliers

  @property
  def initial_state_prior(self):
    """Prior distribution on the initial latent states."""
    return self._initial_state_prior

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map,
                              initial_state_prior=None,
                              **linear_gaussian_ssm_kwargs):

    if initial_state_prior is None:
      initial_state_prior = self.initial_state_prior

    if not self.allow_drift:
      param_map['drift_scale'] = 0.
    linear_gaussian_ssm_kwargs.update(param_map)
    return SmoothSeasonalStateSpaceModel(
        num_timesteps=num_timesteps,
        period=self.period,
        frequency_multipliers=self.frequency_multipliers,
        initial_state_prior=initial_state_prior,
        **linear_gaussian_ssm_kwargs)
