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
"""Semi-Local Linear Trend model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import distribution_util as dist_util

from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


class SemiLocalLinearTrendStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
  """State space model for a semi-local linear trend.

  A state space model (SSM) posits a set of latent (unobserved) variables that
  evolve over time with dynamics specified by a probabilistic transition model
  `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
  observation model conditioned on the current state, `p(x[t] | z[t])`. The
  special case where both the transition and observation models are Gaussians
  with mean specified as a linear function of the inputs, is known as a linear
  Gaussian state space model and supports tractable exact probabilistic
  calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
  details.

  The semi-local linear trend model is a special case of a linear Gaussian
  SSM, in which the latent state posits a `level` and `slope`. The `level`
  evolves via a Gaussian random walk centered at the current `slope`, while
  the `slope` follows a first-order autoregressive (AR1) process with
  mean `slope_mean`:

  ```python
  level[t] = level[t-1] + slope[t-1] + Normal(0., level_scale)
  slope[t] = (slope_mean +
              autoregressive_coef * (slope[t-1] - slope_mean) +
              Normal(0., slope_scale))
  ```

  The latent state is the two-dimensional tuple `[level, slope]`. The
  `level` is observed at each timestep.

  The parameters `level_scale`, `slope_mean`, `slope_scale`,
  `autoregressive_coef`, and `observation_noise_scale` are each (a batch of)
  scalars. The batch shape of this `Distribution` is the broadcast batch shape
  of these parameters and of the `initial_state_prior`.

  #### Mathematical Details

  The semi-local linear trend model implements a
  `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size = 2`
  and `observation_size = 1`, following the transition model:

  ```
  transition_matrix = [[1., 1.]
                       [0., autoregressive_coef]]
  transition_noise ~ N(loc=slope_mean - autoregressive_coef * slope_mean,
                       scale=diag([level_scale, slope_scale]))
  ```

  which implements the evolution of `[level, slope]` described above, and
  the observation model:

  ```
  observation_matrix = [[1., 0.]]
  observation_noise ~ N(loc=0, scale=observation_noise_scale)
  ```

  which picks out the first latent component, i.e., the `level`, as the
  observation at each timestep.

  #### Examples

  A simple model definition:

  ```python
  semilocal_trend_model = SemiLocalLinearTrendStateSpaceModel(
      num_timesteps=50,
      level_scale=0.5,
      slope_mean=0.2,
      slope_scale=0.5,
      autoregressive_coef=0.9,
      initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=[1., 1.]))

  y = semilocal_trend_model.sample() # y has shape [50, 1]
  lp = semilocal_trend_model.log_prob(y) # log_prob is scalar
  ```

  Passing additional parameter dimensions constructs a batch of models. The
  overall batch shape is the broadcast batch shape of the parameters:

  ```python
  semilocal_trend_model = SemiLocalLinearTrendStateSpaceModel(
      num_timesteps=50,
      level_scale=tf.ones([10]),
      slope_mean=0.2,
      slope_scale=0.5,
      autoregressive_coef=0.9,
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([10, 10, 2])))

  y = semilocal_trend_model.sample(5)    # y has shape [5, 10, 10, 50, 1]
  lp = semilocal_trend_model.log_prob(y) # lp has shape [5, 10, 10]
  ```

  """

  def __init__(self,
               num_timesteps,
               level_scale,
               slope_mean,
               slope_scale,
               autoregressive_coef,
               initial_state_prior,
               observation_noise_scale=0.,
               name=None,
               **linear_gaussian_ssm_kwargs):
    """Build a state space model implementing a semi-local linear trend.

    Args:
      num_timesteps: Scalar `int` `Tensor` number of timesteps to model
        with this distribution.
      level_scale: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the standard deviation of the
        level transitions.
      slope_mean: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the expected long-term mean of
        the latent slope.
      slope_scale: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the standard deviation of the
        slope transitions.
      autoregressive_coef: Scalar (any additional dimensions are treated as
        batch dimensions) `float` `Tensor` defining the AR1 process on the
        latent slope.
      initial_state_prior: instance of `tfd.MultivariateNormal`
        representing the prior distribution on latent states; must
        have event shape `[2]`.
      observation_noise_scale: Scalar (any additional dimensions are
        treated as batch dimensions) `float` `Tensor` indicating the standard
        deviation of the observation noise.
      name: Python `str` name prefixed to ops created by this class.
        Default value: "SemiLocalLinearTrendStateSpaceModel".
      **linear_gaussian_ssm_kwargs: Optional additional keyword arguments to
        to the base `tfd.LinearGaussianStateSpaceModel` constructor.
    """
    parameters = dict(locals())
    parameters.update(linear_gaussian_ssm_kwargs)
    del parameters['linear_gaussian_ssm_kwargs']
    with tf.name_scope(name or 'SemiLocalLinearTrendStateSpaceModel') as name:
      dtype = initial_state_prior.dtype

      level_scale = tf.convert_to_tensor(
          value=level_scale, dtype=dtype, name='level_scale')
      slope_mean = tf.convert_to_tensor(
          value=slope_mean, dtype=dtype, name='slope_mean')
      slope_scale = tf.convert_to_tensor(
          value=slope_scale, dtype=dtype, name='slope_scale')
      autoregressive_coef = tf.convert_to_tensor(
          value=autoregressive_coef, dtype=dtype, name='autoregressive_coef')
      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          dtype=dtype,
          name='observation_noise_scale')

      self._level_scale = level_scale
      self._slope_mean = slope_mean
      self._slope_scale = slope_scale
      self._autoregressive_coef = autoregressive_coef
      self._observation_noise_scale = observation_noise_scale

      super(SemiLocalLinearTrendStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=semilocal_linear_trend_transition_matrix(
              autoregressive_coef),
          transition_noise=semilocal_linear_trend_transition_noise(
              level_scale, slope_mean, slope_scale, autoregressive_coef),
          observation_matrix=tf.constant(
              [[1., 0.]], dtype=dtype),
          observation_noise=tfd.MultivariateNormalDiag(
              scale_diag=observation_noise_scale[..., tf.newaxis]),
          initial_state_prior=initial_state_prior,
          name=name,
          **linear_gaussian_ssm_kwargs)
      self._parameters = parameters

  @property
  def level_scale(self):
    return self._level_scale

  @property
  def slope_mean(self):
    return self._slope_mean

  @property
  def slope_scale(self):
    return self._slope_scale

  @property
  def autoregressive_coef(self):
    return self._autoregressive_coef

  @property
  def observation_noise_scale(self):
    return self._observation_noise_scale


def semilocal_linear_trend_transition_matrix(autoregressive_coef):
  """Build the transition matrix for a semi-local linear trend model."""
  # We want to write the following 2 x 2 matrix:
  #  [[1., 1., ],    # level(t+1) = level(t) + slope(t)
  #   [0., ar_coef], # slope(t+1) = ar_coef * slope(t)
  # but it's slightly tricky to properly incorporate the batch shape of
  # autoregressive_coef. E.g., if autoregressive_coef has shape [4,6], we want
  # to return shape [4, 6, 2, 2]. We do this by breaking the matrix into its
  # fixed entries, written explicitly, and then the autoregressive_coef part
  # which we add in after using a mask to broadcast to the correct matrix shape.

  fixed_entries = tf.constant(
      [[1., 1.],
       [0., 0.]],
      dtype=autoregressive_coef.dtype)

  autoregressive_coef_mask = tf.constant([[0., 0.],
                                          [0., 1.]],
                                         dtype=autoregressive_coef.dtype)
  bottom_right_entry = (autoregressive_coef[..., tf.newaxis, tf.newaxis] *
                        autoregressive_coef_mask)
  return tf.linalg.LinearOperatorFullMatrix(
      fixed_entries + bottom_right_entry)


def semilocal_linear_trend_transition_noise(level_scale,
                                            slope_mean,
                                            slope_scale,
                                            autoregressive_coef):
  """Build the transition noise model for a semi-local linear trend model."""

  # At each timestep, the stochasticity of `level` and `slope` are given
  # by `level_scale` and `slope_scale` respectively.
  broadcast_batch_shape = dist_util.get_broadcast_shape(
      level_scale, slope_mean, slope_scale, autoregressive_coef)
  broadcast_ones = tf.ones(broadcast_batch_shape, dtype=level_scale.dtype)
  scale_diag = tf.stack([level_scale * broadcast_ones,
                         slope_scale * broadcast_ones],
                        axis=-1)

  # We additionally fold in a bias term implementing the nonzero `slope_mean`.
  # The overall `slope` update is (from `SemiLocalLinearTrend` docstring)
  #   slope[t] = (slope_mean +
  #               autoregressive_coef * (slope[t-1] - slope_mean) +
  #               Normal(0., slope_scale))
  # which we rewrite as
  #   slope[t] = (
  #    autoregressive_coef * slope[t-1] +                  # linear transition
  #    Normal(loc=slope_mean - autoregressive_coef * slope_mean,  # noise bias
  #           scale=slope_scale))                                 # noise scale
  bias = tf.stack([tf.zeros_like(broadcast_ones),
                   slope_mean * (1 - autoregressive_coef) * broadcast_ones],
                  axis=-1)
  return tfd.MultivariateNormalDiag(
      loc=bias,
      scale_diag=scale_diag)


class SemiLocalLinearTrend(StructuralTimeSeries):
  """Formal representation of a semi-local linear trend model.

  Like the `LocalLinearTrend` model, a semi-local linear trend posits a
  latent `level` and `slope`, with the level component updated according to
  the current slope plus a random walk:

  ```
  level[t] = level[t-1] + slope[t-1] + Normal(0., level_scale)
  ```

  The slope component in a `SemiLocalLinearTrend` model evolves according to
  a first-order autoregressive (AR1) process with potentially nonzero mean:

  ```
  slope[t] = (slope_mean +
              autoregressive_coef * (slope[t-1] - slope_mean) +
              Normal(0., slope_scale))
  ```

  Unlike the random walk used in `LocalLinearTrend`, a stationary
  AR1 process (coefficient in `(-1, 1)`) maintains bounded variance over time,
  so a `SemiLocalLinearTrend` model will often produce more reasonable
  uncertainties when forecasting over long timescales.
  """

  def __init__(self,
               level_scale_prior=None,
               slope_mean_prior=None,
               slope_scale_prior=None,
               autoregressive_coef_prior=None,
               initial_level_prior=None,
               initial_slope_prior=None,
               observed_time_series=None,
               constrain_ar_coef_stationary=True,
               constrain_ar_coef_positive=False,
               name=None):
    """Specify a semi-local linear trend model.

    Args:
      level_scale_prior: optional `tfd.Distribution` instance specifying a prior
        on the `level_scale` parameter. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      slope_mean_prior: optional `tfd.Distribution` instance specifying a prior
        on the `slope_mean` parameter. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      slope_scale_prior: optional `tfd.Distribution` instance specifying a prior
        on the `slope_scale` parameter. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      autoregressive_coef_prior: optional `tfd.Distribution` instance specifying
        a prior on the `autoregressive_coef` parameter. If `None`, the default
        prior is a standard `Normal(0., 1.)`. Note that the prior may be
        implicitly truncated by `constrain_ar_coef_stationary` and/or
        `constrain_ar_coef_positive`.
        Default value: `None`.
      initial_level_prior: optional `tfd.Distribution` instance specifying a
        prior on the initial level. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      initial_slope_prior: optional `tfd.Distribution` instance specifying a
        prior on the initial slope. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      observed_time_series: optional `float` `Tensor` of shape
        `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
        supported when `T > 1`), specifying an observed time series.
        Any priors not explicitly set will be given default values according to
        the scale of the observed time series (or batch of time series). May
        optionally be an instance of `tfp.sts.MaskedTimeSeries`, which includes
        a mask `Tensor` to specify timesteps with missing observations.
        Default value: `None`.
      constrain_ar_coef_stationary: if `True`, perform inference using a
        parameterization that restricts `autoregressive_coef` to the interval
        `(-1, 1)`, or `(0, 1)` if `force_positive_ar_coef` is also `True`,
        corresponding to stationary processes. This will implicitly truncates
        the support of `autoregressive_coef_prior`.
        Default value: `True`.
      constrain_ar_coef_positive: if `True`, perform inference using a
        parameterization that restricts `autoregressive_coef` to be positive,
        or in `(0, 1)` if `constrain_ar_coef_stationary` is also `True`. This
        will implicitly truncate the support of `autoregressive_coef_prior`.
        Default value: `False`.
      name: the name of this model component.
        Default value: 'SemiLocalLinearTrend'.
    """

    with tf.name_scope(name or 'SemiLocalLinearTrend') as name:
      if observed_time_series is not None:
        _, observed_stddev, observed_initial = sts_util.empirical_statistics(
            observed_time_series)
      else:
        observed_stddev, observed_initial = 1., 0.

      # Heuristic default priors. Overriding these may dramatically
      # change inference performance and results.
      if level_scale_prior is None:
        level_scale_prior = tfd.LogNormal(
            loc=tf.math.log(.01 * observed_stddev), scale=2.)
      if slope_mean_prior is None:
        slope_mean_prior = tfd.Normal(loc=0.,
                                      scale=observed_stddev)
      if slope_scale_prior is None:
        slope_scale_prior = tfd.LogNormal(
            loc=tf.math.log(.01 * observed_stddev), scale=2.)
      if autoregressive_coef_prior is None:
        autoregressive_coef_prior = tfd.Normal(
            loc=0., scale=tf.ones_like(observed_initial))
      if initial_level_prior is None:
        initial_level_prior = tfd.Normal(
            loc=observed_initial,
            scale=tf.abs(observed_initial) + observed_stddev)
      if initial_slope_prior is None:
        initial_slope_prior = tfd.Normal(loc=0., scale=observed_stddev)

      self._initial_state_prior = tfd.MultivariateNormalDiag(
          loc=tf.stack(
              [initial_level_prior.mean(),
               initial_slope_prior.mean()
              ], axis=-1),
          scale_diag=tf.stack([
              initial_level_prior.stddev(),
              initial_slope_prior.stddev()
          ], axis=-1))

      # Constrain the support of the autoregressive coefficient.
      if constrain_ar_coef_stationary and constrain_ar_coef_positive:
        autoregressive_coef_bijector = tfb.Sigmoid()   # support in (0, 1)
      elif constrain_ar_coef_positive:
        autoregressive_coef_bijector = tfb.Softplus()  # support in (0, infty)
      elif constrain_ar_coef_stationary:
        autoregressive_coef_bijector = tfb.Tanh()      # support in (-1, 1)
      else:
        autoregressive_coef_bijector = tfb.Identity()  # unconstrained

      stddev_preconditioner = tfb.Scale(scale=observed_stddev)
      scaled_softplus = tfb.Chain([stddev_preconditioner, tfb.Softplus()])
      super(SemiLocalLinearTrend, self).__init__(
          parameters=[
              Parameter('level_scale', level_scale_prior, scaled_softplus),
              Parameter('slope_mean', slope_mean_prior, stddev_preconditioner),
              Parameter('slope_scale', slope_scale_prior, scaled_softplus),
              Parameter('autoregressive_coef',
                        autoregressive_coef_prior,
                        autoregressive_coef_bijector),
          ],
          latent_size=2,
          name=name)

  @property
  def initial_state_prior(self):
    """Prior distribution on the initial latent state (level and scale)."""
    return self._initial_state_prior

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map,
                              initial_state_prior=None,
                              **linear_gaussian_ssm_kwargs):

    if initial_state_prior is None:
      initial_state_prior = self.initial_state_prior
    linear_gaussian_ssm_kwargs.update(param_map)
    return SemiLocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        initial_state_prior=initial_state_prior,
        **linear_gaussian_ssm_kwargs)
