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
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import linear_gaussian_ssm
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers

from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


class LocalLinearTrendStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
  """State space model for a local linear trend.

    A state space model (SSM) posits a set of latent (unobserved) variables that
    evolve over time with dynamics specified by a probabilistic transition model
    `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
    observation model conditioned on the current state, `p(x[t] | z[t])`. The
    special case where both the transition and observation models are Gaussians
    with mean specified as a linear function of the inputs, is known as a linear
    Gaussian state space model and supports tractable exact probabilistic
    calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
    details.

    The local linear trend model is a special case of a linear Gaussian SSM, in
    which the latent state posits a `level` and `slope`, each evolving via a
    Gaussian random walk:

    ```python
    level[t] = level[t-1] + slope[t-1] + Normal(0., level_scale)
    slope[t] = slope[t-1] + Normal(0., slope_scale)
    ```

    The latent state is the two-dimensional tuple `[level, slope]`. The
    `level` is observed at each timestep.

    The parameters `level_scale`, `slope_scale`, and `observation_noise_scale`
    are each (a batch of) scalars. The batch shape of this `Distribution` is the
    broadcast batch shape of these parameters and of the `initial_state_prior`.

    #### Mathematical Details

    The linear trend model implements a
    `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size = 2`
    and `observation_size = 1`, following the transition model:

    ```
    transition_matrix = [[1., 1.]
                         [0., 1.]]
    transition_noise ~ N(loc=0., scale=diag([level_scale, slope_scale]))
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
    linear_trend_model = LocalLinearTrendStateSpaceModel(
        num_timesteps=50,
        level_scale=0.5,
        slope_scale=0.5,
        initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=[1., 1.]))

    y = linear_trend_model.sample() # y has shape [50, 1]
    lp = linear_trend_model.log_prob(y) # log_prob is scalar
    ```

    Passing additional parameter dimensions constructs a batch of models. The
    overall batch shape is the broadcast batch shape of the parameters:

    ```python
    linear_trend_model = LocalLinearTrendStateSpaceModel(
        num_timesteps=50,
        level_scale=tf.ones([10]),
        slope_scale=0.5,
        initial_state_prior=tfd.MultivariateNormalDiag(
          scale_diag=tf.ones([10, 10, 2])))

    y = linear_trend_model.sample(5) # y has shape [5, 10, 10, 50, 1]
    lp = linear_trend_model.log_prob(y) # has shape [5, 10, 10]
    ```

  """

  def __init__(self,
               num_timesteps,
               level_scale,
               slope_scale,
               initial_state_prior,
               observation_noise_scale=0.,
               name=None,
               **linear_gaussian_ssm_kwargs):
    """Build a state space model implementing a local linear trend.

    Args:
      num_timesteps: Scalar `int` `Tensor` number of timesteps to model
        with this distribution.
      level_scale: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the standard deviation of the
        level transitions.
      slope_scale: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the standard deviation of the
        slope transitions.
      initial_state_prior: instance of `tfd.MultivariateNormal`
        representing the prior distribution on latent states; must
        have event shape `[2]`.
      observation_noise_scale: Scalar (any additional dimensions are
        treated as batch dimensions) `float` `Tensor` indicating the standard
        deviation of the observation noise.
      name: Python `str` name prefixed to ops created by this class.
        Default value: "LocalLinearTrendStateSpaceModel".
      **linear_gaussian_ssm_kwargs: Optional additional keyword arguments to
        to the base `tfd.LinearGaussianStateSpaceModel` constructor.
    """
    parameters = dict(locals())
    parameters.update(linear_gaussian_ssm_kwargs)
    del parameters['linear_gaussian_ssm_kwargs']
    with tf.name_scope(name or 'LocalLinearTrendStateSpaceModel') as name:
      # The initial state prior determines the dtype of sampled values.
      # Other model parameters must have the same dtype.
      dtype = initial_state_prior.dtype

      level_scale = tf.convert_to_tensor(
          value=level_scale, name='level_scale', dtype=dtype)
      slope_scale = tf.convert_to_tensor(
          value=slope_scale, name='slope_scale', dtype=dtype)
      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          name='observation_noise_scale',
          dtype=dtype)

      # Explicitly broadcast all parameters to the same batch shape. This
      # allows us to use `tf.stack` for a compact model specification.
      broadcast_batch_shape = dist_util.get_broadcast_shape(
          level_scale, slope_scale)
      broadcast_ones = tf.ones(broadcast_batch_shape, dtype=dtype)

      self._level_scale = level_scale
      self._slope_scale = slope_scale
      self._observation_noise_scale = observation_noise_scale

      # Construct a linear Gaussian state space model implementing the
      # local linear trend model. See "Mathematical Details" in the
      # class docstring for further explanation.
      super(LocalLinearTrendStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=tf.constant(
              [[1., 1.], [0., 1.]], dtype=dtype, name='transition_matrix'),
          transition_noise=tfd.MultivariateNormalDiag(
              scale_diag=tf.stack(
                  [level_scale * broadcast_ones, slope_scale * broadcast_ones],
                  axis=-1),
              name='transition_noise'),
          observation_matrix=tf.constant(
              [[1., 0.]], dtype=dtype, name='observation_matrix'),
          observation_noise=tfd.MultivariateNormalDiag(
              scale_diag=observation_noise_scale[..., tf.newaxis],
              name='observation_noise'),
          initial_state_prior=initial_state_prior,
          name=name,
          **linear_gaussian_ssm_kwargs)
      self._parameters = parameters

  @property
  def level_scale(self):
    """Standard deviation of the level transitions."""
    return self._level_scale

  @property
  def slope_scale(self):
    """Standard deviation of the slope transitions."""
    return self._slope_scale

  @property
  def observation_noise_scale(self):
    """Standard deviation of the observation noise."""
    return self._observation_noise_scale

  def _joint_sample_n(self, n, seed=None):
    """Draw a joint sample from the prior over latents and observations.

    This sampler is specific to LocalLinearTrend models and is faster than the
    generic LinearGaussianStateSpaceModel implementation.

    Args:
      n: `int` `Tensor` number of samples to draw.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    Returns:
      latents: `float` `Tensor` of shape `concat([[n], self.batch_shape,
        [self.num_timesteps, self.latent_size]], axis=0)` representing samples
        of latent trajectories.
      observations: `float` `Tensor` of shape `concat([[n], self.batch_shape,
        [self.num_timesteps, self.observation_size]], axis=0)` representing
        samples of observed series generated from the sampled `latents`.
    """
    with tf.name_scope('joint_sample_n'):
      (initial_state_seed,
       level_jumps_seed,
       slope_jumps_seed,
       prior_observation_seed) = samplers.split_seed(
           seed, n=4, salt='LocalLinearTrendStateSpaceModel_joint_sample_n')

      if self.batch_shape.is_fully_defined():
        batch_shape = self.batch_shape.as_list()
      else:
        batch_shape = self.batch_shape_tensor()
      sample_and_batch_shape = ps.cast(
          ps.concat([[n], batch_shape], axis=0), tf.int32)

      # Sample the initial timestep from the prior.  Since we want
      # this sample to have full batch shape (not just the batch shape
      # of the self.initial_state_prior object which might in general be
      # smaller), we augment the sample shape to include whatever
      # extra batch dimensions are required.
      initial_level_and_slope = self.initial_state_prior.sample(
          linear_gaussian_ssm._augment_sample_shape(  # pylint: disable=protected-access
              self.initial_state_prior,
              sample_and_batch_shape,
              self.validate_args),
          seed=initial_state_seed)

      # Sample the latent random walk on slopes.
      jumps_shape = ps.concat([sample_and_batch_shape,
                               [self.num_timesteps - 1]], axis=0)
      slope_jumps = samplers.normal(
          jumps_shape,
          dtype=self.dtype,
          seed=slope_jumps_seed) * self.slope_scale[..., tf.newaxis]
      prior_slope_sample = tf.cumsum(
          tf.concat([initial_level_and_slope[..., 1:],
                     slope_jumps],
                    axis=-1),
          axis=-1)

      # Sample latent levels, given latent slopes.
      level_jumps = samplers.normal(
          jumps_shape,
          dtype=self.dtype,
          seed=level_jumps_seed) * self.level_scale[..., tf.newaxis]
      prior_level_sample = tf.cumsum(
          tf.concat([initial_level_and_slope[..., :1],
                     level_jumps + prior_slope_sample[..., :-1]],
                    axis=-1),
          axis=-1)

      # Sample noisy observations, given latent levels.
      prior_observation_sample = prior_level_sample + (
          samplers.normal(ps.shape(prior_level_sample),
                          dtype=self.dtype,
                          seed=prior_observation_seed) *
          self.observation_noise_scale[..., tf.newaxis])

      return (tf.stack([prior_level_sample,
                        prior_slope_sample],
                       axis=-1),
              prior_observation_sample[..., tf.newaxis])


class LocalLinearTrend(StructuralTimeSeries):
  """Formal representation of a local linear trend model.

  The local linear trend model posits a `level` and `slope`, each
  evolving via a Gaussian random walk:

  ```
  level[t] = level[t-1] + slope[t-1] + Normal(0., level_scale)
  slope[t] = slope[t-1] + Normal(0., slope_scale)
  ```

  The latent state is the two-dimensional tuple `[level, slope]`. At each
  timestep we observe a noisy realization of the current level:
  `f[t] = level[t] + Normal(0., observation_noise_scale)`. This model
  is appropriate for data where the trend direction and magnitude (latent
  `slope`) is consistent within short periods but may evolve over time.

  Note that this model can produce very high uncertainty forecasts, as
  uncertainty over the slope compounds quickly. If you expect your data to
  have nonzero long-term trend, i.e. that slopes tend to revert to some mean,
  then the `SemiLocalLinearTrend` model may produce sharper forecasts.
  """

  def __init__(self,
               level_scale_prior=None,
               slope_scale_prior=None,
               initial_level_prior=None,
               initial_slope_prior=None,
               observed_time_series=None,
               name=None):
    """Specify a local linear trend model.

    Args:
      level_scale_prior: optional `tfd.Distribution` instance specifying a prior
        on the `level_scale` parameter. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      slope_scale_prior: optional `tfd.Distribution` instance specifying a prior
        on the `slope_scale` parameter. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
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
        supported when `T > 1`), specifying an observed time series. Any `NaN`s
        are interpreted as missing observations; missingness may be also be
        explicitly specified by passing a `tfp.sts.MaskedTimeSeries` instance.
        Any priors not explicitly set will be given default values according to
        the scale of the observed time series (or batch of time series).
        Default value: `None`.
      name: the name of this model component.
        Default value: 'LocalLinearTrend'.
    """
    init_parameters = dict(locals())
    with tf.name_scope(name or 'LocalLinearTrend') as name:
      _, observed_stddev, observed_initial = (
          sts_util.empirical_statistics(observed_time_series)
          if observed_time_series is not None else (0., 1., 0.))

      # Heuristic default priors. Overriding these may dramatically
      # change inference performance and results.
      if level_scale_prior is None:
        level_scale_prior = tfd.LogNormal(
            loc=tf.math.log(.05 * observed_stddev),
            scale=3.,
            name='level_scale_prior')
      if slope_scale_prior is None:
        slope_scale_prior = tfd.LogNormal(
            loc=tf.math.log(.05 * observed_stddev),
            scale=3.,
            name='slope_scale_prior')
      if initial_level_prior is None:
        initial_level_prior = tfd.Normal(
            loc=observed_initial,
            scale=tf.abs(observed_initial) + observed_stddev,
            name='initial_level_prior')
      if initial_slope_prior is None:
        initial_slope_prior = tfd.Normal(
            loc=0., scale=observed_stddev, name='initial_slope_prior')

      tf.debugging.assert_same_float_dtype([
          level_scale_prior, slope_scale_prior, initial_level_prior,
          initial_slope_prior
      ])

      self._initial_state_prior = tfd.MultivariateNormalDiag(
          loc=tf.stack(
              [initial_level_prior.mean(),
               initial_slope_prior.mean()
              ], axis=-1),
          scale_diag=tf.stack([
              initial_level_prior.stddev(),
              initial_slope_prior.stddev()
          ], axis=-1))

      scaled_softplus = tfb.Chain([tfb.Scale(scale=observed_stddev),
                                   tfb.Softplus()])
      super(LocalLinearTrend, self).__init__(
          parameters=[
              Parameter('level_scale', level_scale_prior, scaled_softplus),
              Parameter('slope_scale', slope_scale_prior, scaled_softplus)
          ],
          latent_size=2,
          init_parameters=init_parameters,
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
    return LocalLinearTrendStateSpaceModel(
        num_timesteps,
        initial_state_prior=initial_state_prior,
        **linear_gaussian_ssm_kwargs)
