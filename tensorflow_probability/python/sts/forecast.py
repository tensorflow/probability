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
"""Methods for forecasting in StructuralTimeSeries models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.sts.internal import util as sts_util


def _prefer_static_event_ndims(distribution):
  if distribution.event_shape.ndims is not None:
    return distribution.event_shape.ndims
  else:
    return tf.size(distribution.event_shape_tensor())


def one_step_predictive(model, observed_time_series, parameter_samples):
  """Compute one-step-ahead predictive distributions for all timesteps.

  Given samples from the posterior over parameters, return the predictive
  distribution over observations at each time `T`, given observations up
  through time `T-1`.

  Args:
    model: An instance of `StructuralTimeSeries` representing a
      time-series model. This represents a joint distribution over
      time-series and their parameters with batch shape `[b1, ..., bN]`.
    observed_time_series: `float` `Tensor` of shape
      `concat([sample_shape, model.batch_shape, [num_timesteps, 1]]) where
      `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
      dimension may (optionally) be omitted if `num_timesteps > 1`. May
      optionally be an instance of `tfp.sts.MaskedTimeSeries` including a
      mask `Tensor` to encode the locations of missing observations.
    parameter_samples: Python `list` of `Tensors` representing posterior samples
      of model parameters, with shapes `[concat([[num_posterior_draws],
      param.prior.batch_shape, param.prior.event_shape]) for param in
      model.parameters]`. This may optionally also be a map (Python `dict`) of
      parameter names to `Tensor` values.

  Returns:
    forecast_dist: a `tfd.MixtureSameFamily` instance with event shape
      [num_timesteps] and
      batch shape `concat([sample_shape, model.batch_shape])`, with
      `num_posterior_draws` mixture components. The `t`th step represents the
      forecast distribution `p(observed_time_series[t] |
      observed_time_series[0:t-1], parameter_samples)`.

  #### Examples

  Suppose we've built a model and fit it to data using HMC:

  ```python
    day_of_week = tfp.sts.Seasonal(
        num_seasons=7,
        observed_time_series=observed_time_series,
        name='day_of_week')
    local_linear_trend = tfp.sts.LocalLinearTrend(
        observed_time_series=observed_time_series,
        name='local_linear_trend')
    model = tfp.sts.Sum(components=[day_of_week, local_linear_trend],
                        observed_time_series=observed_time_series)

    samples, kernel_results = tfp.sts.fit_with_hmc(model, observed_time_series)
  ```

  Passing the posterior samples into `one_step_predictive`, we construct a
  one-step-ahead predictive distribution:

  ```python
    one_step_predictive_dist = tfp.sts.one_step_predictive(
      model, observed_time_series, parameter_samples=samples)

    predictive_means = one_step_predictive_dist.mean()
    predictive_scales = one_step_predictive_dist.stddev()
  ```

  If using variational inference instead of HMC, we'd construct a forecast using
  samples from the variational posterior:

  ```python
    surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(
      model=model)
    loss_curve = tfp.vi.fit_surrogate_posterior(
      target_log_prob_fn=model.joint_log_prob(observed_time_series),
      surrogate_posterior=surrogate_posterior,
      optimizer=tf.optimizers.Adam(learning_rate=0.1),
      num_steps=200)
    samples = surrogate_posterior.sample(30)

    one_step_predictive_dist = tfp.sts.one_step_predictive(
      model, observed_time_series, parameter_samples=samples)
  ```

  We can visualize the forecast by plotting:

  ```python
    from matplotlib import pylab as plt
    def plot_one_step_predictive(observed_time_series,
                                 forecast_mean,
                                 forecast_scale):
      plt.figure(figsize=(12, 6))
      num_timesteps = forecast_mean.shape[-1]
      c1, c2 = (0.12, 0.47, 0.71), (1.0, 0.5, 0.05)
      plt.plot(observed_time_series, label="observed time series", color=c1)
      plt.plot(forecast_mean, label="one-step prediction", color=c2)
      plt.fill_between(np.arange(num_timesteps),
                       forecast_mean - 2 * forecast_scale,
                       forecast_mean + 2 * forecast_scale,
                       alpha=0.1, color=c2)
      plt.legend()

    plot_one_step_predictive(observed_time_series,
                             forecast_mean=predictive_means,
                             forecast_scale=predictive_scales)
  ```

  To detect anomalous timesteps, we check whether the observed value at each
  step is within a 95% predictive interval, i.e., two standard deviations from
  the mean:

  ```python
    z_scores = ((observed_time_series[..., 1:] - predictive_means[..., :-1])
                 / predictive_scales[..., :-1])
    anomalous_timesteps = tf.boolean_mask(
        tf.range(1, num_timesteps),
        tf.abs(z_scores) > 2.0)
  ```

  """

  with tf.name_scope('one_step_predictive'):

    [
        observed_time_series,
        is_missing
    ] = sts_util.canonicalize_observed_time_series_with_mask(
        observed_time_series)

    # Run filtering over the training timesteps to extract the
    # predictive means and variances.
    num_timesteps = dist_util.prefer_static_value(
        tf.shape(observed_time_series))[-2]
    lgssm = model.make_state_space_model(
        num_timesteps=num_timesteps, param_vals=parameter_samples)
    (_, _, _, _, _, observation_means, observation_covs
    ) = lgssm.forward_filter(observed_time_series, mask=is_missing)

    # Squeeze dims to convert from LGSSM's event shape `[num_timesteps, 1]`
    # to a scalar time series.
    return sts_util.mix_over_posterior_draws(
        means=observation_means[..., 0],
        variances=observation_covs[..., 0, 0])


def forecast(model,
             observed_time_series,
             parameter_samples,
             num_steps_forecast,
             include_observation_noise=True):
  """Construct predictive distribution over future observations.

  Given samples from the posterior over parameters, return the predictive
  distribution over future observations for num_steps_forecast timesteps.

  Args:
    model: An instance of `StructuralTimeSeries` representing a
      time-series model. This represents a joint distribution over
      time-series and their parameters with batch shape `[b1, ..., bN]`.
    observed_time_series: `float` `Tensor` of shape
      `concat([sample_shape, model.batch_shape, [num_timesteps, 1]])` where
      `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
      dimension may (optionally) be omitted if `num_timesteps > 1`. May
      optionally be an instance of `tfp.sts.MaskedTimeSeries` including a
      mask `Tensor` to encode the locations of missing observations.
    parameter_samples: Python `list` of `Tensors` representing posterior samples
      of model parameters, with shapes `[concat([[num_posterior_draws],
      param.prior.batch_shape, param.prior.event_shape]) for param in
      model.parameters]`. This may optionally also be a map (Python `dict`) of
      parameter names to `Tensor` values.
    num_steps_forecast: scalar `int` `Tensor` number of steps to forecast.
    include_observation_noise: Python `bool` indicating whether the forecast
      distribution should include uncertainty from observation noise. If `True`,
      the forecast is over future observations, if `False`, the forecast is over
      future values of the latent noise-free time series.
      Default value: `True`.

  Returns:
    forecast_dist: a `tfd.MixtureSameFamily` instance with event shape
      [num_steps_forecast, 1] and batch shape
      `concat([sample_shape, model.batch_shape])`, with `num_posterior_draws`
      mixture components.

  #### Examples

  Suppose we've built a model and fit it to data using HMC:

  ```python
    day_of_week = tfp.sts.Seasonal(
        num_seasons=7,
        observed_time_series=observed_time_series,
        name='day_of_week')
    local_linear_trend = tfp.sts.LocalLinearTrend(
        observed_time_series=observed_time_series,
        name='local_linear_trend')
    model = tfp.sts.Sum(components=[day_of_week, local_linear_trend],
                        observed_time_series=observed_time_series)

    samples, kernel_results = tfp.sts.fit_with_hmc(model, observed_time_series)
  ```

  Passing the posterior samples into `forecast`, we construct a forecast
  distribution:

  ```python
    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                     parameter_samples=samples,
                                     num_steps_forecast=50)

    forecast_mean = forecast_dist.mean()[..., 0]  # shape: [50]
    forecast_scale = forecast_dist.stddev()[..., 0]  # shape: [50]
    forecast_samples = forecast_dist.sample(10)[..., 0]  # shape: [10, 50]
  ```

  If using variational inference instead of HMC, we'd construct a forecast using
  samples from the variational posterior:

  ```python
    surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(
      model=model)
    loss_curve = tfp.vi.fit_surrogate_posterior(
      target_log_prob_fn=model.joint_log_prob(observed_time_series),
      surrogate_posterior=surrogate_posterior,
      optimizer=tf.optimizers.Adam(learning_rate=0.1),
      num_steps=200)
    samples = surrogate_posterior.sample(30)

    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                     parameter_samples=samples,
                                     num_steps_forecast=50)
  ```

  We can visualize the forecast by plotting:

  ```python
    from matplotlib import pylab as plt
    def plot_forecast(observed_time_series,
                      forecast_mean,
                      forecast_scale,
                      forecast_samples):
      plt.figure(figsize=(12, 6))

      num_steps = observed_time_series.shape[-1]
      num_steps_forecast = forecast_mean.shape[-1]
      num_steps_train = num_steps - num_steps_forecast

      c1, c2 = (0.12, 0.47, 0.71), (1.0, 0.5, 0.05)
      plt.plot(np.arange(num_steps), observed_time_series,
               lw=2, color=c1, label='ground truth')

      forecast_steps = np.arange(num_steps_train,
                       num_steps_train+num_steps_forecast)
      plt.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)
      plt.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
               label='forecast')
      plt.fill_between(forecast_steps,
                       forecast_mean - 2 * forecast_scale,
                       forecast_mean + 2 * forecast_scale, color=c2, alpha=0.2)

      plt.xlim([0, num_steps])
      plt.legend()

    plot_forecast(observed_time_series,
                  forecast_mean=forecast_mean,
                  forecast_scale=forecast_scale,
                  forecast_samples=forecast_samples)
  ```

  """

  with tf.name_scope('forecast'):
    [
        observed_time_series,
        mask
    ] = sts_util.canonicalize_observed_time_series_with_mask(
        observed_time_series)

    # Run filtering over the observed timesteps to extract the
    # latent state posterior at timestep T+1 (i.e., the final
    # filtering distribution, pushed through the transition model).
    # This is the prior for the forecast model ("today's prior
    # is yesterday's posterior").
    num_observed_steps = dist_util.prefer_static_value(
        tf.shape(observed_time_series))[-2]
    observed_data_ssm = model.make_state_space_model(
        num_timesteps=num_observed_steps, param_vals=parameter_samples)
    (_, _, _, predictive_means, predictive_covs, _, _
    ) = observed_data_ssm.forward_filter(observed_time_series, mask=mask)

    # Build a batch of state-space models over the forecast period. Because
    # we'll use MixtureSameFamily to mix over the posterior draws, we need to
    # do some shenanigans to move the `[num_posterior_draws]` batch dimension
    # from the leftmost to the rightmost side of the model's batch shape.
    # TODO(b/120245392): enhance `MixtureSameFamily` to reduce along an
    # arbitrary axis, and eliminate `move_dimension` calls here.
    parameter_samples = model._canonicalize_param_vals_as_map(parameter_samples)  # pylint: disable=protected-access
    parameter_samples_with_reordered_batch_dimension = {
        param.name: dist_util.move_dimension(
            parameter_samples[param.name],
            0, -(1 + _prefer_static_event_ndims(param.prior)))
        for param in model.parameters}
    forecast_prior = tfd.MultivariateNormalFullCovariance(
        loc=dist_util.move_dimension(predictive_means[..., -1, :], 0, -2),
        covariance_matrix=dist_util.move_dimension(
            predictive_covs[..., -1, :, :], 0, -3))

    # Ugly hack: because we moved `num_posterior_draws` to the trailing (rather
    # than leading) dimension of parameters, the parameter batch shapes no
    # longer broadcast against the `constant_offset` attribute used in `sts.Sum`
    # models. We fix this by manually adding an extra broadcasting dim to
    # `constant_offset` if present.
    # The root cause of this hack is that we mucked with param dimensions above
    # and are now passing params that are 'invalid' in the sense that they don't
    # match the shapes of the model's param priors. The fix (as above) will be
    # to update MixtureSameFamily so we can avoid changing param dimensions
    # altogether.
    # TODO(b/120245392): enhance `MixtureSameFamily` to reduce along an
    # arbitrary axis, and eliminate this hack.
    kwargs = {}
    if hasattr(model, 'constant_offset'):
      kwargs['constant_offset'] = tf.convert_to_tensor(
          value=model.constant_offset,
          dtype=forecast_prior.dtype)[..., tf.newaxis, :]

    if not include_observation_noise:
      parameter_samples_with_reordered_batch_dimension[
          'observation_noise_scale'] = tf.zeros_like(
              parameter_samples_with_reordered_batch_dimension[
                  'observation_noise_scale'])

    # We assume that any STS model that has a `constant_offset` attribute
    # will allow it to be overridden as a kwarg. This is currently just
    # `sts.Sum`.
    # TODO(b/120245392): when kwargs hack is removed, switch back to calling
    # the public version of `_make_state_space_model`.
    forecast_ssm = model._make_state_space_model(  # pylint: disable=protected-access
        num_timesteps=num_steps_forecast,
        param_map=parameter_samples_with_reordered_batch_dimension,
        initial_state_prior=forecast_prior,
        initial_step=num_observed_steps,
        **kwargs)

    num_posterior_draws = dist_util.prefer_static_value(
        forecast_ssm.batch_shape_tensor())[-1]
    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            logits=tf.zeros([num_posterior_draws], dtype=forecast_ssm.dtype)),
        components_distribution=forecast_ssm)


def impute_missing_values(model,
                          observed_time_series,
                          parameter_samples,
                          include_observation_noise=False):
  """Runs posterior inference to impute the missing values in a time series.

  This method computes the posterior marginals `p(latent state | observations)`,
  given the time series at observed timesteps (a missingness mask should
  be specified using `tfp.sts.MaskedTimeSeries`). It pushes this posterior back
  through the observation model to impute a predictive distribution on the
  observed time series. At unobserved steps, this is an imputed value; at other
  steps it is interpreted as the model's estimate of the underlying noise-free
  series.

  Args:
    model: `tfp.sts.Sum` instance defining an additive STS model.
    observed_time_series: `float` `Tensor` of shape
      `concat([sample_shape, model.batch_shape, [num_timesteps, 1]])` where
      `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
      dimension may (optionally) be omitted if `num_timesteps > 1`. May
      optionally be an instance of `tfp.sts.MaskedTimeSeries` including a
      mask `Tensor` to encode the locations of missing observations.
    parameter_samples: Python `list` of `Tensors` representing posterior
      samples of model parameters, with shapes `[concat([
      [num_posterior_draws], param.prior.batch_shape,
      param.prior.event_shape]) for param in model.parameters]`. This may
      optionally also be a map (Python `dict`) of parameter names to
      `Tensor` values.
    include_observation_noise: If `False`, the imputed uncertainties
      represent the model's estimate of the noise-free time series at each
      timestep. If `True`, they represent the model's estimate of the range of
      values that could be *observed* at each timestep, including any i.i.d.
      observation noise.
      Default value: `False`.

  Returns:
    imputed_series_dist: a `tfd.MixtureSameFamily` instance with event shape
      [num_timesteps] and batch shape `concat([sample_shape,
      model.batch_shape])`, with `num_posterior_draws` mixture components.

  #### Example

  To specify a time series with missing values, use `tfp.sts.MaskedTimeSeries`:

  ```python
  time_series_with_nans = [-1., 1., np.nan, 2.4, np.nan, 5]
  observed_time_series = tfp.sts.MaskedTimeSeries(
    time_series=time_series_with_nans,
    is_missing=tf.math.is_nan(time_series_with_nans))
  ```

  Masked time series can be passed to `tfp.sts` methods in place of a
  `observed_time_series` `Tensor`:

  ```python
  # Build model using observed time series to set heuristic priors.
  linear_trend_model = tfp.sts.LocalLinearTrend(
    observed_time_series=observed_time_series)
  model = tfp.sts.Sum([linear_trend_model],
                      observed_time_series=observed_time_series)

  # Fit model to data
  parameter_samples, _ = tfp.sts.fit_with_hmc(model, observed_time_series)
  ```

  After fitting a model, `impute_missing_values` will return a distribution
  ```python
  # Impute missing values
  imputed_series_distribution = tfp.sts.impute_missing_values(
    model, observed_time_series, parameter_samples=parameter_samples)
  print('imputed means and stddevs: ',
        imputed_series_distribution.mean(),
        imputed_series_distribution.stddev())
  ```

  """
  with tf.name_scope('impute_missing_values'):

    [
        observed_time_series,
        mask
    ] = sts_util.canonicalize_observed_time_series_with_mask(
        observed_time_series)

    # Run smoothing over the training timesteps to extract the
    # predictive means and variances.
    num_timesteps = dist_util.prefer_static_value(
        tf.shape(observed_time_series))[-2]
    lgssm = model.make_state_space_model(
        num_timesteps=num_timesteps, param_vals=parameter_samples)
    posterior_means, posterior_covs = lgssm.posterior_marginals(
        observed_time_series, mask=mask)

    observation_means, observation_covs = lgssm.latents_to_observations(
        latent_means=posterior_means,
        latent_covs=posterior_covs)

    if not include_observation_noise:
      # Extract just the variance of observation noise by pushing forward
      # zero-variance latents.
      _, observation_noise_covs = lgssm.latents_to_observations(
          latent_means=posterior_means,
          latent_covs=tf.zeros_like(posterior_covs))
      # Subtract out the observation noise that was added in the original
      # pushforward. Note that this could cause numerical issues if the
      # observation noise is very large. If this becomes an issue we could
      # avoid the subtraction by plumbing `include_observation_noise` through
      # `lgssm.latents_to_observations`.
      observation_covs -= observation_noise_covs

    # Squeeze dims to convert from LGSSM's event shape `[num_timesteps, 1]`
    # to a scalar time series.
    return sts_util.mix_over_posterior_draws(
        means=observation_means[..., 0],
        variances=observation_covs[..., 0, 0])
