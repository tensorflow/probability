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
import tensorflow as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.sts.internal import util as sts_util


def _prefer_static_event_ndims(distribution):
  if distribution.event_shape.ndims is not None:
    return distribution.event_shape.ndims
  else:
    return tf.size(input=distribution.event_shape_tensor())


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
      dimension may (optionally) be omitted if `num_timesteps > 1`.
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
    (variational_loss,
     variational_distributions) = tfp.sts.build_factored_variational_loss(
       model=model, observed_time_series=observed_time_series)

    # OMITTED: take steps to optimize variational loss

    samples = {k: q.sample(30) for (k, q) in variational_distributions.items()}
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

  with tf.name_scope('one_step_predictive',
                     values=[observed_time_series,
                             parameter_samples]):
    observed_time_series = tf.convert_to_tensor(
        value=observed_time_series, name='observed_time_series')
    observed_time_series = sts_util.maybe_expand_trailing_dim(
        observed_time_series)

    # Run filtering over the training timesteps to extract the
    # predictive means and variances.
    num_timesteps = dist_util.prefer_static_value(
        tf.shape(input=observed_time_series))[-2]
    lgssm = model.make_state_space_model(
        num_timesteps=num_timesteps, param_vals=parameter_samples)
    (_, _, _, _, _, observation_means, observation_covs
    ) = lgssm.forward_filter(observed_time_series)

    # Construct the predictive distribution by mixing over posterior draws.
    # Unfortunately this requires some shenanigans with shapes. The predictive
    # parameters have shape
    #   `concat([
    #      [num_posterior_draws],
    #      observed_time_series.sample_shape,
    #      lgssm.batch_shape,
    #      lgssm.event_shape  # => [num_timesteps, 1]
    #    ]`
    # Because MixtureSameFamily mixes over the rightmost batch dimension,
    # we need to move the `num_posterior_draws` dimension to be rightmost
    # in the batch shape. This requires use of `Independent` (to preserve
    # `num_timesteps` as part of the event shape) and `move_dimension`.
    # TODO(b/120245392): enhance `MixtureSameFamily` to reduce along an
    # arbitrary axis, and eliminate `move_dimension` calls here.
    predictions = tfd.Independent(
        distribution=tfd.Normal(
            loc=dist_util.move_dimension(observation_means[..., 0], 0, -2),
            scale=tf.sqrt(dist_util.move_dimension(
                observation_covs[..., 0, 0], 0, -2))),
        reinterpreted_batch_ndims=1)

    num_posterior_draws = dist_util.prefer_static_value(
        tf.shape(input=observation_means))[0]
    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            logits=tf.zeros([num_posterior_draws],
                            dtype=predictions.dtype)),
        components_distribution=predictions)


def forecast(model,
             observed_time_series,
             parameter_samples,
             num_steps_forecast):
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
      dimension may (optionally) be omitted if `num_timesteps > 1`.
    parameter_samples: Python `list` of `Tensors` representing posterior samples
      of model parameters, with shapes `[concat([[num_posterior_draws],
      param.prior.batch_shape, param.prior.event_shape]) for param in
      model.parameters]`. This may optionally also be a map (Python `dict`) of
      parameter names to `Tensor` values.
    num_steps_forecast: scalar `int` `Tensor` number of steps to forecast.

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
    (variational_loss,
     variational_distributions) = tfp.sts.build_factored_variational_loss(
       model=model, observed_time_series=observed_time_series)

    # OMITTED: take steps to optimize variational loss

    samples = {k: q.sample(30) for (k, q) in variational_distributions.items()}
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

  with tf.name_scope('forecast',
                     values=[observed_time_series,
                             parameter_samples,
                             num_steps_forecast]):
    observed_time_series = tf.convert_to_tensor(
        value=observed_time_series, name='observed_time_series')
    observed_time_series = sts_util.maybe_expand_trailing_dim(
        observed_time_series)

    # Run filtering over the observed timesteps to extract the
    # latent state posterior at timestep T+1 (i.e., the final
    # filtering distribution, pushed through the transition model).
    # This is the prior for the forecast model ("today's prior
    # is yesterday's posterior").
    num_observed_steps = dist_util.prefer_static_value(
        tf.shape(input=observed_time_series))[-2]
    observed_data_ssm = model.make_state_space_model(
        num_timesteps=num_observed_steps, param_vals=parameter_samples)
    (_, _, _, predictive_means, predictive_covs, _, _
    ) = observed_data_ssm.forward_filter(observed_time_series)

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
    forecast_ssm = model.make_state_space_model(
        num_timesteps=num_steps_forecast,
        param_vals=parameter_samples_with_reordered_batch_dimension,
        initial_state_prior=forecast_prior,
        initial_step=num_observed_steps)

    num_posterior_draws = dist_util.prefer_static_value(
        forecast_ssm.batch_shape_tensor())[-1]
    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            logits=tf.zeros([num_posterior_draws], dtype=forecast_ssm.dtype)),
        components_distribution=forecast_ssm)
