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
"""Methods for decomposing StructuralTimeSeries models."""
import collections

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental import util as tfe_util
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.sts.internal import util as sts_util


def _split_covariance_into_marginals(covariance, block_sizes):
  """Split a covariance matrix into block-diagonal marginals of given sizes."""
  start_dim = 0
  marginals = []
  for size in block_sizes:
    end_dim = start_dim + size
    marginals.append(covariance[..., start_dim:end_dim, start_dim:end_dim])
    start_dim = end_dim
  return marginals


def _decompose_from_posterior_marginals(
    model, posterior_means, posterior_covs, parameter_samples, initial_step=0):
  """Utility method to decompose a joint posterior into components.

  Args:
    model: `tfp.sts.Sum` instance defining an additive STS model.
    posterior_means: float `Tensor` of shape `concat(
      [[num_posterior_draws], batch_shape, num_timesteps, latent_size])`
      representing the posterior mean over latents in an
      `AdditiveStateSpaceModel`.
    posterior_covs: float `Tensor` of shape `concat(
      [[num_posterior_draws], batch_shape, num_timesteps,
      latent_size, latent_size])`
      representing the posterior marginal covariances over latents in an
      `AdditiveStateSpaceModel`.
    parameter_samples: Python `list` of `Tensors` representing posterior
      samples of model parameters, with shapes `[concat([
      [num_posterior_draws], param.prior.batch_shape,
      param.prior.event_shape]) for param in model.parameters]`. This may
      optionally also be a map (Python `dict`) of parameter names to
      `Tensor` values.
    initial_step: optional `int` specifying the initial timestep of the
      decomposition.

  Returns:
    component_dists: A `collections.OrderedDict` instance mapping
      component StructuralTimeSeries instances (elements of `model.components`)
      to `tfd.Distribution` instances representing the posterior marginal
      distributions on the process modeled by each component. Each distribution
      has batch shape matching that of `posterior_means`/`posterior_covs`, and
      event shape of `[num_timesteps]`.
  """

  try:
    model.components
  except AttributeError:
    raise ValueError('Model decomposed into components must be an instance of'
                     '`tfp.sts.Sum` (passed model {})'.format(model))

  with tf.name_scope('decompose_from_posterior_marginals'):

    # Extract the component means/covs from the joint latent posterior.
    latent_sizes = [component.latent_size for component in model.components]
    component_means = tf.split(posterior_means, latent_sizes, axis=-1)
    component_covs = _split_covariance_into_marginals(
        posterior_covs, latent_sizes)

    # Instantiate per-component state space models, and use them to push the
    # posterior means/covs through the observation model for each component.
    num_timesteps = dist_util.prefer_static_value(
        tf.shape(posterior_means))[-2]
    component_ssms = model.make_component_state_space_models(
        num_timesteps=num_timesteps,
        param_vals=parameter_samples, initial_step=initial_step)
    component_predictive_dists = collections.OrderedDict()
    for (component, component_ssm,
         component_mean, component_cov) in zip(model.components, component_ssms,
                                               component_means, component_covs):
      component_obs_mean, component_obs_cov = (
          component_ssm.latents_to_observations(
              latent_means=component_mean,
              latent_covs=component_cov))

      # Using the observation means and covs, build a mixture distribution
      # that integrates over the posterior draws.
      component_predictive_dists[component] = sts_util.mix_over_posterior_draws(
          means=component_obs_mean[..., 0],
          variances=component_obs_cov[..., 0, 0])
  return component_predictive_dists


def decompose_by_component(model, observed_time_series, parameter_samples):
  """Decompose an observed time series into contributions from each component.

  This method decomposes a time series according to the posterior represention
  of a structural time series model. In particular, it:
    - Computes the posterior marginal mean and covariances over the additive
      model's latent space.
    - Decomposes the latent posterior into the marginal blocks for each
      model component.
    - Maps the per-component latent posteriors back through each component's
      observation model, to generate the time series modeled by that component.

  Args:
    model: An instance of `tfp.sts.Sum` representing a structural time series
      model.
    observed_time_series: optional `float` `Tensor` of shape
        `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
        supported when `T > 1`), specifying an observed time series. Any `NaN`s
        are interpreted as missing observations; missingness may be also be
        explicitly specified by passing a `tfp.sts.MaskedTimeSeries` instance.
    parameter_samples: Python `list` of `Tensors` representing posterior
      samples of model parameters, with shapes `[concat([
      [num_posterior_draws], param.prior.batch_shape,
      param.prior.event_shape]) for param in model.parameters]`. This may
      optionally also be a map (Python `dict`) of parameter names to
      `Tensor` values.
  Returns:
    component_dists: A `collections.OrderedDict` instance mapping
      component StructuralTimeSeries instances (elements of `model.components`)
      to `tfd.Distribution` instances representing the posterior marginal
      distributions on the process modeled by each component. Each distribution
      has batch shape matching that of `posterior_means`/`posterior_covs`, and
      event shape of `[num_timesteps]`.

  #### Examples

  Suppose we've built a model and fit it to data:

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

    num_steps_forecast = 50
    samples, kernel_results = tfp.sts.fit_with_hmc(model, observed_time_series)
  ```

  To extract the contributions of individual components, pass the time series
  and sampled parameters into `decompose_by_component`:

  ```python
    component_dists = decompose_by_component(
      model,
      observed_time_series=observed_time_series,
      parameter_samples=samples)

    # Component mean and stddev have shape `[len(observed_time_series)]`.
    day_of_week_effect_mean = component_dists[day_of_week].mean()
    day_of_week_effect_stddev = component_dists[day_of_week].stddev()
  ```

  Using the component distributions, we can visualize the uncertainty for
  each component:

  ```
  from matplotlib import pylab as plt
  num_components = len(component_dists)
  xs = np.arange(len(observed_time_series))
  fig = plt.figure(figsize=(12, 3 * num_components))
  for i, (component, component_dist) in enumerate(component_dists.items()):

    # If in graph mode, replace `.numpy()` with `.eval()` or `sess.run()`.
    component_mean = component_dist.mean().numpy()
    component_stddev = component_dist.stddev().numpy()

    ax = fig.add_subplot(num_components, 1, 1 + i)
    ax.plot(xs, component_mean, lw=2)
    ax.fill_between(xs,
                    component_mean - 2 * component_stddev,
                    component_mean + 2 * component_stddev,
                    alpha=0.5)
    ax.set_title(component.name)
  ```

  """

  with tf.name_scope('decompose_by_component'):
    [
        observed_time_series,
        is_missing
    ] = sts_util.canonicalize_observed_time_series_with_mask(
        observed_time_series)

    # Run smoothing over the training timesteps to extract the
    # posterior on latents.
    num_timesteps = dist_util.prefer_static_value(
        tf.shape(observed_time_series))[-2]
    ssm = tfe_util.JitPublicMethods(
        model.make_state_space_model(num_timesteps=num_timesteps,
                                     param_vals=parameter_samples),
        trace_only=True)  # Avoid eager overhead w/o introducing XLA dependence.
    posterior_means, posterior_covs = ssm.posterior_marginals(
        observed_time_series, mask=is_missing)

    return _decompose_from_posterior_marginals(
        model, posterior_means, posterior_covs, parameter_samples)


def decompose_forecast_by_component(model, forecast_dist, parameter_samples):
  """Decompose a forecast distribution into contributions from each component.

  Args:
    model: An instance of `tfp.sts.Sum` representing a structural time series
      model.
    forecast_dist: A `Distribution` instance returned by `tfp.sts.forecast()`.
      (specifically, must be a `tfd.MixtureSameFamily` over a
      `tfd.LinearGaussianStateSpaceModel` parameterized by posterior samples).
    parameter_samples: Python `list` of `Tensors` representing posterior samples
      of model parameters, with shapes `[concat([[num_posterior_draws],
      param.prior.batch_shape, param.prior.event_shape]) for param in
      model.parameters]`. This may optionally also be a map (Python `dict`) of
      parameter names to `Tensor` values.
  Returns:
    component_forecasts: A `collections.OrderedDict` instance mapping
      component StructuralTimeSeries instances (elements of `model.components`)
      to `tfd.Distribution` instances representing the marginal forecast for
      each component. Each distribution has batch and event shape matching
      `forecast_dist` (specifically, the event shape is
      `[num_steps_forecast]`).

  #### Examples

  Suppose we've built a model, fit it to data, and constructed a forecast
  distribution:

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

    num_steps_forecast = 50
    samples, kernel_results = tfp.sts.fit_with_hmc(model, observed_time_series)
    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                 parameter_samples=samples,
                                 num_steps_forecast=num_steps_forecast)
  ```

  To extract the forecast for individual components, pass the forecast
  distribution into `decompose_forecast_by_components`:

  ```python
    component_forecasts = decompose_forecast_by_component(
      model, forecast_dist, samples)

    # Component mean and stddev have shape `[num_steps_forecast]`.
    day_of_week_effect_mean = forecast_components[day_of_week].mean()
    day_of_week_effect_stddev = forecast_components[day_of_week].stddev()
  ```

  Using the component forecasts, we can visualize the uncertainty for each
  component:

  ```
  from matplotlib import pylab as plt
  num_components = len(component_forecasts)
  xs = np.arange(num_steps_forecast)
  fig = plt.figure(figsize=(12, 3 * num_components))
  for i, (component, component_dist) in enumerate(component_forecasts.items()):

    # If in graph mode, replace `.numpy()` with `.eval()` or `sess.run()`.
    component_mean = component_dist.mean().numpy()
    component_stddev = component_dist.stddev().numpy()

    ax = fig.add_subplot(num_components, 1, 1 + i)
    ax.plot(xs, component_mean, lw=2)
    ax.fill_between(xs,
                    component_mean - 2 * component_stddev,
                    component_mean + 2 * component_stddev,
                    alpha=0.5)
    ax.set_title(component.name)
  ```

  """

  with tf.name_scope('decompose_forecast_by_component'):
    try:
      forecast_lgssm = forecast_dist.components_distribution
      forecast_latent_mean, _ = forecast_lgssm._joint_mean()  # pylint: disable=protected-access
      forecast_latent_covs, _ = forecast_lgssm._joint_covariances()  # pylint: disable=protected-access
    except AttributeError as e:
      raise ValueError(
          'Forecast distribution must be a MixtureSameFamily of'
          'LinearGaussianStateSpaceModel distributions, such as returned by'
          '`tfp.sts.forecast()`. (saw exception: {})'.format(e))

    # Since `parameter_samples` will have sample shape `[num_posterior_draws]`,
    # we need to move the `num_posterior_draws` dimension of the forecast
    # moments from the trailing batch dimension, where it's currently put by
    # `sts.forecast`, back to the leading (sample shape) dimension.
    forecast_latent_mean = dist_util.move_dimension(
        forecast_latent_mean, source_idx=-3, dest_idx=0)
    forecast_latent_covs = dist_util.move_dimension(
        forecast_latent_covs, source_idx=-4, dest_idx=0)
    return _decompose_from_posterior_marginals(
        model, forecast_latent_mean, forecast_latent_covs, parameter_samples,
        initial_step=forecast_lgssm.initial_step)
