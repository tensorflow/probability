# Copyright 2021 The TensorFlow Probability Authors.
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
"""Utilities for anomaly detection with STS models and Gibbs sampling."""

import collections
import datetime

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.experimental.sts_gibbs import gibbs_sampler
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.sts import regularization
from tensorflow_probability.python.sts.forecast import one_step_predictive
from tensorflow_probability.python.sts.internal import seasonality_util
from tensorflow_probability.python.sts.internal import util as sts_util

__all__ = [
    'PredictionOutput',
    'detect_anomalies',
    'compute_predictive_bounds',
    'plot_predictions'
]


# Structure of outputs from Autofocus anomaly detection / prediction.
# TODO(davmre): add fields for an optional STS model and parameter samples if
# a flag is passed to `detect_anomalies` to return these.
class PredictionOutput(collections.namedtuple(
    'PredictionOutput',
    ['times', 'observed_time_series', 'mean', 'upper_limit', 'lower_limit',
     'tail_probabilities', 'is_anomaly'])):
  """Predictive probabilities and intervals from `detect_anomalies`.

  Attributes:
    times: Pandas `DatetimeIndex` of the regularized series.
    observed_time_series: float `Tensor` values of the regularized series.
      If the input series was a `DataFrame` with multiple columns, this will
      have shape `[num_columns, num_steps]`; otherwise, it has shape
      `[num_steps]`.
    mean: float `Tensor` of the same shape as `observed_time_series` containing
      predicted values.
    upper_limit: float `Tensor` upper limit of credible intervals for
      `observed_time_series`.
    lower_limit: float `Tensor` lower limit of credible intervals for
      `observed_time_series`.
    tail_probabilities: float `Tensor` p-values of `observed_time_series` under
      the predictive distribution.
    is_anomaly: bool `Tensor` equal to `tail_probabilities < anomaly_threshold`.
  """
  pass


def detect_anomalies(series,
                     anomaly_threshold=0.01,
                     use_gibbs_predictive_dist=False,
                     num_warmup_steps=50,
                     num_samples=100,
                     jit_compile=False,
                     seed=None):
  """Detects anomalies in a Pandas time series using a default seasonal model.

  This function fits a `LocalLinearTrend` model with automatically determined
  seasonal effects, and returns a predictive credible interval at each step
  of the series. The fitting is done via Gibbs sampling, implemented
  specifically for this model class, which sometimes gives useful results more
  quickly than other fitting methods such as VI or HMC.

  Args:
    series: a Pandas `pd.Series` or `pd.DataFrame` instance indexed by a
      `pd.DateTimeIndex`. This may be irregular (missing timesteps) and/or
      contain unobserved steps indicated by `NaN` values (`NaN` values may also
      be provided to indicate future steps at which a forecast is desired).
      Multiple columns in a `pd.DataFrame` will generate results with a batch
      dimension.
    anomaly_threshold: float, confidence level for anomaly detection. An
        anomaly will be detected if the observed series falls outside the
        equal-tailed credible interval containing `(1 - anomaly_threshold)` of
        the posterior predictive probability mass.
    use_gibbs_predictive_dist: Python `bool`. If `True`, the predictive
      distribution is derived from Gibbs samples of the latent level, which
      incorporate information from the entire series *including future
      timesteps*. Otherwise, the predictive distribution is the 'filtering'
      distribution in which (conditioned on sampled parameters) the prediction
      at each step depends only on values observed at previous steps.
      Default value: `False`.
    num_warmup_steps: `int` number of steps to take before starting to collect
      samples.
      Default value: `50`.
    num_samples: `int` number of steps to take while sampling parameter
      values.
      Default value: `100`.
    jit_compile: Python `bool`. If `True`, compile the sampler with XLA. This
      adds overhead to the first call, but may speed up subsequent calls with
      series of the same shape and frequency.
      Default value: `True`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
  Returns:
    prediction_output: instance of `PredictionOutput` named tuple containing
      the predicted credible intervals for each point (omitting the first) in
      the series.
  """
  regularized_series = regularization.regularize_series(series)
  observed_time_series = sts_util.canonicalize_observed_time_series_with_mask(
      regularized_series)
  anomaly_threshold = tf.convert_to_tensor(
      anomaly_threshold, dtype=observed_time_series.time_series.dtype,
      name='anomaly_threshold')

  seasonal_structure = seasonality_util.create_seasonal_structure(
      frequency=regularized_series.index.freq,
      num_steps=len(regularized_series))
  # Convert SeasonType keys into strings, because `tf.function` doesn't like
  # enum-valued arguments.
  seasonal_structure = {str(k): v for (k, v) in seasonal_structure.items()}
  inner_fn = (_detect_anomalies_inner_compiled if jit_compile
              else _detect_anomalies_inner)
  lower_limit, upper_limit, mean, tail_probabilities = inner_fn(
      observed_time_series,
      seasonal_structure=seasonal_structure,
      use_gibbs_predictive_dist=use_gibbs_predictive_dist,
      num_warmup_steps=num_warmup_steps,
      num_samples=num_samples,
      seed=seed)
  return PredictionOutput(
      times=regularized_series.index,
      observed_time_series=observed_time_series.time_series[..., 0],
      mean=mean,
      lower_limit=lower_limit,
      upper_limit=upper_limit,
      tail_probabilities=tail_probabilities,
      is_anomaly=tail_probabilities < anomaly_threshold)


@tf.function(autograph=False)
def _detect_anomalies_inner(observed_time_series,
                            seasonal_structure,
                            anomaly_threshold=0.01,
                            num_warmup_steps=50,
                            num_samples=100,
                            use_gibbs_predictive_dist=True,
                            seed=None):
  """Helper function for `detect_anomalies` to cache `tf.function` traces."""
  with tf.name_scope('build_default_model_for_gibbs_sampling'):
    observed_mean, observed_stddev, _ = sts_util.empirical_statistics(
        observed_time_series)
    # Center the series to have mean 0 and stddev 1. Alternately, we could
    # rescale the priors to match the series, but this is simpler.
    observed_mean = observed_mean[..., tf.newaxis]  # Broadcast with num_steps.
    observed_stddev = observed_stddev[..., tf.newaxis]
    observed_time_series = observed_time_series._replace(
        time_series=(
            observed_time_series.time_series - observed_mean[..., tf.newaxis]
            ) / observed_stddev[..., tf.newaxis])

    model, posterior_samples = _fit_seasonal_model_with_gibbs_sampling(
        observed_time_series,
        seasonal_structure=seasonal_structure,
        num_results=num_samples,
        num_warmup_steps=num_warmup_steps,
        seed=seed)
    parameter_samples = _parameter_samples_from_gibbs_posterior(
        model, posterior_samples)
    if use_gibbs_predictive_dist:
      predictive_dist = gibbs_sampler.one_step_predictive(model,
                                                          posterior_samples)
    else:
      predictive_dist = one_step_predictive(model,
                                            observed_time_series,
                                            timesteps_are_event_shape=False,
                                            parameter_samples=parameter_samples)

    prob_lower = predictive_dist.cdf(observed_time_series.time_series[..., 0])
    tail_probabilities = 2 * tf.minimum(prob_lower, 1 - prob_lower)
    lower_limit, upper_limit, predictive_mean = compute_predictive_bounds(
        predictive_dist, anomaly_threshold=anomaly_threshold)
    restore_scale = lambda x: x * observed_stddev + observed_mean
    return (restore_scale(lower_limit),
            restore_scale(upper_limit),
            restore_scale(predictive_mean),
            tail_probabilities)

_detect_anomalies_inner_compiled = tf.function(
    _detect_anomalies_inner, autograph=False, jit_compile=True)


def _fit_seasonal_model_with_gibbs_sampling(observed_time_series,
                                            seasonal_structure,
                                            num_warmup_steps=50,
                                            num_results=100,
                                            seed=None):
  """Builds a seasonality-as-regression model and fits it by Gibbs sampling."""
  with tf.name_scope('fit_seasonal_model_with_gibbs_sampling'):
    observed_time_series = sts_util.canonicalize_observed_time_series_with_mask(
        observed_time_series)
    dtype = observed_time_series.time_series.dtype
    design_matrix = seasonality_util.build_fixed_effects(
        num_steps=ps.shape(observed_time_series.time_series)[-2],
        seasonal_structure=seasonal_structure,
        dtype=dtype)

    # Default priors.
    # pylint: disable=protected-access
    one = tf.ones([], dtype=dtype)
    level_variance_prior = tfd.InverseGamma(concentration=16,
                                            scale=16. * 0.001**2 * one)
    level_variance_prior._upper_bound = one
    slope_variance_prior = tfd.InverseGamma(concentration=16,
                                            scale=16. * 0.05**2 * one)
    slope_variance_prior._upper_bound = 0.01 * one
    observation_noise_variance_prior = tfd.InverseGamma(
        concentration=0.05, scale=0.05 * one)
    observation_noise_variance_prior._upper_bound = 1.2 * one
    # pylint: enable=protected-access

  model = gibbs_sampler.build_model_for_gibbs_fitting(
      observed_time_series=observed_time_series,
      design_matrix=design_matrix,
      weights_prior=tfd.Normal(loc=0., scale=one),
      level_variance_prior=level_variance_prior,
      slope_variance_prior=slope_variance_prior,
      observation_noise_variance_prior=observation_noise_variance_prior)
  return [
      model,
      gibbs_sampler.fit_with_gibbs_sampling(model,
                                            observed_time_series,
                                            num_results=num_results,
                                            num_warmup_steps=num_warmup_steps,
                                            seed=seed)
  ]


def _parameter_samples_from_gibbs_posterior(model, posterior_samples):
  """Extracts samples of model parameters from Gibbs posterior samples."""
  posterior_samples_dict = posterior_samples._asdict()
  get_param = lambda param_name: posterior_samples_dict[  # pylint: disable=g-long-lambda
      [k for k in posterior_samples_dict if k in param_name][0]]
  return [get_param(param.name) for param in model.parameters]


def compute_predictive_bounds(predictive_dist, anomaly_threshold=0.01):
  """Computes upper and lower bounds (and mean) of the predicted series.

  Args:
    predictive_dist: instance of `tfd.Distribution` having event shape `[]` and
      batch shape `[..., num_steps]`, as returned by
      `tfp.sts.one_step_predictive`.
    anomaly_threshold: scalar float `Tensor` probability mass to leave uncovered
      by the returned intervals. For example, given
      `anomaly_threshold=0.01`, the interval from `lower_limit` to `upper_limit`
      will contain 99% of the probability mass of `predictive_dist`.
      Default value: `0.01`.
  Returns:
    lower_limit: float `Tensor` lower bound of shape
      `predictive_dist.batch_shape`.
    upper_limit: float `Tensor` upper bound of shape
      `predictive_dist.batch_shape`.
    predictive_mean: float `Tensor` predictive mean of shape
      `predictive_dist.batch_shape`. Note that this is the mean, not the
      median, so it is theoretically possible (though unlikely) that it can lie
      outside of the bounds.
  """
  anomaly_threshold = tf.convert_to_tensor(anomaly_threshold,
                                           dtype_hint=predictive_dist.dtype,
                                           name='anomaly_threshold')

  # Since quantiles of a mixture distribution are not analytically available,
  # use scalar root search to compute the upper and lower bounds. The search
  # assumes that the lower and upper bounds are no more than 100 standard
  # deviations away from the mean.
  predictive_mean = predictive_dist.mean()
  predictive_stddev = predictive_dist.stddev()
  target_log_cdfs = tf.reshape(
      [tf.math.log(anomaly_threshold / 2.),
       tf.math.log1p(-anomaly_threshold / 2.)],
      ps.concat([[2], ps.ones_like(ps.shape(predictive_mean))], axis=0))
  limits, _, _ = tfp_math.find_root_chandrupatla(
      lambda x: target_log_cdfs - predictive_dist.log_cdf(x),
      low=predictive_mean - 100 * predictive_stddev,
      high=predictive_mean + 100 * predictive_stddev)
  return limits[0], limits[1], predictive_mean


def plot_predictions(predictions,
                     view_date_begin=None,
                     view_date_end=None,
                     ax=None):
  """Creates a plot of the observed series and model predictions.

  Creates a `matplotlib` plot of the observed time series with intervals from
  the model, predictions for any unobserved points, and the locations of any
  anomalies.

  Args:
    predictions: instance of `PredictionOutput` as returned by
      `detect_anomalies`. This should contain predictions for a single series
      with no batch dimensions.
    view_date_begin: Optional `datetime.datetime` instance.
    view_date_end: Optional `datetime.datetime` instance.
    ax: Optional `matplotlib` figure axis.
  """
  # pylint: disable=g-import-not-at-top
  from matplotlib import pylab as plt
  from matplotlib import dates as mdates
  # pylint: enable=g-import-not-at-top

  if len(predictions.observed_time_series.shape) > 1:
    raise ValueError('Time series must be one-dimensional; batches are not '
                     'supported. Saw shape: {}.'.format(
                         predictions.observed_time_series.shape))

  num_steps = len(predictions.times)
  time_delta = predictions.times[1] - predictions.times[0]
  time_period_length = predictions.times[-1] - predictions.times[0]
  if view_date_begin is None:
    view_date_begin = predictions.times[0] - 0.04 * time_period_length
  if view_date_end is None:
    view_date_end = predictions.times[-1] + 0.04 * time_period_length
  if not (isinstance(view_date_begin, datetime.datetime)
          and isinstance(view_date_end, datetime.datetime)):
    raise ValueError('View date start and end must be `datetime.datetime` '
                     'instances.')

  if ax is None:  # Create default axis.
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

  # Plot series with upper and lower limits.
  ax.plot(predictions.times,
          predictions.observed_time_series,
          color='black', alpha=0.8)
  ax.fill_between(predictions.times,
                  predictions.lower_limit,
                  predictions.upper_limit,
                  color='tab:blue', alpha=0.3)
  # At steps where no time series was observed, plot the predictive mean.
  ax.plot(predictions.times,
          np.where(np.isnan(predictions.observed_time_series),
                   predictions.mean,
                   np.nan),
          color='black', alpha=0.8, ls='--')

  # Highlight anomalies.
  for anomaly_idx in np.flatnonzero(predictions.is_anomaly):
    x = predictions.times[anomaly_idx]
    y = predictions.observed_time_series[anomaly_idx]
    ax.scatter(x, y, s=100, alpha=0.4, c='red')
    ax.annotate(str(x), (x, y))
  ax.set_ylabel('Series')
  ax.label_outer()

  # Use smart date formatting for the x axis.
  locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
  ax.xaxis.set_major_locator(locator)
  ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
  ax.set_xlim([view_date_begin, view_date_end])
  ax.grid(True, color='whitesmoke')
  # Display the grid *underneath* the rest of the plot
  # (see https://github.com/matplotlib/matplotlib/issues/5045/).
  ax.set_axisbelow(True)
