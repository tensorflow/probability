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
"""Utilities for automatically building StructuralTimeSeries models."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.sts import structural_time_series
from tensorflow_probability.python.sts.components import local_linear_trend
from tensorflow_probability.python.sts.components import seasonal
from tensorflow_probability.python.sts.components import sum as sum_lib
from tensorflow_probability.python.sts.internal import seasonality_util
from tensorflow_probability.python.sts.internal import util as sts_util

__all__ = [
    'build_default_model',
    'model_from_seasonal_structure'
]


# TODO(davmre): before exposing publicly, consider simplifying this function
# and/or renaming it to something like `auto_build_model`.
def build_default_model(observed_time_series,
                        base_component=local_linear_trend.LocalLinearTrend,
                        name=None):
  """Builds a model with seasonality from a Pandas Series or DataFrame.

  Returns a model of the form
  `tfp.sts.Sum([base_component] + seasonal_components)`, where
  `seasonal_components` are automatically selected using the frequency from the
  `DatetimeIndex` of the provided `pd.Series` or `pd.DataFrame`. If the index
  does not have a set frequency, one will be inferred from the index dates, and

  Args:
    observed_time_series: Instance of `pd.Series` or `pd.DataFrame` containing
      one or more time series indexed by a `pd.DatetimeIndex`.
    base_component: Optional subclass of `tfp.sts.StructuralTimeSeries`
      specifying the model used for residual variation in the series not
      explained by seasonal or other effects. May also be an *instance* of such
      a class with specific priors set; if not provided, such an instance will
      be constructed with heuristic default priors.
      Default value: `tfp.sts.LocalLinearTrend`.
    name: Python `str` name for ops created by this function.
      Default value: `None` (i.e., 'build_default_model').
  Returns:
    model: instance of `tfp.sts.Sum` representing a model for the given data.

  #### Example

  Consider a series of eleven data points, covering a period of two weeks
  with three missing days.

  ```python
  import pandas as pd
  import tensorflow as tf
  import tensorflow_probability as tfp

  series = pd.Series(
    [100., 27., 92., 66., 51., 126., 113., 95., 48., 20., 59.,],
    index=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04',
                          '2020-01-05', '2020-01-06', '2020-01-07',
                          '2020-01-10', '2020-01-11', '2020-01-12',
                          '2020-01-13', '2020-01-14']))
  ```

  Before calling `build_default_model`, we must regularize the series to follow
  a fixed frequency (here, daily observations):

  ```python
  series = tfp.sts.regularize_series(series)
  # len(series) ==> 14
  ```

  The default model will combine a LocalLinearTrend baseline with a Seasonal
  component to capture day-of-week effects. We can then fit this model to our
  observed data. Here we'll use variational inference:

  ```python
  model = tfp.sts.build_default_model(series)
  # len(model.components) == 2

  # Fit the model using variational inference.
  surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(model)
  losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=model.joint_distribution(series).log_prob,
    surrogate_posterior=surrogate_posterior,
    optimizer=tf_keras.optimizers.Adam(0.1),
    num_steps=1000,
    convergence_criterion=(
      tfp.optimizer.convergence_criteria.SuccessiveGradientsAreUncorrelated(
        window_size=20, min_num_steps=50)),
    jit_compile=True)
  parameter_samples = surrogate_posterior.sample(50)
  ```

  Finally, use the fitted parameters to forecast the next week of data:

  ```python
  forecast_dist = tfp.sts.forecast(model,
                                   observed_time_series=series,
                                   parameter_samples=parameter_samples,
                                   num_steps_forecast=7)
  # Strip trailing unit dimension from LinearGaussianStateSpaceModel events.
  forecast_mean = forecast_dist.mean()[..., 0]
  forecast_stddev = forecast_dist.stddev()[..., 0]

  forecast = pd.DataFrame(
      {'mean': forecast_mean,
       'lower_bound': forecast_mean - 2. * forecast_stddev,
       'upper_bound': forecast_mean + 2. * forecast_stddev}
      index=pd.date_range(start=series.index[-1] + series.index.freq,
                          periods=7,
                          freq=series.index.freq))
  ```

  """
  with tf.name_scope(name or 'build_default_model'):
    frequency = getattr(observed_time_series.index, 'freq', None)
    if frequency is None:
      raise ValueError('Provided series has no set frequency. Consider '
                       'using `tfp.sts.regularize_series` to infer a frequency '
                       'and build a regularly spaced series.')
    observed_time_series = sts_util.canonicalize_observed_time_series_with_mask(
        observed_time_series)

    if not isinstance(base_component,
                      structural_time_series.StructuralTimeSeries):
      # Build a component of the given type using default priors.
      base_component = base_component(observed_time_series=observed_time_series)

    seasonal_structure = seasonality_util.create_seasonal_structure(
        frequency=frequency,
        num_steps=int(observed_time_series.time_series.shape[-2]))
    return base_component + model_from_seasonal_structure(
        seasonal_structure, observed_time_series)


def model_from_seasonal_structure(seasonal_structure,
                                  observed_time_series,
                                  allow_drift=True):
  return sum_lib.Sum(
      [  # pylint:disable=g-complex-comprehension
          seasonal.Seasonal(
              num_seasons=season.num,  # pylint: disable=g-complex-comprehension
              num_steps_per_season=season.duration,
              allow_drift=allow_drift,
              observed_time_series=observed_time_series,
              name=str(season_type))
          for season_type, season in seasonal_structure.items()
      ],
      observed_time_series=observed_time_series)
