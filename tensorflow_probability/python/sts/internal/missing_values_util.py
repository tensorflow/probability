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
"""Utilities for time series with missing values."""
import collections

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

tfl = tf.linalg


class MaskedTimeSeries(collections.namedtuple('MaskedTimeSeries',
                                              ['time_series', 'is_missing'])):
  """Named tuple encoding a time series `Tensor` and optional missingness mask.

  Structural time series models handle missing values naturally, following the
  rules of conditional probability. Posterior inference can be used to impute
  missing values, with uncertainties. Forecasting and posterior decomposition
  are also supported for time series with missing values; the missing values
  will generally lead to corresponding higher forecast uncertainty.

  All methods in the `tfp.sts` API that accept an `observed_time_series`
  `Tensor` should optionally also accept a `MaskedTimeSeries` instance.

  The time series should be a float `Tensor` of shape `[..., num_timesteps]` or
  `[..., num_timesteps, 1]`. The `is_missing` mask must be either a boolean
  `Tensor` of shape `[..., num_timesteps]`, or `None`. `True` values in
  `is_missing` denote missing (masked) observations; `False` denotes observed
  (unmasked) values. Note that these semantics are opposite that of low-level
  TensorFlow methods like `tf.boolean_mask`, but consistent with the behavior
  of Numpy masked arrays.

  The batch dimensions of `is_missing` must broadcast with the batch
  dimensions of `time_series`.

  A `MaskedTimeSeries` is just a `collections.namedtuple` instance, i.e., a dumb
  container. Although the convention for the elements is as described here, it's
  left to downstream methods to validate or convert the elements as required.
  In particular, most downstream methods will call `tf.convert_to_tensor`
  on the components. In order to prevent duplicate `Tensor` creation, you may
  (if memory is an issue) wish to ensure that the components are *already*
  `Tensors`, as opposed to numpy arrays or similar.

  #### Examples

  To construct a simple MaskedTimeSeries instance:

  ```
  observed_time_series = tfp.sts.MaskedTimeSeries(
    time_series=tf.random.normal([3, 4, 5]),
    is_missing=[True, False, False, True, False])
  ```

  Note that the mask we specified will broadcast against the batch dimensions of
  the time series.

  For time series with missing entries specified as NaN 'magic values', you can
  generate a mask using `tf.is_nan`:

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp

  time_series_with_nans = [-1., 1., np.nan, 2.4, np.nan, 5]
  observed_time_series = tfp.sts.MaskedTimeSeries(
    time_series=time_series_with_nans,
    is_missing=tf.is_nan(time_series_with_nans))

  # Build model using observed time series to set heuristic priors.
  linear_trend_model = tfp.sts.LocalLinearTrend(
    observed_time_series=observed_time_series)
  model = tfp.sts.Sum([linear_trend_model],
                      observed_time_series=observed_time_series)

  # Fit model to data
  parameter_samples, _ = tfp.sts.fit_with_hmc(model, observed_time_series)

  # Forecast
  forecast_dist = tfp.sts.forecast(
    model, observed_time_series, num_steps_forecast=5)

  # Impute missing values
  observations_dist = tfp.sts.impute_missing_values(model, observed_time_series)
  print('imputed means and stddevs: ',
        observations_dist.mean(),
        observations_dist.stddev())
  ```

  """
  # This class is just a namedtuple, but we use a subclass to override the
  # docstring, following
  # https://stackoverflow.com/questions/1606436/adding-docstrings-to-namedtuples
  # To keep the child class from constructing instance dicts and losing the
  # lightweightness/immutability of a namedtuple, we need the following magic:
  __slots__ = ()


def moments_of_masked_time_series(time_series_tensor, broadcast_mask):
  """Compute mean and variance, accounting for a mask.

  Args:
    time_series_tensor: float `Tensor` time series of shape
      `concat([batch_shape, [num_timesteps]])`.
    broadcast_mask: bool `Tensor` of the same shape as `time_series`.
  Returns:
    mean: float `Tensor` of shape `batch_shape`.
    variance: float `Tensor` of shape `batch_shape`.
  """
  num_unmasked_entries = ps.cast(
      ps.reduce_sum(ps.cast(~broadcast_mask, np.int32), axis=-1),
      time_series_tensor.dtype)

  # Manually compute mean and variance, excluding masked entries.
  mean = (
      tf.reduce_sum(
          tf.where(
              broadcast_mask,
              tf.zeros([], dtype=time_series_tensor.dtype),
              time_series_tensor),
          axis=-1) / num_unmasked_entries)
  variance = (
      tf.reduce_sum(
          tf.where(
              broadcast_mask,
              tf.zeros([], dtype=time_series_tensor.dtype),
              (time_series_tensor - mean[..., tf.newaxis])**2),
          axis=-1) / num_unmasked_entries)
  return mean, variance


def initial_value_of_masked_time_series(time_series_tensor, broadcast_mask):
  """Get the first unmasked entry of each time series in the batch.

  Args:
    time_series_tensor: float `Tensor` of shape `batch_shape + [num_timesteps]`.
    broadcast_mask: bool `Tensor` of same shape as `time_series`.
  Returns:
    initial_values: float `Tensor` of shape `batch_shape`.  If a batch element
      has no unmasked entries, the corresponding return value for that element
      is undefined.
    first_unmasked_indices: int `Tensor` of shape `batch_shape` -- the index of
      the first unmasked entry for each time series in the batch.  If there are
      no unmasked entries, the returned index is the length of the series.
  """

  num_timesteps = ps.shape(time_series_tensor)[-1]

  # Compute the index of the first unmasked entry for each series in the batch.
  unmasked_negindices = (
      ps.cast(~broadcast_mask, np.int32) *
      ps.range(num_timesteps, 0, -1))
  first_unmasked_indices = num_timesteps - ps.reduce_max(
      unmasked_negindices, axis=-1)
  # Avoid out-of-bounds errors if all indices are masked.
  safe_unmasked_indices = ps.minimum(first_unmasked_indices, num_timesteps - 1)

  batch_dims = tensorshape_util.rank(safe_unmasked_indices.shape)
  if batch_dims is None:
    raise NotImplementedError(
        'Cannot compute initial values of a masked time series with'
        'dynamic rank.')  # `batch_gather` requires static rank

  # Extract the initial value for each series in the batch.
  initial_values = tf.squeeze(
      tf.gather(params=time_series_tensor,
                indices=safe_unmasked_indices[..., np.newaxis],
                batch_dims=batch_dims,
                axis=-1),
      # Since we've gathered exactly one step from the `num_timesteps` axis,
      # we can remove that axis entirely.
      axis=-1)

  return initial_values, first_unmasked_indices


def differentiate_masked_time_series(masked_time_series):
  time_series, is_missing = masked_time_series
  return MaskedTimeSeries(
      time_series=time_series[..., 1:, :] - time_series[..., :-1, :],
      is_missing=(None if is_missing is None
                  else is_missing[..., 1:] | is_missing[..., :-1]))
