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
"""Structural Time Series utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf


def empirical_statistics(observed_time_series):
  """Compute statistics of a provided time series, as heuristic initialization.

  Args:
    observed_time_series: `Tensor` representing a time series, or batch of time
       series, of shape either `batch_shape + [num_timesteps, 1]` or
       `batch_shape + [num_timesteps]` (allowed if `num_timesteps > 1`).

  Returns:
    observed_stddev: `Tensor` of shape `batch_shape`, giving the empirical
      standard deviation of each time series in the batch.
    observed_initial: `Tensor of shape `batch_shape`, giving the initial value
      of each time series in the batch.
  """

  with tf.name_scope("empirical_statistics", values=[observed_time_series]):
    observed_time_series = tf.convert_to_tensor(
        observed_time_series, name="observed_time_series")
    observed_time_series = maybe_expand_trailing_dim(observed_time_series)
    _, observed_variance = tf.nn.moments(
        tf.squeeze(observed_time_series, -1), axes=-1)
    observed_stddev = tf.sqrt(observed_variance)
    observed_initial = observed_time_series[..., 0, 0]
    return observed_stddev, observed_initial


def maybe_expand_trailing_dim(observed_time_series):
  """Ensures `observed_time_series` has a trailing dimension of size 1.

  This utility method tries to make time-series shape handling more ergonomic.
  The `tfd.LinearGaussianStateSpaceModel` Distribution has event shape of
  `[num_timesteps, observation_size]`, but canonical BSTS models
  are univariate, so their observation_size is always `1`. The extra trailing
  dimension gets annoying, so this method allows arguments with or without the
  extra dimension. There is no ambiguity except in the trivial special case
  where  `num_timesteps = 1`; this can be avoided by specifying any unit-length
  series in the explicit `[num_timesteps, 1]` style.

  Args:
    observed_time_series: `Tensor` of shape `batch_shape + [num_timesteps, 1]`
      or `batch_shape + [num_timesteps]`, where `num_timesteps > 1`.

  Returns:
    expanded_time_series: `Tensor` of shape `batch_shape + [num_timesteps, 1]`.
  """

  with tf.name_scope(
      "maybe_expand_trailing_dim", values=[observed_time_series]):
    if (observed_time_series.shape.ndims is not None and
        observed_time_series.shape[-1].value is not None):
      expanded_time_series = (
          observed_time_series if observed_time_series.shape[-1] == 1 else
          observed_time_series[..., tf.newaxis])
    else:
      expanded_time_series = tf.cond(
          tf.equal(tf.shape(observed_time_series)[-1], 1),
          lambda: observed_time_series,
          lambda: observed_time_series[..., tf.newaxis])
    return expanded_time_series
