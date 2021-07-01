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
"""Utilities for regularizing a time series to a fixed frequency."""
import collections
import itertools
import math
import warnings

import numpy as np


__all__ = [
    'MissingValuesTolerance',
    'regularize_series',
]


MissingValuesTolerance = collections.namedtuple(
    'MissingValuesTolerance',
    ['overall_fraction',
     'fraction_low_missing_number',
     'fraction_high_missing_number',
     'low_missing_number',
     'high_missing_number'])


# Map from Pandas to Numpy timedelta identifiers.
_PD_TO_NP_DELTAS = {'weeks': 'W', 'days': 'D', 'hours': 'h', 'minutes': 'm',
                    'seconds': 's', 'milliseconds': 'ms', 'microseconds': 'us',
                    'nanoseconds': 'ns'}


# This defines valid dtypes for time series values.
_VALID_DATA_TYPES = (np.float16, np.float32, np.float64, np.int32, np.int64)


def regularize_series(series,
                      frequency=None,
                      warn_missing_tolerance=None,
                      err_missing_tolerance=None,
                      max_series_length=None):
  """Infers frequency and makes an irregular time series regular.

  Converts a time series into a regular time series having the same
  period between successive time points (e.g. 5 seconds, or 1 day).  If
  the frequency is known, it can be supplied through the 'frequency'
  argument; otherwise it will be inferred.

  If multiple values share the same timestamp, they are summed into a single
  value.

  Args:
    series: a Pandas `pd.Series` instance indexed by a `pd.DateTimeIndex`. This
      may also be a single-column `pd.DataFrame`.
    frequency: a Pandas DateOffset object, e.g. `pd.DateOffset(seconds=1)`. If
      no frequency is specified, and the index of `series` does not have a
      frequency populated, the granularity of the time series will be inferred
      automatically.
      Default value: `None`.
    warn_missing_tolerance: optional instance of
      `tfp.sts.MissingValuesTolerance`, specifying warning thresholds for
      too many missing values.
      Default value: `None`. (do not warn).
    err_missing_tolerance: optional instance of
      `tfp.sts.MissingValuesTolerance`, specifying error thresholds for
      too many missing values.
      Default value: `None`. (do not raise errors).
    max_series_length: `int` maximum length of the regularized
      series (note that regularization may increase the length of the series).
      Used to bound the resources used per invocation.
      Default value: `None`.
  Returns:
    regularized_series: instance of the same type as `series`
      (`pd.Series` or `pd.DataFrame`) whose index follows a regular
      frequency (`regularized_series.index.freq is not None`). Any values
      not provided are filled in as `NaN`.
  Raises:
    TypeError: if `data` is not an instance of `pd.Series` or `pd.DataFrame`.
    ValueError:  if `data` is empty, `data.index` is not a DatetimeIndex,
                 `data.index` is not sorted, or if applying the inferred
                 frequency would exceed the `max_series_length` or create
                 more missing values than allowed by `err_missing_vals`.
  """
  # pylint: disable=unused-import,g-import-not-at-top
  import pandas as pd  # Defer import to avoid a package-level Pandas dep.
  from pandas.core.resample import asfreq  # see b/169217869
  # pylint: enable=unused-import,g-import-not-at-top

  _check_data(series)

  # Sum all values provided at each time step, if there is more than one.
  series = series.groupby(
      by=lambda x: x
      # Use numpy sum because just calling `groupby().sum()` would drop NaNs.
      ).agg(lambda x: np.sum(x.values))

  if not frequency:
    frequency = _infer_frequency(series.index)

  # If the frequency is monthly and the first date is the end of a
  # month, follow that convention in future months. Note that this condition
  # is not triggered if the frequency is already `MonthEnd` (which has
  # `kwds == {}`).
  if ('months' in frequency.kwds and
      len(frequency.kwds) == 1 and
      np.all(series.index.is_month_end)):
    frequency = pd.offsets.MonthEnd(n=frequency.kwds['months'])

  if (max_series_length is not None and
      max_series_length < _estimate_num_steps(
          series.index[0], series.index[-1], frequency)):
    raise ValueError("Applying inferred frequency {} to the time period "
                     "starting at '{}' and ending at '{}' would exceed the "
                     "maximum series length ({}).".format(
                         frequency,
                         series.index[0],
                         series.index[-1],
                         max_series_length))

  regularized_series = series.asfreq(frequency)
  if warn_missing_tolerance or err_missing_tolerance:
    _check_missing_values(regularized_series,
                          warn_vals=warn_missing_tolerance,
                          err_vals=err_missing_tolerance)
  return regularized_series


def _check_missing_values(series, warn_vals, err_vals=None):
  """Checks for excess missing values after making a series regular.

  After setting of automatic granularity and/or making a time series regular,
  it may contain a large number of missing values. This method will throw
  an error or warning if a series contains a high fraction or raw number
  missing values. See _ERR_VALS and WARN_VALS for thresholds.

  Args:
    series: instance of `pd.Series` or `pd.DataFrame`.
    warn_vals: optional instance of `tfp.sts.MissingValuesTolerance`
      specifying thresholds at which to warn about too many missing values.
    err_vals: optional instance of `tfp.sts.MissingValuesTolerance`
      specifying thresholds at which to raise an error about too many missing
      values.
      Default value: `None`.

  Raises:
    ValueError: if the series contains too many missing values.
  """
  missing_number = np.sum(np.isnan(series.values))
  total_number = np.prod(series.shape)
  missing_fraction = missing_number / total_number
  missing_msg = 'Too many missing values: {} out of {}.'.format(
      missing_number, total_number)
  warning_msg = 'Large number of missing values: {}'.format(missing_msg)

  # Raise an Error (default) or warning if too many missing values.
  if err_vals:
    if (missing_fraction >= err_vals.overall_fraction or
        (missing_fraction >= err_vals.fraction_low_missing_number and
         missing_number >= err_vals.low_missing_number) or
        (missing_fraction >= err_vals.fraction_high_missing_number and
         missing_number >= err_vals.high_missing_number)):
      raise ValueError('Too many missing values: ' + missing_msg)

  # Raise a warning in case of a lot of missing values.
  if (missing_fraction >= warn_vals.overall_fraction or
      (missing_fraction >= warn_vals.fraction_low_missing_number and
       missing_number >= warn_vals.low_missing_number) or
      (missing_fraction >= warn_vals.fraction_high_missing_number and
       missing_number >= warn_vals.high_missing_number)):
    warnings.warn(warning_msg)


def _check_data(data):
  """Performs validation checks on pandas input data."""
  # Defer import to avoid a package-level Pandas dep.
  import pandas as pd  # pylint: disable=g-import-not-at-top
  if not isinstance(data, (pd.Series, pd.DataFrame)):
    raise TypeError('Expected a pandas Series or DataFrame.')
  if data.empty:
    raise ValueError('Input data is empty')
  if not isinstance(data.index,
                    pd.core.indexes.datetimes.DatetimeIndex):
    raise ValueError('Input data index is not a DatetimeIndex')
  if data.values.dtype not in _VALID_DATA_TYPES:
    raise ValueError('Invalid data type. '
                     'Valid types are: {}. '.format(_VALID_DATA_TYPES) +
                     'Received: {}'.format(data.values.dtype))
  if not data.index.is_monotonic_increasing:
    raise ValueError('Input data index is not sorted.')


def _infer_frequency(date_time_index):
  """Infers frequency from a Pandas DatetimeIndex.

  The frequency is automatically inferred as follows:
      1. Computes the time differences between all time points and determine
         the smallest difference.
      2. For the smallest time difference determine the smallest time
         component from 'seconds', 'minutes, 'hours', 'days', and 'weeks'.
      3. Convert all time differences to the smallest time component determined
         in (2).
      4. Find the greatest common denominator (gcd) determined from the
         resulting time differences in (3). This is used to automatically
         set a time series frequency.

  Args:
    date_time_index: instance of Pandas.DatetimeIndex. Typically this is
      `df.index` for an appropriate dateframe `df`.
  Returns:
    frequency: The inferred frequency as a `pd.DateOffset` instance. This will
      either be a special offset, like `pd.offsets.MonthEnd()`, or will be a
      base `pd.DateOffset` instance with a single keyword component
      (e.g., `DateOffset(hours=26)` rather than `DateOffset(days=1, hours=2)`).
  """
  # Defer import to avoid a package-level Pandas dep.
  import pandas as pd  # pylint: disable=g-import-not-at-top

  # Compute series time deltas and get their minimum.
  diffs = pd.Series(date_time_index).diff()[1:]
  diffs_table = diffs.value_counts()
  min_diff = diffs_table.index.min()

  # Extract datetime components and identify smallest time component.
  min_diff_components = pd.Series(min_diff).dt.components
  available_components = list(min_diff_components)
  components_present = min_diff_components.values[0] > 0
  smallest_unit = list(itertools.compress(available_components,
                                          components_present))[-1]

  irregular_freqs = []
  if smallest_unit == 'days' and min_diff.days >= 365:
    # Attempt to infer a yearly frequency.
    irregular_freqs += [pd.DateOffset(years=min_diff.days // 365)]
  if smallest_unit == 'days' and min_diff.days >= 28:
    # Attempt to infer a monthly frequency. Note that the candidate interval of
    # `days // 28` will fail for intervals larger than 11 months.
    irregular_freqs += [pd.DateOffset(months=min_diff.days // 28),
                        pd.offsets.MonthEnd(n=min_diff.days // 28)]
  for candidate_freq in irregular_freqs:
    # If the candidate frequency explains all of the provided dates, it's
    # probably a reasonable choice.
    if set(pd.date_range(date_time_index.min(),
                         date_time_index.max(),
                         freq=candidate_freq)).issuperset(date_time_index):
      return candidate_freq

  # Pandas Timedelta does not support 'weeks' time components by default.
  # Deal with that special case.
  if smallest_unit == 'days' and min_diff.days % 7 == 0:
    smallest_unit = 'weeks'

  # Express time differences in common unit
  series_divider = np.timedelta64(1, _PD_TO_NP_DELTAS[smallest_unit])
  diffs_common = list((diffs / series_divider).astype(int))

  # Compute the greatest common denominator of time differences.
  diffs_gcd = diffs_common[0]
  for d in diffs_common[1:]:
    diffs_gcd = math.gcd(diffs_gcd, d)

  return pd.DateOffset(**{smallest_unit: diffs_gcd})


def _estimate_num_steps(start_time, end_time, freq):
  """Estimates the number of steps between the given times at the given freq."""
  # Unfortunately `(end_time - start_time) / freq` doesn't work in general,
  # because some frequencies (e.g., MonthEnd) don't correspond to a fixed length
  # of time. Instead, we use a 'typical' length estimated by averaging over a
  # small number of steps. This recovers the exact calculation when `freq` does
  # have a fixed length (e.g., is measured in seconds, minutes, etc.).
  timedelta = ((start_time + 10 * freq) - start_time) / 10.
  return (end_time - start_time) / timedelta
