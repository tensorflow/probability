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
"""Utilities for inferring and representing seasonality."""
import collections
import enum

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps


class SeasonTypes(enum.Enum):
  SECOND_OF_MINUTE = 0,
  MINUTE_OF_HOUR = 1,
  HOUR_OF_DAY = 2,
  DAY_OF_WEEK = 3
  MONTH_OF_YEAR = 4


SeasonConversion = collections.namedtuple(
    'SeasonConversion', ['num', 'duration'])


_SEASONAL_PROTOTYPES = collections.OrderedDict({
    SeasonTypes.SECOND_OF_MINUTE: SeasonConversion(num=60, duration=1),
    SeasonTypes.MINUTE_OF_HOUR: SeasonConversion(num=60, duration=60),
    SeasonTypes.HOUR_OF_DAY: SeasonConversion(num=24, duration=3600),
    SeasonTypes.DAY_OF_WEEK: SeasonConversion(num=7, duration=86400)
})


def create_seasonal_structure(frequency, num_steps, min_cycles=2):
  """Creates a set of suitable seasonal structures for a time series.

  Args:
    frequency: a Pandas `pd.DateOffset` instance.
    num_steps: Python `int` number of steps at the given frequency.
    min_cycles: Python `int` minimum number of cycles to include an effect.
  Returns:
    A dictionary of SeasonConversion instances representing different
    seasonal components.

  Example 1: For data.index.freq: pd.DateOffset(hours=1)
  Seasonal components:
  {
    SeasonTypes.HOUR_OF_DAY: SeasonConversion(num=24, duration=1),
    SeasonTypes.DAY_OF_WEEK: SeasonConversion(num=7, duration=24)
  }

  Example 2: For data.index.freq: pd.DateOffset(seconds=30)
  Seasonal components:
  {
    SeasonTypes.SECOND_OF_MINUTE: SeasonConversion(num=2, duration=1),
    SeasonTypes.MINUTE_OF_HOUR: SeasonConversion(num=60, duration=2),
    SeasonTypes.HOUR_OF_DAY: SeasonConversion(num=24, duration=120),
    SeasonTypes.DAY_OF_WEEK: SeasonConversion(num=7, duration=2880)
  }

  If the frequency is N times per year, for integer 2 <= N <= 12 (e.g.,
  12 for monthly or 4 for quarterly), then a fixed structure of (N, 1)
  will be created.

  """
  num_periods = periods_per_year(frequency)
  if num_periods is not None:
    # Fixed structure for monthly or quarterly data.
    return {
        SeasonTypes.MONTH_OF_YEAR: SeasonConversion(num=num_periods, duration=1)
    }

  # Compute seasonal components by cycling through _SEASONAL_PROTOTYPES and
  # filter out None components.
  components = {  # pylint: disable=g-complex-comprehension
      k: make_component(v,
                        frequency=frequency,
                        num_steps=num_steps,
                        min_cycles=min_cycles)
      for k, v in _SEASONAL_PROTOTYPES.items()}
  return {k: v for (k, v) in components.items() if v is not None}


def make_component(season_tuple, frequency, num_steps, min_cycles=2):
  """Make a seasonal component from a template component.

  This is a helper function to the _create_seasonal_structure() method. It
  takes a SeasonConversion instance from _SEASONAL_PROTOTYPES and
  creates a seasonal component based on the number of observations
  `num_steps` in the data and the time series frequency `freq_sec`. A
  custom seasonal component is created if it fulfills 4 conditions:
    Condition 1: time series must cover at least _MIN_CYCLES full cycles.
    Condition 2: full cycle must be a multiple of the granularity.
    Condition 3: if the season is longer than the granularity, it must be a
      multiple of the granularity.
    Condition 4: number of seasons must be greater than 1.

  Args:
    season_tuple: an `SeasonConversion` instance the number of
      seasons, and season duration for a template seasonal component e.g.
      (60, 1) for seconds-of-minute or (60, 60) for minute-of-hour.
      See _SEASONAL_PROTOTYPES for more details.
    frequency: a `pd.DateOffset` instance.
    num_steps: Python `int` number of steps at the given frequency.
    min_cycles: Python `int` minimum number of cycles to include an effect.

  Returns:
    An `SeasonConversion` instance, where num and duration
    is the inferred structure for the seasonal component. If a seasonal
    component can not be created it returns None for that component.
  """
  freq_sec = freq_to_seconds(frequency)
  if not freq_sec:
    return None

  num_seasons = season_tuple.num
  duration_seconds = season_tuple.duration
  #  None component returned if no component can be created below.
  component = None
  # Condition 1: time series must cover at least _MIN_CYCLES full cycles.
  minimum_observations = ((num_seasons * duration_seconds * min_cycles) /
                          freq_sec)
  cond1 = num_steps >= minimum_observations

  # Condition 2: full cycle must be a multiple of the granularity.
  cond2 = (num_seasons * duration_seconds) % freq_sec == 0
  # Condition 3: if the season is longer than the granularity, it must be a
  # multiple of the granularity.
  cond3 = ((duration_seconds <= freq_sec) or
           (duration_seconds % freq_sec == 0))
  if cond1 and cond2 and cond3:
    nseasons = min(num_seasons * duration_seconds /
                   freq_sec, num_seasons)
    season_duration = max(duration_seconds / freq_sec, 1)
    # Condition 4: number of seasons must be greater than 1.
    cond4 = ((nseasons > 1) and (nseasons <= num_seasons))
    if cond4:
      component = SeasonConversion(
          num=int(nseasons),
          duration=int(season_duration))
  return component


def _design_matrix_for_one_seasonal_effect(num_steps, duration, period, dtype):
  current_period = np.int32(np.arange(num_steps) / duration) % period
  return np.transpose([
      ps.where(current_period == p,  # pylint: disable=g-complex-comprehension
               ps.ones([], dtype=dtype),
               ps.zeros([], dtype=dtype))
      for p in range(period)])


def build_fixed_effects(num_steps,
                        seasonal_structure=None,
                        covariates=None,
                        dtype=tf.float32):
  """Builds a design matrix treating seasonality as fixed-effects regression."""
  if seasonal_structure is None:
    seasonal_structure = {}
  if seasonal_structure:
    design_matrix = ps.concat(
        [
            _design_matrix_for_one_seasonal_effect(
                num_steps, seasonal_effect.duration, seasonal_effect.num, dtype)
            for seasonal_effect in seasonal_structure.values()
        ], axis=-1)
  else:
    design_matrix = ps.ones([num_steps, 1], dtype=dtype)

  if covariates:
    design_matrix = ps.concat(
        [design_matrix] +
        [tf.convert_to_tensor(x)[..., :num_steps, :] for x in covariates],
        axis=-1)
  return design_matrix


def freq_to_seconds(freq):
  """Converts time series DateOffset frequency to seconds."""
  if not freq:
    return None
  if not is_fixed_duration(freq):
    return None

  freq_secs = 0.
  for kwds_unit, kwds_value in freq.kwds.items():
    switch_to_seconds = {
        'weeks': kwds_value * 60 * 60 * 24 * 7,
        'days': kwds_value * 60 * 60 * 24,
        'hours': kwds_value * 60 * 60,
        'minutes': kwds_value * 60,
        'seconds': kwds_value
    }
    freq_secs += switch_to_seconds[kwds_unit]
  return freq_secs


def periods_per_year(frequency):
  """Counts number of steps that equal a year, if defined and 2 <= N <= 12."""
  # pylint: disable=unused-import,g-import-not-at-top
  import pandas as pd  # Defer import to avoid a package-level Pandas dep.
  # pylint: enable=unused-import,g-import-not-at-top

  if is_fixed_duration(frequency):
    return None  # No fixed duration divides both leap and non-leap years.

  start = pd.Timestamp('1900-01-01')
  # Align the start date with any constraints imposed by the frequency, e.g.,
  # `pd.offsets.MonthEnd()`.
  start = (start + frequency) - frequency
  end = start + pd.DateOffset(years=1)
  for num_steps in range(2, 13):
    if start + num_steps * frequency == end:
      return num_steps
  return None


def is_fixed_duration(frequency):
  """Determines if a `pd.DateOffset` represents a fixed number of seconds."""
  # pylint: disable=unused-import,g-import-not-at-top
  import pandas as pd  # Defer import to avoid a package-level Pandas dep.
  # pylint: enable=unused-import,g-import-not-at-top

  # Most Pandas offsets define `self.nanos` if and only if they are
  # fixed-duration (this is checked below), but `pd.DateOffset` doesn't do
  # this for some reason, so handle this case explicitly.
  if type(frequency) == pd.DateOffset:  # pylint: disable=unidiomatic-typecheck
    if frequency.kwds.get('months', 0) != 0:
      return False
    if frequency.kwds.get('years', 0) != 0:
      return False
    return True
  # Handle custom frequencies like `pd.offsets.MonthsEnd()`.
  try:
    frequency.nanos
  except ValueError:
    return False
  return True
