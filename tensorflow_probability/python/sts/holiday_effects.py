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
"""Utilities for holiday regressors."""

__all__ = [
    'get_default_holidays',
    'create_holiday_regressors',
]

# Defines expected holiday file fields.
_HOLIDAY_FILE_FIELDS = frozenset({'geo', 'holiday', 'date'})


# TODO(b/195771744): Add holidays as unofficial TFP dependency.
def get_default_holidays(times, country):
  """Creates default holidays for a specific country.

  Args:
    times: a Pandas `DatetimeIndex` that indexes time series data.
    country: `str`, two-letter upper-case [ISO 3166-1 alpha-2 country code](
      https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) for using holiday
        regressors from a particular country.

  Returns:
    holidays: a Pandas `DataFrame` with default holidays relevant to the input
      times. The `DataFrame` should have the following columns:
      * `geo`: `str`, two-letter upper-case country code
      * `holiday`: `str`, the name of the holiday
      * `date`: `str`, dates in the form of `YYYY-MM-DD`
  """
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  import holidays
  # pylint: enable=g-import-not-at-top

  years = range(times.min().year, times.max().year + 1)
  holidays = holidays.CountryHoliday(country, years=years, expand=False)
  holidays = pd.DataFrame(
      [(country, holidays.get_list(date), date) for date in holidays],
      columns=['geo', 'holiday', 'date'])
  holidays = holidays.explode('holiday')
  # Ensure that only holiday dates covered by times are used.
  holidays = holidays[(pd.to_datetime(holidays['date']) >= times.min())
                      & (pd.to_datetime(holidays['date']) <= times.max())]
  holidays = holidays.reset_index(drop=True)
  holidays['date'] = pd.to_datetime(holidays['date'])
  holidays = holidays.sort_values('date')
  holidays['date'] = holidays['date'].dt.strftime('%Y-%m-%d')
  return holidays


def create_holiday_regressors(times, holidays):
  """Creates a design matrix of holiday regressors for a given time series.

  Args:
    times: a Pandas `DatetimeIndex` that indexes time series data.
    holidays: a Pandas `DataFrame` containing the dates of holidays. The
      `DataFrame` should have the following columns:
      * `geo`: `str`, two-letter upper-case country code
      * `holiday`: `str`, the name of the holiday
      * `date`: `str`, dates in the form of `YYYY-MM-DD`

  Returns:
    holiday_regressors: a Pandas `DataFrame` where the columns are the names of
      holidays. This matrix of one hot encodings is shape
      (N, H), where N is the length of `times` and H is the number of unique
      holiday names in `holidays.holiday`.
  """
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  # pylint: enable=g-import-not-at-top

  # TODO(b/195346554): Expand fixed holidays.
  _check_times(times)
  _check_holidays(holidays)

  holidays = holidays.sort_values('date')

  holiday_types = list(holidays.holiday.unique())
  holiday_regressors = pd.DataFrame()
  for holiday in holiday_types:
    holiday_dates = holidays.loc[holidays.holiday == holiday]
    holiday_dates = pd.to_datetime(
        list(holiday_dates.date), errors='raise', format='%Y-%m-%d')
    holiday_regressors.loc[:, holiday] = _match_dates(times, holiday_dates)

  # Remove all regressors with only zeros.
  holiday_regressors = (
      holiday_regressors.loc[:, (holiday_regressors != 0).any(axis=0)])
  return holiday_regressors


def _check_times(times):
  """Checks that times are in the correct format.

  Args:
    times: a Pandas `DatetimeIndex` that indexes time series data.

  Raises:
    ValueError: if times is not a Pandas `DatetimeIndex` or does not have a
    frequency.
  """
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  # pylint: enable=g-import-not-at-top
  if not isinstance(times, pd.core.indexes.datetimes.DatetimeIndex):
    raise ValueError('Times is not a Pandas DatetimeIndex.')
  if not times.freq:
    raise ValueError('Times does not have a frequency.')


def _check_holidays(holidays):
  """Checks that holiday files are in the correct format.

  Args:
    holidays: a Pandas `DataFrame` containing the dates of holidays.

  Raises:
    ValueError: if the holidays column names are improperly formatted.
  """
  all_column_names = _HOLIDAY_FILE_FIELDS.issubset(holidays.columns)
  if not all_column_names:
    raise ValueError(
        'Holidays column names must contain: {0}.'.format(_HOLIDAY_FILE_FIELDS))


def _match_dates(times, dates):
  """Creates a 0-1 dummy variable that marks every instance of dates that also occurs in times.

  Args:
    times: a Pandas `DatetimeIndex` for observed data.
    dates: a Pandas `DatetimeIndex` for relevant dates of a single holiday.

  Returns:
    regressor: a list the same length as `times`, with a 1 where there is a date
    match, and otherwise 0.
  """
  regressor = [0] * len(times)
  rounded_times = times.floor('d')
  rounded_dates = dates.floor('d')

  # TODO(b/195347492): Approximate to the nearest prior day
  # for greater than daily granularity.
  # TODO(b/195347492): Add _MIN_HOLIDAY_OCCURRENCES.
  date_intersection = rounded_times.intersection(rounded_dates).unique()
  for date in date_intersection:
    date_slice = rounded_times.get_loc(date)
    regressor_slice = regressor[date_slice]
    if isinstance(regressor_slice, int):
      replacement = 1
    else:
      replacement = [1] * len(regressor[date_slice])
    regressor[date_slice] = replacement
  return regressor
