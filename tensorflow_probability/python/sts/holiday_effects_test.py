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
"""Tests for holiday_effects."""
from absl.testing import parameterized
import pandas as pd
import tensorflow as tf
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import holiday_effects

HOLIDAY_FILE_FIELDS = ['geo', 'holiday', 'date']


class HolidayEffectsTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('date_wrong_order',
       pd.DataFrame([['US', 'TestHoliday', '12-20-2013']],
                    columns=HOLIDAY_FILE_FIELDS)),
      ('date_invalid',
       pd.DataFrame([['US', 'TestHoliday', '12-00-2013']],
                    columns=HOLIDAY_FILE_FIELDS)),
      ('bad_column_names',
       pd.DataFrame([['US', 'TestHoliday', '2013-12-20']],
                    columns=['geo', 'wrong', 'date'])))
  def test_holidays_raise_error(self, holidays):
    times = pd.date_range(
        start='2013-12-20', end='2015-12-20', freq=pd.DateOffset(years=1))
    with self.assertRaises(ValueError):
      holiday_effects.create_holiday_regressors(times, holidays)

  @parameterized.named_parameters(
      ('data_wrong_format', pd.Series(['2013-12-20'])),
      ('data_no_frequency', pd.DatetimeIndex(['2013-12-20'])))
  def test_times_raise_error(self, times):
    holidays = pd.DataFrame([['US', 'TestHoliday', '2013-12-20']],
                            columns=HOLIDAY_FILE_FIELDS)
    with self.assertRaises(ValueError):
      holiday_effects.create_holiday_regressors(times, holidays)

  @parameterized.named_parameters(
      ('holiday_daily', pd.DateOffset(days=1), '2012-01-01', '2012-12-31',
       [0] * 359 + [1] + [0] * 6),
      ('holiday_hourly', pd.DateOffset(hours=1), '2012-01-01',
       '2012-12-31 23:00:00', [0] * 359 * 24 + [1] * 24 + [0] * 6 * 24),
      # Note that expected should be `[0] * 51 + [1] + [0]` if
      # _match_dates supports rounding timestamps to the nearest prior day
      ('holiday_weekly', pd.DateOffset(weeks=1), '2012-01-01', '2012-12-31',
       [0] * 51 + [0] + [0]))
  def test_match_dates_by_frequency(self, freq, start, end, expected):
    holiday_dates = pd.to_datetime(['2012-12-25'])
    index = pd.date_range(start, end, freq=freq)
    matched_dates = holiday_effects._match_dates(index, holiday_dates)
    self.assertEqual(matched_dates, expected)

  @parameterized.named_parameters(
      ('holiday_disjoint', '2011-01-01', '2011-12-31', [0] * 365),
      ('holiday_intersection', '2011-02-01', '2012-01-31',
       [0] * 334 + [1] * 31),
      ('holiday_subset', '2012-01-01', '2012-01-31', [1] * 31))
  def test_match_dates_by_overlap(self, start, end, expected):
    holiday_dates = pd.date_range(
        '2012-01-01', '2012-12-31', freq=pd.DateOffset(days=1))
    index = pd.date_range(start, end, freq=pd.DateOffset(days=1))
    matched_dates = holiday_effects._match_dates(index, holiday_dates)
    self.assertEqual(matched_dates, expected)

  @parameterized.named_parameters(
      ('diagonal_pattern', [('H1', 0), ('H2', 1)], [[1, 0], [0, 1]]),
      ('row_pattern', [('H1', 0), ('H2', 0)], [[1, 1], [0, 0]]),
      ('column_pattern', [('H1', 0), ('H1', 1)], [[1], [1]]))
  def test_create_holiday_regressors(self, holiday_patterns, expected):
    times = pd.date_range(
        '2011-01-01', '2012-01-01', freq=pd.DateOffset(years=1))
    holidays_list = []
    for name, date_index in holiday_patterns:
      holidays_list.append(['US', name, times[date_index]])
    holidays = pd.DataFrame(holidays_list, columns=HOLIDAY_FILE_FIELDS)
    holiday_regressors = holiday_effects.create_holiday_regressors(
        times, holidays)
    self.assertEqual(holiday_regressors.values.tolist(), expected)


if __name__ == '__main__':
  tf.test.main()
