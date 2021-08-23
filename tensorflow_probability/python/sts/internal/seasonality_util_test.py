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
"""Tests for seasonality utilities."""

from absl.testing import parameterized

import numpy as np
import pandas as pd

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts.internal import seasonality_util


_TRUTH_2WK_30SEC_SEASONS = {
    seasonality_util.SeasonTypes.SECOND_OF_MINUTE:
        seasonality_util.SeasonConversion(num=2, duration=1),
    seasonality_util.SeasonTypes.MINUTE_OF_HOUR:
        seasonality_util.SeasonConversion(num=60, duration=2),
    seasonality_util.SeasonTypes.HOUR_OF_DAY:
        seasonality_util.SeasonConversion(num=24, duration=120),
    seasonality_util.SeasonTypes.DAY_OF_WEEK:
        seasonality_util.SeasonConversion(num=7, duration=2880)
    }
_TRUTH_2WK_1DAY_SEASONS = {
    seasonality_util.SeasonTypes.DAY_OF_WEEK:
        seasonality_util.SeasonConversion(num=7, duration=1)
    }
_TRUTH_2WK_1HOUR_SEASONS = {
    seasonality_util.SeasonTypes.HOUR_OF_DAY:
        seasonality_util.SeasonConversion(num=24, duration=1),
    seasonality_util.SeasonTypes.DAY_OF_WEEK:
        seasonality_util.SeasonConversion(num=7, duration=24)
    }
_TRUTH_1YR_QUARTERLY_SEASONS = {
    seasonality_util.SeasonTypes.MONTH_OF_YEAR:
        seasonality_util.SeasonConversion(num=4, duration=1),
    }


@test_util.test_graph_and_eager_modes
class SeasonalityUtilsTest(test_util.TestCase):

  @parameterized.parameters(
      (pd.DateOffset(days=13), None),
      (pd.DateOffset(days=31), None),
      (pd.DateOffset(days=73), None),
      (pd.DateOffset(hours=2400), None),
      (pd.DateOffset(months=1), 12),
      (pd.DateOffset(months=3), 4),
      (pd.DateOffset(years=1), None),
      (pd.DateOffset(month=2), None),
      (pd.offsets.QuarterBegin(), 4),
      (pd.offsets.QuarterEnd(), 4),
      (pd.offsets.MonthBegin(), 12),
      (pd.offsets.MonthEnd(), 12))
  def test_periods_per_year(self, frequency, expected_periods_per_year):
    self.assertEqual(seasonality_util.periods_per_year(frequency),
                     expected_periods_per_year)

  @parameterized.parameters(
      (pd.DateOffset(days=13), True),
      (pd.DateOffset(hours=24), True),
      (pd.DateOffset(seconds=1200), True),
      (pd.DateOffset(months=1), False),
      (pd.DateOffset(years=2), False),
      (pd.DateOffset(days=3, months=1), False),
      (pd.DateOffset(days=13, hours=4), True),
      (pd.offsets.QuarterBegin(), False),
      (pd.offsets.MonthEnd(), False),
      (pd.offsets.Day(), True),
      (pd.offsets.Minute(), True),
      (pd.offsets.Second(), True))
  def test_is_fixed_duration(self, frequency, expected_is_fixed):
    self.assertEqual(seasonality_util.is_fixed_duration(frequency),
                     expected_is_fixed)

  @parameterized.named_parameters(
      ('trivial', 1, None, None, 1), ('no_seasons', 1, {}, None, 1),
      ('just_covariates', 2, None, [np.array([[3.], [-1.]])], 2),
      ('hour_of_day_with_covariates', 17, {
          'hour_of_day': seasonality_util.SeasonConversion(num=24, duration=1)
      }, [np.ones([17, 3]), np.zeros([17, 1])], 24 + 3 + 1),
      ('hour_of_day_with_empty_covariates', 1, {
          'day_of_week': seasonality_util.SeasonConversion(num=7, duration=24)
      }, np.zeros([0, 0]), 7), ('multiple_seasonality', 42, {
          'hour_of_day': seasonality_util.SeasonConversion(num=24, duration=1),
          'day_of_week': seasonality_util.SeasonConversion(num=7, duration=24)
      }, [], 24 + 7))
  def test_build_fixed_effects(self, nticks, seasonal_structure, covariates,
                               expected_num_effects):
    effects = seasonality_util.build_fixed_effects(
        nticks, seasonal_structure, covariates)
    self.assertEqual(effects.shape, (nticks, expected_num_effects))

  def test_design_matrix_for_one_seasonal_effect(self):
    self.assertAllEqual(
        np.transpose([[1, 0, 0, 1, 0],
                      [0, 1, 0, 0, 1],
                      [0, 0, 1, 0, 0]]),
        seasonality_util._design_matrix_for_one_seasonal_effect(
            num_steps=5, duration=1, period=3, dtype=np.int32))
    self.assertAllEqual(
        np.transpose([[1, 1, 0, 0, 1],
                      [0, 0, 1, 1, 0]]),
        seasonality_util._design_matrix_for_one_seasonal_effect(
            num_steps=5, duration=2, period=2, dtype=np.int32))

  @parameterized.named_parameters(
      ('minutes_seconds', pd.DateOffset(minutes=1, seconds=30), 90),
      ('minutes', pd.DateOffset(minutes=3), 180),
      ('hours', pd.DateOffset(hours=2), 2 * 3600),
      ('weeks_days', pd.DateOffset(weeks=1, days=2), 9 * 24 * 3600),
      ('months', pd.DateOffset(months=1), None),
      ('none', None, None))
  def test_freq_to_seconds(self, freq, expected_seconds):
    self.assertAllEqual(expected_seconds,
                        seasonality_util.freq_to_seconds(freq))

  @parameterized.named_parameters(
      ('2_week_30sec', '2018-01-01', '2018-01-15',
       pd.DateOffset(seconds=30), _TRUTH_2WK_30SEC_SEASONS),
      ('2_week_daily', '2018-01-01', '2018-01-15', pd.DateOffset(days=1),
       _TRUTH_2WK_1DAY_SEASONS),
      ('2_week_1hour', '2018-01-01', '2018-01-15',
       pd.DateOffset(hours=1), _TRUTH_2WK_1HOUR_SEASONS),
      ('1_year_quarterly', '2018-01-01', '2019-01-01', pd.DateOffset(months=3),
       _TRUTH_1YR_QUARTERLY_SEASONS),
      ('4_years_annually', '2018-01-01', '2022-01-01',
       pd.DateOffset(years=1), {}))
  def test_create_seasonal_structure(self, start, end, freq, expected):
    """Test seasonal structure creation for a few different scenarios."""
    dates = pd.date_range(start, end, freq=freq)
    seasonal_structure = seasonality_util.create_seasonal_structure(
        frequency=freq, num_steps=len(dates))
    for key, value in expected.items():
      self.assertEqual(value, seasonal_structure[key])

if __name__ == '__main__':
  test_util.main()
