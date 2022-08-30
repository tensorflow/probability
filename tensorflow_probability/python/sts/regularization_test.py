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
"""Tests for regularization."""

import datetime

from absl.testing import parameterized

import numpy as np
import pandas as pd

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import regularization


_TRUTH_REGULARIZE_2SEC = pd.DataFrame([0., 1., 2., np.nan, 3.],
                                      index=pd.to_datetime(
                                          ['2014-01-01 00:00:12',
                                           '2014-01-01 00:00:14',
                                           '2014-01-01 00:00:16',
                                           '2014-01-01 00:00:18',
                                           '2014-01-01 00:00:20']),
                                      columns=['value'])
_TRUTH_REGULARIZE_WEEKLY = pd.DataFrame([0., 1., 2., 3., np.nan, 4.,
                                         np.nan, np.nan, np.nan, 5.],
                                        index=pd.date_range(
                                            '2014-01-01',
                                            '2014-03-05',
                                            freq=pd.DateOffset(weeks=1)),
                                        columns=['value'])
_TRUTH_REGULARIZE_QUARTERS = pd.DataFrame([0., 1., np.nan, 2., 3.],
                                          index=pd.to_datetime(
                                              ['2014-01-01',
                                               '2014-04-01',
                                               '2014-07-01',
                                               '2014-10-01',
                                               '2015-01-01']),
                                          columns=['value'])
_TRUTH_REGULARIZE_ENDS_OF_QUARTERS = pd.DataFrame(
    [0., 1., 2., np.nan, 3.],
    index=pd.to_datetime(
        ['2013-12-31',
         '2014-03-31',
         '2014-06-30',
         '2014-09-30',
         '2014-12-31']),
    columns=['value'])


def _generate_random_data(start, end, freq, remove_freq=False):
  """Generate random series of specified window and frequency."""
  date_range = pd.date_range(start, end, freq=freq)
  date_index = pd.DatetimeIndex(date_range)
  test_series = pd.DataFrame(np.random.randn(date_range.shape[0]),
                             index=date_index, columns=['value']).sort_index()
  if remove_freq:
    test_series.index.freq = None
  return test_series


class RegularizationTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('seconds',
       ['2014-01-01 00:00:12', '2014-01-01 00:00:14',
        '2014-01-01 00:00:16', '2014-01-01 00:00:20'],
       _TRUTH_REGULARIZE_2SEC),
      ('weekly',
       ['2014-01-01', '2014-01-08', '2014-01-15', '2014-01-22',
        '2014-02-05', '2014-03-05'],
       _TRUTH_REGULARIZE_WEEKLY),
      ('quarters',
       ['2014-01-01', '2014-04-01', '2014-10-01', '2015-01-01'],
       _TRUTH_REGULARIZE_QUARTERS),
      ('ends_of_quarters',
       ['2013-12-31', '2014-03-31', '2014-06-30', '2014-12-31'],
       _TRUTH_REGULARIZE_ENDS_OF_QUARTERS))
  def test_regularize_series(self, datetimes, expected):
    """Test filling in missing entries of a series."""
    dt_idx = pd.to_datetime(datetimes)
    test_series = pd.DataFrame(np.arange(len(dt_idx)), index=dt_idx,
                               columns=['value']).sort_index()
    regularized_series = regularization.regularize_series(test_series)
    self.assertTrue(regularized_series.equals(expected))

  @parameterized.named_parameters(
      ('5sec', '2018-01-01', '2018-01-02', pd.DateOffset(seconds=5)),
      ('1hour30mins', '2018-01-01', '2018-01-03', pd.DateOffset(hours=1,
                                                                minutes=30)),
      ('2hour', '2018-01-01', '2018-01-03', pd.DateOffset(hours=2)),
      ('1day', '2018-01-01', '2018-01-15', pd.DateOffset(days=1)),
      ('1day6hours', '2018-01-01', '2018-01-15',
       pd.DateOffset(days=1, hours=6)),
      ('10day', '2018-01-01', '2018-03-31', pd.DateOffset(days=10)),
      ('2week', '2018-01-01', '2018-03-31', pd.DateOffset(weeks=2)),
      ('2year', '2000-05-17', '2018-05-17', pd.DateOffset(years=2)))
  def test_inferred_frequency(self, start, end, freq):
    """Generate data with known frequency, remove the frequency, and infer."""
    test_series_no_freq = _generate_random_data(start=start, end=end, freq=freq,
                                                remove_freq=True)
    test_series_with_missing_points = pd.concat([
        test_series_no_freq.iloc[:2],
        test_series_no_freq.iloc[5:]], axis=0)
    regularized_series = regularization.regularize_series(
        test_series_with_missing_points)
    # Check that the inferred frequency produces the same dates as the true
    # frequency. We avoid checking the inferred frequency directly in order
    # to allow frequencies that are equivalent to (but don't equal) what was
    # specified, e.g., 90 minutes rather than 1h30m.
    for date, inferred_date in zip(test_series_no_freq.index,
                                   regularized_series.index):
      self.assertEqual(date, inferred_date)

  def test_frequency_specified(self):
    dt_idx = pd.to_datetime([
        '2014-01-01 00:00:12', '2014-01-01 00:00:14', '2014-01-01 00:00:16',
        '2014-01-01 00:00:18'
    ])
    test_series = pd.DataFrame(
        np.arange(4), index=dt_idx, columns=['value']).sort_index()
    regularized_data = regularization.regularize_series(
        test_series, frequency=pd.DateOffset(seconds=2))
    self.assertEqual(regularized_data.index.freq, pd.DateOffset(seconds=2))

  def test_values_at_duplicate_indices_are_summed(self):
    test_series = pd.DataFrame(
        [1., 2., 3., 4., np.nan, 0.],
        index=pd.to_datetime(['2014-01-01',
                              '2014-01-02',
                              '2014-01-02',
                              '2014-01-03',
                              '2014-01-03',
                              '2014-01-04']),
        columns=['value'])
    regularized_data = regularization.regularize_series(test_series)
    self.assertLen(regularized_data.values, 4)
    self.assertEqual(regularized_data.values[0, 0], 1)
    self.assertEqual(regularized_data.values[1, 0], 5)
    self.assertTrue(np.isnan(regularized_data.values[2, 0]))
    self.assertEqual(regularized_data.values[3, 0], 0)

  _DATE_INDEX = pd.DatetimeIndex(
      pd.date_range(
          start='2018-01-01', end='2018-01-03', freq=pd.DateOffset(days=1)))

  @parameterized.named_parameters(
      ('incorrect_instance_data', np.random.rand(5), TypeError),
      ('empty_data', pd.DataFrame(columns=['value']), ValueError),
      ('incorrect_shape_data', pd.DataFrame(np.random.randn(5, 2)), ValueError),
      ('incorrect_index_data',
       pd.DataFrame(np.random.randn(5), columns=['value']), ValueError),
      ('incorrect_dtype_data',
       pd.DataFrame(['a', 'b', 'c'], columns=['value'],
                    index=_DATE_INDEX), ValueError),
      ('index is not sorted',
       pd.DataFrame(np.random.randn(3), columns=['data'],
                    index=[_DATE_INDEX[2], _DATE_INDEX[0], _DATE_INDEX[1]]),
       ValueError))
  def test_check_data(self, test_data, exception_type):
    with self.assertRaises(exception_type):
      regularization.regularize_series(test_data)

  def test_regularized_series_max_length(self):
    test_data = pd.DataFrame(
        [0., 1., 2.], columns=['value'],
        # Make a ridiculous request for 20 years at one-second frequency.
        index=[datetime.datetime(2000, 1, 1, hour=0, minute=0, second=0),
               datetime.datetime(2000, 1, 1, hour=0, minute=0, second=1),
               datetime.datetime(2020, 1, 1, hour=0, minute=0, second=0)]
        )
    with self.assertRaisesRegex(ValueError,
                                'would exceed the maximum series length'):
      regularization.regularize_series(test_data, max_series_length=50000)

  def test_accepts_series(self):
    test_data = pd.Series(
        [1., 2., 4.],
        index=pd.to_datetime(['2014-01-01',
                              '2014-01-02',
                              '2014-01-04']))
    regularized_series = regularization.regularize_series(test_data)
    self.assertIsInstance(regularized_series, pd.Series)
    self.assertLen(regularized_series, 4)

  def test_multiple_columns(self):
    test_data = pd.DataFrame(
        [[1., 7.], [2., 8.], [4., 10.]],
        columns=['series1', 'series2'],
        index=pd.to_datetime(['2014-01-01', '2014-01-02', '2014-01-04']))
    regularized_series = regularization.regularize_series(test_data)
    self.assertIsInstance(regularized_series, pd.DataFrame)
    self.assertAllEqual(regularized_series.shape, [4, 2])
    self.assertAllEqual(regularized_series.index,
                        pd.to_datetime(['2014-01-01',
                                        '2014-01-02',
                                        '2014-01-03',
                                        '2014-01-04']))
    self.assertAllEqual(regularized_series.columns, ['series1', 'series2'])
    self.assertAllEqual(regularized_series,
                        [[1., 7.], [2., 8.], [np.nan, np.nan], [4., 10.]])

if __name__ == '__main__':
  test_util.main()
