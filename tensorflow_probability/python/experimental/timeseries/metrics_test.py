# Copyright 2023 The TensorFlow Probability Authors.
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
"""Tests for metrics.py."""

import numpy as np
from tensorflow_probability.python.experimental.timeseries import metrics

from absl.testing import absltest


class MetricsTest(absltest.TestCase):

  def test_smape(self):
    y = np.array([1, 2, 3, 4, 5])
    yhat = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
    self.assertAlmostEqual(4.49651015, metrics.smape(y, yhat))

  def test_horizoned_smape(self):
    y = np.array([1, 2, 3, 4, 5])
    yhat = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
    np.testing.assert_allclose(
        [9.52381, 7.326007, 5.976901, 5.115587, 4.49651],
        metrics.horizoned_smape(y, yhat))

  def test_mase(self):
    y = np.array([1, 2, 3, 4, 5])
    yhat = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
    y_obs = np.array([10, 20, 30, 40, 50])
    self.assertAlmostEqual(0.005, metrics.mase(y, yhat, y_obs, 2))

  def test_horizoned_mase(self):
    y = np.array([1, 2, 3, 4, 5])
    yhat = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
    y_obs = np.array([10, 20, 30, 40, 50])
    np.testing.assert_allclose(
        [0.005, 0.005, 0.005, 0.005, 0.005],
        metrics.horizoned_mase(y, yhat, y_obs, 2))

  def test_msis(self):
    y = np.array([1, 2, 3, 4, 5])
    yhat_lower = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    yhat_upper = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y_obs = np.array([10, 20, 30, 40, 50])
    self.assertAlmostEqual(
        0.285, metrics.msis(y, yhat_lower, yhat_upper, y_obs, 2))

  def test_horizoned_msis(self):
    y = np.array([1, 2, 3, 4, 5])
    yhat_lower = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    yhat_upper = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y_obs = np.array([10, 20, 30, 40, 50])
    np.testing.assert_allclose(
        [0.095, 0.1425, 0.19, 0.2375, 0.285],
        metrics.horizoned_msis(y, yhat_lower, yhat_upper, y_obs, 2))


if __name__ == "__main__":
  absltest.main()
