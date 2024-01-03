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
"""Functions for evaluating metrics on timeseries."""

import numpy as np


def smape(y, yhat):
  """Return the symmetric mean absolute percentage error.

  Args:
    y: An array containing the true values.
    yhat: An array containing the predicted values.

  Returns:
    The scalar SMAPE.
  """
  # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
  assert len(yhat) == len(y)
  h = len(y)
  errors = np.abs(y - yhat) / (np.abs(y) + np.abs(yhat)) * 100
  return 2/h * np.sum(errors)


def horizoned_smape(y, yhat):
  """Return the symmetric mean absolute percentage error over all horizons.

  Args:
    y: An array containing the true values.
    yhat: An array containing the predicted values.

  Returns:
    A list a, with a[i] containing the SMAPE over yhat[0] ... yhat[i].
  """
  return [smape(y[:i+1], yhat[:i+1]) for i in range(len(yhat))]


def mase(y, yhat, y_obs, m):
  """Return the mean absolute scaled error.

  Args:
      y: An array containing the true values.
      yhat: An array containing the predicted values.
      y_obs: An array containing the training values.
      m: The season length.

  Returns:
    The scalar MASE.
  """
  # https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
  assert len(yhat) == len(y)
  n = len(y_obs)
  h = len(y)
  assert 0 < m < len(y_obs)
  numer = np.sum(np.abs(y - yhat))
  denom = np.sum(np.abs(y_obs[m:] - y_obs[:-m])) / (n - m)
  return (1 / h) * (numer / denom)


def horizoned_mase(y, yhat, y_obs, m):
  """Return the mean absolute scaled error over all the horizons.

  Args:
      y: An array containing the true values.
      yhat: An array containing the predicted values.
      y_obs: An array containing the training values.
      m: The season length.

  Returns:
    A list a, with a[i] containing the MASE over yhat[0] ... yhat[i].
  """
  return [mase(y[:i+1], yhat[:i+1], y_obs, m) for i in range(len(yhat))]


def msis(y, yhat_lower, yhat_upper, y_obs, m, a=0.05):
  """Return the mean scaled interval score.

  Args:
    y: An array containing the true values.
    yhat_lower: An array containing the a% quantile of the predicted
      distribution.
    yhat_upper: An array containing the (1-a)% quantile of the
      predicted distribution.
    y_obs: An array containing the training values.
    m: The season length.
    a: A scalar in [0, 1] specifying the quantile window to evaluate.

  Returns:
    The scalar MSIS.
  """
  # https://www.uber.com/blog/m4-forecasting-competition/
  assert len(y) == len(yhat_lower) == len(yhat_upper)
  n = len(y_obs)
  h = len(y)
  numer = np.sum(
      (yhat_upper - yhat_lower)
      + (2 / a) * (yhat_lower - y) * (y < yhat_lower)
      + (2 / a) * (y - yhat_upper) * (yhat_upper < y))
  denom = np.sum(np.abs(y_obs[m:] - y_obs[:-m])) / (n - m)
  return (1 / h) * (numer / denom)


def horizoned_msis(y, yhat_lower, yhat_upper, y_obs, m, a=0.025):
  """Return the mean scaled interval score over all horizons.

  Args:
    y: An array containing the true values.
    yhat_lower: An array containing the a% quantile of the predicted
      distribution.
    yhat_upper: An array containing the (1-a)% quantile of the
      predicted distribution.
    y_obs: An array containing the training values.
    m: The season length.
    a: A scalar in [0, 1] specifying the quantile window to evaluate.

  Returns:
    A list a, with a[i] containing the MSIS over y[0] .. y[i].
  """
  assert len(yhat_lower) == len(yhat_upper)
  return [msis(y[:i+1], yhat_lower[:i+1], yhat_upper[:i+1], y_obs, m, a)
          for i in range(len(yhat_lower))]
