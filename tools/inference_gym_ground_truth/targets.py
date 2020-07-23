# Lint as: python2, python3
# Copyright 2020 The TensorFlow Probability Authors.
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
"""Stan models, used as a source of ground truth."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tools.inference_gym_ground_truth import item_response_theory
from tools.inference_gym_ground_truth import log_gaussian_cox_process
from tools.inference_gym_ground_truth import logistic_regression
from tools.inference_gym_ground_truth import probit_regression
from tools.inference_gym_ground_truth import sparse_logistic_regression
from tools.inference_gym_ground_truth import stochastic_volatility
from tensorflow_probability.python.experimental.inference_gym.internal import data

__all__ = [
    'german_credit_numeric_logistic_regression',
    'german_credit_numeric_probit_regression',
    'german_credit_numeric_sparse_logistic_regression',
    'stochastic_volatility_sp500',
    'stochastic_volatility_sp500_small',
    'synthetic_item_response_theory',
    'synthetic_log_gaussian_cox_process',
]


def german_credit_numeric_logistic_regression():
  """German credit (numeric) logistic regression.

  Returns:
    target: StanModel.
  """
  dataset = data.german_credit_numeric()
  del dataset['test_features']
  del dataset['test_labels']
  return logistic_regression.logistic_regression(**dataset)


def german_credit_numeric_probit_regression():
  """German credit (numeric) probit regression.

  Returns:
    target: StanModel.
  """
  dataset = data.german_credit_numeric()
  del dataset['test_features']
  del dataset['test_labels']
  return probit_regression.probit_regression(**dataset)


def german_credit_numeric_sparse_logistic_regression():
  """German credit (numeric) logistic regression with a sparsity-inducing prior.

  Returns:
    target: StanModel.
  """
  dataset = data.german_credit_numeric()
  del dataset['test_features']
  del dataset['test_labels']
  return sparse_logistic_regression.sparse_logistic_regression(**dataset)


def stochastic_volatility_sp500():
  """Stochastic volatility model.

  This uses a dataset of 2517 daily closing prices of the S&P 500 index,
  representing the time period 6/25/2010-6/24/2020.

  Returns:
    target: StanModel.
  """
  dataset = data.sp500_closing_prices()
  return stochastic_volatility.stochastic_volatility(**dataset)


def stochastic_volatility_sp500_small():
  """Stochastic volatility model.

  This is a smaller version of `stochastic_volatility_model_sp500` using only
  100 days of returns from the S&P 500, ending 6/24/2020.

  Returns:
    target: StanModel.
  """
  dataset = data.sp500_closing_prices(num_points=100)
  return stochastic_volatility.stochastic_volatility(**dataset)


def synthetic_item_response_theory():
  """One-parameter logistic item-response theory (IRT) model.

  This uses a dataset sampled from the prior. This dataset is a simulation of
  400 students each answering a subset of 100 unique questions, with a total of
  30012 questions answered.

  Returns:
    target: StanModel.
  """
  dataset = data.synthetic_item_response_theory()
  del dataset['test_student_ids']
  del dataset['test_question_ids']
  del dataset['test_correct']
  return item_response_theory.item_response_theory(**dataset)


def synthetic_log_gaussian_cox_process():
  """Log-Gaussian Cox Process model.

  This dataset was simulated by constructing a 10 by 10 grid of equidistant 2D
  locations with spacing = 1, and then sampling from the prior to determine the
  counts at those locations.

  Returns:
    target: StanModel.
  """
  dataset = data.synthetic_log_gaussian_cox_process()
  return log_gaussian_cox_process.log_gaussian_cox_process(**dataset)
