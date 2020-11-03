# Lint as: python3
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

from inference_gym.internal import data
from inference_gym.tools.stan import brownian_motion
from inference_gym.tools.stan import eight_schools as eight_schools_lib
from inference_gym.tools.stan import item_response_theory
from inference_gym.tools.stan import log_gaussian_cox_process
from inference_gym.tools.stan import logistic_regression
from inference_gym.tools.stan import lorenz_system
from inference_gym.tools.stan import probit_regression
from inference_gym.tools.stan import radon_contextual_effects
from inference_gym.tools.stan import radon_contextual_effects_halfnormal
from inference_gym.tools.stan import sparse_logistic_regression
from inference_gym.tools.stan import stochastic_volatility

__all__ = [
    'brownian_motion_missing_middle_observations',
    'brownian_motion_unknown_scales_missing_middle_observations',
    'convection_lorenz_bridge',
    'convection_lorenz_bridge_unknown_scales',
    'eight_schools',
    'german_credit_numeric_logistic_regression',
    'german_credit_numeric_probit_regression',
    'german_credit_numeric_sparse_logistic_regression',
    'radon_contextual_effects_minnesota',
    'radon_contextual_effects_minnesota_halfnormal',
    'stochastic_volatility_sp500',
    'stochastic_volatility_sp500_small',
    'synthetic_item_response_theory',
    'synthetic_log_gaussian_cox_process',
]


def brownian_motion_missing_middle_observations():
  """Brownian Motion with missing observations.

  Returns:
    target: StanModel.
  """
  dataset = data.brownian_motion_missing_middle_observations()
  return brownian_motion.brownian_motion(**dataset)


def brownian_motion_unknown_scales_missing_middle_observations():
  """Brownian Motion with missing observations and unknown scale parameters.

  Returns:
    target: StanModel.
  """
  dataset = data.brownian_motion_missing_middle_observations()
  return brownian_motion.brownian_motion_unknown_scales(
      observed_locs=dataset['observed_locs'])


def convection_lorenz_bridge():
  """Lorenz System with observed convection and missing observations.

  Returns:
    target: StanModel.
  """
  dataset = data.convection_lorenz_bridge()
  return lorenz_system.partially_observed_lorenz_system(**dataset)


def convection_lorenz_bridge_unknown_scales():
  """Lorenz System with observed convection and missing observations.

  Returns:
    target: StanModel.
  """
  dataset = data.convection_lorenz_bridge()
  del dataset['innovation_scale']
  del dataset['observation_scale']
  return lorenz_system.partially_observed_lorenz_system_unknown_scales(
      **dataset)


def eight_schools():
  """Eight schools hierarchical regression model."""
  return eight_schools_lib.eight_schools()


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


def radon_contextual_effects_minnesota():
  """Hierarchical radon model with contextual effects, with data from Minnesota.

  Returns:
    target: StanModel.
  """
  dataset = data.radon(state='MN')
  for key in list(dataset.keys()):
    if key.startswith('test_'):
      del dataset[key]
  return radon_contextual_effects.radon_contextual_effects(**dataset)


def radon_contextual_effects_minnesota_halfnormal():
  """Hierarchical radon model with contextual effects, with data from Minnesota.

  Returns:
    target: StanModel.
  """
  dataset = data.radon(state='MN')
  for key in list(dataset.keys()):
    if key.startswith('test_'):
      del dataset[key]
  return radon_contextual_effects_halfnormal.radon_contextual_effects(**dataset)


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
