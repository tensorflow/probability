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
"""Targets package."""

from spinoffs.inference_gym.backends import util

with util.silence_nonrewritten_import_errors():
  # pylint: disable=g-import-not-at-top
  from spinoffs.inference_gym.targets.banana import Banana
  from spinoffs.inference_gym.targets.bayesian_model import BayesianModel
  from spinoffs.inference_gym.targets.ill_conditioned_gaussian import IllConditionedGaussian
  from spinoffs.inference_gym.targets.item_response_theory import ItemResponseTheory
  from spinoffs.inference_gym.targets.item_response_theory import SyntheticItemResponseTheory
  from spinoffs.inference_gym.targets.log_gaussian_cox_process import LogGaussianCoxProcess
  from spinoffs.inference_gym.targets.log_gaussian_cox_process import SyntheticLogGaussianCoxProcess
  from spinoffs.inference_gym.targets.logistic_regression import GermanCreditNumericLogisticRegression
  from spinoffs.inference_gym.targets.logistic_regression import LogisticRegression
  from spinoffs.inference_gym.targets.model import Model
  from spinoffs.inference_gym.targets.neals_funnel import NealsFunnel
  from spinoffs.inference_gym.targets.probit_regression import GermanCreditNumericProbitRegression
  from spinoffs.inference_gym.targets.probit_regression import ProbitRegression
  from spinoffs.inference_gym.targets.sparse_logistic_regression import GermanCreditNumericSparseLogisticRegression
  from spinoffs.inference_gym.targets.sparse_logistic_regression import SparseLogisticRegression
  from spinoffs.inference_gym.targets.stochastic_volatility import StochasticVolatility
  from spinoffs.inference_gym.targets.stochastic_volatility import StochasticVolatilitySP500
  from spinoffs.inference_gym.targets.stochastic_volatility import StochasticVolatilitySP500Small
  from spinoffs.inference_gym.targets.vectorized_stochastic_volatility import VectorizedStochasticVolatility
  from spinoffs.inference_gym.targets.vectorized_stochastic_volatility import VectorizedStochasticVolatilitySP500
  from spinoffs.inference_gym.targets.vectorized_stochastic_volatility import VectorizedStochasticVolatilitySP500Small
  from spinoffs.inference_gym.targets.vector_model import VectorModel

__all__ = [
    'Banana',
    'BayesianModel',
    'GermanCreditNumericLogisticRegression',
    'GermanCreditNumericProbitRegression',
    'GermanCreditNumericSparseLogisticRegression',
    'IllConditionedGaussian',
    'ItemResponseTheory',
    'LogGaussianCoxProcess',
    'LogisticRegression',
    'Model',
    'NealsFunnel',
    'ProbitRegression',
    'SparseLogisticRegression',
    'StochasticVolatility',
    'StochasticVolatilitySP500',
    'StochasticVolatilitySP500Small',
    'SyntheticItemResponseTheory',
    'SyntheticLogGaussianCoxProcess',
    'VectorizedStochasticVolatility',
    'VectorizedStochasticVolatilitySP500',
    'VectorizedStochasticVolatilitySP500Small',
    'VectorModel',
]
