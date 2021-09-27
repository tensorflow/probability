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

from inference_gym.backends import util

with util.silence_nonrewritten_import_errors():
  # pylint: disable=g-import-not-at-top
  from inference_gym.targets.banana import Banana
  from inference_gym.targets.bayesian_model import BayesianModel
  from inference_gym.targets.brownian_motion import BrownianMotion
  from inference_gym.targets.brownian_motion import BrownianMotionMissingMiddleObservations
  from inference_gym.targets.brownian_motion import BrownianMotionUnknownScales
  from inference_gym.targets.brownian_motion import BrownianMotionUnknownScalesMissingMiddleObservations
  from inference_gym.targets.eight_schools import EightSchools
  from inference_gym.targets.ill_conditioned_gaussian import IllConditionedGaussian
  from inference_gym.targets.item_response_theory import ItemResponseTheory
  from inference_gym.targets.item_response_theory import SyntheticItemResponseTheory
  from inference_gym.targets.log_gaussian_cox_process import LogGaussianCoxProcess
  from inference_gym.targets.log_gaussian_cox_process import SyntheticLogGaussianCoxProcess
  from inference_gym.targets.logistic_regression import GermanCreditNumericLogisticRegression
  from inference_gym.targets.logistic_regression import LogisticRegression
  from inference_gym.targets.lorenz_system import ConvectionLorenzBridge
  from inference_gym.targets.lorenz_system import LorenzSystem
  from inference_gym.targets.lorenz_system import ConvectionLorenzBridgeUnknownScales
  from inference_gym.targets.lorenz_system import LorenzSystemUnknownScales
  from inference_gym.targets.model import Model
  from inference_gym.targets.neals_funnel import NealsFunnel
  from inference_gym.targets.non_identifiable_quartic import NonIdentifiableQuarticMeasurementModel
  from inference_gym.targets.plasma_spectroscopy import PlasmaSpectroscopy
  from inference_gym.targets.plasma_spectroscopy import SyntheticPlasmaSpectroscopy
  from inference_gym.targets.probit_regression import GermanCreditNumericProbitRegression
  from inference_gym.targets.probit_regression import ProbitRegression
  from inference_gym.targets.radon_contextual_effects import RadonContextualEffects
  from inference_gym.targets.radon_contextual_effects import RadonContextualEffectsHalfNormalIndiana
  from inference_gym.targets.radon_contextual_effects import RadonContextualEffectsIndiana
  from inference_gym.targets.radon_contextual_effects import RadonContextualEffectsHalfNormalMinnesota
  from inference_gym.targets.radon_contextual_effects import RadonContextualEffectsMinnesota
  from inference_gym.targets.sparse_logistic_regression import GermanCreditNumericSparseLogisticRegression
  from inference_gym.targets.sparse_logistic_regression import SparseLogisticRegression
  from inference_gym.targets.stochastic_volatility import StochasticVolatility
  from inference_gym.targets.stochastic_volatility import StochasticVolatilitySP500
  from inference_gym.targets.stochastic_volatility import StochasticVolatilitySP500Small
  from inference_gym.targets.vectorized_stochastic_volatility import VectorizedStochasticVolatility
  from inference_gym.targets.vectorized_stochastic_volatility import VectorizedStochasticVolatilityLogSP500
  from inference_gym.targets.vectorized_stochastic_volatility import VectorizedStochasticVolatilityLogSP500Small
  from inference_gym.targets.vectorized_stochastic_volatility import VectorizedStochasticVolatilitySP500
  from inference_gym.targets.vectorized_stochastic_volatility import VectorizedStochasticVolatilitySP500Small
  from inference_gym.targets.vector_model import VectorModel

__all__ = [
    'Banana',
    'BayesianModel',
    'BrownianMotion',
    'BrownianMotionMissingMiddleObservations',
    'BrownianMotionUnknownScales',
    'BrownianMotionUnknownScalesMissingMiddleObservations',
    'ConvectionLorenzBridge',
    'ConvectionLorenzBridgeUnknownScales',
    'EightSchools',
    'GermanCreditNumericLogisticRegression',
    'GermanCreditNumericProbitRegression',
    'GermanCreditNumericSparseLogisticRegression',
    'IllConditionedGaussian',
    'ItemResponseTheory',
    'LogGaussianCoxProcess',
    'LogisticRegression',
    'LorenzSystem',
    'LorenzSystemUnknownScales',
    'Model',
    'NealsFunnel',
    'NonIdentifiableQuarticMeasurementModel',
    'PlasmaSpectroscopy',
    'ProbitRegression',
    'RadonContextualEffects',
    'RadonContextualEffectsHalfNormalIndiana',
    'RadonContextualEffectsIndiana',
    'RadonContextualEffectsHalfNormalMinnesota',
    'RadonContextualEffectsMinnesota',
    'SparseLogisticRegression',
    'StochasticVolatility',
    'StochasticVolatilitySP500',
    'StochasticVolatilitySP500Small',
    'SyntheticItemResponseTheory',
    'SyntheticLogGaussianCoxProcess',
    'SyntheticPlasmaSpectroscopy',
    'VectorizedStochasticVolatility',
    'VectorizedStochasticVolatilityLogSP500',
    'VectorizedStochasticVolatilityLogSP500Small',
    'VectorizedStochasticVolatilitySP500',
    'VectorizedStochasticVolatilitySP500Small',
    'VectorModel',
]
