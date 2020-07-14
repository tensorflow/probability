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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.experimental.inference_gym.targets.banana import Banana
from tensorflow_probability.python.experimental.inference_gym.targets.bayesian_model import BayesianModel
from tensorflow_probability.python.experimental.inference_gym.targets.ill_conditioned_gaussian import IllConditionedGaussian
from tensorflow_probability.python.experimental.inference_gym.targets.item_response_theory import ItemResponseTheory
from tensorflow_probability.python.experimental.inference_gym.targets.item_response_theory import SyntheticItemResponseTheory
from tensorflow_probability.python.experimental.inference_gym.targets.log_gaussian_cox_process import LogGaussianCoxProcess
from tensorflow_probability.python.experimental.inference_gym.targets.logistic_regression import GermanCreditNumericLogisticRegression
from tensorflow_probability.python.experimental.inference_gym.targets.logistic_regression import LogisticRegression
from tensorflow_probability.python.experimental.inference_gym.targets.model import Model
from tensorflow_probability.python.experimental.inference_gym.targets.neals_funnel import NealsFunnel
from tensorflow_probability.python.experimental.inference_gym.targets.probit_regression import GermanCreditNumericProbitRegression
from tensorflow_probability.python.experimental.inference_gym.targets.probit_regression import ProbitRegression
from tensorflow_probability.python.experimental.inference_gym.targets.sparse_logistic_regression import GermanCreditNumericSparseLogisticRegression
from tensorflow_probability.python.experimental.inference_gym.targets.sparse_logistic_regression import SparseLogisticRegression
from tensorflow_probability.python.experimental.inference_gym.targets.vector_model import VectorModel

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
    'SyntheticItemResponseTheory',
    'VectorModel',
]
