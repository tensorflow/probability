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
"""Acquisition Functions."""

from tensorflow_probability.python.experimental.bayesopt.acquisition.acquisition_function import AcquisitionFunction
from tensorflow_probability.python.experimental.bayesopt.acquisition.acquisition_function import MCMCReducer
from tensorflow_probability.python.experimental.bayesopt.acquisition.expected_improvement import GaussianProcessExpectedImprovement
from tensorflow_probability.python.experimental.bayesopt.acquisition.expected_improvement import ParallelExpectedImprovement
from tensorflow_probability.python.experimental.bayesopt.acquisition.expected_improvement import StudentTProcessExpectedImprovement
from tensorflow_probability.python.experimental.bayesopt.acquisition.max_value_entropy_search import GaussianProcessMaxValueEntropySearch
from tensorflow_probability.python.experimental.bayesopt.acquisition.probability_of_improvement import GaussianProcessProbabilityOfImprovement
from tensorflow_probability.python.experimental.bayesopt.acquisition.probability_of_improvement import ParallelProbabilityOfImprovement
from tensorflow_probability.python.experimental.bayesopt.acquisition.upper_confidence_bound import GaussianProcessUpperConfidenceBound
from tensorflow_probability.python.experimental.bayesopt.acquisition.upper_confidence_bound import ParallelUpperConfidenceBound
from tensorflow_probability.python.experimental.bayesopt.acquisition.weighted_power_scalarization import WeightedPowerScalarization
from tensorflow_probability.python.internal import all_util

JAX_MODE = False

_allowed_symbols = [
    'AcquisitionFunction',
    'GaussianProcessExpectedImprovement',
    'GaussianProcessMaxValueEntropySearch',
    'GaussianProcessProbabilityOfImprovement',
    'GaussianProcessUpperConfidenceBound',
    'MCMCReducer',
    'ParallelExpectedImprovement',
    'ParallelProbabilityOfImprovement',
    'ParallelUpperConfidenceBound',
    'StudentTProcessExpectedImprovement',
    'WeightedPowerScalarization',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
