# Copyright 2024 The TensorFlow Probability Authors.
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
"""Package for training Gaussian Processes in time less than O(n^3)."""

from tensorflow_probability.python.experimental.fastgp import fast_gp
from tensorflow_probability.python.experimental.fastgp import fast_gprm
from tensorflow_probability.python.experimental.fastgp import fast_log_det
from tensorflow_probability.python.experimental.fastgp import fast_mtgp
from tensorflow_probability.python.experimental.fastgp import linalg
from tensorflow_probability.python.experimental.fastgp import linear_operator_sum
from tensorflow_probability.python.experimental.fastgp import mbcg
from tensorflow_probability.python.experimental.fastgp import partial_lanczos
from tensorflow_probability.python.experimental.fastgp import preconditioners
from tensorflow_probability.python.experimental.fastgp import schur_complement
from tensorflow_probability.python.experimental.fastgp.fast_gp import GaussianProcess
from tensorflow_probability.python.experimental.fastgp.fast_gp import GaussianProcessConfig
from tensorflow_probability.python.experimental.fastgp.fast_gprm import GaussianProcessRegressionModel
from tensorflow_probability.python.experimental.fastgp.fast_log_det import get_log_det_algorithm
from tensorflow_probability.python.experimental.fastgp.fast_log_det import ProbeVectorType
from tensorflow_probability.python.experimental.fastgp.fast_mtgp import MultiTaskGaussianProcess
from tensorflow_probability.python.experimental.fastgp.preconditioners import get_preconditioner
from tensorflow_probability.python.experimental.fastgp.schur_complement import SchurComplement
from tensorflow_probability.python.internal import all_util

_allowed_symbols = [
    'GaussianProcessConfig',
    'GaussianProcess',
    'GaussianProcessRegressionModel',
    'ProbeVectorType',
    'get_log_det_algorithm',
    'MultiTaskGaussianProcess',
    'get_preconditioner',
    'SchurComplement',
    'fast_log_det',
    'fast_gp',
    'fast_gprm',
    'fast_mtgp',
    'linalg',
    'linear_operator_sum',
    'mbcg',
    'partial_lanczos',
    'preconditioners',
    'schur_complement',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
