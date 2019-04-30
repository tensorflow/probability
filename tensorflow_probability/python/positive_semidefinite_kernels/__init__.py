# Copyright 2018 The TensorFlow Probability Authors.
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
"""Positive-semidefinite kernels package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.positive_semidefinite_kernels.exp_sin_squared import ExpSinSquared
from tensorflow_probability.python.positive_semidefinite_kernels.exponentiated_quadratic import ExponentiatedQuadratic
from tensorflow_probability.python.positive_semidefinite_kernels.matern import MaternFiveHalves
from tensorflow_probability.python.positive_semidefinite_kernels.matern import MaternOneHalf
from tensorflow_probability.python.positive_semidefinite_kernels.matern import MaternThreeHalves
from tensorflow_probability.python.positive_semidefinite_kernels.polynomial import Linear
from tensorflow_probability.python.positive_semidefinite_kernels.polynomial import Polynomial
from tensorflow_probability.python.positive_semidefinite_kernels.positive_semidefinite_kernel import PositiveSemidefiniteKernel
from tensorflow_probability.python.positive_semidefinite_kernels.rational_quadratic import RationalQuadratic
from tensorflow_probability.python.positive_semidefinite_kernels.schur_complement import SchurComplement

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'ExponentiatedQuadratic',
    'ExpSinSquared',
    'Linear',
    'MaternFiveHalves',
    'MaternOneHalf',
    'MaternThreeHalves',
    'Polynomial',
    'PositiveSemidefiniteKernel',
    'RationalQuadratic',
    'SchurComplement',
]

remove_undocumented(__name__, _allowed_symbols)
