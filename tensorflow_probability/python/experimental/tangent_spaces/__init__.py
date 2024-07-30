# Copyright 2021 The TensorFlow Probability Authors.
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
"""TensorFlow Probability experimental tangent spaces package."""

from tensorflow_probability.python.experimental.tangent_spaces.simplex import ProbabilitySimplexSpace
from tensorflow_probability.python.experimental.tangent_spaces.spaces import AxisAlignedSpace
from tensorflow_probability.python.experimental.tangent_spaces.spaces import FullSpace
from tensorflow_probability.python.experimental.tangent_spaces.spaces import GeneralSpace
from tensorflow_probability.python.experimental.tangent_spaces.spaces import TangentSpace
from tensorflow_probability.python.experimental.tangent_spaces.spaces import UnspecifiedTangentSpaceError
from tensorflow_probability.python.experimental.tangent_spaces.spaces import ZeroSpace
from tensorflow_probability.python.experimental.tangent_spaces.spherical import SphericalSpace
from tensorflow_probability.python.experimental.tangent_spaces.symmetric_matrix import ConstantDiagonalSymmetricMatrixSpace
from tensorflow_probability.python.experimental.tangent_spaces.symmetric_matrix import SymmetricMatrixSpace

__all__ = [
    'AxisAlignedSpace',
    'ConstantDiagonalSymmetricMatrixSpace',
    'FullSpace',
    'GeneralSpace',
    'ProbabilitySimplexSpace',
    'SphericalSpace',
    'SymmetricMatrixSpace',
    'TangentSpace',
    'UnspecifiedTangentSpaceError',
    'ZeroSpace',
]
