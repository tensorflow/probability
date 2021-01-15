# Copyright 2019 The TensorFlow Probability Authors.
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
"""Experimental methods and objectives for variational inference."""

from tensorflow_probability.python.experimental.vi.util.trainable_linear_operators import build_linear_operator_zeros
from tensorflow_probability.python.experimental.vi.util.trainable_linear_operators import build_trainable_linear_operator_block
from tensorflow_probability.python.experimental.vi.util.trainable_linear_operators import build_trainable_linear_operator_diag
from tensorflow_probability.python.experimental.vi.util.trainable_linear_operators import build_trainable_linear_operator_full_matrix
from tensorflow_probability.python.experimental.vi.util.trainable_linear_operators import build_trainable_linear_operator_tril
from tensorflow_probability.python.internal import all_util


_allowed_symbols = [
    'build_linear_operator_zeros',
    'build_trainable_linear_operator_block',
    'build_trainable_linear_operator_diag',
    'build_trainable_linear_operator_full_matrix',
    'build_trainable_linear_operator_tril',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
