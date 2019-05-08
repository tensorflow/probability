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
"""TensorFlow Probability math functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.math import ode
from tensorflow_probability.python.math.custom_gradient import custom_gradient
from tensorflow_probability.python.math.diag_jacobian import diag_jacobian
from tensorflow_probability.python.math.gradient import value_and_gradient
from tensorflow_probability.python.math.interpolation import batch_interp_regular_1d_grid
from tensorflow_probability.python.math.interpolation import batch_interp_regular_nd_grid
from tensorflow_probability.python.math.interpolation import interp_regular_1d_grid
from tensorflow_probability.python.math.linalg import cholesky_concat
from tensorflow_probability.python.math.linalg import lu_matrix_inverse
from tensorflow_probability.python.math.linalg import lu_reconstruct
from tensorflow_probability.python.math.linalg import lu_solve
from tensorflow_probability.python.math.linalg import matrix_rank
from tensorflow_probability.python.math.linalg import pinv
from tensorflow_probability.python.math.linalg import pivoted_cholesky
from tensorflow_probability.python.math.linalg import sparse_or_dense_matmul
from tensorflow_probability.python.math.linalg import sparse_or_dense_matvecmul
from tensorflow_probability.python.math.numeric import clip_by_value_preserve_gradient
from tensorflow_probability.python.math.numeric import log1psquare
from tensorflow_probability.python.math.numeric import soft_threshold
from tensorflow_probability.python.math.random_ops import random_rademacher
from tensorflow_probability.python.math.random_ops import random_rayleigh
from tensorflow_probability.python.math.root_search import secant_root
from tensorflow_probability.python.math.sparse import dense_to_sparse

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'batch_interp_regular_1d_grid',
    'batch_interp_regular_nd_grid',
    'cholesky_concat',
    'clip_by_value_preserve_gradient',
    'custom_gradient',
    'dense_to_sparse',
    'diag_jacobian',
    'interp_regular_1d_grid',
    'log1psquare',
    'lu_matrix_inverse',
    'lu_reconstruct',
    'lu_solve',
    'matrix_rank',
    'ode',
    'pinv',
    'pivoted_cholesky',
    'random_rademacher',
    'random_rayleigh',
    'secant_root',
    'soft_threshold',
    'sparse_or_dense_matmul',
    'sparse_or_dense_matvecmul',
    'value_and_gradient',
]

remove_undocumented(__name__, _allowed_symbols)
