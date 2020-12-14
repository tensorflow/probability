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

from tensorflow_probability.python.internal import all_util
from tensorflow_probability.python.math import ode
from tensorflow_probability.python.math import psd_kernels
from tensorflow_probability.python.math.bessel import bessel_iv_ratio
from tensorflow_probability.python.math.bessel import bessel_ive
from tensorflow_probability.python.math.bessel import bessel_kve
from tensorflow_probability.python.math.bessel import log_bessel_ive
from tensorflow_probability.python.math.custom_gradient import custom_gradient
from tensorflow_probability.python.math.diag_jacobian import diag_jacobian
from tensorflow_probability.python.math.generic import log1mexp
from tensorflow_probability.python.math.generic import log_add_exp
from tensorflow_probability.python.math.generic import log_combinations
from tensorflow_probability.python.math.generic import log_cosh
from tensorflow_probability.python.math.generic import log_cumsum_exp
from tensorflow_probability.python.math.generic import log_sub_exp
from tensorflow_probability.python.math.generic import reduce_kahan_sum
from tensorflow_probability.python.math.generic import reduce_log_harmonic_mean_exp
from tensorflow_probability.python.math.generic import reduce_logmeanexp
from tensorflow_probability.python.math.generic import reduce_weighted_logsumexp
from tensorflow_probability.python.math.generic import smootherstep
from tensorflow_probability.python.math.generic import soft_sorting_matrix
from tensorflow_probability.python.math.generic import soft_threshold
from tensorflow_probability.python.math.generic import softplus_inverse
from tensorflow_probability.python.math.gradient import value_and_gradient
from tensorflow_probability.python.math.gram_schmidt import gram_schmidt
from tensorflow_probability.python.math.interpolation import batch_interp_regular_1d_grid
from tensorflow_probability.python.math.interpolation import batch_interp_regular_nd_grid
from tensorflow_probability.python.math.interpolation import interp_regular_1d_grid
from tensorflow_probability.python.math.linalg import cholesky_concat
from tensorflow_probability.python.math.linalg import cholesky_update
from tensorflow_probability.python.math.linalg import fill_triangular
from tensorflow_probability.python.math.linalg import fill_triangular_inverse
from tensorflow_probability.python.math.linalg import lu_matrix_inverse
from tensorflow_probability.python.math.linalg import lu_reconstruct
from tensorflow_probability.python.math.linalg import lu_solve
from tensorflow_probability.python.math.linalg import pivoted_cholesky
from tensorflow_probability.python.math.linalg import sparse_or_dense_matmul
from tensorflow_probability.python.math.linalg import sparse_or_dense_matvecmul
from tensorflow_probability.python.math.minimize import minimize
from tensorflow_probability.python.math.minimize import MinimizeTraceableQuantities
from tensorflow_probability.python.math.numeric import clip_by_value_preserve_gradient
from tensorflow_probability.python.math.numeric import log1psquare
from tensorflow_probability.python.math.root_search import bracket_root
from tensorflow_probability.python.math.root_search import find_root_chandrupatla
from tensorflow_probability.python.math.root_search import find_root_secant
from tensorflow_probability.python.math.root_search import secant_root
from tensorflow_probability.python.math.scan_associative import scan_associative
from tensorflow_probability.python.math.sparse import dense_to_sparse
from tensorflow_probability.python.math.special import erfcinv
from tensorflow_probability.python.math.special import lambertw
from tensorflow_probability.python.math.special import lambertw_winitzki_approx
from tensorflow_probability.python.math.special import lbeta
from tensorflow_probability.python.math.special import log_gamma_correction
from tensorflow_probability.python.math.special import log_gamma_difference
from tensorflow_probability.python.math.special import round_exponential_bump_function
from tensorflow_probability.python.random import rademacher as random_rademacher
from tensorflow_probability.python.random import rayleigh as random_rayleigh

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

random_rademacher = deprecation.deprecated(
    '2020-09-20', 'Use tfp.random.rademacher')(random_rademacher)
random_rayleigh = deprecation.deprecated(
    '2020-09-20', 'Use tfp.random.rayleigh')(random_rayleigh)

_allowed_symbols = [
    'round_exponential_bump_function',
    'batch_interp_regular_1d_grid',
    'batch_interp_regular_nd_grid',
    'bessel_iv_ratio',
    'bessel_ive',
    'bessel_kve',
    'bracket_root',
    'cholesky_concat',
    'cholesky_update',
    'clip_by_value_preserve_gradient',
    'custom_gradient',
    'dense_to_sparse',
    'diag_jacobian',
    'erfcinv',
    'find_root_chandrupatla',
    'find_root_secant',
    'fill_triangular',
    'fill_triangular_inverse',
    'gram_schmidt',
    'interp_regular_1d_grid',
    'lambertw',
    'lambertw_winitzki_approx',
    'lbeta',
    'log_bessel_ive',
    'log1mexp',
    'log1psquare',
    'log_add_exp',
    'log_combinations',
    'log_cosh',
    'log_cumsum_exp',
    'log_gamma_correction',
    'log_gamma_difference',
    'log_sub_exp',
    'lu_matrix_inverse',
    'lu_reconstruct',
    'lu_solve',
    'minimize',
    'MinimizeTraceableQuantities',
    'ode',
    'pivoted_cholesky',
    'psd_kernels',
    'random_rademacher',
    'random_rayleigh',
    'reduce_kahan_sum',
    'reduce_log_harmonic_mean_exp',
    'reduce_logmeanexp',
    'reduce_weighted_logsumexp',
    'scan_associative',
    'secant_root',
    'smootherstep',
    'soft_sorting_matrix',
    'soft_threshold',
    'softplus_inverse',
    'sparse_or_dense_matmul',
    'sparse_or_dense_matvecmul',
    'value_and_gradient',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
