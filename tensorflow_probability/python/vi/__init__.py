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
"""Methods and objectives for variational inference."""

from tensorflow_probability.python.vi import mutual_information
from tensorflow_probability.python.vi.csiszar_divergence import amari_alpha
from tensorflow_probability.python.vi.csiszar_divergence import arithmetic_geometric
from tensorflow_probability.python.vi.csiszar_divergence import chi_square
from tensorflow_probability.python.vi.csiszar_divergence import csiszar_vimco
from tensorflow_probability.python.vi.csiszar_divergence import dual_csiszar_function
from tensorflow_probability.python.vi.csiszar_divergence import jeffreys
from tensorflow_probability.python.vi.csiszar_divergence import jensen_shannon
from tensorflow_probability.python.vi.csiszar_divergence import kl_forward
from tensorflow_probability.python.vi.csiszar_divergence import kl_reverse
from tensorflow_probability.python.vi.csiszar_divergence import log1p_abs
from tensorflow_probability.python.vi.csiszar_divergence import modified_gan
from tensorflow_probability.python.vi.csiszar_divergence import monte_carlo_variational_loss
from tensorflow_probability.python.vi.csiszar_divergence import pearson
from tensorflow_probability.python.vi.csiszar_divergence import squared_hellinger
from tensorflow_probability.python.vi.csiszar_divergence import symmetrized_csiszar_function
from tensorflow_probability.python.vi.csiszar_divergence import t_power
from tensorflow_probability.python.vi.csiszar_divergence import total_variation
from tensorflow_probability.python.vi.csiszar_divergence import triangular
from tensorflow_probability.python.vi.optimization import fit_surrogate_posterior

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import


_allowed_symbols = [
    'amari_alpha',
    'arithmetic_geometric',
    'chi_square',
    'csiszar_vimco',
    'dual_csiszar_function',
    'fit_surrogate_posterior',
    'jensen_shannon',
    'jeffreys',
    'kl_forward',
    'kl_reverse',
    'log1p_abs',
    'modified_gan',
    'monte_carlo_variational_loss',
    'mutual_information',
    'pearson',
    'squared_hellinger',
    'symmetrized_csiszar_function',
    'total_variation',
    'triangular',
    't_power',
]

remove_undocumented(__name__, _allowed_symbols)
