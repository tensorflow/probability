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

from tensorflow_probability.python.math.diag_jacobian import diag_jacobian
from tensorflow_probability.python.math.linalg import matvecmul
from tensorflow_probability.python.math.linalg import pinv
from tensorflow_probability.python.math.random_ops import random_rademacher
from tensorflow_probability.python.math.random_ops import random_rayleigh

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'matvecmul',
    'pinv',
    'random_rademacher',
    'random_rayleigh',
    'diag_jacobian',
]

remove_undocumented(__name__, _allowed_symbols)
