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
"""Tools for probabilistic reasoning in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python import bijectors
from tensorflow_probability.python import debugging
from tensorflow_probability.python import distributions
from tensorflow_probability.python import edward2
from tensorflow_probability.python import experimental
from tensorflow_probability.python import glm
from tensorflow_probability.python import layers
from tensorflow_probability.python import math
from tensorflow_probability.python import mcmc
from tensorflow_probability.python import monte_carlo
from tensorflow_probability.python import optimizer
from tensorflow_probability.python import positive_semidefinite_kernels
from tensorflow_probability.python import stats
from tensorflow_probability.python import sts
from tensorflow_probability.python import trainable_distributions
from tensorflow_probability.python import util
from tensorflow_probability.python import vi

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'bijectors',
    'debugging',
    'distributions',
    'edward2',
    'experimental',
    'glm',
    'layers',
    'math',
    'mcmc',
    'monte_carlo',
    'optimizer',
    'positive_semidefinite_kernels',
    'trainable_distributions',
    'stats',
    'sts',
    'util',
    'vi',
]

remove_undocumented(__name__, _allowed_symbols)
