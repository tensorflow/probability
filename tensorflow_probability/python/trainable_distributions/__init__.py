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
"""Support for trainable distributions."""

from tensorflow_probability.python.trainable_distributions.trainable_distributions_lib import bernoulli
from tensorflow_probability.python.trainable_distributions.trainable_distributions_lib import multivariate_normal_tril
from tensorflow_probability.python.trainable_distributions.trainable_distributions_lib import normal
from tensorflow_probability.python.trainable_distributions.trainable_distributions_lib import poisson
from tensorflow_probability.python.trainable_distributions.trainable_distributions_lib import softplus_and_shift
from tensorflow_probability.python.trainable_distributions.trainable_distributions_lib import tril_with_diag_softplus_and_shift

from tensorflow.python.util.all_util import remove_undocumented


_allowed_symbols = [
    'bernoulli',
    'multivariate_normal_tril',
    'normal',
    'poisson',
    'softplus_and_shift',
    'tril_with_diag_softplus_and_shift',
]

remove_undocumented(__name__, _allowed_symbols)
