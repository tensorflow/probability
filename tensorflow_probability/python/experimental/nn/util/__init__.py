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
"""Utilitity functions for building neural networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.experimental.nn.util.im2row import im2row
from tensorflow_probability.python.experimental.nn.util.random_variable import CallOnce
from tensorflow_probability.python.experimental.nn.util.random_variable import RandomVariable
from tensorflow_probability.python.experimental.nn.util.utils import display_imgs
from tensorflow_probability.python.experimental.nn.util.utils import expand_dims
from tensorflow_probability.python.experimental.nn.util.utils import flatten_rightmost
from tensorflow_probability.python.experimental.nn.util.utils import halflife_decay
from tensorflow_probability.python.experimental.nn.util.utils import make_fit_op
from tensorflow_probability.python.experimental.nn.util.utils import make_kernel_bias
from tensorflow_probability.python.experimental.nn.util.utils import make_kernel_bias_posterior_mvn_diag
from tensorflow_probability.python.experimental.nn.util.utils import make_kernel_bias_prior_spike_and_slab
from tensorflow_probability.python.experimental.nn.util.utils import negloglik
from tensorflow_probability.python.experimental.nn.util.utils import tfcompile
from tensorflow_probability.python.experimental.nn.util.utils import trace
from tensorflow_probability.python.experimental.nn.util.utils import tune_dataset
from tensorflow_probability.python.experimental.nn.util.utils import variables_load
from tensorflow_probability.python.experimental.nn.util.utils import variables_save
from tensorflow_probability.python.experimental.nn.util.utils import variables_summary
from tensorflow_probability.python.internal import all_util


_allowed_symbols = [
    'CallOnce',
    'RandomVariable',
    'display_imgs',
    'expand_dims',
    'flatten_rightmost',
    'halflife_decay',
    'im2row',
    'make_fit_op',
    'make_kernel_bias',
    'make_kernel_bias_posterior_mvn_diag',
    'make_kernel_bias_prior_spike_and_slab',
    'negloglik',
    'tfcompile',
    'trace',
    'tune_dataset',
    'variables_load',
    'variables_save',
    'variables_summary',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
