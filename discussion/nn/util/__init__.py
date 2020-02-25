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

from discussion.nn.util.utils import display_imgs
from discussion.nn.util.utils import expand_dims
from discussion.nn.util.utils import flatten_rightmost
from discussion.nn.util.utils import make_fit_op
from discussion.nn.util.utils import make_kernel_bias
from discussion.nn.util.utils import make_kernel_bias_posterior_mvn_diag
from discussion.nn.util.utils import make_kernel_bias_prior_spike_and_slab
from discussion.nn.util.utils import negloglik
from discussion.nn.util.utils import tfcompile
from discussion.nn.util.utils import trace
from discussion.nn.util.utils import tune_dataset
from discussion.nn.util.utils import variables_load
from discussion.nn.util.utils import variables_save
from discussion.nn.util.utils import variables_summary

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'display_imgs',
    'expand_dims',
    'flatten_rightmost',
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

remove_undocumented(__name__, _allowed_symbols)
