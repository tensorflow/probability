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

from tensorflow_probability.python.experimental.vi import util
from tensorflow_probability.python.experimental.vi.automatic_structured_vi import build_asvi_surrogate_posterior
from tensorflow_probability.python.experimental.vi.automatic_structured_vi import register_asvi_substitution_rule
from tensorflow_probability.python.experimental.vi.cascading_flows import build_cascading_flow_surrogate_posterior
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_affine_surrogate_posterior
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_affine_surrogate_posterior_from_base_distribution
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_factored_surrogate_posterior
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_split_flow_surrogate_posterior
from tensorflow_probability.python.internal import all_util


_allowed_symbols = [
    'build_affine_surrogate_posterior',
    'build_affine_surrogate_posterior_from_base_distribution',
    'build_asvi_surrogate_posterior',
    'build_cascading_flow_surrogate_posterior',
    'build_factored_surrogate_posterior',
    'build_split_flow_surrogate_posterior',
    'register_asvi_substitution_rule',
    'util',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
