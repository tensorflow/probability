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
from tensorflow_probability.python.experimental.vi.automatic_structured_vi import ASVI_DEFAULT_PRIOR_SUBSTITUTION_RULES
from tensorflow_probability.python.experimental.vi.automatic_structured_vi import ASVI_DEFAULT_SURROGATE_RULES
from tensorflow_probability.python.experimental.vi.automatic_structured_vi import build_asvi_surrogate_posterior
from tensorflow_probability.python.experimental.vi.automatic_structured_vi import build_asvi_surrogate_posterior_stateless
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_affine_surrogate_posterior
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_affine_surrogate_posterior_from_base_distribution
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_affine_surrogate_posterior_from_base_distribution_stateless
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_affine_surrogate_posterior_stateless
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_factored_surrogate_posterior
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_factored_surrogate_posterior_stateless
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import build_split_flow_surrogate_posterior
from tensorflow_probability.python.internal import all_util

JAX_MODE = False

_allowed_symbols = [
    'ASVI_DEFAULT_PRIOR_SUBSTITUTION_RULES',
    'ASVI_DEFAULT_SURROGATE_RULES',
    'build_affine_surrogate_posterior_stateless',
    'build_affine_surrogate_posterior_from_base_distribution_stateless',
    'build_asvi_surrogate_posterior_stateless',
    'build_factored_surrogate_posterior_stateless',
    'build_split_flow_surrogate_posterior',
    'util',
]

if not JAX_MODE:
  # Expose stateful surrogates in TF only.
  _allowed_symbols = _allowed_symbols + [
      s[:-10] for s in _allowed_symbols if s.endswith('_stateless')]


all_util.remove_undocumented(__name__, _allowed_symbols)
