# Copyright 2020 The TensorFlow Probability Authors.
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
"""TensorFlow Probability experimental python utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.experimental.util.deferred_module import DeferredModule
from tensorflow_probability.python.experimental.util.jit_public_methods import DEFAULT_METHODS_EXCLUDED_FROM_JIT
from tensorflow_probability.python.experimental.util.jit_public_methods import JitPublicMethods
from tensorflow_probability.python.experimental.util.trainable import make_trainable
from tensorflow_probability.python.internal import all_util


_allowed_symbols = [
    'DEFAULT_METHODS_EXCLUDED_FROM_JIT',
    'DeferredModule',
    'JitPublicMethods',
    'make_trainable'
]

all_util.remove_undocumented(__name__, _allowed_symbols)
