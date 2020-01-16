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
"""TensorFlow Probability alternative substrates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import lazy_loader  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

jax = lazy_loader.LazyLoader(
    'jax', globals(),
    'tensorflow_probability.python.experimental.substrates.jax')
numpy = lazy_loader.LazyLoader(
    'numpy', globals(),
    'tensorflow_probability.python.experimental.substrates.numpy')


_allowed_symbols = [
    'jax',
    'numpy',
]

remove_undocumented(__name__, _allowed_symbols)
