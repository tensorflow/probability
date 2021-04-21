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
"""TensorFlow Probability API-unstable package.

This package contains potentially useful code which is under active development
with the intention eventually migrate to TFP proper. All code in
`tfp.experimental` should be of production quality, i.e., idiomatically
consistent, well tested, and extensively documented. `tfp.experimental` code
relaxes the TFP non-experimental contract in two regards:
1. `tfp.experimental` has no API stability guarantee. The public footprint of
   `tfp.experimental` code may change without notice or warning.
2. Code outside `tfp.experimental` cannot depend on code within
   `tfp.experimental`.

You are welcome to try any of this out (and tell us how well it works for you!).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.experimental import auto_batching
from tensorflow_probability.python.experimental import bijectors
from tensorflow_probability.python.experimental import distribute
from tensorflow_probability.python.experimental import distributions
from tensorflow_probability.python.experimental import lazybones
from tensorflow_probability.python.experimental import linalg
from tensorflow_probability.python.experimental import marginalize
from tensorflow_probability.python.experimental import mcmc
from tensorflow_probability.python.experimental import nn
from tensorflow_probability.python.experimental import parallel_filter
from tensorflow_probability.python.experimental import sequential
from tensorflow_probability.python.experimental import stats
from tensorflow_probability.python.experimental import substrates
from tensorflow_probability.python.experimental import util
from tensorflow_probability.python.experimental import vi
from tensorflow_probability.python.experimental.composite_tensor import as_composite
from tensorflow_probability.python.experimental.composite_tensor import register_composite
from tensorflow_probability.python.internal import all_util
from tensorflow_probability.python.internal.auto_composite_tensor import auto_composite_tensor
from tensorflow_probability.python.internal.auto_composite_tensor import AutoCompositeTensor
from tensorflow_probability.python.internal.auto_composite_tensor import register_type_spec


_allowed_symbols = [
    'auto_batching',
    'as_composite',
    'auto_composite_tensor',
    'AutoCompositeTensor',
    'bijectors',
    'distribute',
    'distributions',
    'lazybones',
    'linalg',
    'marginalize',
    'mcmc',
    'nn',
    'parallel_filter',
    'register_composite',
    'register_type_spec',
    'sequential',
    'stats',
    'substrates',
    'unnest',
    'util',
    'vi',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
