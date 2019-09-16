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
"""TensorFlow Probability auto-batching package.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.experimental.auto_batching import allocation_strategy
from tensorflow_probability.python.experimental.auto_batching import dsl
from tensorflow_probability.python.experimental.auto_batching import frontend
from tensorflow_probability.python.experimental.auto_batching import instructions
from tensorflow_probability.python.experimental.auto_batching import liveness
from tensorflow_probability.python.experimental.auto_batching import lowering
from tensorflow_probability.python.experimental.auto_batching import numpy_backend
from tensorflow_probability.python.experimental.auto_batching import stack_optimization
from tensorflow_probability.python.experimental.auto_batching import stackless
from tensorflow_probability.python.experimental.auto_batching import tf_backend
from tensorflow_probability.python.experimental.auto_batching import type_inference
from tensorflow_probability.python.experimental.auto_batching import virtual_machine
from tensorflow_probability.python.experimental.auto_batching import xla
from tensorflow_probability.python.experimental.auto_batching.frontend import Context
from tensorflow_probability.python.experimental.auto_batching.frontend import truthy
from tensorflow_probability.python.experimental.auto_batching.instructions import TensorType
from tensorflow_probability.python.experimental.auto_batching.instructions import Type
from tensorflow_probability.python.experimental.auto_batching.numpy_backend import NumpyBackend
from tensorflow_probability.python.experimental.auto_batching.tf_backend import TensorFlowBackend

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'Context',
    'NumpyBackend',
    'TensorFlowBackend',
    'TensorType',
    'Type',
    'allocation_strategy',
    'dsl',
    'frontend',
    'instructions',
    'liveness',
    'lowering',
    'numpy_backend',
    'stack_optimization',
    'stackless',
    'tf_backend',
    'truthy',
    'type_inference',
    'virtual_machine',
    'xla',
]

remove_undocumented(__name__, _allowed_symbols)
