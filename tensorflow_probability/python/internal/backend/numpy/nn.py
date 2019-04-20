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
"""Numpy implementations of TensorFlow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy.internal import utils
from tensorflow_probability.python.internal.backend.numpy.math import log_softmax
from tensorflow_probability.python.internal.backend.numpy.math import softmax
from tensorflow_probability.python.internal.backend.numpy.math import softplus


__all__ = [
    'log_softmax',
    'relu',
    'softmax',
    'softplus',
    # 'sigmoid_cross_entropy_with_logits',
]


# --- Begin Public Functions --------------------------------------------------


relu = utils.copy_docstring(
    tf.nn.relu,
    lambda features, name=None: np.max(features, 0))
