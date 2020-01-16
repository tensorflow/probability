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
"""Experimental Numpy backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'constant',
]

constant = utils.copy_docstring(
    tf1.initializers.constant,
    lambda value=0, dtype=tf.dtypes.float32, verify_shape=False: (  # pylint: disable=g-long-lambda
        lambda shape, dtype=None, partition_info=None, verify_shape=None: (  # pylint: disable=g-long-lambda
            np.ones(shape, dtype=dtype) * value))
)
