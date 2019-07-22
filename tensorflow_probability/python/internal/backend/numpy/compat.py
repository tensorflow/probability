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

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy import v1
from tensorflow_probability.python.internal.backend.numpy import v2
from tensorflow_probability.python.internal.backend.numpy.internal import utils


__all__ = [
    'dimension_value',
    'function',
    'v1',
    'v2',
]


def _dimension_value(dimension):
  if dimension is None:
    return None
  return int(dimension)


# --- Begin Public Functions --------------------------------------------------


dimension_value = utils.copy_docstring(
    tf.compat.dimension_value,
    _dimension_value)


function = utils.copy_docstring(
    tf.function,
    lambda func=None, input_signature=None, autograph=True,  # pylint: disable=g-long-lambda
           experimental_autograph_options=None,
           experimental_relax_shapes=False: func)

del tf, utils
