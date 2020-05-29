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

import collections

# pylint: disable=unused-import
from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import bitwise
from tensorflow_probability.python.internal.backend.numpy import debugging
from tensorflow_probability.python.internal.backend.numpy import errors
from tensorflow_probability.python.internal.backend.numpy import keras
from tensorflow_probability.python.internal.backend.numpy import linalg
from tensorflow_probability.python.internal.backend.numpy import nest
from tensorflow_probability.python.internal.backend.numpy import nn
from tensorflow_probability.python.internal.backend.numpy import numpy_logging as logging
from tensorflow_probability.python.internal.backend.numpy import numpy_math as math
from tensorflow_probability.python.internal.backend.numpy import numpy_signal as signal
from tensorflow_probability.python.internal.backend.numpy import random_generators as random
from tensorflow_probability.python.internal.backend.numpy import raw_ops
from tensorflow_probability.python.internal.backend.numpy import sets_lib as sets
from tensorflow_probability.python.internal.backend.numpy import sparse_lib as sparse
from tensorflow_probability.python.internal.backend.numpy import test_lib as test
from tensorflow_probability.python.internal.backend.numpy.control_flow import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.dtype import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.functional_ops import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.gen.tensor_shape import TensorShape
from tensorflow_probability.python.internal.backend.numpy.misc import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.numpy_array import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.numpy_math import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.ops import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.tensor_array_ops import TensorArray
# pylint: enable=unused-import


JAX_MODE = False


def _function(func=None, input_signature=None, autograph=True,  # pylint: disable=unused-argument
              experimental_autograph_options=None,  # pylint: disable=unused-argument
              experimental_relax_shapes=False, experimental_compile=None):  # pylint: disable=unused-argument
  """Dummy version of `tf.function`."""
  transform = lambda fn: fn
  if experimental_compile:
    if JAX_MODE:
      from jax import jit  # pylint: disable=g-import-not-at-top
      transform = jit
    else:
      raise NotImplementedError('Could not find compiler: Numpy only.')
  # This code path is for the `foo = tf.function(foo, ...)` use case.
  if func is not None:
    return transform(func)
  # This code path is for the following use case:
  #   @tf.function(...)
  #   def foo(...):
  #      ...
  # This case is equivalent to `foo = tf.function(...)(foo)`.
  return transform


# --- Begin Public Functions --------------------------------------------------


compat = collections.namedtuple('compat', 'dimension_value')(
    lambda dim: None if dim is None else int(dim))

function = utils.copy_docstring(
    'tf.function',
    _function)

eye = linalg.eye
matmul = linalg.matmul

del collections, utils
