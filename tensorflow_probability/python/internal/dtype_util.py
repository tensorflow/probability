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
"""Utility functions for dtypes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf


__all__ = [
    'as_numpy_dtype',
    'base_dtype',
    'base_equal',
    'common_dtype',
    'is_bool',
    'is_complex',
    'is_floating',
    'is_integer',
    'max',
    'min',
    'name',
    'size',
]


def as_numpy_dtype(dtype):
  """Returns a `np.dtype` based on this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'as_numpy_dtype'):
    return dtype.as_numpy_dtype
  return dtype


def base_dtype(dtype):
  """Returns a non-reference `dtype` based on this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'base_dtype'):
    return dtype.base_dtype
  return dtype


def base_equal(a, b):
  """Returns `True` if base dtypes are identical."""
  return base_dtype(a) == base_dtype(b)


def common_dtype(args_list, preferred_dtype=None):
  """Returns explict dtype from `args_list` if there is one."""
  dtype = None
  for a in tf.nest.flatten(args_list):
    if hasattr(a, 'dtype'):
      dt = as_numpy_dtype(a.dtype)
    else:
      continue
    if dtype is None:
      dtype = dt
    elif dtype != dt:
      raise TypeError('Found incompatible dtypes, {} and {}.'.format(dtype, dt))
  return preferred_dtype if dtype is None else tf.as_dtype(dtype)


def is_bool(dtype):
  """Returns whether this is a boolean data type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_bool'):
    return dtype.is_bool
  # We use `kind` because:
  # np.issubdtype(np.uint8, np.bool) == True.
  return np.dtype(dtype).kind == 'b'


def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_complex'):
    return dtype.is_complex
  return np.issubdtype(np.dtype(dtype), np.complex)


def is_floating(dtype):
  """Returns whether this is a (non-quantized, real) floating point type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_floating'):
    return dtype.is_floating
  return np.issubdtype(np.dtype(dtype), np.float)


def is_integer(dtype):
  """Returns whether this is a (non-quantized) integer type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_integer'):
    return dtype.is_integer
  return np.issubdtype(np.dtype(dtype), np.integer)


def max(dtype):  # pylint: disable=redefined-builtin
  """Returns the maximum representable value in this data type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'max'):
    return dtype.max
  use_finfo = is_floating(dtype) or is_complex(dtype)
  return np.finfo(dtype).max if use_finfo else np.iinfo(dtype).max


def min(dtype):  # pylint: disable=redefined-builtin
  """Returns the minimum representable value in this data type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'min'):
    return dtype.min
  use_finfo = is_floating(dtype) or is_complex(dtype)
  return np.finfo(dtype).min if use_finfo else np.iinfo(dtype).min


def name(dtype):
  """Returns the string name for this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'name'):
    return dtype.name
  if hasattr(dtype, '__name__'):
    return dtype.__name__
  return str(dtype)


def size(dtype):
  """Returns the number of bytes to represent this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'size'):
    return dtype.size
  return np.dtype(dtype).itemsize
