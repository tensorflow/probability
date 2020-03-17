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
"""Helper functions for numpy backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import types

import numpy as np

try:
  from tensorflow.python.ops import array_ops  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top,unused-import
  from tensorflow.python.ops import random_ops  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top,unused-import
  import tensorflow.compat.v1 as tf1  # pylint: disable=g-import-not-at-top,unused-import
  import tensorflow.compat.v2 as tf  # pylint: disable=g-import-not-at-top
  import wrapt  # pylint: disable=g-import-not-at-top
  can_copy_docstring = True
except ImportError:
  can_copy_docstring = False


__all__ = [
    'common_dtype',
    'copy_docstring',
    'numpy_dtype',
    'try_import',
]


def _find_method_from_name(scope, name):
  method = name.split('.', 1)
  if isinstance(scope, dict):
    child = scope[method[0]]
  else:
    child = getattr(scope, method[0])
  if len(method) == 1:
    return child
  return _find_method_from_name(child, method[1])


def copy_docstring(tf_method_name, new_fn):  # pylint: disable=unused-argument
  """Wraps new_fn with the doc of original_fn if TensorFlow can be imported."""
  if not can_copy_docstring:
    return new_fn
  original_fn = _find_method_from_name(globals(), tf_method_name)
  @wrapt.decorator
  def wrap(wrapped, instance, args, kwargs):
    del instance, wrapped
    return new_fn(*args, **kwargs)
  return wrap(original_fn)  # pylint: disable=no-value-for-parameter


def numpy_dtype(dtype):
  if dtype is None:
    return None
  if hasattr(dtype, 'as_numpy_dtype'):
    return dtype.as_numpy_dtype
  return dtype


def common_dtype(args_list, dtype_hint=None):
  """Returns explict dtype from `args_list` if exists, else dtype_hint."""
  dtype = None
  dtype_hint = None if dtype_hint is None else tf.as_dtype(dtype_hint)
  for a in tf.nest.flatten(args_list):
    if hasattr(a, 'dtype'):
      dt = tf.as_dtype(a.dtype)
    else:
      continue
    if dtype is None:
      dtype = dt
    elif dtype != dt:
      raise TypeError('Found incompatible dtypes, {} and {}'.format(dtype, dt))
  if dtype is None and dtype_hint is None:
    return None
  return (dtype_hint if dtype is None else dtype).as_numpy_dtype


def is_complex(dtype):
  """Returns whether this is a complex floating point type."""
  return np.issubdtype(np.dtype(dtype), np.complexfloating)


class _FakeModule(types.ModuleType):
  """Dummy module which raises `NotImplementedError` on `getattr` access."""

  def __init__(self, name, doc):
    self._name = name
    self._doc = doc
    types.ModuleType.__init__(self, name, doc)  # pylint: disable=non-parent-init-called

  def __dir__(self):
    return []

  def __getattr__(self, attr):
    raise NotImplementedError(self._doc)


def try_import(name):  # pylint: disable=invalid-name
  try:
    return importlib.import_module(name)
  except ImportError:
    return _FakeModule(name, 'Error loading module "{}".'.format(name))
