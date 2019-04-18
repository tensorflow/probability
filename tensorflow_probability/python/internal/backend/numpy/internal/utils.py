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

import tensorflow as tf


__all__ = [
    'common_dtype',
    'copy_docstring',
    'numpy_dtype',
    'try_import',
]


# TODO(jvdillon): Get decoration working. Eg,
# # Dependency imports
# import decorator


def copy_docstring(original_fn, new_fn):  # pylint: disable=unused-argument
  return new_fn
  # TODO(jvdillon): Get decoration working. Eg,
  # @decorator.decorator
  # def wrap(wrapped_fn, *args, **kwargs):
  #   del wrapped_fn
  #   return new_fn(*args, **kwargs)
  # return wrap(original_fn)


def numpy_dtype(dtype):
  if dtype is None:
    return None
  if hasattr(dtype, 'as_numpy_dtype'):
    return dtype.as_numpy_dtype
  return dtype


def common_dtype(args_list, preferred_dtype=None):
  """Returns explict dtype from `args_list` if exists, else preferred_dtype."""
  dtype = None
  preferred_dtype = (None if preferred_dtype is None
                     else tf.as_dtype(preferred_dtype))
  for a in tf.nest.flatten(args_list):
    if hasattr(a, 'dtype'):
      dt = tf.as_dtype(a.dtype)
    else:
      continue
    if dtype is None:
      dtype = dt
    elif dtype != dt:
      raise TypeError('Found incompatible dtypes, {} and {}.'.format(dtype, dt))
  if dtype is None and preferred_dtype is None:
    return None
  return (preferred_dtype if dtype is None else dtype).as_numpy_dtype


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
