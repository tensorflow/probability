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

import contextlib

import numpy as np
from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import initializers
from tensorflow_probability.python.internal.backend.numpy import linalg_impl
from tensorflow_probability.python.internal.backend.numpy import numpy_logging as logging
from tensorflow_probability.python.internal.backend.numpy.numpy_array import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.internal.backend.numpy.ops import convert_to_tensor
from tensorflow_probability.python.internal.backend.numpy.ops import Module
from tensorflow_probability.python.internal.backend.numpy.ops import name_scope
from tensorflow_probability.python.internal.backend.numpy.ops import Variable
from tensorflow_probability.python.internal.backend.numpy.random_generators import set_seed
from tensorflow_probability.python.internal.backend.numpy.tensor_array_ops import TensorArray


__all__ = [
    'Module',
    'Session',
    'TensorArray',
    'colocate_with',
    'control_flow_v2_enabled',
    'get_variable',
    'get_variable_scope',
    'global_variables_initializer',
    'initializers',
    'logging',
    'matrix_determinant',
    'matrix_solve',
    'name_scope',
    'placeholder_with_default',
    'set_random_seed',
    'variable_scope',
]


JAX_MODE = False


@contextlib.contextmanager
def _dummy_scope(*_, **__):  # pylint: disable=unused-argument
  yield


def _get_variable(  # pylint: disable=unused-argument
    name, shape=None, dtype=None, initializer=None, regularizer=None,
    trainable=None, collections=None, caching_device=None, partitioner=None,
    validate_shape=True, use_resource=None, custom_getter=None, constraint=None,
    synchronization=None, aggregation=None):
  if callable(initializer):
    initial_value = initializer(shape)
  else:
    initial_value = initializer
  return Variable(
      initial_value=initial_value, trainable=trainable,
      validate_shape=validate_shape, caching_device=caching_device, name=name,
      variable_def=None, dtype=dtype, import_scope=None, constraint=None)


def _placeholder_with_default(input, shape, name=None):  # pylint: disable=redefined-builtin,unused-argument
  x = convert_to_tensor(input)
  if hasattr(shape, 'as_list'):
    shape = shape.as_list()
  if shape is None or any(s is None for s in shape):
    return x
  return np.reshape(x, shape)


# --- Begin Public Functions --------------------------------------------------


matrix_determinant = linalg_impl.det
matrix_solve = linalg_impl.solve

colocate_with = utils.copy_docstring(
    'tf1.colocate_with',
    _dummy_scope)

control_flow_v2_enabled = utils.copy_docstring(
    'tf1.control_flow_v2_enabled',
    lambda: True)

enable_control_flow_v2 = utils.copy_docstring(
    'tf1.enable_control_flow_v2',
    lambda: None)

get_variable = utils.copy_docstring(
    'tf1.get_variable',
    _get_variable)

get_variable_scope = utils.copy_docstring(
    'tf1.get_variable_scope',
    lambda: variable_scope(name_or_scope=None))

placeholder_with_default = utils.copy_docstring(
    'tf1.placeholder_with_default',
    _placeholder_with_default)

global_variables_initializer = utils.copy_docstring(
    'tf1.global_variables_initializer',
    lambda: None)

set_random_seed = utils.copy_docstring(
    'tf1.set_random_seed',
    set_seed)


class Session(object):

  def __enter__(self, *_, **__):
    return self

  def __exit__(self, *_, **__):
    pass

  def run(self, *args, **_):
    return args


class variable_scope(object):  # pylint: disable=invalid-name
  """A context manager for defining ops that creates variables (layers)."""

  def __init__(
      self, name_or_scope, default_name=None, values=None, initializer=None,  # pylint: disable=unused-argument
      regularizer=None, caching_device=None, partitioner=None,  # pylint: disable=unused-argument
      custom_getter=None, reuse=None, dtype=None, use_resource=None,  # pylint: disable=unused-argument
      constraint=None, auxiliary_name_scope=True):  # pylint: disable=unused-argument
    self._caching_device = None

  @property
  def caching_device(self):
    return self._caching_device

  @caching_device.setter
  def caching_device(self, val):
    self._caching_device = val

  def __enter__(self, *_, **__):
    return self

  def __exit__(self, *_, **__):
    pass
