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

__all__ = [
    'broadcast_dynamic_shape',
    'broadcast_static_shape',
    'broadcast_to',
    'cast',
    'constant',
    'control_dependencies',
    'convert_to_tensor',
    'executing_eagerly',
    'get_static_value',
    'group',
    'identity',
    'is_tensor',
    'name_scope',
    'newaxis',
    'stop_gradient',
    'TensorShape',
    # 'gradients',
]


class _NullContext(object):

  def __init__(self, *args, **kwargs):
    pass

  def __enter__(self):
    pass

  def __exit__(self, type_arg, value_arg, traceback_arg):
    return False  # False values do not suppress exceptions.


def _broadcast_static_shape(shape_x, shape_y):
  shape_x = tf.TensorShape(shape_x)
  shape_y = tf.TensorShape(shape_y)
  shape_xy = tf.broadcast_static_shape(shape_x, shape_y)
  return np.array(shape_xy, dtype=np.int32)


def _constant(value, dtype=None, shape=None, name='Const'):  # pylint: disable=unused-argument
  x = np.array(value, dtype=utils.numpy_dtype(dtype))
  if shape is None:
    return x
  return np.reshape(x, shape)


def _control_dependencies(control_inputs):
  if control_inputs:
    for control in control_inputs:
      if callable(control):
        control()
  return _NullContext()


# --- Begin Public Functions --------------------------------------------------

broadcast_dynamic_shape = utils.copy_docstring(
    tf.broadcast_dynamic_shape,
    _broadcast_static_shape)

broadcast_static_shape = utils.copy_docstring(
    tf.broadcast_static_shape,
    _broadcast_static_shape)

broadcast_to = utils.copy_docstring(
    tf.broadcast_to,
    lambda input, shape, name=None: np.broadcast_to(input, shape))

cast = utils.copy_docstring(
    tf.cast,
    lambda x, dtype, name=None: np.array(x, dtype=utils.numpy_dtype(dtype)))

constant = utils.copy_docstring(
    tf.constant,
    _constant)

control_dependencies = utils.copy_docstring(
    tf.control_dependencies,
    _control_dependencies)

convert_to_tensor = utils.copy_docstring(
    tf.convert_to_tensor,
    lambda value, dtype=None, dtype_hint=None, name=None: (  # pylint: disable=g-long-lambda
        np.array(value, dtype=utils.numpy_dtype(dtype or dtype_hint))))

executing_eagerly = utils.copy_docstring(
    tf.executing_eagerly,
    lambda: True)

get_static_value = utils.copy_docstring(
    tf.get_static_value,
    lambda tensor, partial=False: tensor)

group = utils.copy_docstring(
    tf.group,
    lambda *inputs, **kwargs: None)

identity = utils.copy_docstring(
    tf.identity,
    lambda input, name=None: np.array(input))

is_tensor = utils.copy_docstring(
    tf.is_tensor,
    lambda x: isinstance(x, (np.ndarray, np.generic)))

name_scope = lambda name, *args, **kwargs: _NullContext()

newaxis = np.newaxis

stop_gradient = utils.copy_docstring(
    tf.stop_gradient,
    lambda input, name=None: np.array(input))

TensorShape = tf.TensorShape
