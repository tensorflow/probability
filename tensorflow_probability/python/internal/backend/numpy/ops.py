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
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'broadcast_dynamic_shape',
    'broadcast_static_shape',
    'broadcast_to',
    'cast',
    'clip_by_value',
    'constant',
    'control_dependencies',
    'convert_to_tensor',
    'custom_gradient',
    'executing_eagerly',
    'get_static_value',
    'group',
    'identity',
    'is_tensor',
    'name_scope',
    'newaxis',
    'stop_gradient',
    'GradientTape',
    'Module',
    'TensorShape',
    'Variable',
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
  x = np.array(value, dtype=None if dtype is None else utils.numpy_dtype(dtype))
  if shape is None:
    return x
  return np.reshape(x, shape)


def _control_dependencies(control_inputs):
  if control_inputs:
    for control in control_inputs:
      if callable(control):
        control()
  return _NullContext()


def _convert_to_tensor(value, dtype=None, dtype_hint=None, name=None):  # pylint: disable=unused-argument
  assert not tf.is_tensor(value), value
  return np.array(value, dtype=utils.numpy_dtype(dtype or dtype_hint))


# --- Begin Public Functions --------------------------------------------------


class GradientTape(object):
  """tf.GradientTape stub."""

  def __init__(self, persistent=False, watch_accessed_variables=True):  # pylint: disable=unused-argument
    pass

  def __enter__(self):
    return self

  def __exit__(self, typ, value, traceback):  # pylint: disable=unused-argument
    pass

  def watch(self, tensor):  # pylint: disable=unused-argument
    pass

  def gradient(self, target, sources, output_gradients=None,  # pylint: disable=unused-argument
               unconnected_gradients=UnconnectedGradients.NONE):  # pylint: disable=unused-argument
    return sources


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
    lambda x, dtype, name=None: np.array(x).astype(utils.numpy_dtype(dtype)))

clip_by_value = utils.copy_docstring(
    tf.clip_by_value,
    lambda t, clip_value_min, clip_value_max, name=None:  # pylint: disable=g-long-lambda
    np.clip(t, clip_value_min, clip_value_max))

constant = utils.copy_docstring(
    tf.constant,
    _constant)

control_dependencies = utils.copy_docstring(
    tf.control_dependencies,
    _control_dependencies)

convert_to_tensor = utils.copy_docstring(
    tf.convert_to_tensor,
    _convert_to_tensor)

custom_gradient = utils.copy_docstring(
    tf.custom_gradient,
    lambda f: f)

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


class name_scope(object):  # pylint: disable=invalid-name
  """A context manager for use when defining a Python op.

  This context manager pushes a name scope, which will make the name of all
  operations added within it have a prefix.

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope("MyOp") as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")
      c = tf.convert_to_tensor(c, name="c")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```

  When executed, the Tensors `a`, `b`, `c`, will have names `MyOp/a`, `MyOp/b`,
  and `MyOp/c`.

  If the scope name already exists, the name will be made unique by appending
  `_n`. For example, calling `my_op` the second time will generate `MyOp_1/a`,
  etc.
  """

  @property
  def name(self):
    return self._name

  def __init__(self, name, *args, **kwargs):
    del args, kwargs
    self._name = name
    if self._name is not None and not self._name.endswith('/'):
      self._name += '/'

  def __enter__(self):
    return self._name

  def __exit__(self, type_arg, value_arg, traceback_arg):
    return False  # False values do not suppress exceptions.


newaxis = np.newaxis

stop_gradient = utils.copy_docstring(
    tf.stop_gradient,
    lambda input, name=None: np.array(input))

TensorShape = tf.TensorShape


def Variable(initial_value=None, trainable=True, validate_shape=True,  # pylint: disable=unused-argument,invalid-name
             caching_device=None, name=None, variable_def=None, dtype=None,  # pylint: disable=unused-argument
             import_scope=None, constraint=None):  # pylint: disable=unused-argument
  assert constraint is None
  return np.array(initial_value, dtype=dtype or np.float32)


class Module(object):

  _TF_MODULE_IGNORED_PROPERTIES = frozenset()

  def __init__(self, name):
    self._name = name

  def _no_dependency(self, x):
    return x
