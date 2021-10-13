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
"""Tools for processing Tensors."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static

__all__ = [
    'convert_nonref_to_tensor',
    'discover_trainable_variables',
    'discover_variables',
    'identity_as_tensor',
    'is_module',
    'is_ref',
    'is_trainable_variable',
    'is_variable',
]


def convert_nonref_to_tensor(value, dtype=None, dtype_hint=None,
                             as_shape_tensor=False, name=None):
  """Converts the given `value` to a `Tensor` if input is nonreference type.

  This function converts Python objects of various types to `Tensor` objects
  only if the input has nonreference semantics. Reference semantics are
  characterized by `tensor_util.is_ref` and is any object which is a
  `tf.Variable` or instance of `tf.Module`. This function accepts any input
  which `tf.convert_to_tensor` would also.

  Note: This function diverges from default Numpy behavior for `float` and
    `string` types when `None` is present in a Python list or scalar. Rather
    than silently converting `None` values, an error will be thrown.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    dtype_hint: Optional element type for the returned tensor,
      used when dtype is None. In some cases, a caller may not have a
      dtype in mind when converting to a tensor, so dtype_hint
      can be used as a soft preference.  If the conversion to
      `dtype_hint` is not possible, this argument has no effect.
    as_shape_tensor: Optional boolean when if `True` uses
      `prefer_static.convert_to_shape_tensor` instead of `tf.convert_to_tensor`
      for JAX compatibility.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    tensor: A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.


  #### Examples:

  ```python
  from tensorflow_probability.python.internal import tensor_util

  x = tf.Variable(0.)
  y = tensor_util.convert_nonref_to_tensor(x)
  x is y
  # ==> True

  x = tf.constant(0.)
  y = tensor_util.convert_nonref_to_tensor(x)
  x is y
  # ==> True

  x = np.array(0.)
  y = tensor_util.convert_nonref_to_tensor(x)
  x is y
  # ==> False
  tf.is_tensor(y)
  # ==> True

  x = tfp.util.DeferredTensor(13.37, lambda x: x)
  y = tensor_util.convert_nonref_to_tensor(x)
  x is y
  # ==> True
  tf.is_tensor(y)
  # ==> True
  tf.equal(y, 13.37)
  # ==> True
  ```

  """
  # We explicitly do not use a tf.name_scope to avoid graph clutter.
  if value is None:
    return None
  if is_ref(value):
    if dtype is None:
      return value
    dtype_base = dtype_util.base_dtype(dtype)
    value_dtype_base = dtype_util.base_dtype(value.dtype)
    if dtype_base != value_dtype_base:
      raise TypeError('Mutable type must be of dtype "{}" but is "{}".'.format(
          dtype_util.name(dtype_base), dtype_util.name(value_dtype_base)))
    return value
  if as_shape_tensor:
    return prefer_static.convert_to_shape_tensor(
        value, dtype=dtype, dtype_hint=dtype_hint, name=name)
  return tf.convert_to_tensor(
      value, dtype=dtype, dtype_hint=dtype_hint, name=name)


def identity_as_tensor(value):
  """Converts `value` to `Tensor` while ensuring an op is added to the graph."""
  t = tf.convert_to_tensor(value)
  if t is value:
    t = tf.identity(value)
  return t


def is_ref(x):
  """Evaluates if the object has reference semantics.

  An object is deemed "reference" if it is a `tf.Variable` instance or is
  derived from a `tf.Module` with `dtype` and `shape` properties.

  Args:
    x: Any object.

  Returns:
    is_ref: Python `bool` indicating input is has nonreference semantics, i.e.,
      is a `tf.Variable` or a `tf.Module` with `dtype` and `shape` properties.
  """
  # TODO(b/134430874): Consider making this recurse through nests, e.g.,
  # `tensor_util.is_ref([tf.Variable(0.), np.array(1.)])`
  # returns True. Note: we'd need to actually create a tf.Module on user's
  # behalf and it would need a `dtype` and `shape`. (I.e., there would be some
  # work to support this.)
  return (
      is_variable(x) or
      (is_module(x) and hasattr(x, 'dtype') and hasattr(x, 'shape'))
  )


def is_variable(x):
  """Returns `True` when input is a `tf.Variable`, otherwise `False`."""
  return isinstance(x, tf.Variable)


def is_trainable_variable(x):
  """Returns `True` when input is trainable `tf.Variable`, otherwise `False`."""
  return is_variable(x) and getattr(x, 'trainable', False)


def is_module(x):
  """Returns `True` when input is a `tf.Module`, otherwise `False`."""
  return isinstance(x, tf.Module)


class _Track(tf.Module):
  """Bridge to create functional interface for variable tracking."""

  def __init__(self, *args, **kwargs):
    self._args = args
    self._kwargs = kwargs


def discover_trainable_variables(x):
  """Returns `tuple` of all trainable `tf.Variables` discoverable in input.

  Warning: unlike possibly `tf.Module`, use of this function only does a static,
  "one-time" discovery. (This is self-evidently true from its functional
  nature.)

  Args:
    x: An object to inspected for `tf.Variable` dependencies.

  Returns:
    trainable_vars: A Python `tuple` of `tf.Variable`s with `trainable=True`.
  """
  return _Track(x).trainable_variables


def discover_variables(x):
  """Returns `tuple` of all `tf.Variables` discoverable in input.

  Warning: unlike possibly `tf.Module`, use of this function only does a static,
  "one-time" discovery. (This is self-evidently true from its functional
  nature.)

  Args:
    x: An object to inspected for `tf.Variable` dependencies.

  Returns:
    vars: A Python `tuple` of `tf.Variable`s, regardless of their value of
      `trainable`.
  """
  return _Track(x).variables
