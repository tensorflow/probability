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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import tensorflow.compat.v2 as tf

__all__ = [
    'convert_immutable_to_tensor',
    'is_mutable',
]


def convert_immutable_to_tensor(value, dtype=None, dtype_hint=None, name=None):
  """Converts the given `value` to a `Tensor` only if input is immutable.

  This function converts Python objects of various types to `Tensor` objects
  except if the input is mutable. A mutable object is characterized by
  `tensor_util.is_mutable` and is, roughly speaking, any object which is a
  `tf.Variable` or known to depend on a `tf.Variable`. It accepts `Tensor`
  objects, numpy arrays, Python lists, and Python scalars. This function does
  not descend through structured input--it only verifies if the input is mutable
  per `tensor_util.is_mutable`. For example:

  ```python
  from tensorflow_probability.python.internal import tensor_util

  x = tf.Variable(0.)
  y = tensor_util.convert_immutable_to_tensor(x)
  x is y
  # ==> True

  x = tf.constant(0.)
  y = tensor_util.convert_immutable_to_tensor(x)
  x is y
  # ==> True

  x = np.array(0.)
  y = tensor_util.convert_immutable_to_tensor(x)
  x is y
  # ==> False
  tf.is_tensor(y)
  # ==> True
  ```

  This function can be useful when composing a new operation in Python
  (such as `my_func` in the example above). All standard Python op
  constructors apply this function to each of their Tensor-valued
  inputs, which allows those ops to accept numpy arrays, Python lists,
  and scalars in addition to `Tensor` objects.

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
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    tensor: A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.
  """
  # We explicitly do not use a tf.name_scope to avoid graph clutter.
  if value is None:
    return None
  if is_mutable(value):
    if not hasattr(value, 'dtype'):
      raise ValueError(
          'Mutable type ({}) must implement `dtype` property '
          '({}).'.format(type(value).__name__, value))
    if not hasattr(value, 'shape'):
      raise ValueError(
          'Mutable type ({}) must implement `shape` property '
          '({}).'.format(type(value).__name__, value))
    if dtype is not None and dtype.base_dtype != value.dtype.base_dtype:
      raise TypeError('Mutable type must be of dtype "{}" but is "{}".'.format(
          dtype.base_dtype.name, value.dtype.base_dtype.name))
    return value
  return tf.convert_to_tensor(
      value, dtype=dtype, dtype_hint=dtype_hint, name=name)


def is_mutable(x):
  """Evaluates if the object is known to have `tf.Variable` ancestors.

  An object is deemed mutable if it is a `tf.Variable` instance or has a
  properties `variables` or `trainable_variables` one of which is non-empty (as
  might be the case for a subclasses of `tf.Module` or a Keras layer).

  Args:
    x: Python object which may or may not have a `tf.Variable` ancestor.

  Returns:
    is_mutable: Python `bool` indicating input is mutable or is known to depend
      on mutable objects.
  """
  # TODO(b/134430874): Consider making this recurse through nests, e.g.,
  # `tensor_util.is_mutable([tf.Variable(0.), np.array(1.)])`
  # returns True. Note: we'd need to actually create a tf.Module on user's
  # behalf and it would need a `dtype` and `shape`. (I.e., there would be some
  # work to support this.)
  return ((inspect.isclass(tf.Variable) and isinstance(x, tf.Variable)) or
          getattr(x, 'variables', ()) or
          getattr(x, 'trainable_variables', ()))
