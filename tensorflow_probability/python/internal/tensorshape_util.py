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
"""Utility functions for `TensorShape`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

import tensorflow.compat.v2 as tf

from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import tensor_shape  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import tensor_util  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'as_list',
    'assert_has_rank',
    'assert_is_compatible_with',
    'concatenate',
    'constant_value_as_shape',
    'dims',
    'is_compatible_with',
    'is_fully_defined',
    'merge_with',
    'num_elements',
    'rank',
    'set_shape',
    'with_rank',
    'with_rank_at_least',
]

JAX_MODE = False


def as_list(x):
  """Returns a `list` of integers or `None` for each dimension.

  For more details, see `help(tf.TensorShape.as_list)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.

  Returns:
    shape_as_list: list of `int` or `None` values representing each dimensions
      size if known.

  Raises:
    ValueError: If `x` has unknown rank.
  """
  return tf.TensorShape(x).as_list()


def _cast_tensorshape(x, x_type):
  if issubclass(x_type, tf.TensorShape):
    return x
  if issubclass(x_type, onp.ndarray):
    # np.ndarray default constructor will place x
    # as the shape, which we don't want.
    return onp.array(as_list(x), dtype=onp.int32)
  return x_type(as_list(x))


def assert_has_rank(x, rank):  # pylint: disable=redefined-outer-name
  """Raises an exception if `x` is not compatible with the given `rank`.

  For more details, see `help(tf.TensorShape.assert_has_rank)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.
    rank: an `int` representing the minimum required rank of `x`.

  Returns:
    None

  Raises:
    ValueError: If `x` does not represent a shape with the given `rank`.
  """
  tf.TensorShape(x).assert_has_rank(rank)


def assert_is_compatible_with(x, other):
  """Raises exception if `x` and `other` do not represent the same shape.

  This method can be used to assert that there exists a shape that both
  `x` and `other` represent.

  For more details, see `help(tf.TensorShape.assert_is_compatible_with)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.
    other: object representing a shape; convertible to `tf.TensorShape`.

  Returns:
    None

  Raises:
    ValueError: If `x` and `other` do not represent the same shape.
  """
  tf.TensorShape(x).assert_is_compatible_with(other)


def concatenate(x, other):
  """Returns the concatenation of the dimension in `x` and `other`.

  *Note:* If either `x` or `other` is completely unknown, concatenation will
  discard information about the other shape. In future, we might support
  concatenation that preserves this information for use with slicing.

  For more details, see `help(tf.TensorShape.concatenate)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.
    other: object representing a shape; convertible to `tf.TensorShape`.

  Returns:
    new_shape: an object like `x` whose elements are the concatenation of the
      dimensions in `x` and `other`.
  """
  return _cast_tensorshape(tf.TensorShape(x).concatenate(other), type(x))


def constant_value_as_shape(tensor):  # pylint: disable=invalid-name
  """A version of `constant_value()` that returns a `TensorShape`.

  This version should be used when a constant tensor value is
  interpreted as a (possibly partial) shape, e.g. in the shape
  function for `tf.reshape()`. By explicitly requesting a
  `TensorShape` as the return value, it is possible to represent
  unknown dimensions; by contrast, `constant_value()` is
  all-or-nothing.

  Args:
    tensor: The rank-0 or rank-1 Tensor to be evaluated.

  Returns:
    A `TensorShape` based on the constant value of the given `tensor`.

  Raises:
    ValueError: If the shape is rank-0 and is not statically known to be -1.
  """
  shape = tf.get_static_value(tensor)
  if shape is not None:
    return tensor_shape.as_shape(
        [None if dim == -1 else dim for dim in shape])
  try:
    # Importing here, conditionally, to avoid a hard dependency on
    # DeferredTensor, because that creates a BUILD dependency cycle.
    # Why is it necessary to mention DeferredTensor at all?
    # Because TF's `constant_value_as_shape` barfs on it: b/142254634.
    # pylint: disable=g-import-not-at-top
    from tensorflow_probability.python.util.deferred_tensor import DeferredTensor
    if isinstance(tensor, DeferredTensor):
      # Presumably not constant if deferred
      return tf.TensorShape(None)
  except ImportError:
    # If DeferredTensor doesn't even exist, couldn't have been an instance of
    # it.
    pass
  if tf.executing_eagerly():
    # Working around b/142251799
    if isinstance(tensor, ops.EagerTensor):
      return tensor_shape.as_shape(
          [dim if dim != -1 else None for dim in tensor.numpy()])
    else:
      return tf.TensorShape(None)
  return tensor_util.constant_value_as_shape(tensor)


def dims(x):
  """Returns a list of dimension sizes, or `None` if `rank` is unknown.

  For more details, see `help(tf.TensorShape.dims)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.

  Returns:
    shape_as_list: list of sizes or `None` values representing each
      dimensions size if known. A size is `tf.Dimension` if input is a
      `tf.TensorShape` and an `int` otherwise.
  """
  if isinstance(x, tf.TensorShape):
    return x.dims
  r = tf.TensorShape(x).dims
  return None if r is None else list(map(tf.compat.dimension_value, r))


def is_compatible_with(x, other):
  """Returns `True` iff `x` is compatible with `other`.

  For more details, see `help(tf.TensorShape.is_compatible_with)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.
    other: object representing a shape; convertible to `tf.TensorShape`.

  Returns:
    is_compatible: `bool` indicating of the shapes are compatible.
  """
  return tf.TensorShape(x).is_compatible_with(other)


def is_fully_defined(x):
  """Returns True iff `x` is fully defined in every dimension.

  For more details, see `help(tf.TensorShape.is_fully_defined)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.

  Returns:
    is_fully_defined: `bool` indicating that the shape is fully known.
  """
  return tf.TensorShape(x).is_fully_defined()


def merge_with(x, other):
  """Returns a shape combining the information in `x` and `other`.

  The dimensions in `x` and `other` are merged elementwise, according to the
  rules defined for `tf.Dimension.merge_with()`.

  For more details, see `help(tf.TensorShape.merge_with)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.
    other: object representing a shape; convertible to `tf.TensorShape`.

  Returns:
    merged_shape: shape having `type(x)` containing the combined information of
      `x` and `other`.

  Raises:
    ValueError: If `x` and `other` are not compatible.
  """
  return _cast_tensorshape(tf.TensorShape(x).merge_with(other), type(x))


def num_elements(x):
  """Returns the total number of elements, or `None` for incomplete shapes.

  For more details, see `help(tf.TensorShape.num_elements)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.

  Returns:
    num_elements: `int` representing the total number of elements implied by
      shape `x`.
  """
  return tf.TensorShape(x).num_elements()


def rank(x):
  """Returns the rank implied by this shape, or `None` if it is unspecified.

  For more details, see `help(tf.TensorShape.rank)`.

  Note: This is not the rank of the shape itself, viewed as a Tensor, which
  would always be 1; rather, it's the rank of every Tensor of the shape given by
  `x`.

  Args:
    x: object representing a shape; anything convertible to `tf.TensorShape`,
      or a `Tensor` (interpreted as an in-graph computed shape).

  Returns:
    rank: `int` representing the number of shape dimensions, or `None` if
      not statically known.
  """
  return tf.TensorShape(x).rank


def set_shape(tensor, shape):
  """Updates the shape of this tensor.

  This method can be called multiple times, and will merge the given
  `shape` with the current shape of this tensor. It can be used to
  provide additional information about the shape of this tensor that
  cannot be inferred from the graph alone. For example, this can be used
  to provide additional information about the shapes of images:

  ```python
  _, image_data = tf.TFRecordReader(...).read(...)
  image = tf.image.decode_png(image_data, channels=3)

  # The height and width dimensions of `image` are data dependent, and
  # cannot be computed without executing the op.
  print(image.shape)
  ==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])

  # We know that each image in this dataset is 28 x 28 pixels.
  image.set_shape([28, 28, 3])
  print(image.shape)
  ==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])
  ```

  NOTE: This shape is not enforced at runtime. Setting incorrect shapes can
  result in inconsistencies between the statically-known graph and the runtime
  value of tensors. For runtime validation of the shape, use `tf.ensure_shape`
  instead.

  Args:
    tensor: `Tensor` which will have its static shape set.
    shape: A `TensorShape` representing the shape of this tensor, a
    `TensorShapeProto`, a list, a tuple, or None.

  Raises:
    ValueError: If `shape` is not compatible with the current shape of
      this tensor.
  """
  if hasattr(tensor, 'set_shape'):
    tensor.set_shape(shape)


def with_rank(x, rank):  # pylint: disable=redefined-outer-name
  """Returns a shape based on `x` with the given `rank`.

  This method promotes a completely unknown shape to one with a known rank.

  For more details, see `help(tf.TensorShape.with_rank)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.
    rank: An `int` representing the rank of `x`, or else an assertion is raised.

  Returns:
    shape: a shape having `type(x)` but guaranteed to have given rank (or else
           an assertion was raised).

  Raises:
    ValueError: If `x` does not represent a shape with the given `rank`.
  """
  return _cast_tensorshape(tf.TensorShape(x).with_rank(rank), type(x))


def with_rank_at_least(x, rank):  # pylint: disable=redefined-outer-name
  """Returns a shape based on `x` with at least the given `rank`.

  For more details, see `help(tf.TensorShape.with_rank_at_least)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.
    rank: An `int` representing the minimum rank of `x` or else an assertion is
      raised.

  Returns:
    shape: a shape having `type(x)` but guaranteed to have at least the given
      rank (or else an assertion was raised).

  Raises:
    ValueError: If `x` does not represent a shape with at least the given
      `rank`.
  """
  return _cast_tensorshape(tf.TensorShape(x).with_rank_at_least(rank), type(x))
