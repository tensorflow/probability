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

import tensorflow.compat.v2 as tf

__all__ = [
    'as_list',
    'assert_has_rank',
    'assert_is_compatible_with',
    'concatenate',
    'dims',
    'is_compatible_with',
    'is_fully_defined',
    'merge_with',
    'num_elements',
    'rank',
    'with_rank_at_least',
]


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
  return type(x)(tf.TensorShape(x).concatenate(other))


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
  return type(x)(tf.TensorShape(x).merge_with(other))


def rank(x):
  """Returns the rank of this shape, or `None` if it is unspecified.

  For more details, see `help(tf.TensorShape.rank)`.

  Args:
    x: object representing a shape; convertible to `tf.TensorShape`.

  Returns:
    rank: `int` representing the number of shape dimensions.
  """
  return tf.TensorShape(x).rank


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
  return type(x)(tf.TensorShape(x).with_rank_at_least(rank))
