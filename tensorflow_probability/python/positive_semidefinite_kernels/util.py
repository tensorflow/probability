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
"""Positive-Semidefinite Kernel library utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops.distributions import util as distribution_util

__all__ = [
    'pad_shape_right_with_ones',
    'sum_rightmost_ndims_preserving_shape',
]


def pad_shape_right_with_ones(x, ndims):
  """Maybe add `ndims` ones to `x.shape` on the right.

  If `ndims` is zero, this is a no-op; otherwise, we will create and return a
  new `Tensor` whose shape is that of `x` with `ndims` ones concatenated on the
  right side. If the shape of `x` is known statically, the shape of the return
  value will be as well.

  Args:
    x: The `Tensor` we'll return a reshaping of.
    ndims: Python `integer` number of ones to pad onto `x.shape`.
  Returns:
    If `ndims` is zero, `x`; otherwise, a `Tensor` whose shape is that of `x`
    with `ndims` ones concatenated on the right side. If possible, returns a
    `Tensor` whose shape is known statically.
  Raises:
    ValueError: if `ndims` is not a Python `integer` greater than or equal to
    zero.
  """
  if not (isinstance(ndims, int) and ndims >= 0):
    raise ValueError(
        '`ndims` must be a Python `integer` greater than zero. Got: {}'
        .format(ndims))
  if ndims == 0:
    return x
  x = tf.convert_to_tensor(x)
  original_shape = x.shape
  new_shape = distribution_util.pad(
      tf.shape(x), axis=0, back=True, value=1, count=ndims)
  x = tf.reshape(x, new_shape)
  x.set_shape(original_shape.concatenate([1]*ndims))
  return x


def sum_rightmost_ndims_preserving_shape(x, ndims):
  """Return `Tensor` with right-most ndims summed.

  Args:
    x: the `Tensor` whose right-most `ndims` dimensions to sum
    ndims: number of right-most dimensions to sum.

  Returns:
    A `Tensor` resulting from calling `reduce_sum` on the `ndims` right-most
    dimensions. If the shape of `x` is statically known, the result will also
    have statically known shape. Otherwise, the resulting shape will only be
    known at runtime.
  """
  x = tf.convert_to_tensor(x)
  if x.shape.ndims is not None:
    axes = tf.range(x.shape.ndims - ndims, x.shape.ndims)
  else:
    axes = tf.range(tf.rank(x) - ndims, tf.rank(x))
  return tf.reduce_sum(x, axis=axes)
