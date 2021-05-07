# Copyright 2021 The TensorFlow Probability Authors.
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
"""Integration Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'trapz',
]


def trapz(
    y,
    x=None,
    dx=None,
    axis=-1,
    name=None,
):
  """Integrate y(x) on the specified axis using the trapezoidal rule.

  Computes ∫ y(x) dx ≈ Σ [0.5 (y_k + y_{k+1}) * (x_{k+1} - x_k)]

  Args:
    y: Float `Tensor` of values to integrate.
    x: Optional, Float `Tensor` of points corresponding to the y values. The
      shape of x should match that of y. If x is None, the sample points are
      assumed to be evenly spaced dx apart.
    dx: Scalar float `Tensor`. The spacing between sample points when x is None.
      If neither x nor dx is provided then the default is dx = 1.
    axis: Scalar integer `Tensor`. The axis along which to integrate.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None`, uses name='trapz'.

  Returns:
    Float `Tensor` integral approximated by trapezoidal rule.
      Has the shape of y but with the dimension associated with axis removed.
  """
  with tf.name_scope(name or 'trapz'):
    if not (x is None or dx is None):
      raise ValueError('Not permitted to specify both x and dx input args.')
    dtype = dtype_util.common_dtype([y, x, dx], dtype_hint=tf.float32)
    axis = ps.convert_to_shape_tensor(axis, dtype=tf.int32, name='axis')
    axis_rank = tensorshape_util.rank(axis.shape)
    if axis_rank is None:
      raise ValueError('Require axis to have a static shape.')
    if axis_rank:
      raise ValueError(
          'Only permitted to specify one axis, got axis={}'.format(axis))
    y = tf.convert_to_tensor(y, dtype=dtype, name='y')
    y_shape = ps.convert_to_shape_tensor(ps.shape(y), dtype=tf.int32)
    length = y_shape[axis]
    if x is None:
      if dx is None:
        dx = 1.
      dx = tf.convert_to_tensor(dx, dtype=dtype, name='dx')
      if ps.shape(dx):
        raise ValueError('Expected dx to be a scalar, got dx={}'.format(dx))
      elem_sum = tf.reduce_sum(y, axis=axis)
      elem_sum -= 0.5 * tf.reduce_sum(
          tf.gather(y, [0, length - 1], axis=axis),
          axis=axis)  # half weight endpoints
      return elem_sum * dx
    else:
      x = tf.convert_to_tensor(x, dtype=dtype, name='x')
      tensorshape_util.assert_is_compatible_with(x.shape, y.shape)
      dx = (
          tf.gather(x, ps.range(1, length), axis=axis) -
          tf.gather(x, ps.range(0, length - 1), axis=axis))
      return 0.5 * tf.reduce_sum(
          (tf.gather(y, ps.range(1, length), axis=axis) +
           tf.gather(y, ps.range(0, length - 1), axis=axis)) * dx,
          axis=axis)
