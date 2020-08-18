# Copyright 2019 The TensorFlow Probability Authors.
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
"""Pad bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'Pad',
]


class Pad(bijector.Bijector):
  """Pads a value to the `event_shape` of a `Tensor`.

  The semantics of `tfp.bijectors.Pad` generally follow that of `tf.pad()`
  except that `tfp.bijectors.Pad`'s `paddings` argument applies to the rightmost
  dimensions. Additionally, the new argument `axis` enables overriding the
  dimensions to which `paddings` is applied. Like `paddings`, the `axis`
  argument is also relative to the rightmost dimension and must therefore be
  negative.

  The argument `paddings` is a vector of `int` pairs each representing the
  number of left and/or right `constant_values` to pad to the corresponding
  righmost dimensions. That is, unless `axis` is specified, specifiying `k`
  different `paddings` means the rightmost `k` dimensions will be "grown" by the
  sum of the respective `paddings` row. When `axis` is specified, it indicates
  the dimension to which the corresponding `paddings` element is applied. By
  default `axis` is `None` which means it is logically equivalent to
  `range(start=-len(paddings), limit=0)`, i.e., the rightmost dimensions.

  Example usage:

  ```python
  b = tfp.bijectors.Pad()  # Default arguments.

  b.forward([3., 4.])      # shape: [2]
  # ==> [[3., 4., 0.]]     # shape: [3]

  b.forward([[1., 2.],
             [3., 4.]])    # shape: [2, 2]
  # ==> [[1., 2., 0.],
  #      [3., 4., 0.]]     # shape: [2, 3]

  b.inverse([3., 4., 0.])  # shape: [3]
  # ==> [3., 4.]           # shape: [2]

  b.forward_log_det_jacobian(any_value)
  # ==> 0.

  b.inverse_log_det_jacobian(any_value)
  # ==> 0.
  ```

  ```python
  b = tfp.bijectors.Pad(axis=-2)  # With non-default `axis` arg.

  b.forward([[3., 4.]])    # shape: [1, 2]
  # ==> [[3., 4.],         # shape: [2, 2]
  #      [0., 0.]]

  b.inverse([[3., 4.],     # shape: [2, 2]
             [0., 0.]])
  # ==> [[3., 4.]]         # shape: [1, 2]

  b.forward_log_det_jacobian(any_value)
  # ==> 0.

  b.inverse_log_det_jacobian(any_value)
  # ==> 0.
  ```

  """

  def __init__(self,
               paddings=((0, 1),),
               mode='CONSTANT',
               constant_values=0,
               axis=None,
               validate_args=False,
               name=None):
    """Initializes the `Pad` bijector.

    Args:
      paddings: A vector-shaped `Tensor` of `int` pairs representing the number
        of elements to pad on the left and right, respectively.
        Default value: `((0, 1),)`.
      mode: One of `'CONSTANT'`, `'REFLECT'`, or `'SYMMETRIC'`
        (case-insensitive). For more details, see `tf.pad`.
      constant_values: In "CONSTANT" mode, the scalar pad value to use. Must be
        same type as `tensor`. For more details, see `tf.pad`.
      axis: The dimensions for which `paddings` are applied. Must be 1:1 with
        `paddings` or `None`.
        Default value: `None` (i.e., `tf.range(start=-len(paddings), limit=0)`).
      validate_args: Python `bool` indicating whether arguments should
        be checked for correctness.
        Default value: `False`.
      name: Python `str`, name given to ops managed by this object.
        Default value: `None` (i.e., `'pad'`).
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'pad') as name:
      paddings = tensor_util.convert_nonref_to_tensor(
          paddings, dtype_hint=tf.int32, name='paddings')
      if axis is None:
        axis = prefer_static.range(
            start=-prefer_static.size0(paddings), limit=0,
            dtype=tf.int32, name='axis')
      else:
        axis = tensor_util.convert_nonref_to_tensor(
            axis, dtype_hint=tf.int32, name='axis')
      axis_ = tf.get_static_value(axis)
      if axis_ is None:
        raise NotImplementedError(
            'Argument `axis` must be known statically. If you need this '
            'feature,  please contact `tfprobability@tensorflow.org`.')
      self._axis = axis
      self._paddings = paddings
      self._mode = mode
      self._constant_values = tensor_util.convert_nonref_to_tensor(
          constant_values, dtype_hint=tf.float32, name='constant_values')
      min_event_ndims_ = int(-np.min(np.pad(
          np.reshape(axis_, newshape=[-1]),
          mode='constant', pad_width=[[0, 1]])))
      super(Pad, self).__init__(
          forward_min_event_ndims=min_event_ndims_,
          inverse_min_event_ndims=min_event_ndims_,
          is_constant_jacobian=True,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @property
  def paddings(self):
    return self._paddings

  @property
  def mode(self):
    return self._mode

  @property
  def constant_values(self):
    return self._constant_values

  @property
  def axis(self):
    return self._axis

  def _forward(self, x):
    ndims = prefer_static.rank(x)
    indices = prefer_static.reshape(prefer_static.add(self.axis, ndims),
                                    shape=[-1, 1])
    return tf.pad(
        x,
        paddings=prefer_static.tensor_scatter_nd_update(
            prefer_static.zeros([ndims, 2], dtype=tf.int32),
            indices, self.paddings),
        mode=self.mode,
        constant_values=prefer_static.cast(self.constant_values, dtype=x.dtype))

  def _inverse(self, y):
    ndims = prefer_static.rank(y)
    indices = prefer_static.reshape(prefer_static.add(self.axis, ndims),
                                    shape=[-1, 1])
    num_left, num_right = prefer_static.unstack(self.paddings, num=2, axis=-1)
    x = tf.slice(
        y,
        begin=prefer_static.tensor_scatter_nd_update(
            prefer_static.zeros(ndims, dtype=tf.int32),
            indices, num_left),
        size=prefer_static.tensor_scatter_nd_sub(
            prefer_static.shape(y),
            indices, num_left + num_right))
    if not self.validate_args:
      return x
    assertions = [
        assert_util.assert_equal(
            self._forward(x), y,
            message=('Argument `y` to `inverse` was not padded with '
                     '`constant_values`.')),
    ]
    with tf.control_dependencies(assertions):
      return tf.identity(x)

  def _inverse_log_det_jacobian(self, y):
    # We specifically don't validate `y` here because sometimes folks pass dummy
    # values when `is_constant_jacobian`.
    return tf.zeros([], dtype=y.dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.zeros([], dtype=x.dtype)

  def _forward_event_shape(self, input_shape, is_inverse=False):
    axis = tf.get_static_value(self.axis)
    paddings = tf.get_static_value(self.paddings)
    if input_shape.ndims is None or axis is None or paddings is None:
      return None
    output_shape = [tf.compat.dimension_value(d) for d in list(input_shape)]
    for a, p in zip(list(axis.reshape(-1)), list(paddings.sum(axis=-1))):
      if output_shape[a] is not None:
        output_shape[a] += -p if is_inverse else p
    return output_shape

  def _forward_event_shape_tensor(self, input_shape, is_inverse=False):
    ndims = prefer_static.size(input_shape)
    indices = prefer_static.reshape(prefer_static.add(self.axis, ndims),
                                    shape=[-1, 1])
    extra_sizes = prefer_static.reduce_sum(self.paddings, axis=-1)
    update_fn = (prefer_static.tensor_scatter_nd_sub if is_inverse else
                 prefer_static.tensor_scatter_nd_add)
    return update_fn(prefer_static.identity(input_shape), indices, extra_sizes)

  def _inverse_event_shape(self, output_shape):
    input_shape = self._forward_event_shape(output_shape, is_inverse=True)
    if any(s < 0 for s in input_shape):
      raise ValueError('Invalid inverse shape; {}'.format(input_shape))
    return input_shape

  def _inverse_event_shape_tensor(self, output_shape):
    input_shape = self._forward_event_shape_tensor(
        output_shape, is_inverse=True)
    if not self.validate_args:
      return input_shape
    assertions = [
        assert_util.assert_greater(
            input_shape, -1,
            message='Invalid inverse shape; found negative size.')
    ]
    with tf.control_dependencies(assertions):
      return tf.identity(input_shape)

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    axis = None
    paddings = None

    if is_init != tensor_util.is_ref(self.axis):
      # First we check the shape of the axis argument.
      msg = 'Argument `axis` must be scalar or vector.'
      if tensorshape_util.rank(self.axis.shape) is not None:
        if tensorshape_util.rank(self.axis.shape) > 1:
          raise ValueError(msg)
      elif self.validate_args:
        if axis is None: axis = tf.convert_to_tensor(self.axis)
        assertions.append(assert_util.assert_rank_at_most(
            axis, 1, message=msg))
      # Next we check the values of the axis argument.
      axis_ = tf.get_static_value(self.axis)
      msg = 'Argument `axis` must be negative.'
      if axis_ is not None:
        if np.any(axis_ > -1):
          raise ValueError(msg)
      elif self.validate_args:
        if axis is None: axis = tf.convert_to_tensor(self.axis)
        assertions.append(assert_util.assert_less(axis, 0, message=msg))
      msg = 'Argument `axis` elements must be unique.'
      if axis_ is not None:
        if len(np.array(axis_).reshape(-1)) != len(np.unique(axis_)):
          raise ValueError(msg)
      elif self.validate_args:
        if axis is None: axis = tf.convert_to_tensor(self.axis)
        assertions.append(assert_util.assert_equal(
            prefer_static.size0(axis),
            prefer_static.size0(prefer_static.setdiff1d(axis)),
            message=msg))

    if is_init != tensor_util.is_ref(self.paddings):
      # First we check the shape of the paddings argument.
      msg = 'Argument `paddings` must be a vector of pairs.'
      if tensorshape_util.is_fully_defined(self.paddings.shape):
        shape = np.int32(self.paddings.shape)
        if len(shape) != 2 or shape[0] < 1 or shape[1] != 2:
          raise ValueError(msg)
      elif self.validate_args:
        if paddings is None: paddings = tf.convert_to_tensor(self.paddings)
        with tf.control_dependencies([
            assert_util.assert_equal(tf.rank(paddings), 2, message=msg)]):
          shape = tf.shape(paddings)
          assertions.extend([
              assert_util.assert_greater(shape[0], 0, message=msg),
              assert_util.assert_equal(shape[1], 2, message=msg),
          ])
      # Next we check the values of the paddings argument.
      paddings_ = tf.get_static_value(self.paddings)
      msg = 'Argument `paddings` must be non-negative.'
      if paddings_ is not None:
        if np.any(paddings_ < 0):
          raise ValueError(msg)
      elif self.validate_args:
        if paddings is None: paddings = tf.convert_to_tensor(self.paddings)
        assertions.append(assert_util.assert_greater(
            paddings, -1, message=msg))

    if is_init != (tensor_util.is_ref(self.axis) and
                   tensor_util.is_ref(self.paddings)):
      axis_ = tf.get_static_value(self.axis)
      if axis_ is None and axis is None:
        axis = tf.convert_to_tensor(self.axis)
      len_axis = prefer_static.size0(prefer_static.reshape(
          axis if axis_ is None else axis_, shape=-1))

      paddings_ = tf.get_static_value(self.paddings)
      if paddings_ is None and paddings is None:
        paddings = tf.convert_to_tensor(self.paddings)
      len_paddings = prefer_static.size0(
          paddings if paddings_ is None else paddings_)

      msg = ('Arguments `axis` and `paddings` must have the same number '
             'of elements.')
      if (prefer_static.is_numpy(len_axis) and
          prefer_static.is_numpy(len_paddings)):
        if len_axis != len_paddings:
          raise ValueError(msg + ' Saw: {}, {}.'.format(
              self.axis, self.paddings))
      elif self.validate_args:
        assertions.append(assert_util.assert_equal(
            len_axis, len_paddings, message=msg))

    return assertions
