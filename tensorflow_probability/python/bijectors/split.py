# Copyright 2020 The TensorFlow Probability Authors.
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
"""Split bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'Split',
]


class Split(bijector.Bijector):
  """Split a `Tensor` event along an axis into a list of `Tensor`s.

  Example Use:

  ```python
  split = tfb.Split(
      num_or_size_splits=[4, 1, 3],
      axis=-1
    )
  y = split.forward(tf.zeros([5, 6, 8]))
  ==> [<`Tensor`, shape=(5, 6, 4)>,
       <`Tensor`, shape=(5, 6, 1)>,
       <`Tensor`, shape=(5, 6, 3)>]

  # The inverse of `split` concatenates a list of `Tensor`s along `axis`.
  x_ = split.inverse(y_)
  x_.shape
  ==> TensorShape([5, 6, 8])
  ```
  """

  def __init__(
      self, num_or_size_splits, axis=-1, validate_args=False, name='split'):
    """Creates the bijector.

    Args:
      num_or_size_splits: Either a Python integer indicating the number of
        splits along `axis` or a 1-D integer `Tensor` or Python list containing
        the sizes of each output tensor along `axis`. If a list/`Tensor`, it may
        contain at most one value of `-1`, which indicates a split size that is
        unknown and determined from input.
      axis: A negative integer or scalar `int32` `Tensor`. The dimension along
        which to split. Must be negative to enable the bijector to support
        arbitrary batch dimensions. Defaults to -1 (note that this is different
        from the `tf.Split` default of `0`). Must be statically known.
      validate_args: Python `bool` indicating whether arguments should
        be checked for correctness.
      name: Python `str`, name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:

      if isinstance(num_or_size_splits, numbers.Integral):
        self._num_splits = num_or_size_splits
        self._split_sizes = None
      else:
        self._split_sizes = tensor_util.convert_nonref_to_tensor(
            num_or_size_splits, name='num_or_size_splits', dtype=tf.int32)

        if tensorshape_util.rank(self._split_sizes.shape) != 1:
          raise ValueError(
              '`num_or_size_splits` must be an integer or 1-D `Tensor`.')

        num_splits = tensorshape_util.as_list(self._split_sizes.shape)[0]
        if num_splits is None:
          raise ValueError('If `num_or_size_splits` is a vector of split sizes '
                           'it must have a statically-known number of '
                           'elements.')
        self._num_splits = num_splits

      static_axis = tf.get_static_value(axis)
      if static_axis is None:
        raise ValueError('`axis` must be statically known.')
      if static_axis >= 0:
        raise ValueError('`axis` must be negative. Got {}'.format(axis))

      self._axis = tf.convert_to_tensor(axis, tf.int32)

      super(Split, self).__init__(
          forward_min_event_ndims=-static_axis,
          # TODO(emilyaf): Replace with structured inverse_min_event_ndims when
          # that is supported (for now leave an unused placeholder).
          inverse_min_event_ndims=0,
          is_constant_jacobian=True,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @property
  def num_splits(self):
    return self._num_splits

  @property
  def split_sizes(self):
    return self._split_sizes

  @property
  def axis(self):
    return self._axis

  # TODO(emilyaf): Override the private method instead when the Bijector base
  # class supports nested args.
  def inverse(self, y, name='inverse'):
    """Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

    Args:
      y: List of `Tensor`s. The input to the 'inverse' evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
    """
    with self._name_and_control_scope(name):
      y = [tf.convert_to_tensor(y_, dtype_hint=self.dtype, name='y')
           for y_ in y]

      # TODO(emilyaf): Modify `_maybe_assert_dtype` to operate on structures.
      # self._maybe_assert_dtype(y)

      # Validate `y` statically, if possible, and get assertions.
      is_validated = self._validate_output_shapes([y_.shape for y_ in y])

      if is_validated or not self.validate_args:
        assertions = []
      else:
        assertions = self._validate_output_shape_tensors(
            [prefer_static.shape(y_) for y_ in y])

      with tf.control_dependencies(assertions):
        return tf.concat(y, axis=self.axis)

  # TODO(emilyaf): Override the private method instead when the Bijector base
  # class supports nested args.
  def forward(self, x, name='forward'):
    """Returns the forward `Bijector` evaluation, i.e., X = g(Y).

    Equivalent to `tf.split(x)`.

    Args:
      x: `Tensor`. The input to the 'forward' evaluation.
      name: The name to give this op.

    Returns:
      List of `Tensor`s.

    Raises:
      TypeError: if `self.dtype` is specified and `x.dtype` is not
        `self.dtype`.
      ValueError: if the sum of `split_sizes` does not equal the size of the
        `axis` dimension of `x`.
    """

    with self._name_and_control_scope(name):
      x = tf.convert_to_tensor(x, dtype_hint=self.dtype, name='x')
      self._maybe_assert_dtype(x)

      # Validate `x` statically if possible and get assertions.
      is_validated = self._validate_input_shape(x.shape)
      if is_validated or not self.validate_args:
        assertions = []
      else:
        assertions = self._validate_input_shape_tensor(prefer_static.shape(x))

      with tf.control_dependencies(assertions):
        if self.split_sizes is None:
          return tf.split(x, self.num_splits, axis=self.axis)
        return tf.split(x, self.split_sizes, axis=self.axis)

  # TODO(emilyaf): Override the private method instead when the Bijector base
  # class supports nested args.
  def forward_event_shape(self, input_shape):
    """Shape of a single sample from a single batch as a list of `TensorShape`s.

    Same meaning as `forward_event_shape_tensor`. May be only partially defined.

    Args:
      input_shape: `TensorShape` indicating event-portion shape passed into
        `forward` function.

    Returns:
      forward_event_shape: A list of (possibly unknown) `TensorShape`s
        indicating event-portion shape after applying `forward`. The length of
        the list is equal to the number of splits.
    """
    self._validate_input_shape(input_shape)
    if tensorshape_util.rank(input_shape) is None:
      output_shapes = [None] * self.num_splits
    else:
      input_shape = tf.TensorShape(input_shape).as_list()
      axis = tf.get_static_value(self.axis)

      if self.split_sizes is None:
        # Calculate `split_sizes` from `input_shape` and `num_splits`, if
        # possible.
        split_size = (None if input_shape[axis] is None
                      else input_shape[axis] // self.num_splits)
        split_sizes = [split_size] * self.num_splits

      else:
        static_split_sizes = tf.get_static_value(self.split_sizes)
        if static_split_sizes is None:
          static_split_sizes = [None] * self.num_splits
        split_sizes = tensorshape_util.constant_value_as_shape(
            static_split_sizes).as_list()

        # If there is a single unknown element of `split_sizes` and the input
        # dimension is known, set the unknown element equal to the difference
        # between the input dimension and the sum of the known elements of
        # `split_sizes`.
        if sum(s is None for s in split_sizes) == 1:
          if input_shape is not None and input_shape[axis] is not None:
            total_size = input_shape[axis]
            deduced_split_size = (total_size -
                                  sum(s for s in split_sizes if s is not None))
            split_sizes = [
                deduced_split_size if s is None else s for s in split_sizes]

      output_shapes = []
      for split_size in split_sizes:
        output_shape = input_shape[:]
        output_shape[axis] = split_size
        output_shapes.append(output_shape)

    return [tf.TensorShape(shape) for shape in output_shapes]

  # TODO(emilyaf): Override the private method instead when the Bijector base
  # class supports nested args.
  def forward_event_shape_tensor(self,
                                 input_shape,
                                 name='forward_event_shape_tensor'):
    """Shape of a sample from a single batch as a list of `int32` 1D `Tensor`s.

    Args:
      input_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `forward` function.
      name: name to give to the op

    Returns:
      forward_event_shape_tensor: A list of `Tensor`, `int32` vectors indicating
        event-portion shape after applying `forward`. The length of the list is
        equal to the number of splits.
    """
    with self._name_and_control_scope(name):
      input_shape = tf.convert_to_tensor(
          input_shape, dtype_hint=tf.int32, name='input_shape')

      # Validate `input_shape` statically if possible and get assertions.
      is_validated = self._validate_input_shape(
          tensorshape_util.constant_value_as_shape(input_shape))
      if is_validated or not self.validate_args:
        assertions = []
      else:
        assertions = self._validate_input_shape_tensor(input_shape)

      with tf.control_dependencies(assertions):
        if self.split_sizes is None:
          split_sizes = tf.convert_to_tensor(
              [input_shape[self.axis] // self.num_splits] * self.num_splits)
        else:
          # Deduce the value of the unknown element of `split_sizes`, if any.
          split_sizes = tf.convert_to_tensor(self.split_sizes)
          split_sizes = tf.where(
              split_sizes < 0,
              input_shape[self.axis] -
              tf.reduce_sum(split_sizes) - 1,  # Cancel the unknown size `-1`.
              split_sizes)

        # Each element of the `output_shape_tensor` list is equal to the
        # `input_shape`, with the corresponding element of `split_sizes`
        # substituted in the `axis` position.
        positive_axis = prefer_static.rank_from_shape(input_shape) + self.axis
        tiled_input_shape = tf.tile(
            input_shape[tf.newaxis, :], [self.num_splits, 1])
        fused_output_shapes = tf.concat([
            tiled_input_shape[:, :positive_axis],
            split_sizes[..., tf.newaxis],
            tiled_input_shape[:, positive_axis + 1:]], axis=1)

        output_shapes = tf.unstack(fused_output_shapes, num=self.num_splits)
        return [tf.identity(tf.convert_to_tensor(
            t, dtype_hint=tf.int32, name='forward_event_shape'))
                for t in output_shapes]

  # TODO(emilyaf): Override the private method instead when the Bijector base
  # class supports nested args.
  def inverse_event_shape_tensor(self,
                                 output_shapes,
                                 name='inverse_event_shape_tensor'):
    """Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      output_shapes: An iterable of `Tensor`, `int32` vectors indicating
        event-shapes passed into `inverse` function. The length of the iterable
        must be equal to the number of splits.
      name: Name to give to the op.

    Returns:
      inverse_event_shape_tensor: `Tensor`, `int32` vector indicating
        event-portion shape after applying `inverse`.
    """
    with self._name_and_control_scope(name):
      output_shapes = [
          tf.convert_to_tensor(t, dtype_hint=tf.int32, name='output_shape')
          for t in output_shapes]

      # Validate `output_shapes` statically if possible and get assertions.
      is_validated = self._validate_output_shapes(
          [tensorshape_util.constant_value_as_shape(s) for s in output_shapes])
      if is_validated or not self.validate_args:
        assertions = []
      else:
        assertions = self._validate_output_shape_tensors(output_shapes)

      with tf.control_dependencies(assertions):
        total_size = tf.reduce_sum([t[self.axis] for t in output_shapes])
        inverse_event_shape = tf.tensor_scatter_nd_update(
            output_shapes[0],
            [[prefer_static.rank_from_shape(output_shapes[0]) + self.axis]],
            [total_size])
        return tf.identity(tf.convert_to_tensor(
            inverse_event_shape, dtype_hint=tf.int32,
            name='inverse_event_shape'))

  # TODO(emilyaf): Override the private method instead when the Bijector base
  # class supports nested args.
  def inverse_event_shape(self, output_shapes):
    """Shape of a sample from a single batch as a [nested] `TensorShape`.

    Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

    Args:
      output_shapes: Iterable of `TensorShape`s indicating the event shapes
        passed into `inverse` function. The length of the iterable must be equal
        to the number of splits.

    Returns:
      inverse_event_shape_tensor: `TensorShape` indicating event-portion shape
        after applying `inverse`. Possibly unknown.
    """
    self._validate_output_shapes(output_shapes)
    shapes = [tf.TensorShape(s).as_list() for s in output_shapes]
    axis = tf.get_static_value(self.axis)

    if self.split_sizes is None:
      split_size = None
      for shape in output_shapes:
        if shape[axis] is not None:
          split_size = shape[axis]
      split_sizes = [split_size] * self.num_splits
    else:
      static_split_sizes = tf.get_static_value(self.split_sizes)
      if static_split_sizes is None:
        static_split_sizes = [None] * self.num_splits
      split_sizes = tensorshape_util.constant_value_as_shape(
          static_split_sizes).as_list()

    # Deduce as much static information about `inverse_event_shape` as possible.
    # If all elements of `split_sizes` are known, the concatenated dimension
    # of `inverse_event_shape` is the sum of `split_sizes`.
    if not any(s is None for s in split_sizes):
      total_size = sum(split_sizes)
    else:
      # If at least one of `split_sizes` and `output_shape[axis]` is known
      # for each split, we can determine `total_size`.
      total_size = 0
      for split, output_shape in zip(split_sizes, shapes):
        if split is None and output_shape[axis] is None:
          total_size = None
          break
        total_size += split or output_shape[axis]

    shape = shapes[0]
    shape[axis] = total_size
    return tf.TensorShape(shape)

  # TODO(emilyaf): Override the private method instead when the Bijector base
  # class supports nested args.
  def forward_log_det_jacobian(self,
                               x,
                               event_ndims,
                               name='forward_log_det_jacobian'):
    """Returns the forward_log_det_jacobian.

    Args:
      x: `Tensor`. The input to the 'forward' Jacobian determinant evaluation.
      event_ndims: Number of dimensions in the probabilistic events being
        transformed. Must be greater than or equal to
        `self.forward_min_event_ndims`. The result is summed over the final
        dimensions to produce a scalar Jacobian determinant for each event, i.e.
        it has shape `rank(x) - event_ndims` dimensions.
      name: The name to give this op.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective this is not implemented.
    """
    with self._name_and_control_scope(name), tf.control_dependencies(
        self._check_valid_event_ndims(
            min_event_ndims=self.forward_min_event_ndims,
            event_ndims=event_ndims)):
      x = tf.convert_to_tensor(x, name='x')
      self._maybe_assert_dtype(x)
      return 0.

  # TODO(emilyaf): Override the private method instead when the Bijector base
  # class supports nested args.
  def inverse_log_det_jacobian(self,
                               y,
                               event_ndims,
                               name='inverse_log_det_jacobian'):
    with self._name_and_control_scope(name):
      y = [tf.convert_to_tensor(y_, name='y') for y_ in y]
      # TODO(emilyaf): Modify `_maybe_assert_dtype` to operate on structures.
      # self._maybe_assert_dtype(y)
      return 0.

  def _forward_dtype(self, dtype):
    return [dtype] * self.num_splits

  def _inverse_dtype(self, dtype):
    if any(d != dtype[0] for d in dtype):
      raise ValueError('All dtypes must be equivalent.')
    return dtype[0]

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    if (self.split_sizes is not None and
        is_init != tensor_util.is_ref(self.split_sizes)):
      assertions.extend(self._maybe_validate_split_sizes())
    return assertions

  def _maybe_validate_split_sizes(self):
    """Validations for `split_sizes` property."""
    assertions = []
    split_sizes = tf.convert_to_tensor(self.split_sizes)
    split_sizes_ = tf.get_static_value(split_sizes)

    # Ensure `split_sizes` has no more than one unknown split size (=-1).
    message = '`{}` elements must have at most one `-1`.'
    if split_sizes_ is not None:
      if sum(split_sizes_ == -1) > 1:
        raise ValueError(message.format(split_sizes))
    elif self.validate_args:
      assertions.append(
          assert_util.assert_less(
              tf.reduce_sum(tf.cast(tf.equal(split_sizes, -1), tf.int32)),
              2,
              message=message.format(split_sizes)))

    message = '`{}` elements must be either non-negative integers or `-1`.'
    if split_sizes_ is not None:
      if np.any(split_sizes_ < -1):
        raise ValueError(message.format(split_sizes))
    elif self.validate_args:
      assertions.append(assert_util.assert_greater(
          split_sizes, -2, message=message.format(split_sizes)))

    return assertions

  def _validate_input_shape(self, input_shape):
    """Static validations for `input_shape`."""
    input_shape_ = tf.get_static_value(input_shape)
    if input_shape_ is None:
      return False

    input_split_dim_ = input_shape_[tf.get_static_value(self.axis)]
    # If `split_sizes` is not provided, then we can provide static validation if
    # `input_split_dim` is statically known.
    if self.split_sizes is None:
      if input_split_dim_ is not None:
        if input_split_dim_ % self.num_splits > 0:
          raise ValueError('The number of splits ({}) must divide the `axis` '
                           'dimension of `input_shape` ({})'.format(
                               self.num_splits, input_split_dim_))
        return True
      return False
    else:
      split_sizes_ = tf.get_static_value(self.split_sizes)
      if input_split_dim_ is not None and split_sizes_ is not None:
        if np.any(split_sizes_ == -1):
          if np.sum(split_sizes_[split_sizes_ != -1]) > input_split_dim_:
            raise ValueError('The size of the input along `axis` must be '
                             'greater than the sum of the specified dimensions '
                             'of `split_sizes` if `split_sizes` is not fully '
                             'specified.')
        elif input_split_dim_ != sum(split_sizes_):
          raise ValueError('The size of the input along `axis` must equal the '
                           'sum of `split_sizes`.')
        return True
      return False

  def _validate_input_shape_tensor(self, input_shape):
    input_dim = tf.gather(
        input_shape, [prefer_static.rank_from_shape(input_shape) + self.axis])
    if self.split_sizes is None:
      return [assert_util.assert_equal(
          0,
          tf.math.floormod(input_dim, self.num_splits))]
    else:
      split_sizes = tf.convert_to_tensor(self.split_sizes)
      splits_are_known = tf.reduce_all(tf.greater_equal(split_sizes, 0))
      return [assert_util.assert_equal(
          True,
          ((~splits_are_known &
            (tf.reduce_sum(split_sizes) + 1 <= input_dim)) |
           (splits_are_known & tf.equal(input_dim, tf.reduce_sum(split_sizes)))
          ),
          message='The size of the input along `axis` does not match '
          '`split_sizes`.')]

  def _validate_output_shapes(self, output_shapes):
    """Static validations for `output_shapes`."""
    if self.num_splits != len(output_shapes):
      raise ValueError('Length of the `output_shapes` list (={}) does not '
                       'match the number of splits in `split_sizes` (={})'
                       ''.format(len(output_shapes), self.num_splits))

    split_sizes_ = tf.get_static_value(self.split_sizes)
    is_validated = False
    if split_sizes_ is not None:
      is_validated = True
      for split_size_, shape in zip(split_sizes_, output_shapes):
        output_size_ = tf.get_static_value(
            shape[tf.get_static_value(self.axis)])
        if output_size_ is None:
          is_validated = False
        elif split_size_ != -1 and output_size_ != split_size_:
          raise ValueError('Inverse shape dimension (={}) does not match '
                           'expected `split_size` dimension (={})'.format(
                               output_size_, split_size_))
    return is_validated

  def _validate_output_shape_tensors(self, output_shapes):
    """Dynamic validations for `output_shapes`."""
    assertions = []
    if self.split_sizes is not None:
      split_sizes = tf.convert_to_tensor(self.split_sizes)
      for i, shape in enumerate(output_shapes):
        output_size = tf.gather(
            shape,
            [prefer_static.rank_from_shape(output_shapes[0]) + self.axis])
        split_size = split_sizes[i]
        assertions.append(
            assert_util.assert_equal(
                True,
                tf.equal(split_size, -1) | tf.equal(output_size, split_size),
                message=('Inverse shape dimension (={}) does not match '
                         'expected `split_size` dimension (={})'.format(
                             output_size, split_size))))
    return assertions
