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
"""Blockwise bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.bijectors import bijector as bijector_base

__all__ = [
    'Blockwise',
]


class Blockwise(bijector_base.Bijector):
  """Bijector which applies a list of bijectors to blocks of a `Tensor`.

  More specifically, given [F_0, F_1, ... F_n] which are scalar or vector
  bijectors this bijector creates a transformation which operates on the vector
  [x_0, ... x_n] with the transformation [F_0(x_0), F_1(x_1) ..., F_n(x_n)]
  where x_0, ..., x_n are blocks (partitions) of the vector.

  Example Use:

  ```python
  blockwise = tfb.Blockwise(
      bijectors=[tfb.Exp(), tfb.Sigmoid()], block_sizes=[2, 1]
    )
  y = blockwise.forward(x)

  # Equivalent to:
  x_0, x_1 = tf.split(x, [2, 1], axis=-1)
  y_0 = tfb.Exp().forward(x_0)
  y_1 = tfb.Sigmoid().forward(x_1)
  y = tf.concat([y_0, y_1], axis=-1)
  ```
  """

  def __init__(self,
               bijectors,
               block_sizes=None,
               validate_args=False,
               name=None):
    """Creates the bijector.

    Args:
      bijectors: A non-empty list of bijectors.
      block_sizes: A 1-D integer `Tensor` with each element signifying the
        length of the block of the input vector to pass to the corresponding
        bijector. The length of `block_sizes` must be be equal to the length of
        `bijectors`. If left as None, a vector of 1's is used.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object. Default:
        E.g., `Blockwise([Exp(), Softplus()]).name ==
        'blockwise_of_exp_and_softplus'`.

    Raises:
      NotImplementedError: If a bijector with `event_ndims` > 1 or one that
        reshapes events is passed.
      ValueError: If `bijectors` list is empty.
      ValueError: If size of `block_sizes` does not equal to the length of
        bijectors or is not a vector.
    """
    super(Blockwise, self).__init__(
        forward_min_event_ndims=1,
        validate_args=validate_args,
        name=name or
        'blockwise_of_' + '_and_'.join([b.name for b in bijectors]))

    if not bijectors:
      raise ValueError('`bijectors` must not be empty.')

    for bijector in bijectors:
      if (bijector.forward_min_event_ndims > 1 or
          bijector.inverse_min_event_ndims != bijector.forward_min_event_ndims):
        # TODO(b/) In the future, it can be reasonable to support N-D bijectors
        # by concatenating along some specific axis, broadcasting low-D
        # bijectors appropriately.
        raise NotImplementedError('Only scalar and vector event-shape '
                                  'bijectors that do not alter the '
                                  'shape are supported at this time.')

    self._bijectors = bijectors

    with self._name_scope('init', values=[block_sizes]):
      if block_sizes is None:
        block_sizes = tf.ones(len(bijectors), dtype=tf.int32)
      self._block_sizes = tf.convert_to_tensor(
          value=block_sizes, name='block_sizes', dtype_hint=tf.int32)

      self._block_sizes = _validate_block_sizes(self._block_sizes, bijectors,
                                                validate_args)

  @property
  def bijectors(self):
    return self._bijectors

  @property
  def block_sizes(self):
    return self._block_sizes

  def _forward(self, x):
    split_x = tf.split(x, self.block_sizes, axis=-1, num=len(self.bijectors))
    split_y = [b.forward(x_) for b, x_ in zip(self.bijectors, split_x)]
    y = tf.concat(split_y, axis=-1)
    y.set_shape(y.shape.merge_with(x.shape))
    return y

  def _inverse(self, y):
    split_y = tf.split(y, self.block_sizes, axis=-1, num=len(self.bijectors))
    split_x = [b.inverse(y_) for b, y_ in zip(self.bijectors, split_y)]
    x = tf.concat(split_x, axis=-1)
    x.set_shape(x.shape.merge_with(y.shape))
    return x

  def _forward_log_det_jacobian(self, x):
    split_x = tf.split(x, self.block_sizes, axis=-1, num=len(self.bijectors))
    fldjs = [
        b.forward_log_det_jacobian(x_, event_ndims=1)
        for b, x_ in zip(self.bijectors, split_x)
    ]
    return sum(fldjs)

  def _inverse_log_det_jacobian(self, y):
    split_y = tf.split(y, self.block_sizes, axis=-1, num=len(self.bijectors))
    ildjs = [
        b.inverse_log_det_jacobian(y_, event_ndims=1)
        for b, y_ in zip(self.bijectors, split_y)
    ]
    return sum(ildjs)


def _validate_block_sizes(block_sizes, bijectors, validate_args):
  """Helper to validate block sizes."""
  block_sizes_shape = block_sizes.shape
  if block_sizes_shape.is_fully_defined():
    if block_sizes_shape.ndims != 1 or (block_sizes_shape.num_elements() !=
                                        len(bijectors)):
      raise ValueError(
          '`block_sizes` must be `None`, or a vector of the same length as '
          '`bijectors`. Got a `Tensor` with shape {} and `bijectors` of '
          'length {}'.format(block_sizes_shape, len(bijectors)))
    return block_sizes
  elif validate_args:
    message = ('`block_sizes` must be `None`, or a vector of the same length '
               'as `bijectors`.')
    with tf.control_dependencies([
        tf.compat.v1.assert_equal(
            tf.size(input=block_sizes), len(bijectors), message=message),
        tf.compat.v1.assert_equal(tf.rank(block_sizes), 1)
    ]):
      return tf.identity(block_sizes)
  else:
    return block_sizes
