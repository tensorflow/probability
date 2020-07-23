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

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_base
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'Blockwise',
]


def _get_static_splits(splits):
  # Convert to a static value so that one could run tf.split on TPU.
  static_splits = tf.get_static_value(splits)
  return splits if static_splits is None else static_splits


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

  Keyword arguments can be passed to the inner bijectors by utilizing the inner
  bijector names, e.g.:

  ```python
  blockwise = tfb.Blockwise([Bijector1(name='b1'), Bijector2(name='b2')])
  y = blockwise.forward(x, b1={'arg': 1}, b2={'arg': 2})

  # Equivalent to:
  x_0, x_1 = tf.split(x, [1, 1], axis=-1)
  y_0 = Bijector1().forward(x_0, arg=1)
  y_1 = Bijector2().forward(x_1, arg=2)
  y = tf.concat([y_0, y_1], axis=-1)
  ```

  """

  def __init__(self,
               bijectors,
               block_sizes=None,
               validate_args=False,
               maybe_changes_size=True,
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
      maybe_changes_size: Python `bool` indicating that this bijector might
        change the event size. If this is known to be false and set
        appropriately, then this will lead to improved static shape inference
        when the block sizes are not statically known.
      name: Python `str`, name given to ops managed by this object. Default:
        E.g., `Blockwise([Exp(), Softplus()]).name ==
        'blockwise_of_exp_and_softplus'`.

    Raises:
      NotImplementedError: If there is a bijector with `event_ndims` > 1.
      ValueError: If `bijectors` list is empty.
      ValueError: If size of `block_sizes` does not equal to the length of
        bijectors or is not a vector.
    """
    parameters = dict(locals())
    if not name:
      name = 'blockwise_of_' + '_and_'.join([b.name for b in bijectors])
      name = name.replace('/', '')
    with tf.name_scope(name) as name:
      super(Blockwise, self).__init__(
          forward_min_event_ndims=1,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

      if not bijectors:
        raise ValueError('`bijectors` must not be empty.')

      for bijector in bijectors:
        if (bijector.forward_min_event_ndims > 1 or
            bijector.inverse_min_event_ndims > 1):
          # TODO(siege): In the future, it can be reasonable to support N-D
          # bijectors by concatenating along some specific axis, broadcasting
          # low-D bijectors appropriately.
          raise NotImplementedError('Only scalar and vector event-shape '
                                    'bijectors are supported at this time.')

      self._bijectors = bijectors
      self._maybe_changes_size = maybe_changes_size

      if block_sizes is None:
        block_sizes = tf.ones(len(bijectors), dtype=tf.int32)
      self._block_sizes = tf.convert_to_tensor(
          block_sizes, name='block_sizes', dtype_hint=tf.int32)

      self._block_sizes = _validate_block_sizes(self._block_sizes, bijectors,
                                                validate_args)

  @property
  def bijectors(self):
    return self._bijectors

  @property
  def block_sizes(self):
    return self._block_sizes

  def _output_block_sizes(self):
    return [
        b.forward_event_shape_tensor(bs[tf.newaxis])[0]
        for b, bs in zip(self.bijectors,
                         tf.unstack(self.block_sizes, num=len(self.bijectors)))
    ]

  def _forward_event_shape(self, input_shape):
    input_shape = tensorshape_util.with_rank_at_least(input_shape, 1)
    static_block_sizes = tf.get_static_value(self.block_sizes)
    if static_block_sizes is None:
      return tensorshape_util.concatenate(input_shape[:-1], [None])

    output_size = sum(
        b.forward_event_shape([bs])[0]
        for b, bs in zip(self.bijectors, static_block_sizes))

    return tensorshape_util.concatenate(input_shape[:-1], [output_size])

  def _forward_event_shape_tensor(self, input_shape):
    output_size = ps.reduce_sum(self._output_block_sizes())
    return ps.concat([input_shape[:-1], output_size[tf.newaxis]], -1)

  def _inverse_event_shape(self, output_shape):
    output_shape = tensorshape_util.with_rank_at_least(output_shape, 1)
    static_block_sizes = tf.get_static_value(self.block_sizes)
    if static_block_sizes is None:
      return tensorshape_util.concatenate(output_shape[:-1], [None])

    input_size = sum(static_block_sizes)

    return tensorshape_util.concatenate(output_shape[:-1], [input_size])

  def _inverse_event_shape_tensor(self, output_shape):
    input_size = ps.reduce_sum(self.block_sizes)
    return ps.concat([output_shape[:-1], input_size[tf.newaxis]], -1)

  def _forward(self, x, **kwargs):
    split_x = tf.split(x, _get_static_splits(self.block_sizes), axis=-1,
                       num=len(self.bijectors))
    # TODO(b/162023850): Sanitize the kwargs better.
    split_y = [
        b.forward(x_, **kwargs.get(b.name, {}))
        for b, x_ in zip(self.bijectors, split_x)
    ]
    y = tf.concat(split_y, axis=-1)
    if not self._maybe_changes_size:
      tensorshape_util.set_shape(y, x.shape)
    return y

  def _inverse(self, y, **kwargs):
    split_y = tf.split(y, _get_static_splits(self._output_block_sizes()),
                       axis=-1, num=len(self.bijectors))
    split_x = [
        b.inverse(y_, **kwargs.get(b.name, {}))
        for b, y_ in zip(self.bijectors, split_y)
    ]
    x = tf.concat(split_x, axis=-1)
    if not self._maybe_changes_size:
      tensorshape_util.set_shape(x, y.shape)
    return x

  def _forward_log_det_jacobian(self, x, **kwargs):
    split_x = tf.split(x, _get_static_splits(self.block_sizes), axis=-1,
                       num=len(self.bijectors))
    fldjs = [
        b.forward_log_det_jacobian(x_, event_ndims=1, **kwargs.get(b.name, {}))
        for b, x_ in zip(self.bijectors, split_x)
    ]
    return sum(fldjs)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    split_y = tf.split(y, _get_static_splits(self._output_block_sizes()),
                       axis=-1, num=len(self.bijectors))
    ildjs = [
        b.inverse_log_det_jacobian(y_, event_ndims=1, **kwargs.get(b.name, {}))
        for b, y_ in zip(self.bijectors, split_y)
    ]
    return sum(ildjs)


def _validate_block_sizes(block_sizes, bijectors, validate_args):
  """Helper to validate block sizes."""
  block_sizes_shape = block_sizes.shape
  if tensorshape_util.is_fully_defined(block_sizes_shape):
    if (tensorshape_util.rank(block_sizes_shape) != 1 or
        (tensorshape_util.num_elements(block_sizes_shape) != len(bijectors))):
      raise ValueError(
          '`block_sizes` must be `None`, or a vector of the same length as '
          '`bijectors`. Got a `Tensor` with shape {} and `bijectors` of '
          'length {}'.format(block_sizes_shape, len(bijectors)))
    return block_sizes
  elif validate_args:
    message = ('`block_sizes` must be `None`, or a vector of the same length '
               'as `bijectors`.')
    with tf.control_dependencies([
        assert_util.assert_equal(
            tf.size(block_sizes), len(bijectors), message=message),
        assert_util.assert_equal(tf.rank(block_sizes), 1)
    ]):
      return tf.identity(block_sizes)
  else:
    return block_sizes
