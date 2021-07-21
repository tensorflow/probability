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

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import composition
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import split
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'Blockwise',
]


def _get_static_splits(splits):
  # Convert to a static value so that one could run tf.split on TPU.
  static_splits = tf.get_static_value(splits)
  return splits if static_splits is None else static_splits


class _Blockwise(composition.Composition):
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
      for b in bijectors:
        if (nest.is_nested(b.forward_min_event_ndims)
            or nest.is_nested(b.inverse_min_event_ndims)):
          raise ValueError('Bijectors must all be single-part.')
        elif isinstance(b.forward_min_event_ndims, int):
          if b.forward_min_event_ndims != b.inverse_min_event_ndims:
            raise ValueError('Rank-changing bijectors are not supported.')
          elif b.forward_min_event_ndims > 1:
            raise ValueError('Only scalar and vector event-shape '
                             'bijectors are supported at this time.')

      b_joint = joint_map.JointMap(list(bijectors), name='jointmap')

      block_sizes = (
          np.ones(len(bijectors), dtype=np.int32)
          if block_sizes is None else
          _validate_block_sizes(block_sizes, bijectors, validate_args))
      b_split = split.Split(
          block_sizes, name='split', validate_args=validate_args)

      if maybe_changes_size:
        i_block_sizes = _validate_block_sizes(
            ps.concat(b_joint.forward_event_shape_tensor(
                ps.split(block_sizes, len(bijectors))), axis=0),
            bijectors, validate_args)
        maybe_changes_size = not tf.get_static_value(
            ps.reduce_all(block_sizes == i_block_sizes))
      b_concat = invert.Invert(
          (split.Split(i_block_sizes, name='isplit')
           if maybe_changes_size else b_split),
          name='concat')

      self._maybe_changes_size = maybe_changes_size
      self._chain = chain.Chain(
          [b_concat, b_joint, b_split], validate_args=validate_args)
      super(_Blockwise, self).__init__(
          bijectors=self._chain.bijectors,
          validate_args=validate_args,
          validate_event_size=True,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  @property
  def _b_joint(self):
    return self._bijectors[1]

  @property
  def _b_split(self):
    return self._bijectors[-1]

  @property
  def _b_concat(self):
    return self._bijectors[0].bijector

  @property
  def bijectors(self):
    return self._b_joint.bijectors

  @property
  def block_sizes(self):
    return self._b_split.split_sizes

  @property
  def inverse_block_sizes(self):
    return self._b_concat.split_sizes

  def _forward(self, x, **kwargs):
    y = super(_Blockwise, self)._forward(x, **kwargs)
    if not self._maybe_changes_size:
      tensorshape_util.set_shape(y, x.shape)
    return y

  def _inverse(self, y, **kwargs):
    x = super(_Blockwise, self)._inverse(y, **kwargs)
    if not self._maybe_changes_size:
      tensorshape_util.set_shape(x, y.shape)
    return x

  def _forward_event_shape(self, input_shape):
    if not self._maybe_changes_size:
      return input_shape
    input_shape = tensorshape_util.with_rank_at_least(input_shape, 1)
    static_block_sizes = tf.get_static_value(self.inverse_block_sizes)
    if static_block_sizes is None:
      return tensorshape_util.concatenate(input_shape[:-1], [None])
    output_size = sum(static_block_sizes)
    return tensorshape_util.concatenate(input_shape[:-1], [output_size])

  def _inverse_event_shape(self, output_shape):
    if not self._maybe_changes_size:
      return output_shape
    output_shape = tensorshape_util.with_rank_at_least(output_shape, 1)
    static_block_sizes = tf.get_static_value(self.block_sizes)
    if static_block_sizes is None:
      return tensorshape_util.concatenate(output_shape[:-1], [None])
    input_size = sum(static_block_sizes)
    return tensorshape_util.concatenate(output_shape[:-1], [input_size])

  def _forward_event_shape_tensor(self, x, **kwargs):
    if not self._maybe_changes_size:
      return x
    return super(_Blockwise, self)._forward_event_shape_tensor(x, **kwargs)

  def _inverse_event_shape_tensor(self, y, **kwargs):
    if not self._maybe_changes_size:
      return y
    return super(_Blockwise, self)._inverse_event_shape_tensor(y, **kwargs)

  def _walk_forward(self, step_fn, x, **kwargs):
    return self._chain._walk_forward(  # pylint: disable=protected-access
        step_fn, x, **{self._b_joint.name: kwargs})

  def _walk_inverse(self, step_fn, x, **kwargs):
    return self._chain._walk_inverse(  # pylint: disable=protected-access
        step_fn, x, **{self._b_joint.name: kwargs})


def _validate_block_sizes(block_sizes, bijectors, validate_args):
  """Helper to validate block sizes."""
  block_sizes = ps.convert_to_shape_tensor(
      block_sizes, name='block_sizes', dtype_hint=tf.int32)
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
      block_sizes = tf.identity(block_sizes)

  # Set the shape if missing to pass statically known structure to split.
  tensorshape_util.set_shape(block_sizes, [len(bijectors)])
  return block_sizes


class Blockwise(_Blockwise, bijector_lib.AutoCompositeTensorBijector):

  def __new__(cls, *args, **kwargs):
    """Returns a `_Blockwise` if any of `bijectors` is not `CompositeTensor."""
    if cls is Blockwise:
      if args:
        bijectors = args[0]
      elif 'bijectors' in kwargs:
        bijectors = kwargs['bijectors']
      else:
        raise TypeError(
            '`Blockwise.__new__()` is missing argument `bijectors`.')

      if not all(isinstance(b, tf.__internal__.CompositeTensor)
                 for b in bijectors):
        return _Blockwise(*args, **kwargs)
    return super(Blockwise, cls).__new__(cls)


Blockwise.__doc__ = _Blockwise.__doc__ + '\n' + (
    'If every element of the `bijectors` list is a `CompositeTensor`, the '
    'resulting `Blockwise` bijector is a `CompositeTensor` as well. If any '
    'element of `bijectors` is not a `CompositeTensor`, then a '
    'non-`CompositeTensor` `_Blockwise` instance is created instead. Bijector '
    'subclasses that inherit from `Blockwise` will also inherit from '
    '`CompositeTensor`.')
