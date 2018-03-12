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
"""Internal utiltiy functions for implementing TransitionKernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf


__all__ = [
    'is_list_like',
    'choose',
    'set_doc',
    'safe_sum',
]


def is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))


def choose(is_accepted,
           accepted,
           rejected,
           name=None):
  """Helper to `kernel` which expand_dims `is_accepted` to apply tf.where."""
  def _expand_is_accepted_like(x):
    with tf.name_scope('expand_is_accepted_like'):
      expand_shape = tf.concat([
          tf.shape(is_accepted),
          tf.ones([tf.rank(x) - tf.rank(is_accepted)],
                  dtype=tf.int32),
      ], axis=0)
      multiples = tf.concat([
          tf.ones([tf.rank(is_accepted)], dtype=tf.int32),
          tf.shape(x)[tf.rank(is_accepted):],
      ], axis=0)
      m = tf.tile(tf.reshape(is_accepted, expand_shape),
                  multiples)
      m.set_shape(m.shape.merge_with(x.shape))
      return m
  def _where(accepted, rejected):
    accepted = tf.convert_to_tensor(accepted, name='accepted')
    rejected = tf.convert_to_tensor(rejected, name='rejected')
    r = tf.where(_expand_is_accepted_like(accepted), accepted, rejected)
    r.set_shape(r.shape.merge_with(accepted.shape.merge_with(rejected.shape)))
    return r
  with tf.name_scope(name, 'choose', values=[
      is_accepted, accepted, rejected]):
    if not is_list_like(accepted):
      return _where(accepted, rejected)
    return [_where(a, r) for a, r in zip(accepted, rejected)]


def safe_sum(x, alt_value=-np.inf, name=None):
  """Elementwise adds list members, replacing non-finite results with alt_value.

  Args:
    x: Python `list` of `Tensors` to elementwise add.
    alt_value: Python scalar used to replace any elementwise sums which would
      otherwise be non-finite.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "safe_sum").

  Returns:
    safe_sum: `Tensor` representing the elementwise sum of list of `Tensor`s
      `x` or `alt_value` where sums are non-finite.

  Raises:
    TypeError: if `x` is not list-like.
    ValueError: if `x` is empty.
  """
  with tf.name_scope(name, 'safe_sum', [x, alt_value]):
    if not is_list_like(x):
      raise TypeError('Expected list input.')
    if not x:
      raise ValueError('Input should not be empty.')
    n = np.int32(len(x))
    in_shape = x[0].shape
    x = tf.stack(x, axis=-1)
    # The sum is NaN if any element is NaN or we see both +Inf and -Inf.  Thus
    # we will replace such rows with the `alt_value`. Typically the `alt_value`
    # is chosen so the `MetropolisHastings` `TransitionKernel` always rejects
    # the proposal.  rejection.
    # Regarding the following float-comparisons, recall comparing with NaN is
    # always False, i.e., we're implicitly capturing NaN and explicitly
    # capturing +/- Inf.
    is_sum_determinate = (
        tf.reduce_all(tf.is_finite(x) | (x >= 0.), axis=-1) &
        tf.reduce_all(tf.is_finite(x) | (x <= 0.), axis=-1))
    is_sum_determinate = tf.tile(
        is_sum_determinate[..., tf.newaxis],
        multiples=tf.concat([tf.ones(tf.rank(x) - 1, dtype=tf.int32), [n]],
                            axis=0))
    x = tf.where(
        is_sum_determinate,
        x,
        tf.fill(tf.shape(x), value=x.dtype.as_numpy_dtype(alt_value)))
    x = tf.reduce_sum(x, axis=-1)
    x.set_shape(x.shape.merge_with(in_shape))
    return x


def set_doc(value):
  """Decorator to programmatically set a function docstring."""
  def _doc(func):
    func.__doc__ = value
    return func
  return _doc
