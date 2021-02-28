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
"""Tools for manipulating shapes and broadcasting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps


__all__ = [
    'left_justified_expand_dims_like',
    'left_justified_expand_dims_to',
    'left_justified_broadcast_like',
    'left_justified_broadcast_to',
    'where_left_justified_mask',
]


def left_justified_expand_dims_like(x, reference, name=None):
  """Right pads `x` with `rank(reference) - rank(x)` ones."""
  with tf.name_scope(name or 'left_justified_expand_dims_like'):
    return left_justified_expand_dims_to(x, ps.rank(reference))


def left_justified_expand_dims_to(x, rank, name=None):
  """Right pads `x` with `rank - rank(x)` ones."""
  with tf.name_scope(name or 'left_justified_expand_dims_to'):
    expand_ndims = ps.maximum(rank - ps.rank(x), 0)
    expand_shape = ps.concat(
        [ps.shape(x),
         ps.ones(shape=[expand_ndims], dtype=tf.int32)],
        axis=0)
    return tf.reshape(x, expand_shape)


def left_justified_broadcast_like(x, reference, name=None):
  """Broadcasts `x` to shape of reference, in a left-justified manner."""
  with tf.name_scope(name or 'left_justified_broadcast_like'):
    return left_justified_broadcast_to(x, ps.shape(reference))


def left_justified_broadcast_to(x, shape, name=None):
  """Broadcasts `x` to shape, in a left-justified manner."""
  with tf.name_scope(name or 'left_justified_broadcast_to'):
    return tf.broadcast_to(
        left_justified_expand_dims_to(x, ps.size(shape)), shape)


def where_left_justified_mask(mask, vals1, vals2, name=None):
  """Like `tf.where`, but broadcasts the `mask` left-justified."""
  with tf.name_scope(name or 'where_left_justified_mask'):
    target_rank = ps.maximum(ps.rank(vals1), ps.rank(vals2))
    bcast_mask = left_justified_expand_dims_to(mask, target_rank)
    return tf.where(bcast_mask, vals1, vals2)
