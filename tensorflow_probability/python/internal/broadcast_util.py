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

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps


__all__ = [
    'left_justified_expand_dims_like',
    'left_justified_expand_dims_to',
    'left_justified_broadcast_like',
    'left_justified_broadcast_to',
    'right_justified_unsorted_segment_sum',
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


def right_justified_unsorted_segment_sum(
    data, segment_ids, num_segments, name=None):
  """Same as tf.segment_sum, except the segment ids line up on the right."""
  with tf.name_scope(name or 'right_justified_unsorted_segment_sum'):
    data = tf.convert_to_tensor(data)
    segment_ids = tf.convert_to_tensor(segment_ids)
    n_seg = ps.rank(segment_ids)
    n_data = ps.rank(data)
    # Move the rightmost n_seg dimensions to the left, where
    # segment_sum will find them
    perm = ps.concat(
        [ps.range(n_data - n_seg, n_data), ps.range(0, n_data - n_seg)], axis=0)
    data_justified = tf.transpose(data, perm=perm)
    results_justified = tf.math.unsorted_segment_sum(
        data_justified, segment_ids, num_segments)
    # segment_sum puts the segment dimension of the result on the
    # left; move it to the right.
    inverse_perm = ps.concat([ps.range(1, n_data - n_seg + 1), [0]], axis=0)
    return tf.transpose(results_justified, perm=inverse_perm)
