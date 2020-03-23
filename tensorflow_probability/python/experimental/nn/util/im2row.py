# Lint as: python2, python3
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
"""Functions for framing `conv` as `matmul`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static


__all__ = [
    'im2row',
]


def im2row(x,
           block_shape,
           slice_step=(1, 1),
           data_format='NHWC',
           padding='VALID',
           name=None):
  """Rearrange image blocks into rows.

  This function can be used to implement 2D convolution as a `matml`, e.g.,

  `tf.nn.conv2d(x, k) = tf.matmul(im2row(x), tf.reshape(k, [-1, out_size]))`.

  Args:
    x: Rank 3 (or more) Tensor representing 2D images.
    block_shape: Length-2 vector representing the block or "filter" shape.
    slice_step: Length-2 vector specifying the convolution stride length.
      Default value: `(1, 1)`.
    data_format: One of `'NHWC'` or `'NCHW'` (case insensitive).
      Default value: `'NHWC'`.
    padding: One of `'VALID'` or `'SAME'` (case insensitive).
      Default value: `'VALID'`.
    name: Python `str` used to describe ops created by this function.
      Default value: `None` (i.e., `'im2col'`).

  Returns:
    im2row_x: batch of matrices representing subblock copies of `x`.
      Same batch shape as `x` but with rightmost shape:
      `batch_shape + [oh * ow, block_shape[0] * block_shape[1] * channels]`,
      where `oh = (h - block_shape[0] + 1) // slice_step[0]` and
      `ow = (w - block_shape[1] + 1) // slice_step[1]` when `padding = 'VALID'`
      and `oh = h` and `ow = w` when `padding = 'SAME'`.
    shape: shape `Tensor` equivalent to:
      `batch_shape + [oh, ow, block_shape[0] * block_shape[1] * channels]` where
      `oh, ow` are defined as above.
  """
  with tf.name_scope(name or 'im2row'):
    data_format = _validate_data_format(data_format)
    padding = _validate_padding(padding)
    if padding == 'VALID':
      pass  # Do nothing.
    elif padding == 'SAME':
      raise NotImplementedError(
          'Argument padding="SAME" not implemented.')
      # TODO(jvdillon): See if the following works:
      # fh, fw = block_shape
      # o = 1 if data_format == 'NHWC' else 0
      # n = prefer_static.maximum(0, prefer_static.rank(x) - 3)
      # paddings = prefer_static.pad(
      #     [[0, fh - 1], [0, fw - 1]],
      #     paddings=[[n + 1 - o, o], [0, 0]],
      #     constant_values=0)
      # x = tf.pad(x, paddings=paddings, constant_values=0)
      # padding = 'VALID'
    else:
      assert False  # Can't be here.
    x_shape = prefer_static.shape(x)
    idx, s = _im2row_index(
        x_shape, block_shape, slice_step, data_format, padding)
    flat_shape = prefer_static.pad(
        x_shape[:-3], paddings=[[0, 1]], constant_values=-1)
    x = tf.gather(tf.reshape(x, flat_shape), idx, axis=-1)  # == np.take
    return tf.reshape(x, s)


def _im2row_index(input_shape,
                  block_shape,
                  slice_step=(1, 1),
                  data_format='NHWC',
                  padding='VALID',
                  dtype=tf.int64,
                  name=None):
  """Computes indexes into a flattened image for building `im2col`."""
  with tf.name_scope(name or 'im2row_index'):
    # 1) Process input arguments.
    batch_shape, s3, s2, s1 = prefer_static.split(
        prefer_static.cast(input_shape, tf.int32),
        num_or_size_splits=[-1, 1, 1, 1])
    fh, fw = _split_pair(block_shape)
    sh, sw = _split_pair(slice_step)
    data_format = _validate_data_format(data_format)
    padding = _validate_padding(padding)

    # 2) Assemble all block start positions as indexes into the flattened image.
    if data_format == 'NHWC':
      h, w, c = s3[0], s2[0], s1[0]
      # start_idx.shape = [fh, fw, c]
      start_idx = _cartesian_add([
          prefer_static.range(c * w * fh, delta=c * w, dtype=dtype),
          prefer_static.range(c * fw, delta=c, dtype=dtype),
          prefer_static.range(c, delta=1, dtype=dtype),
      ])
    elif data_format == 'NCHW':
      c, h, w = s3[0], s2[0], s1[0]
      # start_idx.shape = [c, fh, fw]
      start_idx = _cartesian_add([
          prefer_static.range(w * h * c, delta=w * h, dtype=dtype),
          prefer_static.range(w * fh, delta=w, dtype=dtype),
          prefer_static.range(fw, delta=1, dtype=dtype),
      ])
    else:
      assert False  # Can't be here.

    # 3) Assemble all block offsets (into flattened image).
    if padding == 'VALID':
      eh = h - fh + 1  # extent height
      ew = w - fw + 1  # extent width
      # offset_idx.shape = [eh // sh, ew // sw]
      offset_idx = _cartesian_add([
          prefer_static.range(w * eh, delta=w * sh, dtype=dtype),
          prefer_static.range(ew, delta=sw, dtype=dtype),
      ])
      if data_format == 'NHWC':
        offset_idx *= c
      oh = eh // sh  # out height
      ow = ew // sw  # out width
    else:
      assert False  # Can't be here.

    # 4) Combine block start/offset pairs.
    # shape = [(eh // sh) * (ew // sw), fh * fw * c]
    idx = _cartesian_add([offset_idx, start_idx])
    new_shape = [oh, ow, fh * fw * c]
    new_shape = prefer_static.concat([batch_shape, new_shape], axis=0)
    return idx, new_shape


def _split_pair(x):
  """Splits a length two vector into two scalars."""
  x = prefer_static.cast(x, dtype=tf.int32)
  a, b = prefer_static.split(x, num_or_size_splits=[1, 1])
  return a[0], b[0]


def _cartesian_add(xs):
  """Adds a list of vectors by cumulatively expanding a dimension."""
  return sum(prefer_static.reshape(x, shape=[-1] + [1]*(len(xs) - 1 - i))
             for i, x in enumerate(xs))


def _validate_data_format(data_format):
  """Verify correctness of `data_format` argument."""
  data_format_ = str(data_format).upper()
  if data_format_ in {'NHWC', 'NCHW'}:
    return data_format_
  raise ValueError(
      'Argument data_format="{}" not recognized; must be one of '
      '{{"NHWC", "NCHW"}} (case insensitive).'.format(data_format))


def _validate_padding(padding):
  """Verify correctness of `padding` argument."""
  padding_ = str(padding).upper()
  if padding_ in {'SAME', 'VALID'}:
    return padding_
  raise ValueError(
      'Argument padding="{}" not recognized; must be one of '
      '{{"VALID", "SAME"}} (case insensitive).'.format(padding))
