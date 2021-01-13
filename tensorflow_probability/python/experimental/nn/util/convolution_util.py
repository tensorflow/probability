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

from tensorflow_probability.python.experimental.nn.util import utils
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps

__all__ = [
    'im2row',
    'im2row_index',
    'make_convolution_fn',
    'make_convolution_transpose_fn_with_dilation',
    'make_convolution_transpose_fn_with_subkernels',
    'make_convolution_transpose_fn_with_subkernels_matrix',
]


def im2row(x,
           block_shape,
           slice_step=(1, 1),
           padding='VALID',
           name=None):
  """Rearrange image blocks into rows.

  This function can be used to implement 2D convolution as a `matmul`, e.g.,

  `tf.nn.conv2d(x, k) = tf.matmul(
      tf.experimental.nn.util.im2row(x), tf.reshape(k, shape=[-1, out_size]))`.

  Args:
    x: Rank 3 (or more) Tensor representing 2D images.
    block_shape: Length-2 vector representing the block or "filter" shape.
    slice_step: Length-2 vector specifying the convolution stride length.
      Default value: `(1, 1)`.
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
    padding = _validate_padding(padding)
    if padding == 'VALID':
      pass  # Do nothing.
    elif padding == 'SAME':
      raise NotImplementedError(
          'Argument padding="SAME" not implemented.')
      # TODO(jvdillon): See if the following works:
      # fh, fw = block_shape
      # o = 1 if data_format == 'NHWC' else 0
      # n = ps.maximum(0, ps.rank(x) - 3)
      # paddings = ps.pad(
      #     [[0, fh - 1], [0, fw - 1]],
      #     paddings=[[n + 1 - o, o], [0, 0]],
      #     constant_values=0)
      # x = tf.pad(x, paddings=paddings, constant_values=0)
      # padding = 'VALID'
    else:
      assert False  # Can't be here.
    x_shape = ps.shape(x)
    idx, s = im2row_index(
        x_shape, block_shape=block_shape, slice_step=slice_step)
    flat_shape = ps.pad(
        x_shape[:-3], paddings=[[0, 1]], constant_values=-1)
    x = tf.gather(tf.reshape(x, flat_shape), idx, axis=-1)  # == np.take
    return tf.reshape(x, s)


def im2row_index(input_shape,
                 block_shape,
                 rank=2,
                 slice_step=(1, 1),
                 dilations=(1, 1),
                 dtype=tf.int32,
                 transpose=False,
                 validate_args=False,
                 name=None):
  """Computes indexes into a flattened image for building `im2row`."""
  with tf.name_scope(name or 'im2row_index'):
    if tf.get_static_value(rank) != 2:
      raise NotImplementedError('Argument `rank` currently only supports `2`; '
                                'saw "{}".'.format(rank))
    fh, fw = prepare_tuple_argument(
        block_shape, n=rank, arg_name='block_shape',
        validate_args=validate_args)
    sh, sw = prepare_tuple_argument(
        slice_step, n=rank, arg_name='slice_step', validate_args=validate_args)
    dh, dw = prepare_tuple_argument(
        dilations, n=rank, arg_name='dilations', validate_args=validate_args)

    # 1) Process input arguments.
    batch_shape, h, w, c = ps.split(
        ps.reshape(ps.cast(input_shape, dtype=dtype), shape=[-1]),
        num_or_size_splits=[-1, 1, 1, 1])
    h, w, c = h[0], w[0], c[0]

    tot_fh = dh * (fh - 1) + 1
    tot_fw = dw * (fw - 1) + 1

    # 2) Assemble all block start positions as indexes into the flattened image.
    # start_idx.shape = [fh, fw, c]
    if transpose:
      last_element = lambda size, step: size - (size - 1) % step - 1
      w_step = c * dw
      h_step = c * w * dh
      last_w = last_element(c * tot_fw, w_step)
      last_h = last_element(c * w * tot_fh, h_step)
      start_idx = cartesian_add([
          ps.range(last_h, -1, delta=-h_step, dtype=dtype),
          ps.range(last_w, -1, delta=-w_step, dtype=dtype),
          ps.range(c, delta=1, dtype=dtype),
      ])
    else:
      start_idx = cartesian_add([
          ps.range(c * w * tot_fh, delta=c * w * dh, dtype=dtype),
          ps.range(c * tot_fw, delta=c * dw, dtype=dtype),
          ps.range(c, delta=1, dtype=dtype),
      ])

    # 3) Assemble all block offsets (into flattened image).
    eh = h - tot_fh + 1
    ew = w - tot_fw + 1

    offset_idx = cartesian_add([
        ps.range(w * eh, delta=w * sh, dtype=dtype),
        ps.range(ew, delta=sw, dtype=dtype),
    ])

    offset_idx = offset_idx * c
    oh = (eh - 1) // sh + 1  # out height
    ow = (ew - 1) // sw + 1  # out width

    # 4) Combine block start/offset pairs.
    # shape = [(eh // sh) * (ew // sw), fh * fw * c]
    idx = cartesian_add([offset_idx, start_idx])
    new_shape = ps.concat(
        [batch_shape, ps.convert_to_shape_tensor([oh, ow, fh * fw * c])],
        axis=0)
    return idx, new_shape


def cartesian_add(xs):
  """Adds a list of vectors by cumulatively expanding a dimension."""
  return sum(ps.reshape(x, shape=[-1] + [1] * (len(xs) - 1 - i))
             for i, x in enumerate(xs))


def _validate_padding(padding):
  """Verify correctness of `padding` argument."""
  padding_ = str(padding).upper()
  if padding_ in {'SAME', 'VALID'}:
    return padding_
  raise ValueError(
      'Argument padding="{}" not recognized; must be one of '
      '{{"VALID", "SAME"}} (case insensitive).'.format(padding))


# TODO(emilyaf): Finish docstrings.
def make_convolution_fn(
    filter_shape, rank, strides, padding, dilations=None, dtype=tf.int32,
    validate_args=False, name=None):
  """Like `tf.nn.conv2d` except applies batch of kernels to batch of `x`."""
  with tf.name_scope(name or 'conv2d'):
    if tf.get_static_value(rank) != 2:
      raise NotImplementedError('Argument `rank` currently only supports `2`; '
                                'saw "{}".'.format(rank))
    [
        filter_shape,
        rank,
        strides,
        padding,
        dilations,
    ] = prepare_conv_args(
        filter_shape, rank=rank, strides=strides, padding=padding,
        dilations=dilations, validate_args=validate_args)

  def op(x, kernel):
    input_dtype = dtype_util.common_dtype([x, kernel], dtype_hint=tf.float32)
    x = tf.convert_to_tensor(x, dtype=input_dtype, name='x')
    kernel = tf.convert_to_tensor(kernel, dtype=input_dtype, name='kernel')

    batch_shape, event_shape = ps.split(
        ps.shape(x), num_or_size_splits=[-1, 3])
    xh, xw, c_in = ps.unstack(event_shape, num=3)
    fh, fw = filter_shape

    assertions = _maybe_validate_input_shapes(
        ps.shape(kernel), channels_in=c_in, filter_height=fh,
        filter_width=fw, validate_args=validate_args)

    with tf.control_dependencies(assertions):
      if tf.get_static_value(ps.rank(kernel)) == 2:
        flat_x = tf.reshape(x, shape=ps.concat([[-1], event_shape], axis=0))
        flat_y = tf.nn.conv2d(
            x,
            filters=tf.reshape(kernel, shape=[fh, fw, c_in, -1]),
            strides=strides,
            padding=padding,
            data_format='NHWC',
            dilations=dilations)
        output_shape = ps.shape(flat_y)[-3:]
        return tf.reshape(
            flat_y, shape=ps.concat([batch_shape, output_shape], axis=0))

      pad_values = [
          _get_conv_padding(
              xdim, filter_dim=k, stride=s, dilation=d, padding=padding)
          for (xdim, k, s, d) in zip((xh, xw), filter_shape, strides, dilations)
      ]

      idx, shape = im2row_index(
          (xh + sum(pad_values[0]), xw + sum(pad_values[1]), c_in),
          block_shape=filter_shape, slice_step=strides, dilations=dilations,
          dtype=dtype)

      if padding == 'SAME':
        n = ps.maximum(0, ps.rank(x) - 3)
        paddings = ps.pad(
            pad_values, paddings=[[n, 1], [0, 0]], constant_values=0)
        x = tf.pad(x, paddings=paddings, constant_values=0)

      flat_shape = ps.pad(
          batch_shape, paddings=[[0, 1]], constant_values=-1)
      flat_x = tf.gather(tf.reshape(x, shape=flat_shape), indices=idx, axis=-1)
      im_x = tf.reshape(flat_x, shape=ps.concat([batch_shape, shape], axis=0))
      return tf.matmul(im_x, kernel[..., tf.newaxis, :, :])
  return op


def _get_conv_padding(xdim, filter_dim, stride, dilation, padding):
  """Returns the number of zeros to pad at the start and end of an axis."""
  if padding == 'VALID':
    return (0, 0)
  elif padding == 'SAME':
    tot_k = dilation * (filter_dim - 1) + 1
    tot_pad = tf.maximum(tot_k - ((xdim - 1) % stride + 1), 0)
    pad_start = tot_pad // 2
    return pad_start, tot_pad - pad_start


def make_convolution_transpose_fn_with_dilation(
    filter_shape, strides, padding, rank=2, dilations=None, dtype=tf.int32,
    validate_args=False, name=None):
  """Like `tf.nn.conv2d` except applies batch of kernels to batch of `x`.

  This version tends to be fastest on GPU. It implements the transposed
  convolution as a regular convolution of an image that is dilated by
  interleaving rows and columns of zeros equal to the number of strides.

  Args:
    filter_shape: ...
    strides: ...
    padding: ...
    rank: ...
    dilations: ...
    dtype: ...
    validate_args: ...
    name: ...
  Returns:
    convolution_transpose_fn: A callable that takes an input `Tensor` and kernel
      and applies the transpose convolution operation.
  """
  with tf.name_scope(name or 'make_convolution_transpose_fn_with_dilation'):

    if tf.get_static_value(rank) != 2:
      raise NotImplementedError('Argument `rank` currently only supports `2`; '
                                'saw "{}".'.format(rank))
    [
        filter_shape,
        rank,
        strides,
        padding,
        dilations,
    ] = prepare_conv_args(
        filter_shape, rank=rank, strides=strides, padding=padding,
        dilations=dilations, is_transpose=True, validate_args=validate_args)

    sh, sw = strides
    fh, fw = filter_shape

    pad_values = [
        _get_transpose_conv_dilated_padding(
            k, stride=s, dilation=d, padding=padding)
        for (k, s, d) in zip(filter_shape, strides, dilations)]

    def op(x, kernel):
      input_dtype = dtype_util.common_dtype([x, kernel], dtype_hint=tf.float32)
      x = tf.convert_to_tensor(x, dtype=input_dtype, name='x')
      kernel = tf.convert_to_tensor(kernel, dtype=input_dtype, name='kernel')

      batch_shape, event_shape = ps.split(
          ps.shape(x), num_or_size_splits=[-1, 3])
      xh, xw, c_in = ps.unstack(event_shape, num=3)
      kernel_shape = ps.shape(kernel)
      assertions = _maybe_validate_input_shapes(
          kernel_shape, channels_in=c_in, filter_height=fh, filter_width=fw,
          validate_args=validate_args)

      with tf.control_dependencies(assertions):
        # If the kernel does not have batch shape, fall back to
        # `conv2d_transpose` (unless dilations > 1, which is not implemented in
        # `conv2d_transpose`).
        if (tf.get_static_value(ps.rank(kernel)) == 2
            and all(d == 1 for d in dilations)):
          return _call_conv2d_transpose(
              x, kernel, filter_shape, strides, padding, dilations,
              kernel_shape[-1], batch_shape, event_shape)

        idx, shape = im2row_index(
            (xh * sh + sum(pad_values[0]), xw * sw + sum(pad_values[1]), c_in),
            block_shape=filter_shape, slice_step=(1, 1), dilations=dilations,
            dtype=dtype, transpose=True)

        n = ps.maximum(0, ps.rank(x) - 3)
        paddings = ps.pad(
            pad_values, paddings=[[n, 1], [0, 0]], constant_values=0)

        # Interleave the rows and columns of the input with rows and columns of
        # zeros equal to the number of strides.
        x_half_dilated = tf.concat(
            [tf.zeros(ps.concat([batch_shape, (xh * xw, sw - 1, c_in)], axis=0),
                      dtype=input_dtype),
             tf.reshape(
                 x, shape=ps.concat([batch_shape, (xh * xw, 1, c_in)], axis=0))
             ], axis=-2)
        y = tf.reshape(
            x_half_dilated,
            shape=ps.concat([batch_shape, (xh, 1, xw * sw, c_in)], axis=0))

        x = tf.reshape(
            tf.concat(
                [tf.zeros(
                    ps.concat(
                        [batch_shape, (xh, sh - 1, xw * sw, c_in)], axis=0),
                    dtype=input_dtype), y], axis=-3),
            shape=ps.concat([batch_shape, (xh * sh, xw * sw, c_in)], axis=0))
        x_pad = tf.pad(x, paddings=paddings, constant_values=0)
        flat_shape = ps.pad(batch_shape, paddings=[[0, 1]], constant_values=-1)
        flat_x = tf.gather(
            tf.reshape(x_pad, shape=flat_shape), indices=idx, axis=-1)
        im_x = tf.reshape(flat_x, shape=ps.concat([batch_shape, shape], axis=0))
        return tf.matmul(im_x, kernel[..., tf.newaxis, :, :])
    return op


def make_convolution_transpose_fn_with_subkernels_matrix(
    filter_shape, strides, padding, rank=2, dilations=None, dtype=tf.int32,
    validate_args=False, name=None):
  """Like `tf.nn.conv2d` except applies batch of kernels to batch of `x`."""
  with tf.name_scope(name or 'make_convolution_transpose_fn_with_dilation'):

    if tf.get_static_value(rank) != 2:
      raise NotImplementedError('Argument `rank` currently only supports `2`; '
                                'saw "{}".'.format(rank))

    strides = tf.get_static_value(strides)
    if not isinstance(strides, int):
      raise ValueError('Argument `strides` must be a statically known integer.'
                       'Saw: {}'.format(strides))

    [
        filter_shape,
        rank,
        _,
        padding,
        dilations,
    ] = prepare_conv_args(
        filter_shape, rank=rank, strides=strides, padding=padding,
        dilations=dilations, is_transpose=True, validate_args=validate_args)

    fh, fw = filter_shape
    dh, dw = dilations

    # Determine maximum filter height and filter width of sub-kernels.
    sub_fh = (fh - 1) // strides + 1
    sub_fw = (fw - 1) // strides + 1

    def loop_body(i_, event_ind):
      i = i_ // strides
      j = i_ % strides

      i_ind = ps.range(i * fw, fw * fh, delta=strides * fw, dtype=dtype)
      j_ind = ps.range(j, fw, delta=strides, dtype=dtype)

      nc = cartesian_add([i_ind, j_ind])
      ind = ps.reverse(ps.reshape(nc, shape=[-1]), axis=[0])

      k = ps.reshape(
          cartesian_add(
              [ps.range(ps.shape(nc)[0] * sub_fw, delta=sub_fw, dtype=dtype),
               ps.range(ps.shape(nc)[1], dtype=dtype)]),
          shape=[-1])
      last_j = strides - (fw - j - 1) % strides - 1
      last_i = strides - (fh - i - 1) % strides - 1
      kernel_ind = ps.stack(
          [k, ps.ones_like(k) * last_i * strides + last_j], axis=1)
      event_ind = ps.tensor_scatter_nd_update(
          event_ind, ind[..., tf.newaxis], kernel_ind)

      return i_ + 1, event_ind

    event_ind = ps.zeros((fh * fw, 2), dtype=dtype)
    _, event_ind = tf.while_loop(
        lambda i, _: i < strides ** 2,
        loop_body,
        [tf.zeros([], dtype=dtype), event_ind])

    tot_pad_top, tot_pad_bottom = _get_transpose_conv_dilated_padding(
        fh, stride=strides, dilation=dh, padding=padding)
    tot_pad_left, tot_pad_right = _get_transpose_conv_dilated_padding(
        fw, stride=strides, dilation=dw, padding=padding)

    pad_bottom = (tot_pad_bottom - 1) // strides + 1
    pad_top = (tot_pad_top - 1) // strides + 1
    pad_right = (tot_pad_right - 1) // strides + 1
    pad_left = (tot_pad_left - 1) // strides + 1
    padding_vals = ((pad_top, pad_bottom), (pad_left, pad_right))

    truncate_top = pad_top * strides - tot_pad_top
    truncate_left = pad_left * strides - tot_pad_left

    def op(x, kernel):
      input_dtype = dtype_util.common_dtype([x, kernel], dtype_hint=tf.float32)
      x = tf.convert_to_tensor(x, dtype=input_dtype, name='x')
      kernel = tf.convert_to_tensor(kernel, dtype=input_dtype, name='kernel')

      batch_shape, event_shape = ps.split(
          ps.shape(x), num_or_size_splits=[-1, 3])
      xh, xw, c_in = ps.unstack(event_shape, num=3)

      kernel_shape = ps.shape(kernel)
      c_out = kernel_shape[-1]
      kernel_batch = kernel_shape[:-2]
      assertions = _maybe_validate_input_shapes(
          kernel_shape, channels_in=c_in, filter_height=fh, filter_width=fw,
          validate_args=validate_args)

      with tf.control_dependencies(assertions):

        # If the kernel does not have batch shape, fall back to
        # `conv2d_transpose` (unless dilations > 1, which is not implemented in
        # `conv2d_transpose`).
        if (tf.get_static_value(ps.rank(kernel)) == 2
            and all(d == 1 for d in dilations)):
          return _call_conv2d_transpose(
              x, kernel=kernel, filter_shape=filter_shape,
              strides=(strides,) * rank, padding=padding, dilations=dilations,
              c_out=c_out, batch_shape=batch_shape, event_shape=event_shape)

        n = ps.maximum(0, ps.rank(x) - 3)
        paddings = ps.pad(
            padding_vals,
            paddings=[[n, 1], [0, 0]],
            constant_values=0)

        x_pad = tf.pad(x, paddings=paddings, constant_values=0)
        x_pad_shape = ps.shape(x_pad)[:-3]
        flat_shape = ps.pad(x_pad_shape, paddings=[[0, 1]], constant_values=-1)
        flat_x = tf.reshape(x_pad, shape=flat_shape)

        idx, s = im2row_index(
            (xh + tf.reduce_sum(padding_vals[0]),
             xw + tf.reduce_sum(padding_vals[1]), c_in),
            block_shape=(sub_fh, sub_fw), slice_step=(1, 1), dilations=dilations
            )

        x_ = tf.gather(flat_x, indices=idx, axis=-1)
        im_x = tf.reshape(x_, shape=ps.concat([x_pad_shape, s], axis=0))

        # Add channels to subkernel indices
        idx_event = event_ind * [[c_in, 1]]
        idx_event_channels = (
            idx_event[tf.newaxis]
            + tf.stack([ps.range(c_in), tf.zeros((c_in,), dtype=dtype)],
                       axis=-1)[:, tf.newaxis, :])
        idx_event = tf.squeeze(
            tf.batch_to_space(
                idx_event_channels, block_shape=[c_in], crops=[[0, 0]]), axis=0)
        idx_event_broadcast = tf.broadcast_to(
            idx_event,
            shape=ps.concat([kernel_batch, ps.shape(idx_event)], axis=0))

        # Add cartesian product of batch indices, since scatter_nd can only be
        # applied to leading dimensions.
        idx_batch = tf.stack(
            tf.meshgrid(
                *[ps.range(b_, delta=1, dtype=dtype)
                  for b_ in tf.unstack(kernel_batch)], indexing='ij'),
            axis=ps.size(kernel_batch))

        idx_batch = tf.cast(idx_batch, dtype=dtype)  # empty tensor is float

        idx_batch_broadcast = idx_batch[..., tf.newaxis, :] + tf.zeros(
            (ps.shape(idx_event)[0], 1), dtype=dtype)
        idx_kernel = tf.concat(
            [idx_batch_broadcast, idx_event_broadcast], axis=-1)

        kernel_mat = tf.scatter_nd(
            idx_kernel,
            updates=kernel,
            shape=ps.cast(
                ps.concat([kernel_batch,
                           [sub_fh * sub_fw * c_in, strides ** 2, c_out]],
                          axis=0),
                dtype=dtype))

        kernel_mat = tf.reshape(
            kernel_mat,
            shape=ps.concat(
                [ps.shape(kernel_mat)[:-2], [strides ** 2 * c_out]], axis=0))

        kernel_mat = kernel_mat[..., tf.newaxis, :, :]
        out = tf.matmul(im_x, kernel_mat)
        broadcast_batch_shape = ps.broadcast_shape(batch_shape, kernel_batch)

        if strides > 1:
          tot_size = tf.reduce_prod(broadcast_batch_shape)
          flat_out = tf.reshape(
              out,
              shape=ps.concat([[tot_size], ps.shape(out)[-3:]], axis=0))
          out = tf.nn.depth_to_space(flat_out, block_size=strides)

        if padding == 'VALID':
          out_height = fh + strides * (xh - 1)
          out_width = fw + strides * (xw - 1)
        elif padding == 'SAME':
          out_height = xh * strides
          out_width = xw * strides

        out = out[..., truncate_top:truncate_top + out_height,
                  truncate_left:truncate_left + out_width, :]
        out = tf.reshape(
            out, shape=ps.concat(
                [broadcast_batch_shape, [out_height, out_width, c_out]],
                axis=0))
        return out
    return op


def make_convolution_transpose_fn_with_subkernels(
    filter_shape, strides, padding, rank=2, dilations=None, dtype=tf.int32,
    validate_args=False, name=None):
  """Like `tf.nn.conv2d` except applies batch of kernels to batch of `x`."""
  with tf.name_scope(name or 'make_convolution_transpose_fn_with_dilation'):

    if tf.get_static_value(rank) != 2:
      raise NotImplementedError('Argument `rank` currently only supports `2`; '
                                'saw "{}".'.format(rank))
    [
        filter_shape,
        rank,
        strides,
        padding,
        dilations,
    ] = prepare_conv_args(
        filter_shape, rank=rank, strides=strides, padding=padding,
        dilations=dilations, is_transpose=True, validate_args=validate_args)

    sh, sw = strides
    fh, fw = filter_shape
    dh, dw = dilations

    # Determine maximum filter height and filter width of sub-kernels.
    sub_fh = (fh - 1) // sh + 1
    sub_fw = (fw - 1) // sw + 1

    def loop_body(i_, kernels_ind):
      i = i_ // sw
      j = i_ % sw
      i_ind = ps.range((sh - i - 1)*fw, fw * fh, delta=sh*fw, dtype=dtype)
      j_ind = ps.range((sw - j - 1), fw, delta=sw, dtype=dtype)

      last_j = sw - (fw - j - 1) % sw - 1
      last_i = sh - (fh - i - 1) % sh - 1
      pos = last_i * sw + last_j

      nc = cartesian_add([i_ind, j_ind])
      kernels_ind = kernels_ind.write(
          sh * sw - pos - 1, ps.reverse(ps.reverse(nc, [0]), [1]))

      return i_ + 1, kernels_ind

    kernels_ind = tf.TensorArray(dtype=dtype, infer_shape=False, size=1,
                                 dynamic_size=True)

    _, kernels_ind = tf.while_loop(
        lambda i, _: i < sh * sw,
        loop_body,
        [0, kernels_ind])

    tot_pad_top, tot_pad_bottom = _get_transpose_conv_dilated_padding(
        fh, stride=sh, dilation=dh, padding=padding)
    tot_pad_left, tot_pad_right = _get_transpose_conv_dilated_padding(
        fw, stride=sw, dilation=dw, padding=padding)

    pad_bottom = (tot_pad_bottom - 1) // sh + 1
    pad_top = (tot_pad_top - 1) // sh + 1
    pad_right = (tot_pad_right - 1) // sw + 1
    pad_left = (tot_pad_left - 1) // sw + 1
    padding_vals = ((pad_top, pad_bottom), (pad_left, pad_right))

    truncate_top = pad_top * sh - tot_pad_top
    truncate_left = pad_left * sw - tot_pad_left

    def op(x, kernel):
      input_dtype = dtype_util.common_dtype([x, kernel], dtype_hint=tf.float32)
      x = tf.convert_to_tensor(x, dtype=input_dtype, name='x')
      kernel = tf.convert_to_tensor(kernel, dtype=input_dtype, name='kernel')

      batch_shape, event_shape = ps.split(
          ps.shape(x), num_or_size_splits=[-1, 3])
      xh, xw, c_in = ps.unstack(event_shape, num=3)

      kernel_shape = ps.shape(kernel)
      c_out = kernel_shape[-1]
      kernel_batch = kernel_shape[:-2]
      assertions = _maybe_validate_input_shapes(
          kernel_shape, channels_in=c_in, filter_height=fh, filter_width=fw,
          validate_args=validate_args)

      with tf.control_dependencies(assertions):
        # If the kernel does not have batch shape, fall back to
        # `conv2d_transpose` (unless dilations > 1, which is not implemented in
        # `conv2d_transpose`).
        if (tf.get_static_value(ps.rank(kernel)) == 2
            and all(d == 1 for d in dilations)):
          return _call_conv2d_transpose(
              x, kernel, filter_shape, strides, padding, dilations, c_out,
              batch_shape, event_shape)

        n = ps.maximum(0, ps.rank(x) - 3)
        paddings = ps.pad(
            padding_vals,
            paddings=[[n, 1], [0, 0]],
            constant_values=0)
        x_pad = tf.pad(x, paddings=paddings, constant_values=0)

        ex_h = xh + tf.reduce_sum(padding_vals[0]) - sub_fh + 1
        ex_w = xw + tf.reduce_sum(padding_vals[1]) - sub_fw + 1

        def loop_body(i, outputs):
          subkernel_ind = kernels_ind.read(i)
          fh_, fw_ = ps.unstack(ps.shape(subkernel_ind), num=2)
          eh = ex_h + fh_ - 1
          ew = ex_w + fw_ - 1

          subkernel_ind = ps.reshape(
              ps.reshape(subkernel_ind * c_in, shape=[-1])[:, tf.newaxis]
              + ps.range(c_in), shape=[-1])

          k = tf.gather(kernel, subkernel_ind, axis=-2)
          ind, shape = im2row_index(
              [eh, ew, c_in],
              block_shape=(fh_, fw_),
              slice_step=(1, 1),
              dilations=dilations)
          x_i = x_pad[..., :eh, :ew, :]
          x_i_shape = ps.shape(x_i)
          flat_shape = ps.pad(
              x_i_shape[:-3], paddings=[[0, 1]], constant_values=-1)
          flat_x = tf.reshape(x_i, flat_shape)
          x_ = tf.gather(flat_x, ind, axis=-1)
          im_x = tf.reshape(x_, ps.concat([x_i_shape[:-3], shape], axis=0))
          outputs = outputs.write(
              i,
              tf.matmul(
                  im_x,
                  tf.reshape(
                      k, ps.concat(
                          [kernel_batch, [1, fh_ * fw_* c_in, c_out]], axis=0)))
              )
          return i + 1, outputs

        outputs = tf.TensorArray(dtype=input_dtype, infer_shape=False, size=1,
                                 dynamic_size=True)

        _, outputs = tf.while_loop(
            lambda i, _: i < sh * sw,
            loop_body,
            [0, outputs])

        y = outputs.concat()

        m = tf.reduce_prod(ps.shape(y)[:-3])
        y_ = tf.reshape(y, shape=ps.concat([[m], ps.shape(y)[-3:]], axis=0))
        y2 = tf.batch_to_space(
            y_, strides, crops=tf.zeros([2, 2], dtype=tf.int64))
        broadcast_batch_shape = ps.broadcast_shape(batch_shape, kernel_batch)
        y2 = tf.reshape(y2, ps.concat(
            [broadcast_batch_shape, ps.shape(y2)[-3:]], axis=0))

        if padding == 'VALID':
          out_height = fh + sh * (xh - 1)
          out_width = fw + sw * (xw - 1)
        elif padding == 'SAME':
          out_height = xh * sh
          out_width = xw * sw

        return y2[..., truncate_top:truncate_top+out_height,
                  truncate_left:truncate_left+out_width, :]
    return op


def _maybe_validate_input_shapes(
    kernel_shape, channels_in, filter_height, filter_width, validate_args):
  """Validate shapes of inputs to convolution op."""
  k_dim = kernel_shape[-2]
  k_dim_ = tf.get_static_value(k_dim)
  expected_k_dim = filter_height * filter_width * channels_in
  expected_k_dim_ = tf.get_static_value(expected_k_dim)
  assertions = []
  if expected_k_dim_ is not None and k_dim_ is not None:
    if expected_k_dim_ != k_dim_:
      raise ValueError(
          'The size of the second-to-rightmost dimension of `kernel` ( ={}) '
          ' must equal `filter_height * filter_width * channels_in` ( ={}), '
          'where `channels_in` is the size of the rightmost dimension of the '
          'input.'.format(k_dim_, expected_k_dim_))
  elif validate_args:
    assertions.append(
        assert_util.assert_equal(
            k_dim, expected_k_dim,
            message=('The size of the second-to-rightmost dimension of `kernel`'
                     ' must equal `filter_height * filter_width * channels_in`,'
                     ' where `channels_in` is the size of the rightmost '
                     'dimension of the input.')))
  return assertions


def _get_transpose_conv_dilated_padding(filter_dim, stride, dilation, padding):
  """Zero-padding for inputs dilated by strides."""
  tot_filter_dim = filter_dim + (filter_dim - 1) * (dilation - 1)
  if padding == 'VALID':
    tot_pad = 2 * (tot_filter_dim - 1)
  elif padding == 'SAME':
    tot_pad = tot_filter_dim + stride - 2

  # TODO(emilyaf): Support stride > kernel_dim.
  # if filter_dim > 1:
  pad_end = tot_pad // 2
  pad_start = tot_pad - pad_end - (stride - 1)  # implicit pad
  # else:
  #   pad_end = pad_start = 0
  return pad_start, pad_end


def _get_output_shape(rank, strides, padding, dilations, input_shape,
                      output_size, filter_shape, output_padding=None):
  """Compute the `output_shape` and `strides` arg used by `conv_transpose`."""
  if output_padding is None:
    output_padding = (None,) * rank
  else:
    output_padding = utils.prepare_tuple_argument(
        output_padding, n=rank, arg_name='output_padding')
    for stride, out_pad in zip(strides, output_padding):
      if out_pad >= stride:
        raise ValueError('Stride {} must be greater than output '
                         'padding {}.'.format(strides, output_padding))
  event_shape = []
  for i in range(-rank, 0):
    event_shape.append(_deconv_output_length(
        input_shape[i - 1],
        filter_size=filter_shape[i],
        padding=padding,
        output_padding=output_padding[i],
        stride=strides[i],
        dilation=dilations[i]))
  event_shape.append(output_size)
  batch_shape = input_shape[:-rank-1]
  output_shape = ps.concat([batch_shape, event_shape], axis=0)
  strides = ps.pad(strides, paddings=[[1, 1]], constant_values=1)
  return output_shape, strides


def _deconv_output_length(input_size, filter_size, padding, output_padding,
                          stride, dilation):
  """Determines output length of a transposed convolution given input length.

  Args:
    input_size: `int`.
    filter_size: `int`.
    padding: one of `"SAME"`, `"VALID"`, `"FULL"`.
    output_padding: `int`, amount of padding along the output dimension. Can
      be set to `None` in which case the output length is inferred.
    stride: `int`.
    dilation: `int`.

  Returns:
    output_length: The output length (`int`).
  """
  assert padding in {'SAME', 'VALID', 'FULL'}
  if input_size is None:
    return None
  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'VALID':
      return input_size * stride + max(filter_size - stride, 0)
    elif padding == 'FULL':
      return input_size * stride - (stride + filter_size - 2)
    elif padding == 'SAME':
      return input_size * stride
  if padding == 'SAME':
    pad = filter_size // 2
  elif padding == 'VALID':
    pad = 0
  elif padding == 'FULL':
    pad = filter_size - 1
  return (input_size - 1) * stride + filter_size - 2 * pad + output_padding


def prepare_conv_args(
    filter_shape, rank, strides, padding, dilations,
    is_transpose=False, validate_args=False):
  """Sanitizes use provided input."""
  padding = _validate_padding(padding)
  try:
    rank = int(tf.get_static_value(rank))
  except TypeError:
    raise TypeError('Argument `rank` must be statically known `int`.')
  valid_rank = {1, 2, 3}
  if rank not in valid_rank:
    raise ValueError('Argument `rank` must be in {}.'.format(valid_rank))
  filter_shape = prepare_tuple_argument(
      filter_shape, n=rank, arg_name='filter_shape',
      validate_args=validate_args)
  strides = prepare_tuple_argument(
      strides, n=rank, arg_name='strides', validate_args=validate_args)
  padding = _prepare_padding_argument(padding)
  dilations = prepare_tuple_argument(
      dilations, n=rank, arg_name='dilations', validate_args=validate_args)

  strides_ = [tf.get_static_value(s) for s in strides]
  dilations_ = [tf.get_static_value(d) for d in dilations]
  assertions = []
  if is_transpose:
    if (all(s is not None for s in strides_)
        and all(d is not None for d in dilations_)):
      if any(s > 1 for s in strides_) and any(d > 1 for d in dilations_):
        raise NotImplementedError('At least one of `dilations` and `strides` '
                                  'must equal `1` for each dimension. Saw: '
                                  '`strides={}`, `dilations={}`'.format(
                                      strides, dilations))
    elif validate_args:
      assertions.append(
          assert_util.assert_equal(
              tf.logical_or(
                  tf.equal(tf.reduce_max(strides), 1),
                  tf.equal(tf.reduce_max(dilations), 1)),
              True,
              message='At least one of `dilations` and `strides` must equal `1` '
              'for each dimension.'))

    # TODO(emilyaf): Remove this once strides > filter_dim is supported.
    filter_shape_ = [tf.get_static_value(s) for s in filter_shape]
    if any(s is not None and f is not None and s > f
           for s, f in zip(strides_, filter_shape_)):
      raise NotImplementedError('Stride must be less than or equal to the '
                                'filter size along each dimension.')

  with tf.control_dependencies(assertions):
    return filter_shape, rank, strides, padding, dilations


def prepare_tuple_argument(arg, n, arg_name, validate_args=False):
  """Helper which processes `Tensor`s to tuples in standard form."""
  arg_size = ps.size(arg)
  arg_size_ = tf.get_static_value(arg_size)
  assertions = []
  if arg_size_ is not None:
    if arg_size_ not in (1, n):
      raise ValueError('The size of `{}` must be equal to `1` or to the rank '
                       'of the convolution (={}). Saw size = {}'.format(
                           arg_name, n, arg_size_))
  elif validate_args:
    assertions.append(assert_util.assert_equal(
        ps.logical_or(arg_size == 1, arg_size == n),
        True,
        message=('The size of `{}` must be equal to `1` or to the rank of the '
                 'convolution (={})'.format(arg_name, n))))

  with tf.control_dependencies(assertions):
    arg = ps.broadcast_to(arg, shape=[n])
    arg = ps.unstack(arg, num=n)
    return arg


def _prepare_padding_argument(x):
  """Helper which processes the padding argument."""
  if not hasattr(x, 'upper'):
    return tuple(x)
  padding = x.upper()
  if padding in {'CAUSAL', 'FULL'}:
    raise NotImplementedError(
        'Argument `padding` value "{}" currently not supported. If you '
        'require this feature, please create an issue on '
        '`https://github.com/tensorflow/probability` or email '
        '`tfprobability@tensorflow.org`.'.format(padding))
  valid_values = {'VALID', 'SAME'}
  if padding not in valid_values:
    raise ValueError('Argument `padding` must be convertible to a tuple '
                     'or one of {}; saw: "{}".'.format(valid_values, padding))
  return padding


def _call_conv2d_transpose(x, kernel, filter_shape, strides, padding, dilations,
                           c_out, batch_shape, event_shape):
  """Call `tf.nn.conv2d_transpose` (for kernels with no batch dimensions)."""
  fh, fw = filter_shape
  flat_x = tf.reshape(x, shape=ps.concat([[-1], event_shape], axis=0))
  output_shape, strides_ = _get_output_shape(
      rank=2, strides=strides, padding=padding, dilations=dilations,
      input_shape=ps.shape(flat_x), output_size=c_out,
      filter_shape=filter_shape)
  flat_y = tf.nn.conv2d_transpose(
      flat_x,
      filters=tf.transpose(
          tf.reshape(
              kernel, shape=[fh, fw, event_shape[-1], -1]),
          perm=[0, 1, 3, 2]),
      output_shape=output_shape,
      strides=strides_,
      padding=padding,
      data_format='NHWC',
      dilations=dilations)
  return tf.reshape(
      flat_y, shape=ps.concat([batch_shape, output_shape[-3:]], axis=0))
