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
"""Interpolation Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util


__all__ = [
    'interp_regular_1d_grid',
    'batch_interp_regular_1d_grid',
]


def _interp_regular_1d_grid_impl(x,
                                 x_ref_min,
                                 x_ref_max,
                                 y_ref,
                                 axis=-1,
                                 batch_y_ref=False,
                                 fill_value='constant_extension',
                                 fill_value_below=None,
                                 fill_value_above=None,
                                 grid_regularizing_transform=None,
                                 name=None):
  """1-D interpolation that works with/without batching."""
  # To understand the implemention differences between the batch/no-batch
  # versions of this function, you should probably understand the difference
  # between tf.gather and tf.batch_gather.  In particular, we do *not* make the
  # no-batch version a special case of the batch version, because that would
  # an inefficient use of batch_gather with unnecessarily broadcast args.
  with tf.name_scope(
      name,
      values=[
          x, x_ref_min, x_ref_max, y_ref, axis, fill_value, fill_value_below,
          fill_value_above
      ]):

    # Arg checking.
    allowed_fv_st = ('constant_extension', 'extrapolate')
    for fv in (fill_value, fill_value_below, fill_value_above):
      if isinstance(fv, str) and fv not in allowed_fv_st:
        raise ValueError(
            'A fill value ({}) was not an allowed string ({})'.format(
                fv, allowed_fv_st))

    # Separate value fills for below/above incurs extra cost, so keep track of
    # whether this is needed.
    need_separate_fills = (
        fill_value_above is not None or fill_value_below is not None or
        fill_value == 'extrapolate'  # always requries separate below/above
    )
    if need_separate_fills and fill_value_above is None:
      fill_value_above = fill_value
    if need_separate_fills and fill_value_below is None:
      fill_value_below = fill_value

    dtype = dtype_util.common_dtype([x, x_ref_min, x_ref_max, y_ref],
                                    preferred_dtype=tf.float32)
    x = tf.convert_to_tensor(x, name='x', dtype=dtype)

    x_ref_min = tf.convert_to_tensor(x_ref_min, name='x_ref_min', dtype=dtype)
    x_ref_max = tf.convert_to_tensor(x_ref_max, name='x_ref_max', dtype=dtype)
    if not batch_y_ref:
      _assert_ndims_statically(x_ref_min, expect_ndims=0)
      _assert_ndims_statically(x_ref_max, expect_ndims=0)

    y_ref = tf.convert_to_tensor(y_ref, name='y_ref', dtype=dtype)

    if batch_y_ref:
      # If we're batching,
      #   x.shape ~ [A1,...,AN, D],  x_ref_min/max.shape ~ [A1,...,AN]
      # So to add together we'll append a singleton.
      # If not batching, x_ref_min/max are scalar, so this isn't an issue,
      # moreover, if not batching, x can be scalar, and expanding x_ref_min/max
      # would cause a bad expansion of x when added to x (confused yet?).
      x_ref_min = x_ref_min[..., tf.newaxis]
      x_ref_max = x_ref_max[..., tf.newaxis]

    axis = tf.convert_to_tensor(axis, name='axis', dtype=tf.int32)
    axis = distribution_util.make_non_negative_axis(axis, tf.rank(y_ref))
    _assert_ndims_statically(axis, expect_ndims=0)

    ny = tf.cast(tf.shape(y_ref)[axis], dtype)

    # Map [x_ref_min, x_ref_max] to [0, ny - 1].
    # This is the (fractional) index of x.
    if grid_regularizing_transform is None:
      g = lambda x: x
    else:
      g = grid_regularizing_transform
    fractional_idx = ((g(x) - g(x_ref_min)) / (g(x_ref_max) - g(x_ref_min)))
    x_idx_unclipped = fractional_idx * (ny - 1)

    # Wherever x is NaN, x_idx_unclipped will be NaN as well.
    # Keep track of the nan indices here (so we can impute NaN later).
    # Also eliminate any NaN indices, since there is not NaN in 32bit.
    nan_idx = tf.is_nan(x_idx_unclipped)
    x_idx_unclipped = tf.where(nan_idx, tf.zeros_like(x_idx_unclipped),
                               x_idx_unclipped)

    x_idx = tf.clip_by_value(x_idx_unclipped, tf.zeros((), dtype=dtype), ny - 1)

    # Get the index above and below x_idx.
    # Naively we could set idx_below = floor(x_idx), idx_above = ceil(x_idx),
    # however, this results in idx_below == idx_above whenever x is on a grid.
    # This in turn results in y_ref_below == y_ref_above, and then the gradient
    # at this point is zero.  So here we "jitter" one of idx_below, idx_above,
    # so that they are at different values.  This jittering does not affect the
    # interpolated value, but does make the gradient nonzero (unless of course
    # the y_ref values are the same).
    idx_below = tf.floor(x_idx)
    idx_above = tf.minimum(idx_below + 1, ny - 1)
    idx_below = tf.maximum(idx_above - 1, 0)

    # These are the values of y_ref corresponding to above/below indices.
    idx_below_int32 = tf.to_int32(idx_below)
    idx_above_int32 = tf.to_int32(idx_above)
    if batch_y_ref:
      # If y_ref.shape ~ [A1,...,AN, C, B1,...,BN],
      # and x.shape, x_ref_min/max.shape ~ [A1,...,AN, D]
      # Then y_ref_below.shape ~ [A1,...,AN, D, B1,...,BN]
      y_ref_below = _batch_gather_with_broadcast(y_ref, idx_below_int32, axis)
      y_ref_above = _batch_gather_with_broadcast(y_ref, idx_above_int32, axis)
    else:
      # Here, y_ref_below.shape =
      #   y_ref.shape[:axis] + x.shape + y_ref.shape[axis + 1:]
      y_ref_below = tf.gather(y_ref, idx_below_int32, axis=axis)
      y_ref_above = tf.gather(y_ref, idx_above_int32, axis=axis)

    # Use t to get a convex combination of the below/above values.
    t = x_idx - idx_below

    # x, and tensors shaped like x, need to be added to, and selected with
    # (using tf.where) the output y.  This requires appending singletons.
    # Make functions appropriate for batch/no-batch.
    if batch_y_ref:
      # In the non-batch case, the output shape is going to be
      #   y_ref.shape[:axis] + x.shape + y_ref.shape[axis+1:]
      expand_x_fn = _make_expand_x_fn_for_batch_interpolation(y_ref, axis)
    else:
      # In the batch case, the output shape is going to be
      #   Broadcast(y_ref.shape[:axis], x.shape[:-1]) +
      #   x.shape[-1:] +  y_ref.shape[axis+1:]
      expand_x_fn = _make_expand_x_fn_for_non_batch_interpolation(y_ref, axis)

    t = expand_x_fn(t)
    nan_idx = expand_x_fn(nan_idx, broadcast=True)
    x_idx_unclipped = expand_x_fn(x_idx_unclipped, broadcast=True)

    y = t * y_ref_above + (1 - t) * y_ref_below

    # Now begins a long excursion to fill values outside [x_min, x_max].

    # Re-insert NaN wherever x was NaN.
    y = tf.where(nan_idx, tf.fill(tf.shape(y), tf.constant(np.nan, y.dtype)), y)

    if not need_separate_fills:
      if fill_value == 'constant_extension':
        pass  # Already handled by clipping x_idx_unclipped.
      else:
        y = tf.where((x_idx_unclipped < 0) | (x_idx_unclipped > ny - 1),
                     fill_value + tf.zeros_like(y), y)
    else:
      # Fill values below x_ref_min <==> x_idx_unclipped < 0.
      if fill_value_below == 'constant_extension':
        pass  # Already handled by the clipping that created x_idx_unclipped.
      elif fill_value_below == 'extrapolate':
        if batch_y_ref:
          # For every batch member, gather the first two elements of y across
          # `axis`.
          y_0 = tf.gather(y_ref, [0], axis=axis)
          y_1 = tf.gather(y_ref, [1], axis=axis)
        else:
          # If not batching, we want to gather the first two elements, just like
          # above.  However, these results need to be replicated for every
          # member of x.  An easy way to do that is to gather using
          # indices = zeros/ones(x.shape).
          y_0 = tf.gather(
              y_ref, tf.zeros(tf.shape(x), dtype=tf.int32), axis=axis)
          y_1 = tf.gather(
              y_ref, tf.ones(tf.shape(x), dtype=tf.int32), axis=axis)
        x_delta = (x_ref_max - x_ref_min) / (ny - 1)
        x_factor = expand_x_fn((x - x_ref_min) / x_delta, broadcast=True)
        y = tf.where(x_idx_unclipped < 0, y_0 + x_factor * (y_1 - y_0), y)
      else:
        y = tf.where(x_idx_unclipped < 0, fill_value_below + tf.zeros_like(y),
                     y)
      # Fill values above x_ref_min <==> x_idx_unclipped > ny - 1.
      if fill_value_above == 'constant_extension':
        pass  # Already handled by the clipping that created x_idx_unclipped.
      elif fill_value_above == 'extrapolate':
        ny_int32 = tf.shape(y_ref)[axis]
        if batch_y_ref:
          y_n1 = tf.gather(y_ref, [tf.shape(y_ref)[axis] - 1], axis=axis)
          y_n2 = tf.gather(y_ref, [tf.shape(y_ref)[axis] - 2], axis=axis)
        else:
          y_n1 = tf.gather(y_ref, tf.fill(tf.shape(x), ny_int32 - 1), axis=axis)
          y_n2 = tf.gather(y_ref, tf.fill(tf.shape(x), ny_int32 - 2), axis=axis)
        x_delta = (x_ref_max - x_ref_min) / (ny - 1)
        x_factor = expand_x_fn((x - x_ref_max) / x_delta, broadcast=True)
        y = tf.where(x_idx_unclipped > ny - 1,
                     y_n1 + x_factor * (y_n1 - y_n2), y)
      else:
        y = tf.where(x_idx_unclipped > ny - 1,
                     fill_value_above + tf.zeros_like(y), y)

    return y


def interp_regular_1d_grid(x,
                           x_ref_min,
                           x_ref_max,
                           y_ref,
                           axis=-1,
                           fill_value='constant_extension',
                           fill_value_below=None,
                           fill_value_above=None,
                           grid_regularizing_transform=None,
                           name=None):
  """Linear `1-D` interpolation on a regular (constant spacing) grid.

  Given reference values, this function computes a piecewise linear interpolant
  and evaluates it on a new set of `x` values.

  The interpolant is built from `C` reference values indexed by one dimension
  of `y_ref` (specified by the `axis` kwarg).

  If `y_ref` is a vector, then each value `y_ref[i]` is considered to be equal
  to `f(x_ref[i])`, for `C` (implicitly defined) reference values between
  `x_ref_min` and `x_ref_max`:

  ```none
  x_ref[i] = x_ref_min + i * (x_ref_max - x_ref_min) / (C - 1),
  i = 0, ..., C - 1.
  ```

  If `rank(y_ref) > 1`, then dimension `axis` indexes `C` reference values of
  a shape `y_ref.shape[:axis] + y_ref.shape[axis + 1:]` `Tensor`.

  If `rank(x) > 1`, then the output is obtained by effectively flattening `x`,
  interpolating along `axis`, then expanding the result to shape
  `y_ref.shape[:axis] + x.shape + y_ref.shape[axis + 1:]`.

  These shape semantics are equivalent to `scipy.interpolate.interp1d`.

  Args:
    x: Numeric `Tensor` The x-coordinates of the interpolated output values.
    x_ref_min:  Scalar `Tensor` of same `dtype` as `x`.  The minimum value of
      the (implicitly defined) reference `x_ref`.
    x_ref_max:  Scalar `Tensor` of same `dtype` as `x`.  The maximum value of
      the (implicitly defined) reference `x_ref`.
    y_ref:  `N-D` `Tensor` (`N > 0`) of same `dtype` as `x`. The reference
      output values.
    axis:  Scalar `Tensor` designating the dimension of `y_ref` that indexes
      values of the interpolation variable.
      Default value: `-1`, the rightmost axis.
    fill_value:  Determines what values output should take for `x` values that
      are below `x_ref_min` or above `x_ref_max`. `Tensor` or one of the strings
      "constant_extension" ==> Extend as constant function. "extrapolate" ==>
      Extrapolate in a linear fashion.
      Default value: `"constant_extension"`
    fill_value_below:  Optional override of `fill_value` for `x < x_ref_min`.
    fill_value_above:  Optional override of `fill_value` for `x > x_ref_max`.
    grid_regularizing_transform:  Optional transformation `g` which regularizes
      the implied spacing of the x reference points.  In other words, if
      provided, we assume `g(x_ref_i)` is a regular grid between `g(x_ref_min)`
      and `g(x_ref_max)`.
    name:  A name to prepend to created ops.
      Default value: `"interp_regular_1d_grid"`.

  Returns:
    y_interp:  Interpolation between members of `y_ref`, at points `x`.
      `Tensor` of same `dtype` as `x`, and shape
      `y.shape[:axis] + x.shape + y.shape[axis + 1:]`

  Raises:
    ValueError:  If `fill_value` is not an allowed string.
    ValueError:  If `axis` is not a scalar.

  #### Examples

  Interpolate a function of one variable:

  ```python
  y_ref = tf.exp(tf.linspace(start=0., stop=10., num=200))

  interp_regular_1d_grid(
      x=[6.0, 0.5, 3.3], x_ref_min=0., x_ref_max=1., y_ref=y_ref)
  ==> approx [exp(6.0), exp(0.5), exp(3.3)]
  ```

  Interpolate a matrix-valued function of one variable:

  ```python
  mat_0 = [[1., 0.], [0., 1.]]
  mat_1 = [[0., -1], [1, 0]]
  y_ref = [mat_0, mat_1]

  # Get three output matrices at once.
  tfp.math.interp_regular_1d_grid(
      x=[0., 0.5, 1.], x_ref_min=0., x_ref_max=1., y_ref=y_ref, axis=0)
  ==> [mat_0, 0.5 * mat_0 + 0.5 * mat_1, mat_1]
  ```

  Interpolate a scalar valued function, and get a matrix of results:

  ```python
  y_ref = tf.exp(tf.linspace(start=0., stop=10., num=200))
  x = [[1.1, 1.2], [2.1, 2.2]]
  tfp.math.interp_regular_1d_grid(x, x_ref_min=0., x_ref_max=10., y_ref=y_ref)
  ==> tf.exp(x)
  ```

  Interpolate a function of one variable on a log-spaced grid:

  ```python
  x_ref = tf.exp(tf.linspace(tf.log(1.), tf.log(100000.), num_pts))
  y_ref = tf.log(x_ref + x_ref**2)

  interp_regular_1d_grid(x=[1.1, 2.2], x_ref_min=1., x_ref_max=100000., y_ref,
      grid_regularizing_transform=tf.log)
  ==> [tf.log(1.1 + 1.1**2), tf.log(2.2 + 2.2**2)]
  ```

  """

  return _interp_regular_1d_grid_impl(
      x,
      x_ref_min,
      x_ref_max,
      y_ref,
      axis=axis,
      batch_y_ref=False,
      fill_value=fill_value,
      fill_value_below=fill_value_below,
      fill_value_above=fill_value_above,
      grid_regularizing_transform=grid_regularizing_transform,
      name=name or 'interp_regular_1d_grid')


def batch_interp_regular_1d_grid(x,
                                 x_ref_min,
                                 x_ref_max,
                                 y_ref,
                                 axis=-1,
                                 fill_value='constant_extension',
                                 fill_value_below=None,
                                 fill_value_above=None,
                                 grid_regularizing_transform=None,
                                 name=None):
  """Linear `1-D` interpolation on a regular (constant spacing) grid.

  Given [batch of] reference values, this function computes a piecewise linear
  interpolant and evaluates it on a [batch of] of new `x` values.

  The interpolant is built from `C` reference values indexed by one dimension
  of `y_ref` (specified by the `axis` kwarg).

  If `y_ref` is a vector, then each value `y_ref[i]` is considered to be equal
  to `f(x_ref[i])`, for `C` (implicitly defined) reference values between
  `x_ref_min` and `x_ref_max`:

  ```none
  x_ref[i] = x_ref_min + i * (x_ref_max - x_ref_min) / (C - 1),
  i = 0, ..., C - 1.
  ```

  In the general case, dimensions to the left of `axis` in `y_ref` are broadcast
  with leading dimensions in `x`, `x_ref_min`, `x_ref_max`.

  Args:
    x: Numeric `Tensor` The x-coordinates of the interpolated output values
      for each batch.  Shape broadcasts with `[A1, ..., AN, D]`, `N >= 0`.
    x_ref_min:  `Tensor` of same `dtype` as `x`.  The minimum value of the
      each batch of the (implicitly defined) reference `x_ref`.
      Shape broadcasts with `[A1, ..., AN]`, `N >= 0`.
    x_ref_max:  `Tensor` of same `dtype` as `x`.  The maximum value of the
      each batch of the (implicitly defined) reference `x_ref`.
      Shape broadcasts with `[A1, ..., AN]`, `N >= 0`.
    y_ref:  `Tensor` of same `dtype` as `x`.  The reference output values.
      `y_ref.shape[:axis]` broadcasts with the batch shape `[A1, ..., AN]`, and
      `y_ref.shape[axis:]` is `[C, B1, ..., BM]`, so the trailing dimensions
      index `C` reference values of a rank `M` `Tensor` (`M >= 0`).
    axis:  Scalar `Tensor` designating the dimension of `y_ref` that indexes
      values of the interpolation variable.
      Default value: `-1`, the rightmost axis.
    fill_value:  Determines what values output should take for `x` values that
      are below `x_ref_min` or above `x_ref_max`. `Tensor` or one of the strings
      "constant_extension" ==> Extend as constant function. "extrapolate" ==>
      Extrapolate in a linear fashion.
      Default value: `"constant_extension"`
    fill_value_below:  Optional override of `fill_value` for `x < x_ref_min`.
    fill_value_above:  Optional override of `fill_value` for `x > x_ref_max`.
    grid_regularizing_transform:  Optional transformation `g` which regularizes
      the implied spacing of the x reference points.  In other words, if
      provided, we assume `g(x_ref_i)` is a regular grid between `g(x_ref_min)`
      and `g(x_ref_max)`.
    name:  A name to prepend to created ops.
      Default value: `"batch_interp_regular_1d_grid"`.

  Returns:
    y_interp:  Interpolation between members of `y_ref`, at points `x`.
      `Tensor` of same `dtype` as `x`, and shape `[A1, ..., AN, D, B1, ..., BM]`

  Raises:
    ValueError:  If `fill_value` is not an allowed string.
    ValueError:  If `axis` is not a scalar.

  #### Examples

  Interpolate a function of one variable:

  ```python
  y_ref = tf.exp(tf.linspace(start=0., stop=10., 20))

  batch_interp_regular_1d_grid(
      x=[6.0, 0.5, 3.3], x_ref_min=0., x_ref_max=1., y_ref=y_ref)
  ==> approx [exp(6.0), exp(0.5), exp(3.3)]
  ```

  Interpolate a batch of functions of one variable.

  ```python
  # First batch member is an exponential function, second is a log.
  implied_x_ref = [tf.linspace(-3., 3.2, 200), tf.linspace(0.5, 3., 200)]
  y_ref = tf.stack(  # Shape [2, 200], 2 batches, 200 reference values per batch
      [tf.exp(implied_x_ref[0]), tf.log(implied_x_ref[1])], axis=0)

  x = [[-1., 1., 0.],  # Shape [2, 3], 2 batches, 3 values per batch.
       [1., 2., 3.]]

  y = tfp.math.batch_interp_regular_1d_grid(  # Shape [2, 3]
      x,
      x_ref_min=[-3., 0.5],
      x_ref_max=[3.2, 3.],
      y_ref=y_ref,
      axis=-1)

  # y[0] approx tf.exp(x[0])
  # y[1] approx tf.log(x[1])
  ```

  Interpolate a function of one variable on a log-spaced grid:

  ```python
  x_ref = tf.exp(tf.linspace(tf.log(1.), tf.log(100000.), num_pts))
  y_ref = tf.log(x_ref + x_ref**2)

  batch_interp_regular_1d_grid(x=[1.1, 2.2], x_ref_min=1., x_ref_max=100000.,
      y_ref, grid_regularizing_transform=tf.log)
  ==> [tf.log(1.1 + 1.1**2), tf.log(2.2 + 2.2**2)]
  ```

  """

  return _interp_regular_1d_grid_impl(
      x,
      x_ref_min,
      x_ref_max,
      y_ref,
      axis=axis,
      batch_y_ref=True,
      fill_value=fill_value,
      fill_value_below=fill_value_below,
      fill_value_above=fill_value_above,
      grid_regularizing_transform=grid_regularizing_transform,
      name=name or 'batch_interp_regular_1d_grid')


def _assert_ndims_statically(x, expect_ndims=None):
  ndims = x.shape.ndims
  if ndims is None:
    return
  if expect_ndims is not None and ndims != expect_ndims:
    raise ValueError('ndims must be {}.  Found: {}'.format(expect_ndims, ndims))


def _make_expand_x_fn_for_non_batch_interpolation(y_ref, axis):
  """Make func to expand left/right (of axis) dims of tensors shaped like x."""
  # This expansion is to help x broadcast with `y`, the output.
  # In the non-batch case, the output shape is going to be
  #   y_ref.shape[:axis] + x.shape + y_ref.shape[axis+1:]

  # Recall we made axis non-negative
  y_ref_shape = tf.shape(y_ref)
  y_ref_shape_left = y_ref_shape[:axis]
  y_ref_shape_right = y_ref_shape[axis + 1:]

  def expand_ends(x, broadcast=False):
    """Expand x so it can bcast w/ tensors of output shape."""
    # Assume out_shape = A + x.shape + B, and rank(A) = axis.
    # Expand with singletons with same rank as A, B.
    expanded_shape = tf.pad(
        tf.shape(x),
        paddings=[[axis, tf.size(y_ref_shape_right)]], constant_values=1)
    x_expanded = tf.reshape(x, expanded_shape)

    if broadcast:
      out_shape = tf.concat((
          y_ref_shape_left,
          tf.shape(x),
          y_ref_shape_right,
      ), axis=0)
      if x.dtype.is_bool:
        x_expanded = x_expanded | tf.cast(tf.zeros(out_shape), tf.bool)
      else:
        x_expanded += tf.zeros(out_shape, dtype=x.dtype)
    return x_expanded

  return expand_ends


def _make_expand_x_fn_for_batch_interpolation(y_ref, axis):
  """Make func to expand left/right (of axis) dims of tensors shaped like x."""
  # This expansion is to help x broadcast with `y`, the output.
  # In the batch case, the output shape is going to be
  #   Broadcast(y_ref.shape[:axis], x.shape[:-1]) +
  #   x.shape[-1:] +  y_ref.shape[axis+1:]

  # Recall we made axis non-negative
  y_ref_shape = tf.shape(y_ref)
  y_ref_shape_left = y_ref_shape[:axis]
  y_ref_shape_right = y_ref_shape[axis + 1:]

  def expand_right_dims(x, broadcast=False):
    """Expand x so it can bcast w/ tensors of output shape."""
    expanded_shape_left = tf.broadcast_dynamic_shape(
        tf.shape(x)[:-1], tf.ones([tf.size(y_ref_shape_left)], dtype=tf.int32))
    expanded_shape = tf.concat(
        (expanded_shape_left, tf.shape(x)[-1:],
         tf.ones([tf.size(y_ref_shape_right)], dtype=tf.int32)),
        axis=0)
    x_expanded = tf.reshape(x, expanded_shape)
    if broadcast:
      broadcast_shape_left = tf.broadcast_dynamic_shape(
          tf.shape(x)[:-1], y_ref_shape_left)
      broadcast_shape = tf.concat(
          (broadcast_shape_left, tf.shape(x)[-1:], y_ref_shape_right), axis=0)
      if x.dtype.is_bool:
        x_expanded = x_expanded | tf.cast(tf.zeros(broadcast_shape), tf.bool)
      else:
        x_expanded += tf.zeros(broadcast_shape, dtype=x.dtype)
    return x_expanded

  return expand_right_dims


def _batch_gather_with_broadcast(params, indices, axis):
  """Like batch_gather, but broadcasts to the left of axis."""
  # batch_gather assumes...
  #   params.shape =  [A1,...,AN, B1,...,BM]
  #   indices.shape = [A1,...,AN, C]
  # which gives output of shape
  #                   [A1,...,AN, C, B1,...,BM]
  # Here we broadcast dims of each to the left of `axis` in params, and left of
  # the rightmost dim in indices, e.g. we can
  # have
  #   params.shape =  [A1,...,AN, B1,...,BM]
  #   indices.shape = [a1,...,aN, C],
  # where Ai broadcasts with Ai.

  # leading_bcast_shape is the broadcast of [A1,...,AN] and [a1,...,aN].
  leading_bcast_shape = tf.broadcast_dynamic_shape(
      tf.shape(params)[:axis],
      tf.shape(indices)[:-1])
  params += tf.zeros(
      tf.concat((leading_bcast_shape, tf.shape(params)[axis:]), axis=0),
      dtype=params.dtype)
  indices += tf.zeros(
      tf.concat((leading_bcast_shape, tf.shape(indices)[-1:]), axis=0),
      dtype=indices.dtype)
  return tf.batch_gather(params, indices)
