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

import itertools

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'interp_regular_1d_grid',
    'batch_interp_regular_1d_grid',
    'batch_interp_regular_nd_grid',
    'batch_interp_rectilinear_nd_grid',
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
  # Note: we do *not* make the no-batch version a special case of the batch
  # version, because that would an inefficient use of batch_gather with
  # unnecessarily broadcast args.
  with tf.name_scope(name or 'interp_regular_1d_grid_impl'):

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
                                    dtype_hint=tf.float32)
    x = tf.convert_to_tensor(x, name='x', dtype=dtype)

    x_ref_min = tf.convert_to_tensor(
        x_ref_min, name='x_ref_min', dtype=dtype)
    x_ref_max = tf.convert_to_tensor(
        x_ref_max, name='x_ref_max', dtype=dtype)
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

    axis = ps.convert_to_shape_tensor(axis, name='axis', dtype=tf.int32)
    axis = ps.non_negative_axis(axis, ps.rank(y_ref))
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
    nan_idx = tf.math.is_nan(x_idx_unclipped)
    zero = tf.zeros((), dtype=dtype)
    x_idx_unclipped = tf.where(nan_idx, zero, x_idx_unclipped)
    x_idx = tf.clip_by_value(x_idx_unclipped, zero, ny - 1)

    # Get the index above and below x_idx.
    # Naively we could set idx_below = floor(x_idx), idx_above = ceil(x_idx),
    # however, this results in idx_below == idx_above whenever x is on a grid.
    # This in turn results in y_ref_below == y_ref_above, and then the gradient
    # at this point is zero.  So here we 'jitter' one of idx_below, idx_above,
    # so that they are at different values.  This jittering does not affect the
    # interpolated value, but does make the gradient nonzero (unless of course
    # the y_ref values are the same).
    idx_below = tf.floor(x_idx)
    idx_above = tf.minimum(idx_below + 1, ny - 1)
    idx_below = tf.maximum(idx_above - 1, 0)

    # These are the values of y_ref corresponding to above/below indices.
    idx_below_int32 = tf.cast(idx_below, dtype=tf.int32)
    idx_above_int32 = tf.cast(idx_above, dtype=tf.int32)
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
    y = tf.where(nan_idx, tf.constant(np.nan, y.dtype), y)

    if not need_separate_fills:
      if fill_value == 'constant_extension':
        pass  # Already handled by clipping x_idx_unclipped.
      else:
        y = tf.where(
            (x_idx_unclipped < 0) | (x_idx_unclipped > ny - 1),
            fill_value, y)
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
        y = tf.where(x_idx_unclipped < 0, fill_value_below, y)
      # Fill values above x_ref_min <==> x_idx_unclipped > ny - 1.
      if fill_value_above == 'constant_extension':
        pass  # Already handled by the clipping that created x_idx_unclipped.
      elif fill_value_above == 'extrapolate':
        ny_int32 = tf.shape(y_ref)[axis]
        if batch_y_ref:
          y_n1 = tf.gather(y_ref, [tf.shape(y_ref)[axis] - 1], axis=axis)
          y_n2 = tf.gather(y_ref, [tf.shape(y_ref)[axis] - 2], axis=axis)
        else:
          y_n1 = tf.gather(
              y_ref, tf.fill(tf.shape(x), ny_int32 - 1), axis=axis)
          y_n2 = tf.gather(
              y_ref, tf.fill(tf.shape(x), ny_int32 - 2), axis=axis)
        x_delta = (x_ref_max - x_ref_min) / (ny - 1)
        x_factor = expand_x_fn((x - x_ref_max) / x_delta, broadcast=True)
        y = tf.where(x_idx_unclipped > ny - 1,
                     y_n1 + x_factor * (y_n1 - y_n2), y)
      else:
        y = tf.where(x_idx_unclipped > ny - 1, fill_value_above, y)

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
      values of the interpolation table.
      Default value: `-1`, the rightmost axis.
    fill_value:  Determines what values output should take for `x` values that
      are below `x_ref_min` or above `x_ref_max`. `Tensor` or one of the strings
      'constant_extension' ==> Extend as constant function. 'extrapolate' ==>
      Extrapolate in a linear fashion.
      Default value: `'constant_extension'`
    fill_value_below:  Optional override of `fill_value` for `x < x_ref_min`.
    fill_value_above:  Optional override of `fill_value` for `x > x_ref_max`.
    grid_regularizing_transform:  Optional transformation `g` which regularizes
      the implied spacing of the x reference points.  In other words, if
      provided, we assume `g(x_ref_i)` is a regular grid between `g(x_ref_min)`
      and `g(x_ref_max)`.
    name:  A name to prepend to created ops.
      Default value: `'interp_regular_1d_grid'`.

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
      x=[6.0, 0.5, 3.3], x_ref_min=0., x_ref_max=10., y_ref=y_ref)
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
    x: Numeric `Tensor` The x-coordinates of the interpolated output values for
      each batch.  Shape broadcasts with `[A1, ..., AN, D]`, `N >= 0`.
    x_ref_min:  `Tensor` of same `dtype` as `x`.  The minimum value of the each
      batch of the (implicitly defined) reference `x_ref`. Shape broadcasts with
      `[A1, ..., AN]`, `N >= 0`.
    x_ref_max:  `Tensor` of same `dtype` as `x`.  The maximum value of the each
      batch of the (implicitly defined) reference `x_ref`. Shape broadcasts with
      `[A1, ..., AN]`, `N >= 0`.
    y_ref:  `Tensor` of same `dtype` as `x`.  The reference output values.
      `y_ref.shape[:axis]` broadcasts with the batch shape `[A1, ..., AN]`, and
      `y_ref.shape[axis:]` is `[C, B1, ..., BM]`, so the trailing dimensions
        index `C` reference values of a rank `M` `Tensor` (`M >= 0`).
    axis:  Scalar `Tensor` designating the dimension of `y_ref` that indexes
      values of the interpolation table.
      Default value: `-1`, the rightmost axis.
    fill_value:  Determines what values output should take for `x` values that
      are below `x_ref_min` or above `x_ref_max`. `Tensor` or one of the strings
      'constant_extension' ==> Extend as constant function. 'extrapolate' ==>
      Extrapolate in a linear fashion.
      Default value: `'constant_extension'`
    fill_value_below:  Optional override of `fill_value` for `x < x_ref_min`.
    fill_value_above:  Optional override of `fill_value` for `x > x_ref_max`.
    grid_regularizing_transform:  Optional transformation `g` which regularizes
      the implied spacing of the x reference points.  In other words, if
      provided, we assume `g(x_ref_i)` is a regular grid between `g(x_ref_min)`
      and `g(x_ref_max)`.
    name:  A name to prepend to created ops.
      Default value: `'batch_interp_regular_1d_grid'`.

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
      x=[6.0, 0.5, 3.3], x_ref_min=0., x_ref_max=10., y_ref=y_ref)
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


def batch_interp_regular_nd_grid(x,
                                 x_ref_min,
                                 x_ref_max,
                                 y_ref,
                                 axis,
                                 fill_value='constant_extension',
                                 name=None):
  """Multi-linear interpolation on a regular (constant spacing) grid.

  Given [a batch of] reference values, this function computes a multi-linear
  interpolant and evaluates it on [a batch of] of new `x` values. This is a
  multi-dimensional generalization of [Bilinear Interpolation](
  https://en.wikipedia.org/wiki/Bilinear_interpolation).

  The interpolant is built from reference values indexed by `nd` dimensions
  of `y_ref`, starting at `axis`.

  The x grid span is defined by `x_ref_min`, `x_ref_max`. The number of grid
  points is inferred from the shape of `y_ref`.

  For example, take the case of a `2-D` scalar valued function and no leading
  batch dimensions.  In this case, `y_ref.shape = [C1, C2]` and `y_ref[i, j]`
  is the reference value corresponding to grid point

  ```
  [x_ref_min[0] + i * (x_ref_max[0] - x_ref_min[0]) / (C1 - 1),
   x_ref_min[1] + j * (x_ref_max[1] - x_ref_min[1]) / (C2 - 1)]
  ```

  In the general case, dimensions to the left of `axis` in `y_ref` are broadcast
  with leading dimensions in `x`, `x_ref_min`, `x_ref_max`.

  Args:
    x: Numeric `Tensor` The x-coordinates of the interpolated output values for
      each batch.  Shape `[..., D, nd]`, designating [a batch of] `D`
      coordinates in `nd` space.  `D` must be `>= 1` and is not a batch dim.
    x_ref_min:  `Tensor` of same `dtype` as `x`.  The minimum values of the
      (implicitly defined) reference `x_ref`.  Shape `[..., nd]`.
    x_ref_max:  `Tensor` of same `dtype` as `x`.  The maximum values of the
      (implicitly defined) reference `x_ref`.  Shape `[..., nd]`.
    y_ref:  `Tensor` of same `dtype` as `x`.  The reference output values. Shape
      `[..., C1, ..., Cnd, B1,...,BM]`, designating [a batch of] reference
      values indexed by `nd` dimensions, of a shape `[B1,...,BM]` valued
      function (for `M >= 0`).
    axis:  Scalar integer `Tensor`.  Dimensions `[axis, axis + nd)` of `y_ref`
      index the interpolation table.  E.g. `3-D` interpolation of a scalar
      valued function requires `axis=-3` and a `3-D` matrix valued function
      requires `axis=-5`.
    fill_value:  Determines what values output should take for `x` values that
      are below `x_ref_min` or above `x_ref_max`. Scalar `Tensor` or
      'constant_extension' ==> Extend as constant function.
      Default value: `'constant_extension'`
    name:  A name to prepend to created ops.
      Default value: `'batch_interp_regular_nd_grid'`.

  Returns:
    y_interp:  Interpolation between members of `y_ref`, at points `x`.
      `Tensor` of same `dtype` as `x`, and shape `[..., D, B1, ..., BM].`

  Exceptions will be raised if shapes are statically determined to be wrong.

  Raises:
    ValueError:  If `rank(x) < 2`.
    ValueError:  If `axis` is not a scalar.
    ValueError:  If `axis + nd > rank(y_ref)`.

  #### Examples

  Interpolate a function of one variable.

  ```python
  y_ref = tf.exp(tf.linspace(start=0., stop=10., num=20))

  tfp.math.batch_interp_regular_nd_grid(
      # x.shape = [3, 1], x_ref_min/max.shape = [1].  Trailing `1` for `1-D`.
      x=[[6.0], [0.5], [3.3]], x_ref_min=[0.], x_ref_max=[10.], y_ref=y_ref,
      axis=0)
  ==> approx [exp(6.0), exp(0.5), exp(3.3)]
  ```

  Interpolate a scalar function of two variables.

  ```python
  x_ref_min = [0., 0.]
  x_ref_max = [2 * np.pi, 2 * np.pi]

  # Build y_ref.
  x0s, x1s = tf.meshgrid(
      tf.linspace(x_ref_min[0], x_ref_max[0], num=100),
      tf.linspace(x_ref_min[1], x_ref_max[1], num=100),
      indexing='ij')

  def func(x0, x1):
    return tf.sin(x0) * tf.cos(x1)

  y_ref = func(x0s, x1s)

  x = 2 * np.pi * tf.random.uniform(shape=(10, 2))

  tfp.math.batch_interp_regular_nd_grid(x, x_ref_min, x_ref_max, y_ref, axis=-2)
  ==> tf.sin(x[:, 0]) * tf.cos(x[:, 1])
  ```

  """
  with tf.name_scope(name or 'batch_interp_regular_nd_grid'):
    dtype = dtype_util.common_dtype([x, x_ref_min, x_ref_max, y_ref],
                                    dtype_hint=tf.float32)

    # Arg checking.
    fill_value = _intake_fill_value_for_nd_interp(fill_value, dtype)

    # x.shape = [..., nd].
    x = tf.convert_to_tensor(x, name='x', dtype=dtype)
    _assert_ndims_statically(x, expect_ndims_at_least=2)

    # y_ref.shape = [..., C1,...,Cnd, B1,...,BM]
    y_ref = tf.convert_to_tensor(y_ref, name='y_ref', dtype=dtype)

    # x_ref_min.shape = [nd]
    x_ref_min = tf.convert_to_tensor(
        x_ref_min, name='x_ref_min', dtype=dtype)
    x_ref_max = tf.convert_to_tensor(
        x_ref_max, name='x_ref_max', dtype=dtype)
    _assert_ndims_statically(
        x_ref_min, expect_ndims_at_least=1, expect_static=True)
    _assert_ndims_statically(
        x_ref_max, expect_ndims_at_least=1, expect_static=True)

    # nd is the number of dimensions indexing the interpolation table, it's the
    # 'nd' in the function name.
    nd = tf.compat.dimension_value(x_ref_min.shape[-1])
    if nd is None:
      raise ValueError('`x_ref_min.shape[-1]` must be known statically.')
    tensorshape_util.assert_is_compatible_with(
        x_ref_max.shape[-1:], x_ref_min.shape[-1:])

    # Convert axis and check it statically.
    axis = _intake_axis_for_nd_interp(axis, y_ref, nd)

    x_batch_shape = ps.shape_slice(x, np.s_[:-2])
    x_ref_min_batch_shape = ps.shape_slice(x_ref_min, np.s_[:-1])
    x_ref_max_batch_shape = ps.shape_slice(x_ref_max, np.s_[:-1])
    y_ref_batch_shape = ps.shape_slice(y_ref, np.s_[:axis])

    # Do a brute-force broadcast of batch dims (add zeros).
    batch_shape = y_ref_batch_shape
    for tensor in [x_batch_shape, x_ref_min_batch_shape, x_ref_max_batch_shape]:
      batch_shape = ps.broadcast_shape(batch_shape, tensor)

    def _batch_shape_of_zeros_with_rightmost_singletons(n_singletons):
      """Return Tensor of zeros with some singletons on the rightmost dims."""
      return ps.concat([batch_shape, _int32ones(n_singletons)], axis=0)

    x = _broadcast_with(
        x, _batch_shape_of_zeros_with_rightmost_singletons(n_singletons=2))
    x_ref_min = _broadcast_with(
        x_ref_min,
        _batch_shape_of_zeros_with_rightmost_singletons(n_singletons=1))
    x_ref_max = _broadcast_with(
        x_ref_max,
        _batch_shape_of_zeros_with_rightmost_singletons(n_singletons=1))
    y_ref = _broadcast_with(
        y_ref,
        _batch_shape_of_zeros_with_rightmost_singletons(
            n_singletons=ps.rank(y_ref) - axis))

    # At this point,
    # x.shape = [A1, ..., An, D, nd], where n = batch_ndims
    # and
    # y_ref.shape = [A1, ..., An, C1, C2,..., Cnd, B1,...,BM]
    # y_ref[A1, ..., An, i1,...,ind] is a shape [B1,...,BM] Tensor with value
    # at index [i1,...,ind] in the interpolation table.
    #  and x_ref_max have shapes [A1, ..., An, nd].

    batch_ndims = ps.rank(x) - 2

    # ny[k] is number of y reference points in interp dim k.
    # It is used to indicate the dimension sizes.
    ny = tf.cast(
        # After broadcasting y_ref with x, slice(batch_ndims, batch_ndims + nd)
        # is the proper way to extract ny. Before broadcasting, use
        # slice(axis, axis + nd)
        ps.shape_slice(y_ref, np.s_[batch_ndims:batch_ndims + nd]), dtype)

    # Map [x_ref_min, x_ref_max] to [0, ny - 1].
    # This is the (fractional) index of x.
    # x_idx_unclipped[A1, ..., An, d, k] is the fractional index into dim k of
    # interpolation table for the dth x value.
    x_ref_min_expanded = tf.expand_dims(x_ref_min, axis=-2)
    x_ref_max_expanded = tf.expand_dims(x_ref_max, axis=-2)
    x_idx_unclipped = (ny - 1) * (x - x_ref_min_expanded) / (
        x_ref_max_expanded - x_ref_min_expanded)

    return _batch_interp_with_gather_nd(
        x=x,
        x_idx_unclipped=x_idx_unclipped,
        y_ref=y_ref,
        nd=nd,
        fill_value=fill_value,
        batch_ndims=batch_ndims)


def batch_interp_rectilinear_nd_grid(x,
                                     x_grid_points,
                                     y_ref,
                                     axis,
                                     fill_value='constant_extension',
                                     name=None):
  """Multi-linear interpolation on a rectilinear grid.

  Given [a batch of] reference values, this function computes a multi-linear
  interpolant and evaluates it on [a batch of] new `x` values. This is a
  multi-dimensional generalization of [Bilinear Interpolation](
  https://en.wikipedia.org/wiki/Bilinear_interpolation).

  The interpolant is built from reference values indexed by `nd` dimensions
  of `y_ref`, starting at `axis`.

  The x grid is defined by `1-D` points along each dimension. These points must
  be sorted, but may have unequal spacing.

  For example, take the case of a `2-D` scalar valued function and no leading
  batch dimensions.  In this case, `y_ref.shape = [C1, C2]` and `y_ref[i, j]`
  is the reference value corresponding to grid point

  ```[x_grid_points[0][i], x_grid_points[1][j]]```

  In the general case, dimensions to the left of `axis` in `y_ref` are broadcast
  with leading dimensions in `x`, and `x_grid_points[k]`, `k = 0, ..., nd - 1`.

  Args:
    x: Numeric `Tensor` The x-coordinates of the interpolated output values for
      each batch.  Shape `[..., D, nd]`, designating [a batch of] `D`
      coordinates in `nd` space.  `D` must be `>= 1` and is not a batch dim.
    x_grid_points: Tuple of dimension points. `x_grid_points[k]` are a shape
      `[..., Ck]` `Tensor` of the same dtype as `x` that must be sorted along
      the innermost (-1) axis. These represent [a batch of] points defining the
      `kth` dimension values.
    y_ref:  `Tensor` of same `dtype` as `x`.  The reference output values. Shape
      `[..., C1, ..., Cnd, B1,...,BM]`, designating [a batch of] reference
      values indexed by `nd` dimensions, of a shape `[B1,...,BM]` valued
      function (for `M >= 0`).
    axis:  Scalar integer `Tensor`.  Dimensions `[axis, axis + nd)` of `y_ref`
      index the interpolation table.  E.g. `3-D` interpolation of a scalar
      valued function requires `axis=-3` and a `3-D` matrix valued function
      requires `axis=-5`.
    fill_value:  Determines what values output should take for `x` values that
      are below/above the min/max values in `x_grid_points`.
      'constant_extension' ==> Extend as constant function.
      Default value: `'constant_extension'`
    name:  A name to prepend to created ops.
      Default value: `'batch_interp_rectilinear_nd_grid'`.

  Returns:
    y_interp:  Interpolation between members of `y_ref`, at points `x`.
      `Tensor` of same `dtype` as `x`, and shape `[..., D, B1, ..., BM].`

  Exceptions will be raised if shapes are statically determined to be wrong.

  Raises:
    ValueError:  If `rank(x) < 2`
    ValueError:  If `axis` is not a scalar.
    ValueError:  If `axis + nd > rank(y_ref)`.
    ValueError:  If `x_grid_points[k].shape[-1] != y_ref.shape[axis + k]`.

  #### Examples

  Interpolate a function of one variable.

  ```python
  x_grid = tf.linspace(0., 1., 20)**2   # Nonlinearly spaced
  y_ref = tf.exp(x_grid)

  tfp.math.batch_interp_rectilinear_nd_grid(
      # x.shape = [3, 1], with the trailing `1` for `1-D`.
      x=[[6.0], [0.5], [3.3]], x_grid_points=(x_grid,), y_ref=y_ref, axis=0)
  ==> approx [exp(6.0), exp(0.5), exp(3.3)]
  ```

  Interpolate a scalar function of two variables.

  ```python
  x0_grid = tf.linspace(0., 2 * np.pi, num=100),
  x1_grid = tf.linspace(0., 2 * np.pi, num=100),

  # Build y_ref.
  x0s, x1s = tf.meshgrid(x0_grid, x1_grid, indexing='ij')

  def func(x0, x1):
    return tf.sin(x0) * tf.cos(x1)

  y_ref = func(x0s, x1s)

  x = np.pi * tf.random.uniform(shape=(10, 2))

  tfp.math.batch_interp_regular_nd_grid(x, x_grid_points=(x0_grid, x1_grid),
                                        y_ref, axis=-2)
  ==> tf.sin(x[:, 0]) * tf.cos(x[:, 1])
  ```

  """
  with tf.name_scope(name or 'batch_interp_rectilinear_nd_grid'):
    if not isinstance(x_grid_points, tuple):
      raise ValueError(
          f'`x_grid_points` must be a tuple. Found {type(x_grid_points)}')

    dtype = dtype_util.common_dtype([x, y_ref] + list(x_grid_points),
                                    dtype_hint=tf.float32)

    # Arg checking.
    fill_value = _intake_fill_value_for_nd_interp(fill_value, dtype)

    # x.shape = [..., nd].
    x = tf.convert_to_tensor(x, name='x', dtype=dtype)
    _assert_ndims_statically(x, expect_ndims_at_least=2)

    # y_ref.shape = [..., C1,...,Cnd, B1,...,BM]
    y_ref = tf.convert_to_tensor(y_ref, name='y_ref', dtype=dtype)

    # x_ref_min.shape = [nd]
    x_grid_points = tuple(
        tf.convert_to_tensor(p, dtype=dtype) for p in x_grid_points)
    for p in x_grid_points:
      _assert_ndims_statically(p, expect_ndims_at_least=1, expect_static=True)

    # nd is the number of dimensions indexing the interpolation table, it's the
    # 'nd' in the function name.
    nd = len(x_grid_points)

    # Convert axis and check it statically.
    axis = _intake_axis_for_nd_interp(axis, y_ref, nd)

    # Check that the number of grid points implied by x_grid_points and y_ref
    # match.
    for k, p_k in enumerate(x_grid_points):
      nx_k = p_k.shape[-1]
      ny_k = y_ref.shape[axis + k]
      if ny_k is not None and ny_k is not None and nx_k != ny_k:
        raise ValueError(
            f'x_grid_points[{k}] contained {nx_k} points, which differed from '
            f'{ny_k}, the number of points in the {k}th table dimension of '
            f'y_ref.')

    x_batch_shape = ps.shape_slice(x, np.s_[:-2])
    x_grid_points_batch_shapes = list(
        ps.shape_slice(p, np.s_[:-1]) for p in x_grid_points)
    y_ref_batch_shape = ps.shape_slice(y_ref, np.s_[:axis])

    # Do a brute-force broadcast of batch dims (add zeros).
    batch_shape = y_ref_batch_shape
    for tensor in [x_batch_shape] + x_grid_points_batch_shapes:
      batch_shape = ps.broadcast_shape(batch_shape, tensor)

    def _batch_shape_of_zeros_with_rightmost_singletons(n_singletons):
      """Return Tensor of zeros with some singletons on the rightmost dims."""
      return ps.concat([batch_shape, _int32ones(n_singletons)], axis=0)

    x = _broadcast_with(
        x, _batch_shape_of_zeros_with_rightmost_singletons(n_singletons=2))
    x_grid_points = tuple(
        _broadcast_with(
            p, _batch_shape_of_zeros_with_rightmost_singletons(n_singletons=1))
        for p in x_grid_points)
    y_ref = _broadcast_with(
        y_ref,
        _batch_shape_of_zeros_with_rightmost_singletons(
            n_singletons=ps.rank(y_ref) - axis))

    # At this point,
    # x.shape = [A1, ..., An, D, nd], where n = batch_ndims
    # and
    # y_ref.shape = [A1, ..., An, C1, C2,..., Cnd, B1,...,BM]
    # y_ref[A1, ..., An, i1,...,ind] is a shape [B1,...,BM] Tensor with value
    # at index [i1,...,ind] in the interpolation table.
    # and `p_k = x_grid_points[k]` has shape [A1, ..., An, Ck].

    batch_ndims = ps.rank(x) - 2

    # ny[k] is number of y reference points in interp dim k.
    # It is used to indicate the dimension sizes...
    # It could also be called nx, if we actually materialized a grid of x
    # points. We don't though, as x points are given only as axis values.
    ny = tf.cast(
        ps.shape_slice(y_ref, np.s_[batch_ndims:batch_ndims + nd]), tf.int32)

    # Map the `kth` point `x_grid_points[k]` to [0, ny[k] - 1].
    # This is the (fractional) index of x, "unclipped" meaning it may take
    # values outside [0, ..., ny[k]].
    # x_idx_unclipped[A1, ..., An, d, k] is the fractional index into dim k of
    # interpolation table for the dth x value.
    x_idx_unclipped = []
    for k, p_k in enumerate(x_grid_points):
      # x_k and x_k_clipped shape [A1, ..., An, D].
      # Clip x_k below...no need to clip above since, in the place it is used
      # below, we have a tf.minimum(ny[k] - 1,...)
      x_k = x[..., k]
      x_k_clipped = tf.maximum(x_k, tf.reduce_min(p_k, axis=-1, keepdims=True))

      # This construction of indices ensures that idx_below_k < idx_above_k.
      # In particular, the use of x_k_clipped ensures this, even if x_k is OOB.
      idx_above_k = tf.minimum(
          ny[k] - 1, tf.searchsorted(p_k, x_k_clipped, side='right'))
      idx_below_k = tf.maximum(idx_above_k - 1, 0)
      x_above_k = tf.gather(p_k, idx_above_k, batch_dims=batch_ndims)
      x_below_k = tf.gather(p_k, idx_below_k, batch_dims=batch_ndims)

      # The use of x_k (not clipped) here allows x_idx_unclipped to be < 0 or >
      # ny[k] - 1.
      x_idx_unclipped.append(
          tf.cast(idx_below_k, dtype) + (x_k - x_below_k) /
          (x_above_k - x_below_k))

    x_idx_unclipped = tf.stack(x_idx_unclipped, axis=-1)

    return _batch_interp_with_gather_nd(
        x=x,
        x_idx_unclipped=x_idx_unclipped,
        y_ref=y_ref,
        nd=nd,
        fill_value=fill_value,
        batch_ndims=batch_ndims)


def _batch_interp_with_gather_nd(x, x_idx_unclipped, y_ref, nd, fill_value,
                                 batch_ndims):
  """Batch interpolation starting with indices."""
  dtype = x.dtype
  # Wherever x is NaN, x_idx_unclipped will be NaN as well.
  # Keep track of the nan indices here (so we can impute NaN later).
  # Also eliminate any NaN indices, since there is not NaN in 32bit.
  nan_idx = tf.math.is_nan(x_idx_unclipped)
  x_idx_unclipped = tf.where(nan_idx, tf.cast(0., dtype=dtype), x_idx_unclipped)

  # ny[k] is number of y reference points in interp dim k.
  # It is used to indicate the dimension sizes.
  ny = tf.cast(
      ps.shape_slice(y_ref, np.s_[batch_ndims:batch_ndims + nd]), dtype)

  # x_idx.shape = [A1, ..., An, D, nd]
  x_idx = tf.clip_by_value(x_idx_unclipped, tf.zeros((), dtype=dtype), ny - 1)

  # Get the index above and below x_idx.
  # Naively we could set idx_below = floor(x_idx), idx_above = ceil(x_idx),
  # however, this results in idx_below == idx_above whenever x is on a grid.
  # This in turn results in y_ref_below == y_ref_above, and then the gradient
  # at this point is zero.  So here we 'jitter' one of idx_below, idx_above,
  # so that they are at different values.  This jittering does not affect the
  # interpolated value, but does make the gradient nonzero (unless of course
  # the y_ref values are the same).
  idx_below = tf.floor(x_idx)
  idx_above = tf.minimum(idx_below + 1, ny - 1)
  idx_below = tf.maximum(idx_above - 1, 0)

  # These are the values of y_ref corresponding to above/below indices.
  # idx_below_int32.shape = x.shape[:-1] + [nd]
  idx_below_int32 = tf.cast(idx_below, dtype=tf.int32)
  idx_above_int32 = tf.cast(idx_above, dtype=tf.int32)

  # idx_below_list is a length nd list of shape x.shape[:-1] int32 tensors.
  idx_below_list = tf.unstack(idx_below_int32, axis=-1)
  idx_above_list = tf.unstack(idx_above_int32, axis=-1)

  # Use t to get a convex combination of the below/above values.
  # t.shape = [A1, ..., An, D, nd]
  t = x_idx - idx_below

  # x, and tensors shaped like x, need to be added to, and selected with
  # (using tf.where) the output y.  This requires appending singletons.
  def _expand_x_fn(tensor):
    # Reshape tensor to tensor.shape + [1] * M.
    extended_shape = ps.concat(
        [
            ps.shape(tensor),
            ps.ones_like(
                ps.convert_to_shape_tensor(
                    ps.shape_slice(y_ref, np.s_[batch_ndims + nd:])))
        ],
        axis=0,
    )
    return tf.reshape(tensor, extended_shape)

  # Now, t.shape = [A1, ..., An, D, nd] + [1] * (rank(y_ref) - nd - batch_ndims)
  t = _expand_x_fn(t)
  s = 1 - t

  # Re-insert NaN wherever x was NaN.
  nan_idx = _expand_x_fn(nan_idx)
  t = tf.where(nan_idx, tf.constant(np.nan, dtype), t)

  # Initialize y and accumulate in a loop. An alternative would be to store
  # summands in a list. However, without XLA compilation, the "list method"
  # results in storage of 2^nd (possibly large) summands, which could OOM.
  # Thus, if you are not XLA compiling, the method below is highly preferred.
  # With XLA compilation, both methods are equivalent.
  y = tf.zeros((), dtype=dtype)

  # Our work above has located x's fractional index inside a cube of above/below
  # indices. The distance to the below indices is t, and to the above indices
  # is s.
  # Drawing lines from x to the cube walls, we get 2**nd smaller cubes. Each
  # term in the result is a product of a reference point, gathered from y_ref,
  # multiplied by a volume.  The volume is that of the cube opposite to the
  # reference point.  E.g. if the reference point is below x in every axis, the
  # volume is that of the cube with corner above x in every axis, s[0]*...*s[nd]
  # We could probably do this with one massive gather, but that would be very
  # unreadable and un-debuggable.  It also would create a large Tensor.
  for zero_ones_list in _binary_count(nd):
    gather_from_y_ref_idx = []
    opposite_volume_t_idx = []
    opposite_volume_s_idx = []
    for k, zero_or_one in enumerate(zero_ones_list):
      if zero_or_one == 0:
        # If the kth iterate has zero_or_one = 0,
        # Will gather from the 'below' reference point along axis k.
        gather_from_y_ref_idx.append(idx_below_list[k])
        # Now append the index to gather for computing opposite_volume.
        # This could be done by initializing opposite_volume to 1, then here:
        #  opposite_volume *= tf.gather(s, indices=k, axis=tf.rank(x) - 1)
        # but that puts a gather in the 'inner loop.'  Better to append the
        # index and do one larger gather down below.
        opposite_volume_s_idx.append(k)
      else:
        gather_from_y_ref_idx.append(idx_above_list[k])
        # Append an index to gather, having the same effect as
        #   opposite_volume *= tf.gather(t, indices=k, axis=tf.rank(x) - 1)
        opposite_volume_t_idx.append(k)

    # Compute opposite_volume (volume of cube opposite the ref point):
    # Recall t.shape = s.shape = [D, nd] + [1, ..., 1]
    # Gather from t and s along the 'nd' axis, which is rank(x) - 1.
    ov_axis = ps.cast(ps.rank(x) - 1, tf.int32)
    opposite_volume = (
        tf.reduce_prod(
            tf.gather(
                t, indices=tf.cast(opposite_volume_t_idx, dtype=tf.int32),
                axis=ov_axis),
            axis=ov_axis) *
        tf.reduce_prod(
            tf.gather(
                s, indices=tf.cast(opposite_volume_s_idx, dtype=tf.int32),
                axis=ov_axis),
            axis=ov_axis)
    )  # pyformat: disable

    y_ref_pt = tf.gather_nd(
        y_ref, tf.stack(gather_from_y_ref_idx, axis=-1), batch_dims=batch_ndims)

    y = y + y_ref_pt * opposite_volume

  if tf.debugging.is_numeric_tensor(fill_value):
    # Recall x_idx_unclipped.shape = [D, nd],
    # so here we check if it was out of bounds in any of the nd dims.
    # Thus, oob_idx.shape = [D].
    oob_idx = tf.reduce_any(
        (x_idx_unclipped < 0) | (x_idx_unclipped > ny - 1), axis=-1)

    # Now, y.shape = [D, B1,...,BM], so we'll have to broadcast oob_idx.

    oob_idx = _expand_x_fn(oob_idx)  # Shape [D, 1,...,1]
    oob_idx = _broadcast_with(oob_idx, ps.shape(y))
    y = tf.where(oob_idx, fill_value, y)
  return y


def _assert_ndims_statically(x,
                             expect_ndims=None,
                             expect_ndims_at_least=None,
                             expect_static=False):
  """Assert that Tensor x has expected number of dimensions."""
  ndims = tensorshape_util.rank(x.shape)
  if ndims is None:
    if expect_static:
      raise ValueError('Expected static ndims. Found: {}'.format(x))
    return
  if expect_ndims is not None and ndims != expect_ndims:
    raise ValueError('ndims must be {}.  Found: {}'.format(expect_ndims, ndims))
  if expect_ndims_at_least is not None and ndims < expect_ndims_at_least:
    raise ValueError('ndims must be at least {}. Found {}'.format(
        expect_ndims_at_least, ndims))


def _intake_fill_value_for_nd_interp(fill_value, dtype):
  """Check `fill_value` and return after converting numeric to tensor."""
  if isinstance(fill_value, str):
    if fill_value != 'constant_extension':
      raise ValueError(
          'A fill value ({}) was not an allowed string ({})'.format(
              fill_value, 'constant_extension'))
  else:
    fill_value = tf.convert_to_tensor(
        fill_value, name='fill_value', dtype=dtype)
    _assert_ndims_statically(fill_value, expect_ndims=0)
  return fill_value


def _intake_axis_for_nd_interp(axis, y_ref, nd):
  """Convert `axis` to its non-negative value and return after validation."""
  axis = ps.convert_to_shape_tensor(axis, dtype=tf.int32, name='axis')
  axis = ps.non_negative_axis(axis, ps.rank(y_ref))
  tensorshape_util.assert_has_rank(axis.shape, 0)
  axis_ = tf.get_static_value(axis)
  y_ref_rank_ = tf.get_static_value(tf.rank(y_ref))
  if axis_ is not None and y_ref_rank_ is not None:
    if axis_ + nd > y_ref_rank_:
      raise ValueError(
          'Since dims `[axis, axis + nd)` index the interpolation table, we '
          'must have `axis + nd <= rank(y_ref)`.  Found: '
          '`axis`: {},  rank(y_ref): {}, and inferred `nd` from trailing '
          'dimensions of `x_ref_min` to be {}.'.format(
              axis_, y_ref_rank_, nd))
  return axis


def _make_expand_x_fn_for_non_batch_interpolation(y_ref, axis):
  """Make func to expand left/right (of axis) dims of tensors shaped like x."""
  # This expansion is to help x broadcast with `y`, the output.
  # In the non-batch case, the output shape is going to be
  #   y_ref.shape[:axis] + x.shape + y_ref.shape[axis+1:]

  # Recall we made axis non-negative
  y_ref_shape_left = ps.shape_slice(y_ref, np.s_[:axis])
  y_ref_shape_right = ps.shape_slice(y_ref, np.s_[axis + 1:])

  def expand_ends(x, broadcast=False):
    """Expand x so it can bcast w/ tensors of output shape."""
    # Assume out_shape = A + x.shape + B, and rank(A) = axis.
    # Expand with singletons with same rank as A, B.
    expanded_shape = ps.concat(
        [ps.shape(x), _int32ones(ps.size(y_ref_shape_right))], axis=0)
    x_expanded = tf.reshape(x, expanded_shape)

    if broadcast:
      out_bcaster = ps.concat((
          y_ref_shape_left,
          _int32ones(ps.rank(x)),
          y_ref_shape_right,
      ),
                              axis=0)
      x_expanded = _broadcast_with(x_expanded, out_bcaster)
    return x_expanded

  return expand_ends


def _make_expand_x_fn_for_batch_interpolation(y_ref, axis):
  """Make func to expand left/right (of axis) dims of tensors shaped like x."""
  # This expansion is to help x broadcast with `y`, the output.
  # In the batch case, the output shape is going to be
  #   Broadcast(y_ref.shape[:axis], x.shape[:-1]) +
  #   x.shape[-1:] +  y_ref.shape[axis+1:]

  # Recall we made axis non-negative
  y_ref_shape_left = ps.shape_slice(y_ref, np.s_[:axis])
  y_ref_shape_right = ps.shape_slice(y_ref, np.s_[axis + 1:])

  def expand_right_dims(x, broadcast=False):
    """Expand x so it can bcast w/ tensors of output shape."""
    x_shape_left = ps.shape_slice(x, np.s_[:-1])
    x_shape_right = ps.shape_slice(x, np.s_[-1:])
    expanded_shape_left = ps.broadcast_shape(
        x_shape_left, _int32ones(ps.size(y_ref_shape_left)))
    expanded_shape = ps.concat((expanded_shape_left, x_shape_right,
                                _int32ones(ps.size(y_ref_shape_right))),
                               axis=0)
    x_expanded = tf.reshape(x, expanded_shape)
    if broadcast:
      broadcast_shape_left = ps.broadcast_shape(x_shape_left, y_ref_shape_left)
      ones_like_x_shape_right = _int32ones(1)
      broadcast_shape = ps.concat(
          (broadcast_shape_left, ones_like_x_shape_right, y_ref_shape_right),
          axis=0)
      x_expanded = _broadcast_with(x_expanded, broadcast_shape)
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
  # where ai broadcasts with Ai.

  # leading_bcast_shape is the broadcast of [A1,...,AN] and [a1,...,aN].
  leading_bcast_shape = ps.broadcast_shape(
      ps.shape_slice(params, np.s_[:axis]), ps.shape_slice(indices, np.s_[:-1]))
  params = _broadcast_with(
      params,
      ps.concat((leading_bcast_shape, _int32ones(ps.rank(params) - axis)),
                axis=0))
  indices = _broadcast_with(
      indices, ps.concat((leading_bcast_shape, _int32ones(1)), axis=0))
  return tf.gather(
      params, indices, batch_dims=tensorshape_util.rank(indices.shape) - 1)


def _binary_count(n):
  """Count `n` binary digits from [0...0] to [1...1]."""
  return list(itertools.product([0, 1], repeat=n))


def _broadcast_with(tensor, shape):
  """Like broadcast_to, but allows singletons in the destination shape."""
  res = tf.broadcast_to(
      tensor, ps.broadcast_shape(ps.shape(tensor), shape))
  # We need this done explicitly because ps.broadcast_shape cannot deal with
  # partially specified shapes.
  tensorshape_util.set_shape(
      res,
      tf.broadcast_static_shape(tensor.shape,
                                tf.TensorShape(tf.get_static_value(shape))))
  return res


def _int32ones(n):
  return ps.ones(n, dtype=tf.int32)
