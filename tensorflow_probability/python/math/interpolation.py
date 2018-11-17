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
]


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

  The interpolant is built from `M` reference values indexed by one dimension
  of `y_ref` (specified by the `axis` kwarg).

  If `y_ref` is a vector, then each value `y_ref[i]` is considered to be equal
  to `f(x_ref[i])`, for `M` (implicitly defined) reference values between
  `x_ref_min` and `x_ref_max`:

  ```none
  x_ref[i] = x_ref_min + i * (x_ref_max - x_ref_min) / (M - 1),
  i = 0, ..., M - 1.
  ```

  If `rank(y_ref) > 1`, then `y_ref` contains `M` reference values of a
  `rank(y_ref) - 1` rank tensor valued function of one variable.
  `x_ref` is a `Tensor` of values of that variable (any shape allowed).

  Args:
    x: Numeric `Tensor` The x-coordinates of the interpolated output values.
    x_ref_min:  `Tensor` of same `dtype` as `x`.  The minimum value of the
      (implicitly defined) reference `x_ref`.
    x_ref_max:  `Tensor` of same `dtype` as `x`.  The maximum value of the
      (implicitly defined) reference `x_ref`.
    y_ref:  `N-D` `Tensor` (`N > 0`) of same `dtype` as `x`.
      The reference output values.
    axis:  Scalar `Tensor` designating the dimension of `y_ref` that indexes
      values of the interpolation variable.
      Default value: `-1`, the rightmost axis.
    fill_value:  Determines what values output should take for `x` values that
      are below `x_ref_min` or above `x_ref_max`.
      `Tensor` or one of the strings
        "constant_extension" ==> Extend as constant function.
        "extrapolate" ==> Extrapolate in a linear fashion.
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
  y_ref = tf.exp(tf.linspace(start=0., stop=10., 20))

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

  Interpolate a function of one variable on a log-spaced grid:

  ```python
  x_ref = tf.exp(tf.linspace(tf.log(1.), tf.log(100000.), num_pts))
  y_ref = tf.log(x_ref + x_ref**2)

  interp_regular_1d_grid(x=[1.1, 2.2], x_ref_min=1., x_ref_max=100000., y_ref,
      grid_regularizing_transform=tf.log)
  ==> [tf.log(1.1 + 1.1**2), tf.log(2.2 + 2.2**2)]
  ```

  """

  with tf.name_scope(
      name,
      'interp_regular_1d_grid',
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

    axis = tf.convert_to_tensor(axis, name='axis', dtype=tf.int32)
    _assert_ndims_statically(axis, expect_ndims=0)
    axis = distribution_util.make_non_negative_axis(axis, tf.rank(y_ref))

    dtype = dtype_util.common_dtype([x, x_ref_min, x_ref_max, y_ref],
                                    preferred_dtype=tf.float32)
    x = tf.convert_to_tensor(x, name='x', dtype=dtype)
    x_ref_min = tf.convert_to_tensor(x_ref_min, name='x_ref_min', dtype=dtype)
    x_ref_max = tf.convert_to_tensor(x_ref_max, name='x_ref_max', dtype=dtype)
    y_ref = tf.convert_to_tensor(y_ref, name='y_ref', dtype=dtype)

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
    y_ref_below = tf.gather(y_ref, idx_below_int32, axis=axis)
    y_ref_above = tf.gather(y_ref, idx_above_int32, axis=axis)

    # out_shape = y_ref.shape[:axis] + x.shape + y_ref.shape[axis + 1:]
    out_shape = tf.shape(y_ref_below)

    # Return a convex combination.
    t = x_idx - idx_below

    t = _expand_ends(t, out_shape, axis)

    y = t * y_ref_above + (1 - t) * y_ref_below

    # Now begins a long excursion to fill values outside [x_min, x_max].

    # Re-insert NaN wherever x was NaN.
    y = tf.where(
        _expand_ends(nan_idx, out_shape, axis, broadcast=True),
        tf.fill(tf.shape(y), tf.constant(np.nan, y.dtype)), y)

    x_idx_unclipped = _expand_ends(
        x_idx_unclipped, out_shape, axis, broadcast=True)

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
        y_0 = tf.gather(y_ref, tf.zeros(tf.shape(x), dtype=tf.int32), axis=axis)
        y_1 = tf.gather(y_ref, tf.ones(tf.shape(x), dtype=tf.int32), axis=axis)
        x_delta = (x_ref_max - x_ref_min) / (ny - 1)
        x_factor = (x - x_ref_min) / x_delta
        x_factor = _expand_ends(x_factor, out_shape, axis, broadcast=True)
        y = tf.where(x_idx_unclipped < 0, y_0 + x_factor * (y_1 - y_0), y)
      else:
        y = tf.where(x_idx_unclipped < 0, fill_value_below + tf.zeros_like(y),
                     y)
      # Fill values above x_ref_min <==> x_idx_unclipped > ny - 1.
      if fill_value_above == 'constant_extension':
        pass  # Already handled by the clipping that created x_idx_unclipped.
      elif fill_value_above == 'extrapolate':
        ny_int32 = tf.shape(y_ref)[axis]
        y_n1 = tf.gather(y_ref, tf.fill(tf.shape(x), ny_int32 - 1), axis=axis)
        y_n2 = tf.gather(y_ref, tf.fill(tf.shape(x), ny_int32 - 2), axis=axis)
        x_delta = (x_ref_max - x_ref_min) / (ny - 1)
        x_factor = (x - x_ref_max) / x_delta
        x_factor = _expand_ends(x_factor, out_shape, axis, broadcast=True)
        y = tf.where(x_idx_unclipped > ny - 1,
                     y_n1 + x_factor * (y_n1 - y_n2), y)
      else:
        y = tf.where(x_idx_unclipped > ny - 1,
                     fill_value_above + tf.zeros_like(y), y)

    return y


def _assert_ndims_statically(x, expect_ndims=None):
  ndims = x.shape.ndims
  if ndims is None:
    return
  if expect_ndims is not None and ndims != expect_ndims:
    raise ValueError('ndims must be {}.  Found: {}'.format(expect_ndims, ndims))


def _expand_ends(x, out_shape, axis, broadcast=False):
  """Expand x with singletons so it can bcast w/ tensors of shape out_shape."""
  # Assume out_shape = A + x.shape + B, and rank(A) = axis.
  # Expand with singletons with same rank as A, B.
  new_shape = tf.concat((
      tf.ones([axis], dtype=tf.int32),
      tf.shape(x),
      tf.ones([tf.size(out_shape) - axis - tf.rank(x)], dtype=tf.int32),
  ),
                        axis=0)
  x_expanded = tf.reshape(x, new_shape)

  if broadcast:
    if x.dtype.is_bool:
      x_expanded = x_expanded | tf.cast(tf.zeros(out_shape), tf.bool)
    else:
      x_expanded += tf.zeros(out_shape, dtype=x.dtype)
  return x_expanded
