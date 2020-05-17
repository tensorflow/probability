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
"""Functions for computing statistics of samples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'count_integers',
    'find_bins',
    'histogram',
    'percentile',
    'quantiles',
]


# TODO(b/124015136) This function isn't necessary once tf.math.bincount supports
# an `axis` kwarg.
def count_integers(arr,
                   weights=None,
                   minlength=None,
                   maxlength=None,
                   axis=None,
                   dtype=tf.int32,
                   name=None):
  """Counts the number of occurrences of each value in an integer array `arr`.

  Works like `tf.math.bincount`, but provides an `axis` kwarg that specifies
  dimensions to reduce over.  With
    `~axis = [i for i in range(arr.ndim) if i not in axis]`,
  this function returns a `Tensor` of shape `[K] + arr.shape[~axis]`.

  If `minlength` and `maxlength` are not given, `K = tf.reduce_max(arr) + 1`
  if `arr` is non-empty, and 0 otherwise.
  If `weights` are non-None, then index `i` of the output stores the sum of the
  value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Args:
    arr: An `int32` `Tensor` of non-negative values.
    weights: If non-None, must be the same shape as arr. For each value in
      `arr`, the bin will be incremented by the corresponding weight instead of
      1.
    minlength: If given, ensures the output has length at least `minlength`,
      padding with zeros at the end if necessary.
    maxlength: If given, skips values in `arr` that are equal or greater than
      `maxlength`, ensuring that the output has length at most `maxlength`.
    axis: A `0-D` or `1-D` `int32` `Tensor` (with static values) designating
      dimensions in `arr` to reduce over.
      `Default value:` `None`, meaning reduce over all dimensions.
    dtype: If `weights` is None, determines the type of the output bins.
    name: A name scope for the associated operations (optional).

  Returns:
    A vector with the same dtype as `weights` or the given `dtype`. The bin
    values.
  """
  with tf.name_scope(name or 'count_integers'):
    if axis is None:
      return tf.math.bincount(
          arr,
          weights=weights,
          minlength=minlength,
          maxlength=maxlength,
          dtype=dtype)

    arr = tf.convert_to_tensor(arr, dtype=tf.int32, name='arr')
    arr_ndims = _get_static_ndims(arr, expect_static=True)

    axis = _make_static_axis_non_negative_list(axis, arr_ndims)

    # ~axis from docstring.  Dims in arr that are not in axis.
    not_axis = sorted(set(range(arr_ndims)).difference(axis))

    # If we're reducing over everything, just use standard bincount.
    if not not_axis:
      return tf.math.bincount(
          arr,
          weights=weights,
          minlength=minlength,
          maxlength=maxlength,
          dtype=dtype)

    # Move dims in ~axis to the left, so we can tf.map_fn bincount over them,
    # Producing counts for every index I in ~axis.
    # Thus, flat_arr is not totally flat, it just has the dims in ~axis
    # flattened.
    flat_arr = _move_dims_to_flat_end(arr, not_axis, arr_ndims, right_end=False)

    # tf.map_fn over dim 0.
    if weights is None:

      def one_bincount(arr_slice):
        return tf.math.bincount(
            arr_slice,
            weights=None,
            minlength=minlength,
            maxlength=maxlength,
            dtype=dtype)

      flat_counts = tf.map_fn(one_bincount, elems=flat_arr,
                              fn_output_signature=dtype)
    else:
      weights = tf.convert_to_tensor(weights, name='weights')
      _get_static_ndims(weights, expect_static=True, expect_ndims=arr_ndims)
      flat_weights = _move_dims_to_flat_end(
          weights, not_axis, arr_ndims, right_end=False)

      def one_bincount(arr_and_weights_slices):
        arr_slice, weights_slice = arr_and_weights_slices
        return tf.math.bincount(
            arr_slice,
            weights=weights_slice,
            minlength=minlength,
            maxlength=maxlength,
            dtype=dtype)

      flat_counts = tf.map_fn(
          one_bincount, elems=[flat_arr, flat_weights],
          fn_output_signature=weights.dtype)

    # flat_counts.shape = [prod(~axis), K], because map_fn stacked on axis 0.
    # bincount needs to have the K bins in axis 0, so transpose...
    flat_counts_t = tf.transpose(a=flat_counts, perm=[1, 0])

    # Throw in this assert, to ensure shape assumptions are correct.
    _get_static_ndims(flat_counts_t, expect_ndims=2, expect_static=True)

    # not_axis_shape = arr.shape[~axis]
    not_axis_shape = tf.gather(tf.shape(arr), indices=not_axis)

    # The first index of flat_counts_t indexes bins 0,..,K-1, the rest are ~axis
    out_shape = tf.concat([[-1], not_axis_shape], axis=0)

    return tf.reshape(flat_counts_t, out_shape)


def find_bins(x,
              edges,
              extend_lower_interval=False,
              extend_upper_interval=False,
              dtype=None,
              name=None):
  """Bin values into discrete intervals.

  Given `edges = [c0, ..., cK]`, defining intervals
  `I0 = [c0, c1)`, `I1 = [c1, c2)`, ..., `I_{K-1} = [c_{K-1}, cK]`,
  This function returns `bins`, such that:
  `edges[bins[i]] <= x[i] < edges[bins[i] + 1]`.

  Args:
    x:  Numeric `N-D` `Tensor` with `N > 0`.
    edges:  `Tensor` of same `dtype` as `x`.  The first dimension indexes edges
      of intervals.  Must either be `1-D` or have
      `x.shape[1:] == edges.shape[1:]`.  If `rank(edges) > 1`, `edges[k]`
      designates a shape `edges.shape[1:]` `Tensor` of bin edges for the
      corresponding dimensions of `x`.
    extend_lower_interval:  Python `bool`.  If `True`, extend the lowest
      interval `I0` to `(-inf, c1]`.
    extend_upper_interval:  Python `bool`.  If `True`, extend the upper
      interval `I_{K-1}` to `[c_{K-1}, +inf)`.
    dtype: The output type (`int32` or `int64`). `Default value:` `x.dtype`.
      This effects the output values when `x` is below/above the intervals,
      which will be `-1/K+1` for `int` types and `NaN` for `float`s.
      At indices where `x` is `NaN`, the output values will be `0` for `int`
      types and `NaN` for floats.
    name:  A Python string name to prepend to created ops. Default: 'find_bins'

  Returns:
    bins: `Tensor` with same `shape` as `x` and `dtype`.
      Has whole number values.  `bins[i] = k` means the `x[i]` falls into the
      `kth` bin, ie, `edges[bins[i]] <= x[i] < edges[bins[i] + 1]`.

  Raises:
    ValueError:  If `edges.shape[0]` is determined to be less than 2.

  #### Examples

  Cut a `1-D` array

  ```python
  x = [0., 5., 6., 10., 20.]
  edges = [0., 5., 10.]
  tfp.stats.find_bins(x, edges)
  ==> [0., 0., 1., 1., np.nan]
  ```

  Cut `x` into its deciles

  ```python
  x = tf.random_uniform(shape=(100, 200))
  decile_edges = tfp.stats.quantiles(x, num_quantiles=10)
  bins = tfp.stats.find_bins(x, edges=decile_edges)
  bins.shape
  ==> (100, 200)
  tf.reduce_mean(bins == 0.)
  ==> approximately 0.1
  tf.reduce_mean(bins == 1.)
  ==> approximately 0.1
  ```

  """
  # TFP users may be surprised to see the "action" in the leftmost dim of
  # edges, rather than the rightmost (event) dim.  Why?
  # 1. Most likely you created edges by getting quantiles over samples, and
  #    quantile/percentile return these edges in the leftmost (sample) dim.
  # 2. Say you have event_shape = [5], then we expect the bin will be different
  #    for all 5 events, so the index of the bin should not be in the event dim.
  with tf.name_scope(name or 'find_bins'):
    in_type = dtype_util.common_dtype([x, edges], dtype_hint=tf.float32)
    edges = tf.convert_to_tensor(edges, name='edges', dtype=in_type)
    x = tf.convert_to_tensor(x, name='x', dtype=in_type)

    if (tf.compat.dimension_value(edges.shape[0]) is not None and
        tf.compat.dimension_value(edges.shape[0]) < 2):
      raise ValueError(
          'First dimension of `edges` must have length > 1 to index 1 or '
          'more bin. Found: {}'.format(edges.shape))

    flattening_x = (tensorshape_util.rank(edges.shape) == 1 and
                    tensorshape_util.rank(x.shape) > 1)

    if flattening_x:
      x_orig_shape = tf.shape(x)
      x = tf.reshape(x, [-1])

    if dtype is None:
      dtype = in_type
    dtype = tf.as_dtype(dtype)

    # Move first dims into the rightmost.
    x_permed = distribution_util.rotate_transpose(x, shift=-1)
    edges_permed = distribution_util.rotate_transpose(edges, shift=-1)

    # If...
    #   x_permed = [0, 1, 6., 10]
    #   edges = [0, 5, 10.]
    #   ==> almost_output = [0, 1, 2, 2]
    searchsorted_type = dtype if dtype in [tf.int32, tf.int64] else None
    almost_output_permed = tf.searchsorted(
        sorted_sequence=edges_permed,
        values=x_permed,
        side='right',
        out_type=searchsorted_type)
    # Move the rightmost dims back to the leftmost.
    almost_output = tf.cast(
        distribution_util.rotate_transpose(almost_output_permed, shift=1),
        dtype)

    # In above example, we want [0, 0, 1, 1], so correct this here.
    bins = tf.clip_by_value(almost_output - 1, tf.cast(0, dtype),
                            tf.cast(tf.shape(edges)[0] - 2, dtype))

    if not extend_lower_interval:
      low_fill = np.nan if dtype_util.is_floating(dtype) else -1
      bins = tf.where(x < tf.expand_dims(edges[0], 0),
                      tf.cast(low_fill, dtype), bins)

    if not extend_upper_interval:
      up_fill = (np.nan if dtype_util.is_floating(dtype)
                 else tf.shape(edges)[0] - 1)
      bins = tf.where(x > tf.expand_dims(edges[-1], 0),
                      tf.cast(up_fill, dtype), bins)

    if flattening_x:
      bins = tf.reshape(bins, x_orig_shape)

    return bins


def histogram(x,
              edges,
              axis=None,
              extend_lower_interval=False,
              extend_upper_interval=False,
              dtype=None,
              name=None):
  """Count how often `x` falls in intervals defined by `edges`.

  Given `edges = [c0, ..., cK]`, defining intervals
  `I0 = [c0, c1)`, `I1 = [c1, c2)`, ..., `I_{K-1} = [c_{K-1}, cK]`,
  This function counts how often `x` falls into each interval.

  Values of `x` outside of the intervals cause errors.  Consider using
  `extend_lower_interval`, `extend_upper_interval` to deal with this.

  Args:
    x:  Numeric `N-D` `Tensor` with `N > 0`.  If `axis` is not
      `None`, must have statically known number of dimensions. The
      `axis` kwarg determines which dimensions index iid samples.
      Other dimensions of `x` index "events" for which we will compute different
      histograms.
    edges:  `Tensor` of same `dtype` as `x`.  The first dimension indexes edges
      of intervals.  Must either be `1-D` or have `edges.shape[1:]` the same
      as the dimensions of `x` excluding `axis`.
      If `rank(edges) > 1`, `edges[k]` designates a shape `edges.shape[1:]`
      `Tensor` of interval edges for the corresponding dimensions of `x`.
    axis:  Optional `0-D` or `1-D` integer `Tensor` with constant
      values. The axis in `x` that index iid samples.
      `Default value:` `None` (treat every dimension as sample dimension).
    extend_lower_interval:  Python `bool`.  If `True`, extend the lowest
      interval `I0` to `(-inf, c1]`.
    extend_upper_interval:  Python `bool`.  If `True`, extend the upper
      interval `I_{K-1}` to `[c_{K-1}, +inf)`.
    dtype: The output type (`int32` or `int64`). `Default value:` `x.dtype`.
    name:  A Python string name to prepend to created ops.
      `Default value:` 'histogram'

  Returns:
    counts: `Tensor` of type `dtype` and, with
      `~axis = [i for i in range(arr.ndim) if i not in axis]`,
      `counts.shape = [edges.shape[0]] + x.shape[~axis]`.
      With `I` a multi-index into `~axis`, `counts[k][I]` is the number of times
      event(s) fell into the `kth` interval of `edges`.

  #### Examples

  ```python
  # x.shape = [1000, 2]
  # x[:, 0] ~ Uniform(0, 1), x[:, 1] ~ Uniform(1, 2).
  x = tf.stack([tf.random_uniform([1000]), 1 + tf.random_uniform([1000])],
               axis=-1)

  # edges ==> bins [0, 0.5), [0.5, 1.0), [1.0, 1.5), [1.5, 2.0].
  edges = [0., 0.5, 1.0, 1.5, 2.0]

  tfp.stats.histogram(x, edges)
  ==> approximately [500, 500, 500, 500]

  tfp.stats.histogram(x, edges, axis=0)
  ==> approximately [[500, 500, 0, 0], [0, 0, 500, 500]]
  ```

  """
  with tf.name_scope(name or 'histogram'):

    # Tensor conversions.
    in_dtype = dtype_util.common_dtype([x, edges], dtype_hint=tf.float32)

    x = tf.convert_to_tensor(x, name='x', dtype=in_dtype)
    edges = tf.convert_to_tensor(edges, name='edges', dtype=in_dtype)

    # Move dims in axis to the left end as one flattened dim.
    # After this, x.shape = [n_samples] + E.
    if axis is None:
      x = tf.reshape(x, shape=[-1])
    else:
      x_ndims = _get_static_ndims(
          x, expect_static=True, expect_ndims_at_least=1)
      axis = _make_static_axis_non_negative_list(axis, x_ndims)
      if not axis:
        raise ValueError('`axis` cannot be empty.  Found: {}'.format(axis))
      x = _move_dims_to_flat_end(x, axis, x_ndims, right_end=False)

    # bins.shape = x.shape = [n_samples] + E,
    # and bins[i] is a shape E Tensor of the bins that sample `i` fell into.
    # E is the "event shape", which is [] if axis is None.
    bins = find_bins(
        x,
        edges=edges,
        # If not extending intervals, then values outside the edges will return
        # -1, which gives an error when fed to bincount.
        extend_lower_interval=extend_lower_interval,
        extend_upper_interval=extend_upper_interval,
        dtype=tf.int32)

    # TODO(b/124015136) Use standard tf.math.bincount once it supports `axis`.
    counts = count_integers(
        bins,
        # Ensure we get correct output, even if x did not fall into every bin
        minlength=tf.shape(edges)[0] - 1,
        maxlength=tf.shape(edges)[0] - 1,
        axis=0,
        dtype=dtype or in_dtype)
    n_edges = tf.compat.dimension_value(edges.shape[0])
    if n_edges is not None:
      tensorshape_util.set_shape(
          counts,
          tf.TensorShape([n_edges - 1]).concatenate(counts.shape[1:]))
    return counts


@deprecation.deprecated_args(
    '2020-05-03',
    '`keep_dims` is deprecated, use `keepdims` instead.',
    'keep_dims')
def percentile(x,
               q,
               axis=None,
               interpolation=None,
               keepdims=False,
               validate_args=False,
               preserve_gradients=True,
               keep_dims=None,
               name=None):
  """Compute the `q`-th percentile(s) of `x`.

  Given a vector `x`, the `q`-th percentile of `x` is the value `q / 100` of the
  way from the minimum to the maximum in a sorted copy of `x`.

  The values and distances of the two nearest neighbors as well as the
  `interpolation` parameter will determine the percentile if the normalized
  ranking does not match the location of `q` exactly.

  This function is the same as the median if `q = 50`, the same as the minimum
  if `q = 0` and the same as the maximum if `q = 100`.

  Multiple percentiles can be computed at once by using `1-D` vector `q`.
  Dimension zero of the returned `Tensor` will index the different percentiles.

  Compare to `numpy.percentile`.

  Args:
    x:  Numeric `N-D` `Tensor` with `N > 0`.  If `axis` is not `None`,
      `x` must have statically known number of dimensions.
    q:  Scalar or vector `Tensor` with values in `[0, 100]`. The percentile(s).
    axis:  Optional `0-D` or `1-D` integer `Tensor` with constant values. The
      axis that index independent samples over which to return the desired
      percentile.  If `None` (the default), treat every dimension as a sample
      dimension, returning a scalar.
    interpolation : {'nearest', 'linear', 'lower', 'higher', 'midpoint'}.
      Default value: 'nearest'.  This specifies the interpolation method to
      use when the desired quantile lies between two data points `i < j`:
        * linear: i + (j - i) * fraction, where fraction is the fractional part
          of the index surrounded by i and j.
        * lower: `i`.
        * higher: `j`.
        * nearest: `i` or `j`, whichever is nearest.
        * midpoint: (i + j) / 2.
      `linear` and `midpoint` interpolation do not work with integer dtypes.
    keepdims:  Python `bool`. If `True`, the last dimension is kept with size 1
      If `False`, the last dimension is removed from the output shape.
    validate_args:  Whether to add runtime checks of argument validity. If
      False, and arguments are incorrect, correct behavior is not guaranteed.
    preserve_gradients:  Python `bool`.  If `True`, ensure that gradient w.r.t
      the percentile `q` is preserved in the case of linear interpolation.
      If `False`, the gradient will be (incorrectly) zero when `q` corresponds
      to a point in `x`.
    keep_dims: deprecated, use keepdims instead.
    name:  A Python string name to give this `Op`.  Default is 'percentile'

  Returns:
    A `(rank(q) + N - len(axis))` dimensional `Tensor` of same dtype as `x`, or,
      if `axis` is `None`, a `rank(q)` `Tensor`.  The first `rank(q)` dimensions
      index quantiles for different values of `q`.

  Raises:
    ValueError:  If argument 'interpolation' is not an allowed type.
    ValueError:  If interpolation type not compatible with `dtype`.

  #### Examples

  ```python
  # Get 30th percentile with default ('nearest') interpolation.
  x = [1., 2., 3., 4.]
  tfp.stats.percentile(x, q=30.)
  ==> 2.0

  # Get 30th percentile with 'linear' interpolation.
  x = [1., 2., 3., 4.]
  tfp.stats.percentile(x, q=30., interpolation='linear')
  ==> 1.9

  # Get 30th and 70th percentiles with 'lower' interpolation
  x = [1., 2., 3., 4.]
  tfp.stats.percentile(x, q=[30., 70.], interpolation='lower')
  ==> [1., 3.]

  # Get 100th percentile (maximum).  By default, this is computed over every dim
  x = [[1., 2.]
       [3., 4.]]
  tfp.stats.percentile(x, q=100.)
  ==> 4.

  # Treat the leading dim as indexing samples, and find the 100th quantile (max)
  # over all such samples.
  x = [[1., 2.]
       [3., 4.]]
  tfp.stats.percentile(x, q=100., axis=[0])
  ==> [3., 4.]
  ```

  """
  keepdims = keepdims if keep_dims is None else keep_dims
  del keep_dims
  name = name or 'percentile'
  allowed_interpolations = {'linear', 'lower', 'higher', 'nearest', 'midpoint'}

  if interpolation is None:
    interpolation = 'nearest'
  else:
    if interpolation not in allowed_interpolations:
      raise ValueError(
          'Argument `interpolation` must be in {}. Found {}.'.format(
              allowed_interpolations, interpolation))

  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')

    if (interpolation in {'linear', 'midpoint'} and
        dtype_util.is_integer(x.dtype)):
      raise TypeError('{} interpolation not allowed with dtype {}'.format(
          interpolation, x.dtype))

    # Double is needed here and below, else we get the wrong index if the array
    # is huge along axis.
    q = tf.cast(q, tf.float64)
    _get_static_ndims(q, expect_ndims_no_more_than=1)

    if validate_args:
      q = distribution_util.with_dependencies([
          assert_util.assert_rank_in(q, [0, 1]),
          assert_util.assert_greater_equal(q, tf.cast(0., tf.float64)),
          assert_util.assert_less_equal(q, tf.cast(100., tf.float64))
      ], q)

    # Move `axis` dims of `x` to the rightmost, call it `y`.
    if axis is None:
      y = tf.reshape(x, [-1])
    else:
      x_ndims = _get_static_ndims(
          x, expect_static=True, expect_ndims_at_least=1)
      axis = _make_static_axis_non_negative_list(axis, x_ndims)
      y = _move_dims_to_flat_end(x, axis, x_ndims, right_end=True)

    frac_at_q_or_below = q / 100.

    # Sort (in ascending order) everything which allows multiple calls to sort
    # only once (under the hood) and use CSE.
    sorted_y = tf.sort(y, axis=-1, direction='ASCENDING')

    d = tf.cast(tf.shape(y)[-1], tf.float64)

    def _get_indices(interp_type):
      """Get values of y at the indices implied by interp_type."""
      if interp_type == 'lower':
        indices = tf.math.floor((d - 1) * frac_at_q_or_below)
      elif interp_type == 'higher':
        indices = tf.math.ceil((d - 1) * frac_at_q_or_below)
      elif interp_type == 'nearest':
        indices = tf.round((d - 1) * frac_at_q_or_below)
      # d - 1 will be distinct from d in int32, but not necessarily double.
      # So clip to avoid out of bounds errors.
      return tf.clip_by_value(
          tf.cast(indices, tf.int32), 0,
          tf.shape(y)[-1] - 1)

    if interpolation in ['nearest', 'lower', 'higher']:
      gathered_y = tf.gather(sorted_y, _get_indices(interpolation), axis=-1)
    elif interpolation == 'midpoint':
      gathered_y = 0.5 * (
          tf.gather(sorted_y, _get_indices('lower'), axis=-1) +
          tf.gather(sorted_y, _get_indices('higher'), axis=-1))
    elif interpolation == 'linear':
      # Copy-paste of docstring on interpolation:
      # linear: i + (j - i) * fraction, where fraction is the fractional part
      # of the index surrounded by i and j.
      larger_y_idx = _get_indices('higher')
      exact_idx = (d - 1) * frac_at_q_or_below
      if preserve_gradients:
        # If q corresponds to a point in x, we will initially have
        # larger_y_idx == smaller_y_idx.
        # This results in the gradient w.r.t. fraction being zero (recall `q`
        # enters only through `fraction`...and see that things cancel).
        # The fix is to ensure that smaller_y_idx and larger_y_idx are always
        # separated by exactly 1.
        smaller_y_idx = tf.maximum(larger_y_idx - 1, 0)
        larger_y_idx = tf.minimum(smaller_y_idx + 1, tf.shape(y)[-1] - 1)
        fraction = tf.cast(larger_y_idx, tf.float64) - exact_idx
      else:
        smaller_y_idx = _get_indices('lower')
        fraction = tf.math.ceil((d - 1) * frac_at_q_or_below) - exact_idx

      fraction = tf.cast(fraction, y.dtype)
      gathered_y = (
          tf.gather(sorted_y, larger_y_idx, axis=-1) * (1 - fraction) +
          tf.gather(sorted_y, smaller_y_idx, axis=-1) * fraction)

    # Propagate NaNs
    if x.dtype in (tf.bfloat16, tf.float16, tf.float32, tf.float64):
      # Apparently tf.is_nan doesn't like other dtypes
      nan_batch_members = tf.reduce_any(tf.math.is_nan(x), axis=axis)
      right_rank_matched_shape = tf.pad(
          tf.shape(nan_batch_members),
          paddings=[[0, tf.rank(q)]],
          constant_values=1)
      nan_batch_members = tf.reshape(
          nan_batch_members, shape=right_rank_matched_shape)
      nan = np.array(np.nan, dtype_util.as_numpy_dtype(gathered_y.dtype))
      gathered_y = tf.where(nan_batch_members, nan, gathered_y)

    # Expand dimensions if requested
    if keepdims:
      if axis is None:
        ones_vec = tf.ones(
            shape=[_get_best_effort_ndims(x) + _get_best_effort_ndims(q)],
            dtype=tf.int32)
        gathered_y *= tf.ones(ones_vec, dtype=x.dtype)
      else:
        gathered_y = _insert_back_keepdims(gathered_y, axis)

    # If q is a scalar, then result has the right shape.
    # If q is a vector, then result has trailing dim of shape q.shape, which
    # needs to be rotated to dim 0.
    return distribution_util.rotate_transpose(gathered_y, tf.rank(q))


@deprecation.deprecated_args(
    '2020-05-03',
    '`keep_dims` is deprecated, use `keepdims` instead.',
    'keep_dims')
def quantiles(x,
              num_quantiles,
              axis=None,
              interpolation=None,
              keepdims=False,
              validate_args=False,
              keep_dims=None,
              name=None):
  """Compute quantiles of `x` along `axis`.

  The quantiles of a distribution are cut points dividing the range into
  intervals with equal probabilities.

  Given a vector `x` of samples, this function estimates the cut points by
  returning `num_quantiles + 1` cut points, `(c0, ..., cn)`, such that, roughly
  speaking, equal number of sample points lie in the `num_quantiles` intervals
  `[c0, c1), [c1, c2), ..., [c_{n-1}, cn]`.  That is,

  * About `1 / n` fraction of the data lies in `[c_{k-1}, c_k)`, `k = 1, ..., n`
  * About `k / n` fraction of the data lies below `c_k`.
  * `c0` is the sample minimum and `cn` is the maximum.

  The exact number of data points in each interval depends on the size of
  `x` (e.g. whether the size is divisible by `n`) and the `interpolation` kwarg.

  Args:
    x:  Numeric `N-D` `Tensor` with `N > 0`.  If `axis` is not `None`,
      `x` must have statically known number of dimensions.
    num_quantiles:  Scalar `integer` `Tensor`.  The number of intervals the
      returned `num_quantiles + 1` cut points divide the range into.
    axis:  Optional `0-D` or `1-D` integer `Tensor` with constant values. The
      axis that index independent samples over which to return the desired
      percentile.  If `None` (the default), treat every dimension as a sample
      dimension, returning a scalar.
    interpolation : {'nearest', 'linear', 'lower', 'higher', 'midpoint'}.
      Default value: 'nearest'.  This specifies the interpolation method to
      use when the fractions `k / n` lie between two data points `i < j`:
        * linear: i + (j - i) * fraction, where fraction is the fractional part
          of the index surrounded by i and j.
        * lower: `i`.
        * higher: `j`.
        * nearest: `i` or `j`, whichever is nearest.
        * midpoint: (i + j) / 2. `linear` and `midpoint` interpolation do not
          work with integer dtypes.
    keepdims:  Python `bool`. If `True`, the last dimension is kept with size 1
      If `False`, the last dimension is removed from the output shape.
    validate_args:  Whether to add runtime checks of argument validity. If
      False, and arguments are incorrect, correct behavior is not guaranteed.
    keep_dims: deprecated, use keepdims instead.
    name:  A Python string name to give this `Op`.  Default is 'percentile'

  Returns:
    cut_points:  A `rank(x) + 1 - len(axis)` dimensional `Tensor` with same
    `dtype` as `x` and shape `[num_quantiles + 1, ...]` where the trailing shape
    is that of `x` without the dimensions in `axis` (unless `keepdims is True`)

  Raises:
    ValueError:  If argument 'interpolation' is not an allowed type.
    ValueError:  If interpolation type not compatible with `dtype`.

  #### Examples

  ```python
  # Get quartiles of x with various interpolation choices.
  x = [0.,  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.]

  tfp.stats.quantiles(x, num_quantiles=4, interpolation='nearest')
  ==> [  0.,   2.,   5.,   8.,  10.]

  tfp.stats.quantiles(x, num_quantiles=4, interpolation='linear')
  ==> [  0. ,   2.5,   5. ,   7.5,  10. ]

  tfp.stats.quantiles(x, num_quantiles=4, interpolation='lower')
  ==> [  0.,   2.,   5.,   7.,  10.]

  # Get deciles of columns of an R x C data set.
  data = load_my_columnar_data(...)
  tfp.stats.quantiles(data, num_quantiles=10)
  ==> Shape [11, C] Tensor
  ```

  """
  keepdims = keepdims if keep_dims is None else keep_dims
  del keep_dims
  with tf.name_scope(name or 'quantiles'):
    x = tf.convert_to_tensor(x, name='x')
    return percentile(
        x,
        q=tf.linspace(
            # percentile casts q to float64 before using it...so may as well use
            # float64 here. Note that  using x.dtype won't work with linspace
            # if x is integral type (which is anothe motivation for hard-coding
            # float64).
            tf.convert_to_tensor(0, dtype=tf.float64),
            tf.convert_to_tensor(100, dtype=tf.float64),
            num=num_quantiles + 1),
        axis=axis,
        interpolation=interpolation,
        keepdims=keepdims,
        validate_args=validate_args,
        preserve_gradients=False)


def _get_static_ndims(x,
                      expect_static=False,
                      expect_ndims=None,
                      expect_ndims_no_more_than=None,
                      expect_ndims_at_least=None):
  """Get static number of dimensions and assert that some expectations are met.

  This function returns the number of dimensions 'ndims' of x, as a Python int.

  The optional expect arguments are used to check the ndims of x, but this is
  only done if the static ndims of x is not None.

  Args:
    x:  A Tensor.
    expect_static:  Expect `x` to have statically defined `ndims`.
    expect_ndims:  Optional Python integer.  If provided, assert that x has
      number of dimensions equal to this.
    expect_ndims_no_more_than:  Optional Python integer.  If provided, assert
      that x has no more than this many dimensions.
    expect_ndims_at_least:  Optional Python integer.  If provided, assert that x
      has at least this many dimensions.

  Returns:
    ndims:  A Python integer.

  Raises:
    ValueError:  If any of the expectations above are violated.
  """
  ndims = tensorshape_util.rank(x.shape)

  if ndims is None:
    if expect_static:
      raise ValueError(
          'Expected argument `x` to have statically defined `ndims`. '
          'Found: {}.'.format(x))
    return

  if expect_ndims is not None:
    ndims_message = (
        'Expected argument `x` to have ndims {}. Found tensor {}.'.format(
            expect_ndims, x))
    if ndims != expect_ndims:
      raise ValueError(ndims_message)

  if expect_ndims_at_least is not None:
    ndims_at_least_message = (
        'Expected argument `x` to have ndims >= {}. Found tensor {}.'.format(
            expect_ndims_at_least, x))
    if ndims < expect_ndims_at_least:
      raise ValueError(ndims_at_least_message)

  if expect_ndims_no_more_than is not None:
    ndims_no_more_than_message = (
        'Expected argument `x` to have ndims <= {}. Found tensor {}.'.format(
            expect_ndims_no_more_than, x))
    if ndims > expect_ndims_no_more_than:
      raise ValueError(ndims_no_more_than_message)

  return ndims


def _get_best_effort_ndims(x,
                           expect_ndims=None,
                           expect_ndims_at_least=None,
                           expect_ndims_no_more_than=None):
  """Get static ndims if possible.  Fallback on `tf.rank(x)`."""
  ndims_static = _get_static_ndims(
      x,
      expect_ndims=expect_ndims,
      expect_ndims_at_least=expect_ndims_at_least,
      expect_ndims_no_more_than=expect_ndims_no_more_than)
  if ndims_static is not None:
    return ndims_static
  return tf.rank(x)


def _insert_back_keepdims(x, axis):
  """Insert the dims in `axis` back as singletons after being removed.

  Args:
    x:  `Tensor`.
    axis:  Python list of integers.

  Returns:
    `Tensor` with same values as `x`, but additional singleton dimensions.
  """
  for i in sorted(axis):
    x = tf.expand_dims(x, axis=i)
  return x


def _make_static_axis_non_negative_list(axis, ndims):
  """Convert possibly negatively indexed axis to non-negative list of ints.

  Args:
    axis:  Integer Tensor.
    ndims:  Number of dimensions into which axis indexes.

  Returns:
    A list of non-negative Python integers.

  Raises:
    ValueError: If `axis` is not statically defined.
  """
  axis = prefer_static.non_negative_axis(axis, ndims)

  axis_const = tf.get_static_value(axis)
  if axis_const is None:
    raise ValueError(
        'Expected argument `axis` to be statically available. '
        'Found: {}.'.format(axis))

  # Make at least 1-D.
  axis = axis_const + np.zeros([1], dtype=axis_const.dtype)

  return list(int(dim) for dim in axis)


def _move_dims_to_flat_end(x, axis, x_ndims, right_end=True):
  """Move dims corresponding to `axis` in `x` to the end, then flatten.

  Args:
    x: `Tensor` with shape `[B0,B1,...,Bb]`.
    axis:  Python list of indices into dimensions of `x`.
    x_ndims:  Python integer holding number of dimensions in `x`.
    right_end:  Python bool.  Whether to move dims to the right end (else left).

  Returns:
    `Tensor` with value from `x` and dims in `axis` moved to end into one single
      dimension.
  """

  if not axis:
    return x

  # Suppose x.shape = [a, b, c, d]
  # Suppose axis = [1, 3]

  # other_dims = [0, 2] in example above.
  other_dims = sorted(set(range(x_ndims)).difference(axis))
  # x_permed.shape = [a, c, b, d]
  perm = other_dims + list(axis) if right_end else list(axis) + other_dims
  x_permed = tf.transpose(a=x, perm=perm)

  if tensorshape_util.is_fully_defined(x.shape):
    x_shape = tensorshape_util.as_list(x.shape)
    # other_shape = [a, c], end_shape = [b * d]
    other_shape = [x_shape[i] for i in other_dims]
    end_shape = [np.prod([x_shape[i] for i in axis])]
    full_shape = (
        other_shape + end_shape if right_end else end_shape + other_shape)
  else:
    other_shape = tf.gather(tf.shape(x), tf.constant(other_dims, tf.int64))
    full_shape = tf.concat(
        [other_shape, [-1]] if right_end else [[-1], other_shape], axis=0)
  return tf.reshape(x_permed, shape=full_shape)
