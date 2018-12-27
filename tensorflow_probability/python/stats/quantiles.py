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
import tensorflow as tf

from tensorflow_probability.python.internal import distribution_util as util
from tensorflow.python.ops import control_flow_ops

__all__ = [
    'percentile',
]


def percentile(x,
               q,
               axis=None,
               interpolation=None,
               keep_dims=False,
               validate_args=False,
               preserve_gradients=True,
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

  Compare to `numpy.percentile`.

  Args:
    x:  Floating point `N-D` `Tensor` with `N > 0`.  If `axis` is not `None`,
      `x` must have statically known number of dimensions.
    q:  Scalar or vector `Tensor` with values in `[0, 100]`. The percentile(s).
    axis:  Optional `0-D` or `1-D` integer `Tensor` with constant values. The
      axis that hold independent samples over which to return the desired
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
    keep_dims:  Python `bool`. If `True`, the last dimension is kept with size 1
      If `False`, the last dimension is removed from the output shape.
    validate_args:  Whether to add runtime checks of argument validity. If
      False, and arguments are incorrect, correct behavior is not guaranteed.
    preserve_gradients:  Python `bool`.  If `True`, ensure that gradient w.r.t
      the percentile `q` is preserved in the case of linear interpolation.
      If `False`, the gradient will be (incorrectly) zero when `q` corresponds
      to a point in `x`.
    name:  A Python string name to give this `Op`.  Default is 'percentile'

  Returns:
    A `(rank(q) + N - len(axis))` dimensional `Tensor` of same dtype as `x`, or,
      if `axis` is `None`, a `rank(q)` `Tensor`.  The first `rank(q)` dimensions
      index quantiles for different values of `q`.

  Raises:
    ValueError:  If argument 'interpolation' is not an allowed type.
    ValueError:  If interpolation type not compatible with `dtype`.
  """
  name = name or 'percentile'
  allowed_interpolations = {'linear', 'lower', 'higher', 'nearest', 'midpoint'}

  if interpolation is None:
    interpolation = 'nearest'
  else:
    if interpolation not in allowed_interpolations:
      raise ValueError('Argument `interpolation` must be in %s.  Found %s' %
                       (allowed_interpolations, interpolation))

  with tf.name_scope(name, values=[x, q]):
    x = tf.convert_to_tensor(x, name='x')

    if interpolation in {'linear', 'midpoint'} and x.dtype.is_integer:
      raise TypeError('{} interpolation not allowed with dtype {}'.format(
          interpolation, x.dtype))

    # Double is needed here and below, else we get the wrong index if the array
    # is huge along axis.
    q = tf.cast(q, tf.float64)
    _get_static_ndims(q, expect_ndims_no_more_than=1)

    if validate_args:
      q = control_flow_ops.with_dependencies([
          tf.assert_rank_in(q, [0, 1]),
          tf.assert_greater_equal(q, tf.cast(0., tf.float64)),
          tf.assert_less_equal(q, tf.cast(100., tf.float64))
      ], q)

    if axis is None:
      y = tf.reshape(x, [-1])
    else:
      axis = tf.convert_to_tensor(axis, name='axis', dtype=tf.int32)
      tf.assert_integer(axis)
      axis_ndims = _get_static_ndims(
          axis, expect_static=True, expect_ndims_no_more_than=1)
      axis_const = tf.contrib.util.constant_value(axis)
      if axis_const is None:
        raise ValueError(
            'Expected argument `axis` to be statically available.  Found: %s' %
            axis)
      axis = axis_const
      if axis_ndims == 0:
        axis = [axis]
      axis = [int(a) for a in axis]
      x_ndims = _get_static_ndims(
          x, expect_static=True, expect_ndims_at_least=1)
      axis = _make_static_axis_non_negative(axis, x_ndims)
      # Move dims in axis to the end, since _sort_tensor, which calls top_k,
      # only sorts the last dim.
      y = _move_dims_to_flat_end(x, axis, x_ndims)

    frac_at_q_or_above = 1. - q / 100.

    # Sort everything, not just the top 'k' entries, which allows multiple calls
    # to sort only once (under the hood) and use CSE.
    sorted_y = _sort_tensor(y)

    d = tf.cast(tf.shape(y)[-1], tf.float64)

    def _get_indices(interp_type):
      """Get values of y at the indices implied by interp_type."""
      # Note `lower` <--> ceiling.  Confusing, huh?  Due to the fact that
      # _sort_tensor sorts highest to lowest, tf.ceil corresponds to the higher
      # index, but the lower value of y!
      if interp_type == 'lower':
        indices = tf.ceil((d - 1) * frac_at_q_or_above)
      elif interp_type == 'higher':
        indices = tf.floor((d - 1) * frac_at_q_or_above)
      elif interp_type == 'nearest':
        indices = tf.round((d - 1) * frac_at_q_or_above)
      # d - 1 will be distinct from d in int32, but not necessarily double.
      # So clip to avoid out of bounds errors.
      return tf.clip_by_value(
          tf.cast(indices, tf.int32), 0, tf.shape(y)[-1] - 1)

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
      larger_y_idx = _get_indices('lower')
      exact_idx = (d - 1) * frac_at_q_or_above
      if preserve_gradients:
        # If q cooresponds to a point in x, we will initially have
        # larger_y_idx == smaller_y_idx.
        # This results in the gradient w.r.t. fraction being zero (recall `q`
        # enters only through `fraction`...and see that things cancel).
        # The fix is to ensure that smaller_y_idx and larger_y_idx are always
        # separated by exactly 1.
        smaller_y_idx = tf.maximum(larger_y_idx - 1, 0)
        larger_y_idx = tf.minimum(smaller_y_idx + 1, tf.shape(y)[-1] - 1)
        fraction = tf.cast(larger_y_idx, tf.float64) - exact_idx
      else:
        smaller_y_idx = _get_indices('higher')
        fraction = tf.ceil((d - 1) * frac_at_q_or_above) - exact_idx

      fraction = tf.cast(fraction, y.dtype)
      gathered_y = (
          tf.gather(sorted_y, larger_y_idx, axis=-1) * (1 - fraction) +
          tf.gather(sorted_y, smaller_y_idx, axis=-1) * fraction)

    if keep_dims:
      if axis is None:
        ones_vec = tf.ones(
            shape=[_get_best_effort_ndims(x) + _get_best_effort_ndims(q)],
            dtype=tf.int32)
        gathered_y *= tf.ones(ones_vec, dtype=x.dtype)
      else:
        gathered_y = _insert_back_keep_dims(gathered_y, axis)

    # If q is a scalar, then result has the right shape.
    # If q is a vector, then result has trailing dim of shape q.shape, which
    # needs to be rotated to dim 0.
    return util.rotate_transpose(gathered_y, tf.rank(q))


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
  ndims = x.shape.ndims
  if ndims is None:
    shape_const = tf.contrib.util.constant_value(tf.shape(x))
    if shape_const is not None:
      ndims = shape_const.ndim

  if ndims is None:
    if expect_static:
      raise ValueError(
          'Expected argument `x` to have statically defined `ndims`.  Found: ' %
          x)
    return

  if expect_ndims is not None:
    ndims_message = ('Expected argument `x` to have ndims %s.  Found tensor %s'
                     % (expect_ndims, x))
    if ndims != expect_ndims:
      raise ValueError(ndims_message)

  if expect_ndims_at_least is not None:
    ndims_at_least_message = (
        'Expected argument `x` to have ndims >= %d.  Found tensor %s' %
        (expect_ndims_at_least, x))
    if ndims < expect_ndims_at_least:
      raise ValueError(ndims_at_least_message)

  if expect_ndims_no_more_than is not None:
    ndims_no_more_than_message = (
        'Expected argument `x` to have ndims <= %d.  Found tensor %s' %
        (expect_ndims_no_more_than, x))
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


def _insert_back_keep_dims(x, axis):
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


def _make_static_axis_non_negative(axis, ndims):
  """Convert possibly negatively indexed axis to non-negative.

  Args:
    axis:  Iterable over Python integers.
    ndims:  Number of dimensions into which axis indexes.

  Returns:
    A list of non-negative Python integers.

  Raises:
    ValueError: If values in `axis` are too big/small to index into `ndims`.
  """
  non_negative_axis = []
  for d in axis:
    if d >= 0:
      if d >= ndims:
        raise ValueError('dim %d not in the interval [0, %d].' % (d, ndims - 1))
      non_negative_axis.append(d)
    else:
      if d < -1 * ndims:
        raise ValueError(
            'Negatively indexed dim %d not in the interval [-%d, -1]' % (d,
                                                                         ndims))
      non_negative_axis.append(ndims + d)
  return non_negative_axis


def _move_dims_to_flat_end(x, axis, x_ndims):
  """Move dims corresponding to `axis` in `x` to the end, then flatten.

  Args:
    x: `Tensor` with shape `[B0,B1,...,Bb]`.
    axis:  Python list of indices into dimensions of `x`.
    x_ndims:  Python integer holding number of dimensions in `x`.

  Returns:
    `Tensor` with value from `x` and dims in `axis` moved to end into one single
      dimension.
  """
  # Suppose x.shape = [a, b, c, d]
  # Suppose axis = [1, 3]

  # front_dims = [0, 2] in example above.
  front_dims = sorted(set(range(x_ndims)).difference(axis))
  # x_permed.shape = [a, c, b, d]
  x_permed = tf.transpose(x, perm=front_dims + list(axis))

  if x.shape.is_fully_defined():
    x_shape = x.shape.as_list()
    # front_shape = [a, c], end_shape = [b * d]
    front_shape = [x_shape[i] for i in front_dims]
    end_shape = [np.prod([x_shape[i] for i in axis])]
    full_shape = front_shape + end_shape
  else:
    front_shape = tf.shape(x_permed)[:x_ndims - len(axis)]
    end_shape = [-1]
    full_shape = tf.concat([front_shape, end_shape], axis=0)
  return tf.reshape(x_permed, shape=full_shape)


def _sort_tensor(tensor):
  """Use `top_k` to sort a `Tensor` along the last dimension."""
  sorted_, _ = tf.nn.top_k(tensor, k=tf.shape(tensor)[-1])
  sorted_.set_shape(tensor.shape)
  return sorted_
