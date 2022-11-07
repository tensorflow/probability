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
"""Utilities for probability distributions."""

import functools
import hashlib
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


def _convert_to_tensor(x, name, dtype=None):
  return None if x is None else tf.convert_to_tensor(x, name=name, dtype=dtype)


def mixture_stddev(mixture_weight_vector, mean_vector, stddev_vector):
  """Computes the standard deviation of a mixture distribution.

  This function works regardless of the component distribution, so long as
  each component's mean and standard deviation can be provided.

  Args:
    mixture_weight_vector: A Tensor with shape `batch_shape + [num_components]`
    mean_vector: A Tensor of mixture component means. Has shape `batch_shape +
      [num_components]`.
    stddev_vector: A Tensor of mixture component standard deviations. Has
      shape `batch_shape + [num_components]`.

  Returns:
    A 1D tensor of shape `batch_shape` representing the standard deviation of
    the mixture distribution with given weights and component means and standard
    deviations.
  Raises:
    ValueError: If the shapes of the input tensors are not as expected.
  """
  if not tensorshape_util.is_compatible_with(mean_vector.shape,
                                             mixture_weight_vector.shape):
    raise ValueError('Expecting means to have same shape as mixture weights.')
  if not tensorshape_util.is_compatible_with(stddev_vector.shape,
                                             mixture_weight_vector.shape):
    raise ValueError('Expecting stddevs to have same shape as mixture weights.')

  weighted_average_means = tf.reduce_sum(
      mixture_weight_vector * mean_vector, axis=-1, keepdims=True)
  deviations = mean_vector - weighted_average_means
  return _hypot(_weighted_norm(mixture_weight_vector, deviations),
                _weighted_norm(mixture_weight_vector, stddev_vector))


def _hypot(x, y):
  """Returns sqrt(x**2 + y**2) elementwise without overflow."""
  # Notably, this implementation avoids overflow better than
  # tf.experimental.numpy.hypot
  mag = tf.maximum(tf.math.abs(x), tf.math.abs(y))
  normalized_result = tf.sqrt(tf.square(x / mag) + tf.square(y / mag))
  return tf.math.multiply_no_nan(normalized_result, mag)


def _weighted_norm(weights, xs):
  """Returns sqrt(sum_{axis=-1}(w_i * x_i**2)) without overflow in x_i**2."""
  # Notably, this implementation differs from tf.norm in supporting weights and
  # avoiding overflow.  The weights are assumed < 1, i.e., incapable of causing
  # overflow themselves.
  magnitude = tf.reduce_max(tf.math.abs(xs), axis=-1)
  xs = xs / magnitude[..., tf.newaxis]
  normalized_result = tf.sqrt(tf.reduce_sum(weights * tf.square(xs), axis=-1))
  return tf.math.multiply_no_nan(normalized_result, magnitude)


def shapes_from_loc_and_scale(loc, scale, name='shapes_from_loc_and_scale'):
  """Infer distribution batch and event shapes from a location and scale.

  Location and scale family distributions determine their batch/event shape by
  broadcasting the `loc` and `scale` args.  This helper does that broadcast,
  statically if possible.

  Batch shape broadcasts as per the normal rules.
  We allow the `loc` event shape to broadcast up to that of `scale`.  We do not
  allow `scale`'s event shape to change.  Therefore, the last dimension of `loc`
  must either be size `1`, or the same as `scale.range_dimension`.

  See `MultivariateNormalLinearOperator` for a usage example.

  Args:
    loc: `Tensor` (already converted to tensor) or `None`. If `None`, or
      `rank(loc)==0`, both batch and event shape are determined by `scale`.
    scale:  A `LinearOperator` instance.
    name:  A string name to prepend to created ops.

  Returns:
    batch_shape:  `TensorShape` (if broadcast is done statically), or `Tensor`.
    event_shape:  `TensorShape` (if broadcast is done statically), or `Tensor`.

  Raises:
    ValueError:  If the last dimension of `loc` is determined statically to be
      different than the range of `scale`.
  """
  if loc is not None and tensorshape_util.rank(loc.shape) == 0:
    loc = None  # scalar loc is irrelevant to determining batch/event shape.
  with tf.name_scope(name):
    # Get event shape.
    event_size = tf.compat.dimension_value(scale.range_dimension)
    if event_size is None:
      event_size = scale.range_dimension_tensor()
    event_size_ = tf.get_static_value(ps.convert_to_shape_tensor(event_size))
    loc_event_size_ = (None if loc is None
                       else tf.compat.dimension_value(loc.shape[-1]))

    if event_size_ is not None and loc_event_size_ is not None:
      # Static check that event shapes match.
      if loc_event_size_ != 1 and loc_event_size_ != event_size_:
        raise ValueError(
            'Event size of `scale` ({}) could not be broadcast up to that '
            'of `loc` ({}).'.format(event_size_, loc_event_size_))
    elif loc_event_size_ is not None and loc_event_size_ != 1:
      event_size_ = loc_event_size_

    if event_size_ is None:
      event_shape = event_size[tf.newaxis]
    else:
      event_shape = ps.convert_to_shape_tensor(
          np.reshape(event_size_, [1]), dtype=tf.int32, name='event_shape')

    # Get batch shape.
    batch_shape = scale.batch_shape
    if not tensorshape_util.is_fully_defined(batch_shape):
      batch_shape = scale.batch_shape_tensor()
    else:
      batch_shape = ps.convert_to_shape_tensor(batch_shape)
    if loc is not None:
      loc_batch_shape = tensorshape_util.with_rank_at_least(loc.shape, 1)[:-1]
      if (tensorshape_util.rank(loc.shape) is None or
          not tensorshape_util.is_fully_defined(loc_batch_shape)):
        loc_batch_shape = tf.shape(loc)[:-1]
      else:
        loc_batch_shape = ps.convert_to_shape_tensor(
            loc_batch_shape, dtype=tf.int32, name='loc_batch_shape')
      # This is defined in the core util module.
      batch_shape = ps.broadcast_shape(batch_shape, loc_batch_shape)
      batch_shape = ps.convert_to_shape_tensor(
          batch_shape, dtype=tf.int32, name='batch_shape')

    return batch_shape, event_shape


def get_broadcast_shape(*tensors):
  """Get broadcast shape as a Python list of integers (preferred) or `Tensor`.

  Args:
    *tensors:  One or more `Tensor` objects (already converted!).

  Returns:
    broadcast shape:  Python list (if shapes determined statically), otherwise
      an `int32` `Tensor`.
  """
  # Try static.
  s_shape = tensors[0].shape
  for t in tensors[1:]:
    s_shape = tf.broadcast_static_shape(s_shape, t.shape)
  if tensorshape_util.is_fully_defined(s_shape):
    return tensorshape_util.as_list(s_shape)

  # Fallback on dynamic.
  d_shape = tf.shape(tensors[0])
  for t in tensors[1:]:
    d_shape = tf.broadcast_dynamic_shape(d_shape, tf.shape(t))
  return d_shape


def shape_may_be_nontrivial(shape):
  """Returns `True` if it's possible that `shape` describes a non-scalar."""
  static_size = tf.get_static_value(ps.size(shape))
  return (static_size is None) or static_size >= 1


def is_diagonal_scale(scale):
  """Returns `True` if `scale` is a `LinearOperator` that is known to be diag.

  Args:
    scale:  `LinearOperator` instance.

  Returns:
    Python `bool`.

  Raises:
    TypeError:  If `scale` is not a `LinearOperator`.
  """
  if not isinstance(scale, tf.linalg.LinearOperator):
    raise TypeError('Expected argument `scale` to be instance of '
                    '`LinearOperator`. Found: `{}`.'.format(scale))
  return (isinstance(scale, tf.linalg.LinearOperatorIdentity) or
          isinstance(scale, tf.linalg.LinearOperatorScaledIdentity) or
          isinstance(scale, tf.linalg.LinearOperatorDiag))


def maybe_check_scalar_distribution(distribution, expected_base_dtype,
                                    validate_args):
  """Helper which checks validity of a scalar `distribution` init arg.

  Valid here means:

  * `distribution` has scalar batch and event shapes.
  * `distribution` is `FULLY_REPARAMETERIZED`
  * `distribution` has expected dtype.

  Args:
    distribution:  `Distribution`-like object.
    expected_base_dtype:  `TensorFlow` `dtype`.
    validate_args:  Python `bool`.  Whether to do additional checks: (i)  check
      that reparameterization_type is `FULLY_REPARAMETERIZED`. (ii) add
      `tf.Assert` ops to the graph to enforce that distribution is scalar in the
      event that this cannot be determined statically.

  Returns:
    List of `tf.Assert` ops to run to enforce validity checks that could not
      be statically determined.  Empty if `not validate_args`.

  Raises:
    ValueError:  If validate_args and distribution is not FULLY_REPARAMETERIZED
    ValueError:  If distribution is statically determined to not have both
      scalar batch and scalar event shapes.
  """
  if distribution.dtype != expected_base_dtype:
    raise TypeError('dtype mismatch; '
                    'distribution.dtype=\'{}\' is not \'{}\''.format(
                        dtype_util.name(distribution.dtype),
                        dtype_util.name(expected_base_dtype)))

  # Although `reparameterization_type` is a static property, we guard it by
  # `validate_args`. This allows users to use a `distribution` which is not
  # reparameterized itself. However, we tacitly assume that although the
  # distribution is not reparameterized, it only depends on non-trainable
  # variables.
  if validate_args and (distribution.reparameterization_type !=
                        reparameterization.FULLY_REPARAMETERIZED):
    raise ValueError('Base distribution should be reparameterized or be '
                     'a function of non-trainable variables; '
                     'distribution.reparameterization_type = \'{}\' '
                     '!= \'FULLY_REPARAMETERIZED\'.'.format(
                         distribution.reparameterization_type))
  with tf.name_scope('check_distribution'):
    assertions = []

    def check_is_scalar(is_scalar, name):
      is_scalar_ = tf.get_static_value(is_scalar)
      if is_scalar_ is not None:
        if not is_scalar_:
          raise ValueError('distribution must be scalar; '
                           'distribution.{}=False is not True'.format(name))
      elif validate_args:
        assertions.append(
            assert_util.assert_equal(
                is_scalar,
                True,
                message=('distribution must be scalar; '
                         'distribution.{}=False is not True'.format(name))))

    check_is_scalar(distribution.is_scalar_event(), 'is_scalar_event')
    check_is_scalar(distribution.is_scalar_batch(), 'is_scalar_batch')
    return assertions


def pad_mixture_dimensions(x, mixture_distribution, categorical_distribution,
                           event_ndims):
  """Pad dimensions of event tensors for mixture distributions.

  See `Mixture._sample_n` and `MixtureSameFamily._sample_n` for usage examples.

  Args:
    x: event tensor to pad.
    mixture_distribution: Base distribution of the mixture.
    categorical_distribution: `Categorical` distribution that mixes the base
      distribution.
    event_ndims: Integer specifying the number of event dimensions in the event
      tensor.

  Returns:
    A padded version of `x` that can broadcast with `categorical_distribution`.
  """
  with tf.name_scope('pad_mix_dims'):

    def _get_ndims(d):
      if tensorshape_util.rank(d.batch_shape) is not None:
        return tensorshape_util.rank(d.batch_shape)
      return tf.shape(d.batch_shape_tensor())[0]

    dist_batch_ndims = _get_ndims(mixture_distribution)
    cat_batch_ndims = _get_ndims(categorical_distribution)
    pad_ndims = tf.where(categorical_distribution.is_scalar_batch(),
                         dist_batch_ndims, dist_batch_ndims - cat_batch_ndims)
    s = tf.shape(x)
    x = tf.reshape(
        x,
        shape=tf.concat([
            s[:-1],
            tf.ones([pad_ndims], dtype=tf.int32),
            s[-1:],
            tf.ones([event_ndims], dtype=tf.int32),
        ],
                        axis=0))
    return x


def pick_scalar_condition(pred, true_value, false_value, name=None):
  """Convenience function that chooses one of two values based on the predicate.

  This utility is equivalent to a version of `tf.where` that accepts only a
  scalar predicate and computes its result statically when possible. It may also
  be used in place of `tf.cond` when both branches yield a `Tensor` of the same
  shape; the operational difference is that `tf.cond` uses control flow to
  evaluate only the branch that's needed, while `tf.where` (and thus
  this method) may evaluate both branches before the predicate's truth is known.
  This means that `tf.cond` is preferred when one of the branches is expensive
  to evaluate (like performing a large matmul), while this method is preferred
  when both branches are cheap, e.g., constants. In the latter case, we expect
  this method to be substantially faster than `tf.cond` on GPU and to give
  similar performance on CPU.

  Args:
    pred: Scalar `bool` `Tensor` predicate.
    true_value: `Tensor` to return if `pred` is `True`.
    false_value: `Tensor` to return if `pred` is `False`. Must have the same
      shape as `true_value`.
    name: Python `str` name given to ops managed by this object.

  Returns:
    result: a `Tensor` (or `Tensor`-convertible Python value) equal to
      `true_value` if `pred` evaluates to `True` and `false_value` otherwise.
      If the condition can be evaluated statically, the result returned is one
      of the input Python values, with no graph side effects.
  """
  with tf.name_scope(name or 'pick_scalar_condition'):
    pred = tf.convert_to_tensor(pred, dtype_hint=tf.bool, name='pred')
    true_value = tf.convert_to_tensor(true_value, name='true_value')
    false_value = tf.convert_to_tensor(false_value, name='false_value')
    pred_ = tf.get_static_value(pred)
    if pred_ is None:
      return tf.where(pred, true_value, false_value)
    return true_value if pred_ else false_value


def move_dimension(x, source_idx, dest_idx):
  """Move a single tensor dimension within its shape.

  This is a special case of `tf.transpose()`, which applies
  arbitrary permutations to tensor dimensions.

  Args:
    x: Tensor of rank `ndims`.
    source_idx: Integer index into `x.shape` (negative indexing is supported).
    dest_idx: Integer index into `x.shape` (negative indexing is supported).

  Returns:
    x_perm: Tensor of rank `ndims`, in which the dimension at original
     index `source_idx` has been moved to new index `dest_idx`, with
     all other dimensions retained in their original order.

  Example:

  ```python
  x = tf.placeholder(shape=[200, 30, 4, 1, 6])
  x_perm = _move_dimension(x, 1, 1) # no-op
  x_perm = _move_dimension(x, 0, 3) # result shape [30, 4, 1, 200, 6]
  x_perm = _move_dimension(x, 0, -2) # equivalent to previous
  x_perm = _move_dimension(x, 4, 2) # result shape [200, 30, 6, 4, 1]
  ```
  """
  dtype = dtype_util.common_dtype([source_idx, dest_idx],
                                  dtype_hint=tf.int32)
  ndims = ps.cast(ps.rank(x), dtype)
  source_idx = ps.convert_to_shape_tensor(source_idx, dtype=dtype)
  dest_idx = ps.convert_to_shape_tensor(dest_idx, dtype=dtype)

  # Handle negative indexing.
  source_idx = ps.where(source_idx < 0, ndims + source_idx, source_idx)
  dest_idx = ps.where(dest_idx < 0, ndims + dest_idx, dest_idx)

  # Construct the appropriate permutation of dimensions, depending
  # whether the source is before or after the destination.
  def move_left_permutation():
    return ps.concat([
        ps.range(0, dest_idx, dtype=dtype),
        [source_idx],
        ps.range(dest_idx, source_idx, dtype=dtype),
        ps.range(source_idx + 1, ndims, dtype=dtype)
    ], axis=0)

  def move_right_permutation():
    return ps.concat([
        ps.range(0, source_idx, dtype=dtype),
        ps.range(source_idx + 1, dest_idx + 1, dtype=dtype),
        [source_idx],
        ps.range(dest_idx + 1, ndims, dtype=dtype)
    ], axis=0)

  def x_permuted():
    return tf.transpose(
        a=x,
        perm=ps.cond(source_idx < dest_idx,
                     move_right_permutation,
                     move_left_permutation))

  # One final conditional to handle the special case where source
  # and destination indices are equal.
  return ps.cond(ps.equal(source_idx, dest_idx), lambda: x, x_permuted)


def assert_integer_form(x,
                        summarize=None,
                        message=None,
                        atol=None,
                        rtol=None,
                        name='assert_integer_form'):
  """Assert that x has integer components (or floats near integers).

  Args:
    x: Floating-point or integer `Tensor`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    atol: Tensor. Same dtype as, and broadcastable to, x. The absolute
      tolerance. Default is 10 * eps.
    rtol: Tensor. Same dtype as, and broadcastable to, x. The relative
      tolerance. Default is 10 * eps.
    name: A name for this operation (optional).

  Returns:
    Op raising `InvalidArgumentError` if `round(x) != x` within tolerance.
  """
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')
    if dtype_util.is_integer(x.dtype):
      return tf.no_op()
    message = message or '{} has non-integer components'.format(x)
    return assert_util.assert_near(
        x, tf.round(x), atol=atol, rtol=rtol,
        summarize=summarize, message=message, name=name)


def assert_casting_closed(x,
                          target_dtype,
                          summarize=None,
                          message=None,
                          name='assert_casting_closed'):
  """Assert that x is fixed under round-trip casting to `target_dtype`.

  Note that even when `target_dtype` is the integer dtype of the same width as
  the dtype of `x`, this is stronger than `assert_integer_form`.  This is
  because any given floating-point format can represent integers outside the
  range of the equally wide integer format.

  Args:
    x: Floating-point `Tensor`
    target_dtype: A `tf.dtype` used to cast `x` to.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).

  Returns:
    Op raising `InvalidArgumentError` if `cast(x, target_dtype) != x`.
  """
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')
    if message is None:
      message = 'Tensor values must be representable as {}.'.format(
          target_dtype)
    return assert_util.assert_equal(
        x,
        tf.cast(tf.cast(x, target_dtype), x.dtype),
        summarize=summarize,
        message=message,
        name=name)


def assert_symmetric(matrix):
  matrix_t = tf.linalg.matrix_transpose(matrix)
  return with_dependencies(
      [assert_util.assert_near(matrix, matrix_t)], matrix)


def assert_nondecreasing(x, summarize=None, message=None, name=None):
  """Assert (batched) elements in `x` are non decreasing."""

  with tf.name_scope(name or 'assert_non_decreasing'):
    if message is None:
      message = '`Tensor` contained decreasing values.'
    x = tf.convert_to_tensor(x)
    x_ = tf.get_static_value(x)
    if x_ is not None:
      if not np.all(x_[..., :-1] <= x_[..., 1:]):
        raise ValueError(message)
      return x
    return assert_util.assert_less_equal(
        x[..., :-1],
        x[..., 1:],
        summarize=summarize,
        message=message)


def assert_nonnegative_integer_form(
    x, atol=None, rtol=None, name='assert_nonnegative_integer_form'):
  """Assert x is a non-negative tensor, and optionally of integers."""
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')
    assertions = [
        assert_util.assert_non_negative(
            x, message='`{}` must be non-negative.'.format(x)),
    ]
    if not dtype_util.is_integer(x.dtype):
      assertions += [
          assert_integer_form(
              x, atol=atol, rtol=rtol,
              message='`{}` cannot contain fractional components.'.format(x)),
      ]
    return assertions


def embed_check_nonnegative_integer_form(
    x, atol=None, rtol=None, name='embed_check_nonnegative_integer_form'):
  """Assert x is a non-negative tensor, and optionally of integers."""
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')
    return with_dependencies(assert_nonnegative_integer_form(
        x, atol=atol, rtol=rtol), x)


def same_dynamic_shape(a, b):
  """Returns whether a and b have the same dynamic shape.

  Args:
    a: `Tensor`
    b: `Tensor`

  Returns:
    `bool` `Tensor` representing if both tensors have the same shape.
  """
  a = tf.convert_to_tensor(a, name='a')
  b = tf.convert_to_tensor(b, name='b')

  # Here we can't just do tf.equal(a.shape, b.shape), since
  # static shape inference may break the equality comparison between
  # shape(a) and shape(b) in tf.equal.
  def all_shapes_equal():
    return tf.reduce_all(
        tf.equal(
            tf.concat([tf.shape(a), tf.shape(b)], 0),
            tf.concat([tf.shape(b), tf.shape(a)], 0)))

  # One of the shapes isn't fully defined, so we need to use the dynamic
  # shape.
  return tf.cond(
      pred=tf.equal(tf.rank(a), tf.rank(b)),
      true_fn=all_shapes_equal,
      false_fn=lambda: tf.constant(False))


def maybe_get_static_value(x, dtype=None):
  """Helper which tries to return a static value.

  Given `x`, extract it's value statically, optionally casting to a specific
  dtype. If this is not possible, None is returned.

  Args:
    x: `Tensor` for which to extract a value statically.
    dtype: Optional dtype to cast to.

  Returns:
    Statically inferred value if possible, otherwise None.
  """
  if x is None:
    return x
  try:
    # This returns an np.ndarray.
    x_ = tf.get_static_value(x)
  except TypeError:
    x_ = x
  if x_ is None or dtype is None:
    return x_
  return np.array(x_, dtype)


def _is_known_unsigned_by_dtype(dt):
  """Helper returning True if dtype is known to be unsigned."""
  return {
      tf.bool: True,
      tf.uint8: True,
      tf.uint16: True,
  }.get(dtype_util.base_dtype(dt), False)


def _is_known_signed_by_dtype(dt):
  """Helper returning True if dtype is known to be signed."""
  return {
      tf.float16: True,
      tf.float32: True,
      tf.float64: True,
      tf.int8: True,
      tf.int16: True,
      tf.int32: True,
      tf.int64: True,
  }.get(dtype_util.base_dtype(dt), False)


def _is_known_dtype(dt):
  """Helper returning True if dtype is known."""
  return _is_known_unsigned_by_dtype(dt) or _is_known_signed_by_dtype(dt)


def _largest_integer_by_dtype(dt):
  """Helper returning the largest integer exactly representable by dtype."""
  if not _is_known_dtype(dt):
    raise TypeError('Unrecognized dtype: {}'.format(dtype_util.name(dt)))
  if dtype_util.is_floating(dt):
    return int(2**(np.finfo(dtype_util.as_numpy_dtype(dt)).nmant + 1))
  if dtype_util.is_integer(dt):
    return np.iinfo(dtype_util.as_numpy_dtype(dt)).max
  if dtype_util.base_dtype(dt) == tf.bool:
    return int(1)
  # We actually can't land here but keep the case for completeness.
  raise TypeError('Unrecognized dtype: {}'.format(dtype_util.name(dt)))


def _smallest_integer_by_dtype(dt):
  """Helper returning the smallest integer exactly representable by dtype."""
  if not _is_known_dtype(dt):
    raise TypeError('Unrecognized dtype: {}'.format(dtype_util.name(dt)))
  if _is_known_unsigned_by_dtype(dt):
    return 0
  return -1 * _largest_integer_by_dtype(dt)


def _is_integer_like_by_dtype(dt):
  """Helper returning True if dtype.is_integer or is `bool`."""
  if not _is_known_dtype(dt):
    raise TypeError('Unrecognized dtype: {}'.format(dtype_util.name(dt)))
  return dtype_util.is_integer(dt) or dtype_util.base_dtype(dt) == tf.bool


def assert_categorical_event_shape(
    categorical_param, name='assert_check_categorical_event_shape'):
  """Embeds checks that categorical distributions don't have too many classes.

  A categorical-type distribution is one which, e.g., returns the class label
  rather than a one-hot encoding.  E.g., `Categorical(probs)`.

  Since distributions output samples in the same dtype as the parameters, we
  must ensure that casting doesn't lose precision. That is, the
  `parameter.dtype` implies a maximum number of classes. However, since shape is
  `int32` and categorical variables are presumed to be indexes into a `Tensor`,
  we must also ensure that the number of classes is no larger than the largest
  possible `int32` index, i.e., `2**31-1`.

  In other words the number of classes, `K`, must satisfy the following
  condition:

  ```python
  K <= min(
      int(2**31 - 1),  # Largest float as an index.
      {
          tf.float16: int(2**11),   # Largest int as a float16.
          tf.float32: int(2**24),
          tf.float64: int(2**53),
      }.get(dtype_util.base_dtype(categorical_param.dtype), 0))
  ```

  Args:
    categorical_param: Floating-point `Tensor` representing parameters of
      distribution over categories. The rightmost shape is presumed to be the
      number of categories.
    name: A name for this operation (optional).

  Returns:
    assertions: Python `list` of assertions.

  Raises:
    TypeError: if `categorical_param` has an unknown `dtype`.
    ValueError: if we can statically identify `categorical_param` as being too
      large (for being closed under int32/float casting).
  """
  with tf.name_scope(name):
    x = tf.convert_to_tensor(categorical_param, name='categorical_param')
    # The size must not exceed both of:
    # - The largest possible int32 (since categorical values are presumed to be
    #   indexes into a Tensor).
    # - The largest possible integer exactly representable under the given
    #   floating-point dtype (since we need to cast to/from).
    #
    # The chosen floating-point thresholds are 2**(1 + mantissa_bits).
    # For more details, see:
    # https://en.wikipedia.org/wiki/Floating-point_arithmetic#Internal_representation
    x_dtype = dtype_util.base_dtype(x.dtype)
    max_event_size = (
        _largest_integer_by_dtype(x_dtype)
        if dtype_util.is_floating(x_dtype) else 0)
    if max_event_size == 0:
      raise TypeError('Unable to validate size of unrecognized dtype '
                      '({}).'.format(dtype_util.name(x_dtype)))
    try:
      x_shape_static = tensorshape_util.with_rank_at_least(x.shape, 1)
    except ValueError:
      raise ValueError('A categorical-distribution parameter must have '
                       'at least 1 dimension.')
    event_size = tf.compat.dimension_value(x_shape_static[-1])
    if event_size is not None:
      if event_size < 2:
        raise ValueError('A categorical-distribution parameter must have at '
                         'least 2 events.')
      if event_size > max_event_size:
        raise ValueError('Number of classes exceeds `dtype` precision, i.e., '
                         '{} implies shape ({}) cannot exceed {}.'.format(
                             dtype_util.name(x_dtype), event_size,
                             max_event_size))
      return []

    event_size = tf.shape(x, out_type=tf.int64, name='x_shape')[-1]
    return [
        assert_util.assert_rank_at_least(
            x,
            1,
            message=('A categorical-distribution parameter must have '
                     'at least 1 dimension.')),
        assert_util.assert_greater_equal(
            tf.shape(x)[-1],
            2,
            message=('A categorical-distribution parameter must have at '
                     'least 2 events.')),
        assert_util.assert_less_equal(
            event_size,
            tf.convert_to_tensor(max_event_size, dtype=tf.int64),
            message='Number of classes exceeds `dtype` precision, '
            'i.e., {} dtype cannot exceed {} shape.'.format(
                dtype_util.name(x_dtype), max_event_size)),
    ]


def embed_check_categorical_event_shape(
    categorical_param, name='embed_check_categorical_event_shape'):
  """Embeds checks that categorical distributions don't have too many classes.

  A categorical-type distribution is one which, e.g., returns the class label
  rather than a one-hot encoding.  E.g., `Categorical(probs)`.

  Since distributions output samples in the same dtype as the parameters, we
  must ensure that casting doesn't lose precision. That is, the
  `parameter.dtype` implies a maximum number of classes. However, since shape is
  `int32` and categorical variables are presumed to be indexes into a `Tensor`,
  we must also ensure that the number of classes is no larger than the largest
  possible `int32` index, i.e., `2**31-1`.

  In other words the number of classes, `K`, must satisfy the following
  condition:

  ```python
  K <= min(
      int(2**31 - 1),  # Largest float as an index.
      {
          tf.float16: int(2**11),   # Largest int as a float16.
          tf.float32: int(2**24),
          tf.float64: int(2**53),
      }.get(dtype_util.base_dtype(categorical_param.dtype), 0))
  ```

  Args:
    categorical_param: Floating-point `Tensor` representing parameters of
      distribution over categories. The rightmost shape is presumed to be the
      number of categories.
    name: A name for this operation (optional).

  Returns:
    categorical_param: Input `Tensor` with appropriate assertions embedded.

  Raises:
    TypeError: if `categorical_param` has an unknown `dtype`.
    ValueError: if we can statically identify `categorical_param` as being too
      large (for being closed under int32/float casting).
  """
  with tf.name_scope(name):
    x = tf.convert_to_tensor(categorical_param, name='categorical_param')
    assertions = assert_categorical_event_shape(x)
    if not assertions:
      return x
    return with_dependencies(assertions, x)


def embed_check_integer_casting_closed(x,
                                       target_dtype,
                                       assert_nonnegative=True,
                                       assert_positive=False,
                                       name='embed_check_casting_closed'):
  """Ensures integers remain unaffected despite casting to/from int/float types.

  Example integer-types: `uint8`, `int32`, `bool`.
  Example floating-types: `float32`, `float64`.

  The largest possible integer representable by an IEEE754 floating-point is
  `2**(1 + mantissa_bits)` yet the largest possible integer as an int-type is
  `2**(bits - 1) - 1`. This function ensures that a `Tensor` purporting to have
  integer-form values can be cast to some other type without loss of precision.

  The smallest representable integer is the negative of the largest
  representable integer, except for types: `uint8`, `uint16`, `bool`. For these
  types, the smallest representable integer is `0`.

  Args:
    x: `Tensor` representing integer-form values.
    target_dtype: TF `dtype` under which `x` should have identical values.
    assert_nonnegative: `bool` indicating `x` should contain nonnegative values.
    assert_positive: `bool` indicating `x` should contain positive values.
    name: A name for this operation (optional).

  Returns:
    x: Input `Tensor` with appropriate assertions embedded.

  Raises:
    TypeError: if `x` is neither integer- nor floating-type.
    TypeError: if `target_dtype` is neither integer- nor floating-type.
    TypeError: if neither `x` nor `target_dtype` are integer-type.
  """

  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')
    if (not _is_integer_like_by_dtype(x.dtype) and
        not dtype_util.is_floating(x.dtype)):
      raise TypeError('{}.dtype must be floating- or '
                      'integer-type.'.format(dtype_util.name(x.dtype)))
    if (not _is_integer_like_by_dtype(target_dtype) and
        not dtype_util.is_floating(target_dtype)):
      raise TypeError('target_dtype ({}) must be floating- or '
                      'integer-type.'.format(dtype_util.name(target_dtype)))
    if (not _is_integer_like_by_dtype(x.dtype) and
        not _is_integer_like_by_dtype(target_dtype)):
      raise TypeError('At least one of {}.dtype ({}) and target_dtype ({}) '
                      'must be integer-type.'.format(
                          x, dtype_util.name(x.dtype),
                          dtype_util.name(target_dtype)))

    assertions = []
    if assert_positive:
      assertions += [
          assert_util.assert_positive(x, message='Elements must be positive.'),
      ]
    elif assert_nonnegative:
      assertions += [
          assert_util.assert_non_negative(
              x, message='Elements must be non-negative.'),
      ]

    if dtype_util.is_floating(x.dtype):
      # Being here means _is_integer_like_by_dtype(target_dtype) = True.
      # Since this check implies the magnitude check below, we need only it.
      assertions += [
          assert_casting_closed(
              x,
              target_dtype,
              message='Elements must be {}-equivalent.'.format(
                  dtype_util.name(target_dtype))),
      ]
    else:
      if (_largest_integer_by_dtype(x.dtype) >
          _largest_integer_by_dtype(target_dtype)):
        # Cast may lose integer precision.
        assertions += [
            assert_util.assert_less_equal(
                x,
                _largest_integer_by_dtype(target_dtype),
                message=('Elements cannot exceed {}.'.format(
                    _largest_integer_by_dtype(target_dtype)))),
        ]
      if (not assert_nonnegative and (_smallest_integer_by_dtype(
          x.dtype) < _smallest_integer_by_dtype(target_dtype))):
        assertions += [
            assert_util.assert_greater_equal(
                x,
                _smallest_integer_by_dtype(target_dtype),
                message=('Elements cannot be smaller than {}.'.format(
                    _smallest_integer_by_dtype(target_dtype)))),
        ]

    if not assertions:
      return x
    return with_dependencies(assertions, x)


def rotate_transpose(x, shift, name='rotate_transpose'):
  """Circularly moves dims left or right.

  Effectively identical to:

  ```python
  numpy.transpose(x, numpy.roll(numpy.arange(len(x.shape)), shift))
  ```

  When `validate_args=False` additional graph-runtime checks are
  performed. These checks entail moving data from to GPU to CPU.

  Example:

  ```python
  x = tf.random.normal([1, 2, 3, 4])  # Tensor of shape [1, 2, 3, 4].
  rotate_transpose(x, -1).shape == [2, 3, 4, 1]
  rotate_transpose(x, -2).shape == [3, 4, 1, 2]
  rotate_transpose(x,  1).shape == [4, 1, 2, 3]
  rotate_transpose(x,  2).shape == [3, 4, 1, 2]
  rotate_transpose(x,  7).shape == rotate_transpose(x, 3).shape  # [2, 3, 4, 1]
  rotate_transpose(x, -7).shape == rotate_transpose(x, -3).shape  # [4, 1, 2, 3]
  ```

  Args:
    x: `Tensor`.
    shift: `Tensor`. Number of dimensions to transpose left (shift<0) or
      transpose right (shift>0).
    name: Python `str`. The name to give this op.

  Returns:
    rotated_x: Input `Tensor` with dimensions circularly rotated by shift.

  Raises:
    TypeError: if shift is not integer type.
  """
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')
    shift = ps.convert_to_shape_tensor(shift, name='shift')
    # We do not assign back to preserve constant-ness.
    assert_util.assert_integer(shift)
    shift_value_static = tf.get_static_value(shift)
    ndims = tensorshape_util.rank(x.shape)
    if ndims is not None and shift_value_static is not None:
      if ndims < 2:
        return x
      shift_value_static = np.sign(shift_value_static) * (
          abs(shift_value_static) % ndims)
      if shift_value_static == 0:
        return x
      perm = np.roll(np.arange(ndims), shift_value_static)
      return tf.transpose(a=x, perm=perm)
    else:
      # Consider if we always had a positive shift, and some specified
      # direction.
      # When shifting left we want the new array:
      #   last(x, n-shift) + first(x, shift)
      # and if shifting right then we want:
      #   last(x, shift) + first(x, n-shift)
      # Observe that last(a) == slice(a, n) and first(a) == slice(0, a).
      # Also, we can encode direction and shift as one: direction * shift.
      # Combining these facts, we have:
      #   a = cond(shift<0, -shift, n-shift)
      #   last(x, n-a) + first(x, a) == x[a:n] + x[0:a]
      # Finally, we transform shift by modulo length so it can be specified
      # independently from the array upon which it operates (like python).
      ndims = tf.rank(x)
      shift = tf.where(
          tf.less(shift, 0), -shift % ndims, ndims - shift % ndims)
      first = tf.range(0, shift)
      last = tf.range(shift, ndims)
      perm = tf.concat([last, first], 0)
      return tf.transpose(a=x, perm=perm)


def pick_vector(cond, true_vector, false_vector, name='pick_vector'):
  """Picks possibly different length row `Tensor`s based on condition.

  Value `Tensor`s should have exactly one dimension.

  If `cond` is a python Boolean or `tf.constant` then either `true_vector` or
  `false_vector` is immediately returned. I.e., no graph nodes are created and
  no validation happens.

  Args:
    cond: `Tensor`. Must have `dtype=tf.bool` and be scalar.
    true_vector: `Tensor` of one dimension. Returned when cond is `True`.
    false_vector: `Tensor` of one dimension. Returned when cond is `False`.
    name: Python `str`. The name to give this op.

  Example:

  ```python
  pick_vector(tf.less(0, 5), tf.range(10, 12), tf.range(15, 18))  # [10, 11]
  pick_vector(tf.less(5, 0), tf.range(10, 12), tf.range(15, 18))  # [15, 16, 17]
  ```

  Returns:
    true_or_false_vector: `Tensor`.

  Raises:
    TypeError: if `cond.dtype != tf.bool`
    TypeError: if `cond` is not a constant and
      `true_vector.dtype != false_vector.dtype`
  """
  with tf.name_scope(name):
    cond = tf.convert_to_tensor(cond, dtype_hint=tf.bool, name='cond')
    if cond.dtype != tf.bool:
      raise TypeError(
          '{}.dtype={} which is not {}'.format(cond, cond.dtype, tf.bool))

    true_vector = tf.convert_to_tensor(true_vector, name='true_vector')
    false_vector = tf.convert_to_tensor(false_vector, name='false_vector')
    if true_vector.dtype != false_vector.dtype:
      raise TypeError(
          '{}.dtype={} does not match {}.dtype={}'.format(
              true_vector, true_vector.dtype, false_vector, false_vector.dtype))

    cond_value_static = tf.get_static_value(cond)
    if cond_value_static is not None:
      return true_vector if cond_value_static else false_vector
    n = tf.shape(true_vector)[0]
    return tf.slice(
        tf.concat([true_vector, false_vector], 0), [tf.where(cond, 0, n)],
        [tf.where(cond, n, -1)])


def prefer_static_broadcast_shape(shape1,
                                  shape2,
                                  name='prefer_static_broadcast_shape'):
  """Convenience function which statically broadcasts shape when possible.

  Args:
    shape1:  `1-D` integer `Tensor`.  Already converted to tensor!
    shape2:  `1-D` integer `Tensor`.  Already converted to tensor!
    name:  A string name to prepend to created ops.

  Returns:
    The broadcast shape, either as `TensorShape` (if broadcast can be done
      statically), or as a `Tensor`.
  """
  with tf.name_scope(name):

    def make_shape_tensor(x):
      return tf.convert_to_tensor(x, name='shape', dtype=tf.int32)

    def get_tensor_shape(s):
      if isinstance(s, tf.TensorShape):
        return s
      s_ = tf.get_static_value(make_shape_tensor(s))
      if s_ is not None:
        return tf.TensorShape(s_)
      return None

    def get_shape_tensor(s):
      if not isinstance(s, tf.TensorShape):
        return make_shape_tensor(s)
      if tensorshape_util.is_fully_defined(s):
        return make_shape_tensor(tensorshape_util.as_list(s))
      raise ValueError('Cannot broadcast from partially '
                       'defined `TensorShape`.')

    shape1_ = get_tensor_shape(shape1)
    shape2_ = get_tensor_shape(shape2)
    if shape1_ is not None and shape2_ is not None:
      return tf.broadcast_static_shape(shape1_, shape2_)

    shape1_ = get_shape_tensor(shape1)
    shape2_ = get_shape_tensor(shape2)
    return tf.broadcast_dynamic_shape(shape1_, shape2_)


def prefer_static_rank(x):
  """Return static rank of tensor `x` if available, else `tf.rank(x)`.

  Args:
    x: `Tensor` (already converted).

  Returns:
    Numpy array (if static rank is obtainable), else `Tensor`.
  """
  return ps.rank(x)


def prefer_static_shape(x):
  """Return static shape of tensor `x` if available, else `tf.shape(x)`.

  Args:
    x: `Tensor` (already converted).

  Returns:
    Numpy array (if static shape is obtainable), else `Tensor`.
  """
  return ps.shape(x)


def prefer_static_value(x):
  """Return static value of tensor `x` if available, else `x`.

  Args:
    x: `Tensor` (already converted).

  Returns:
    Numpy array (if static value is obtainable), else `Tensor`.
  """
  static_x = tf.get_static_value(x)
  if static_x is not None:
    return static_x
  return x


def gen_new_seed(seed, salt):
  """Generate a new seed, from the given seed and salt."""
  if seed is None:
    return None
  string = (str(seed) + salt).encode('utf-8')
  return int(hashlib.md5(string).hexdigest()[:8], 16) & 0x7FFFFFFF


def process_quadrature_grid_and_probs(quadrature_grid_and_probs,
                                      dtype,
                                      validate_args,
                                      name=None):
  """Validates quadrature grid, probs or computes them as necessary.

  Args:
    quadrature_grid_and_probs: Python pair of `float`-like `Tensor`s
      representing the sample points and the corresponding (possibly
      normalized) weight.  When `None`, defaults to:
        `np.polynomial.hermite.hermgauss(deg=8)`.
    dtype: The expected `dtype` of `grid` and `probs`.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
     quadrature_grid_and_probs: Python pair of `float`-like `Tensor`s
      representing the sample points and the corresponding (possibly
      normalized) weight.

  Raises:
    ValueError: if `quadrature_grid_and_probs is not None` and
      `len(quadrature_grid_and_probs[0]) != len(quadrature_grid_and_probs[1])`
  """
  with tf.name_scope(name or 'process_quadrature_grid_and_probs'):
    if quadrature_grid_and_probs is None:
      grid, probs = np.polynomial.hermite.hermgauss(deg=8)
      grid = grid.astype(dtype_util.as_numpy_dtype(dtype))
      probs = probs.astype(dtype_util.as_numpy_dtype(dtype))
      probs /= np.linalg.norm(probs, ord=1, keepdims=True)
      grid = tf.convert_to_tensor(grid, name='grid', dtype=dtype)
      probs = tf.convert_to_tensor(probs, name='probs', dtype=dtype)
      return grid, probs

    grid, probs = tuple(quadrature_grid_and_probs)
    grid = tf.convert_to_tensor(grid, name='grid', dtype=dtype)
    probs = tf.convert_to_tensor(probs, name='unnormalized_probs', dtype=dtype)
    probs /= tf.norm(probs, ord=1, axis=-1, keepdims=True, name='probs')

    def _static_event_size(x):
      """Returns the static size of a specific dimension or `None`."""
      return tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(x.shape, 1)[-1])

    m, n = _static_event_size(probs), _static_event_size(grid)
    if m is not None and n is not None:
      if m != n:
        raise ValueError('`quadrature_grid_and_probs` must be a `tuple` of '
                         'same-length zero-th-dimension `Tensor`s '
                         '(saw lengths {}, {})'.format(m, n))
    elif validate_args:
      assertions = [
          assert_util.assert_equal(
              ps.dimension_size(probs, axis=-1),
              ps.dimension_size(grid, axis=-1),
              message=('`quadrature_grid_and_probs` must be a `tuple` of '
                       'same-length zero-th-dimension `Tensor`s')),
      ]
      with tf.control_dependencies(assertions):
        grid = tf.identity(grid)
        probs = tf.identity(probs)
    return grid, probs


def pad(x, axis, front=False, back=False, value=0, count=1, name=None):
  """Pads `value` to the front and/or back of a `Tensor` dim, `count` times.

  Args:
    x: `Tensor` input.
    axis: Scalar `int`-like `Tensor` representing the single dimension to pad.
      (Negative indexing is supported.)
    front: Python `bool`; if `True` the beginning of the `axis` dimension is
      padded with `value`, `count` times. If `False` no front padding is made.
    back: Python `bool`; if `True` the end of the `axis` dimension is padded
      with `value`, `count` times. If `False` no end padding is made.
    value: Scalar `int`-like `Tensor` representing the actual value added to the
      front and/or back of the `axis` dimension of `x`.
    count: Scalar `int`-like `Tensor` representing number of elements added to
      the front and/or back of the `axis` dimension of `x`. E.g., if `front =
      back = True` then `2 * count` elements are added.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    pad: The padded version of input `x`.

  Raises:
    ValueError: if both `front` and `back` are `False`.
    TypeError: if `count` is not `int`-like.
  """
  with tf.name_scope(name or 'pad'):
    x = tf.convert_to_tensor(x, name='x')
    value = tf.convert_to_tensor(value, dtype=x.dtype, name='value')
    count = ps.convert_to_shape_tensor(count, name='count')
    if not dtype_util.is_integer(count.dtype):
      raise TypeError('`count.dtype` (`{}`) must be `int`-like.'.format(
          dtype_util.name(count.dtype)))
    if not front and not back:
      raise ValueError('At least one of `front`, `back` must be `True`.')
    ndims = (
        tensorshape_util.rank(x.shape)
        if tensorshape_util.rank(x.shape) is not None else tf.rank(
            x, name='ndims'))
    axis = ps.convert_to_shape_tensor(axis, name='axis')
    axis_ = tf.get_static_value(axis)
    if axis_ is not None:
      axis = axis_
      if axis < 0:
        axis = ndims + axis
      count_ = tf.get_static_value(count)
      if axis_ >= 0 or tensorshape_util.rank(x.shape) is not None:
        head = x.shape[:axis]
        mid_dim_value = tf.compat.dimension_value(x.shape[axis])
        if count_ is None or mid_dim_value is None:
          middle = tf.TensorShape(None)
        else:
          middle = tf.TensorShape(mid_dim_value + count_ * (front + back))
        tail = x.shape[axis + 1:]
        final_shape = tensorshape_util.concatenate(
            head, tensorshape_util.concatenate(middle, tail))
      else:
        final_shape = None
    else:
      axis = tf.where(axis < 0, ndims + axis, axis)
      final_shape = None
    x = tf.pad(
        x,
        paddings=ps.one_hot(
            indices=ps.stack([axis if front else -1, axis if back else -1]),
            depth=ndims,
            axis=0,
            on_value=count,
            dtype=tf.int32),
        constant_values=value)
    if final_shape is not None:
      tensorshape_util.set_shape(x, final_shape)
    return x


def parent_frame_arguments():
  """Returns parent frame arguments.

  When called inside a function, returns a dictionary with the caller's function
  arguments. These are positional arguments and keyword arguments (**kwargs),
  while variable arguments (*varargs) are excluded.

  When called at global scope, this will return an empty dictionary, since there
  are no arguments.

  WARNING: If caller function argument names are overloaded before invoking
  this method, then values will reflect the overloaded value. For this reason,
  we recommend calling `parent_frame_arguments` at the beginning of the
  function.
  """
  # All arguments and the names used for *varargs, and **kwargs
  arg_names, variable_arg_name, keyword_arg_name, local_vars = (
      tf_inspect._inspect.getargvalues(  # pylint: disable=protected-access
          # Get the first frame of the caller of this method.
          tf_inspect._inspect.stack()[1][0]))  # pylint: disable=protected-access

  # Remove the *varargs, and flatten the **kwargs. Both are
  # nested lists.
  local_vars.pop(variable_arg_name, {})
  keyword_args = local_vars.pop(keyword_arg_name, {})

  final_args = {}
  # Copy over arguments and their values. In general, local_vars
  # may contain more than just the arguments, since this method
  # can be called anywhere in a function.
  for arg_name in arg_names:
    final_args[arg_name] = local_vars.pop(arg_name)
  final_args.update(keyword_args)

  return final_args


class AppendDocstring(object):
  """Helper class to promote private subclass docstring to public counterpart.

  Example:

  ```python
  class TransformedDistribution(Distribution):
    @AppendDocstring(
      additional_note='A special note!',
      kwargs_dict={'foo': 'An extra arg.'})
    def _prob(self, y, foo=None):
      pass
  ```

  In this case, the `AppendDocstring` decorator appends the `additional_note` to
  the docstring of `prob` (not `_prob`) and adds a new `kwargs`
  section with each dictionary item as a bullet-point.

  For a more detailed example, see `TransformedDistribution`.
  """

  def __init__(self, additional_note='', kwargs_dict=None):
    """Initializes the AppendDocstring object.

    Args:
      additional_note: Python string added as additional docstring to public
        version of function.
      kwargs_dict: Python string/string dictionary representing specific kwargs
        expanded from the **kwargs input.

    Raises:
      ValueError: if kwargs_dict.key contains whitespace.
      ValueError: if kwargs_dict.value contains newlines.
    """
    self._additional_note = additional_note
    if kwargs_dict:
      bullets = []
      for key in sorted(kwargs_dict.keys()):
        value = kwargs_dict[key]
        if any(x.isspace() for x in key):
          raise ValueError('Parameter name \'%s\' contains whitespace.' % key)
        value = value.lstrip()
        if '\n' in value:
          raise ValueError(
              'Parameter description for \'%s\' contains newlines.' % key)
        bullets.append('*  `%s`: %s' % (key, value))
      self._additional_note += ('\n\n##### `kwargs`:\n\n' + '\n'.join(bullets))

  def __call__(self, fn):

    @functools.wraps(fn)
    def _fn(*args, **kwargs):
      return fn(*args, **kwargs)

    if _fn.__doc__ is None:
      _fn.__doc__ = self._additional_note
    else:
      _fn.__doc__ += '\n%s' % self._additional_note
    return _fn


def expand_to_vector(x, tensor_name=None, op_name=None, validate_args=False):
  """Transform a 0-D or 1-D `Tensor` to be 1-D.

  For user convenience, many parts of the TensorFlow Probability API accept
  inputs of rank 0 or 1 -- i.e., allowing an `event_shape` of `[5]` to be passed
  to the API as either `5` or `[5]`.  This function can be used to transform
  such an argument to always be 1-D.

  NOTE: Python or NumPy values will be converted to `Tensor`s with standard type
  inference/conversion.  In particular, an empty list or tuple will become an
  empty `Tensor` with dtype `float32`.  Callers should convert values to
  `Tensor`s before calling this function if different behavior is desired
  (e.g. converting empty lists / other values to `Tensor`s with dtype `int32`).

  Args:
    x: A 0-D or 1-D `Tensor`.
    tensor_name: Python `str` name for `Tensor`s created by this function.
    op_name: Python `str` name for `Op`s created by this function.
    validate_args: Python `bool, default `False`.  When `True`, arguments may be
      checked for validity at execution time, possibly degrading runtime
      performance.  When `False`, invalid inputs may silently render incorrect
        outputs.
  Returns:
    vector: a 1-D `Tensor`.
  """
  with tf.name_scope(op_name or 'expand_to_vector'):
    x_orig = x
    x = ps.convert_to_shape_tensor(x, name='x')
    ndims = tensorshape_util.rank(x.shape)

    if ndims is None:
      # Maybe expand ndims from 0 to 1.
      if validate_args:
        x = with_dependencies([
            assert_util.assert_rank_at_most(
                x, 1, message='Input is neither scalar nor vector.')
        ], x)
      ndims = ps.rank(x)
      expanded_shape = pick_vector(
          ps.equal(ndims, 0), np.array([1], dtype=np.int32), ps.shape(x))
      return ps.reshape(x, expanded_shape)

    elif ndims == 0:
      # Definitely expand ndims from 0 to 1.
      return ps.convert_to_shape_tensor(
          ps.reshape(x_orig, [1]), name=tensor_name)

    elif ndims != 1:
      raise ValueError('Input is neither scalar nor vector.')

    # ndims == 1
    return x


def with_dependencies(dependencies, output_tensor, name=None):
  """Produces the content of `output_tensor` only after `dependencies`.

  In some cases, a user may want the output of an operation to be consumed
  externally only after some other dependencies have run first. This function
  returns `output_tensor`, but only after all operations in `dependencies` have
  run. Note that this means that there is no guarantee that `output_tensor` will
  be evaluated after any `dependencies` have run.

  See also `tf.tuple` and `tf.group`.

  Args:
    dependencies: Iterable of operations to run before this op finishes.
    output_tensor: A `Tensor` or `IndexedSlices` that will be returned.
    name: (Optional) A name for this operation.

  Returns:
    output_with_deps: Same as `output_tensor` but with embedded dependencies.

  Raises:
    TypeError: if `output_tensor` is not a `Tensor` or `IndexedSlices`.
  """
  if tf.executing_eagerly():
    return output_tensor
  with tf.name_scope(name or 'control_dependency') as name:
    with tf.control_dependencies(d for d in dependencies if d is not None):
      output_tensor = tf.convert_to_tensor(output_tensor)
      if isinstance(output_tensor, tf.Tensor):
        return tf.identity(output_tensor, name=name)
      else:
        return tf.IndexedSlices(
            tf.identity(output_tensor.values, name=name),
            output_tensor.indices,
            output_tensor.dense_shape)


def is_distribution_instance(d):
  """Standardizes our definition of being a `tfd.Distribution`."""
  return (not tf_inspect.isclass(d) and
          hasattr(d, 'log_prob') and
          hasattr(d, 'sample'))


def extend_cdf_outside_support(x, computed_cdf, low=None, high=None):
  """Returns a CDF correctly extended outside a distribution's support interval.

  This helper is useful when the natural formula for computing a CDF computes
  the wrong thing outside the distribution's support.  For instance, a `nan` due
  to invoking some special function with parameters out of bounds.

  Note that correct gradients may require the "double-where" trick.  For that,
  the caller must compute the `computed_cdf` Tensor with a doctored input that
  replaces all out-of-support values of `x` with a "safe" in-support value that
  is guaranteed not to produce a `nan` in the `computed_cdf` Tensor.  After
  calling `extend_cdf_outside_support` those doctored CDF values will be ignored
  in the primal computation, and any `nan`s thus avoided will not pollute the
  gradients.

  Args:
    x: Tensor of input values at which the CDF is desired.
    computed_cdf: Tensor of values computed for the CDF.  Must broadcast with
      `x`.  Entries corresponding to points `x` falling below or above the given
      support are ignored and replaced with 0 or 1, respectively.
    low: Tensor of lower bounds for the support.  Must broadcast with `x`.
    high: Tensor of upper bounds for the support.  Must broadcast with `x`.

  Returns:
    cdf: Tensor of corrected CDF values.  Each entry is either 0 if the
      corresponding entry of `x` is outside the support from below, or the
      computed CDF value if `x` is in the support, or 1 if `x` is outside the
      support from above.
  """
  if low is not None:
    computed_cdf = tf.where(x >= low, computed_cdf, 0.)
  if high is not None:
    computed_cdf = tf.where(x < high, computed_cdf, 1.)
  return computed_cdf
