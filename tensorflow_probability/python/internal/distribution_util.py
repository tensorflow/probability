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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import hashlib
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


def _convert_to_tensor(x, name, dtype=None):
  return None if x is None else tf.convert_to_tensor(
      value=x, name=name, dtype=dtype)


def mixture_stddev(mixture_weight_vector, mean_vector, stddev_vector):
  """Computes the standard deviation of a mixture distribution.

  This function works regardless of the component distribution, so long as
  each component's mean and standard deviation can be provided.

  Args:
    mixture_weight_vector: A 2D tensor with shape [batch_size, num_components]
    mean_vector: A 2D tensor of mixture component means. Has shape `[batch_size,
      num_components]`.
    stddev_vector: A 2D tensor of mixture component standard deviations. Has
      shape `[batch_size, num_components]`.

  Returns:
    A 1D tensor of shape `[batch_size]` representing the standard deviation of
    the mixture distribution with given weights and component means and standard
    deviations.
  Raises:
    ValueError: If the shapes of the input tensors are not as expected.
  """
  tensorshape_util.assert_has_rank(mixture_weight_vector.shape, 2)
  if not tensorshape_util.is_compatible_with(mean_vector.shape,
                                             mixture_weight_vector.shape):
    raise ValueError("Expecting means to have same shape as mixture weights.")
  if not tensorshape_util.is_compatible_with(stddev_vector.shape,
                                             mixture_weight_vector.shape):
    raise ValueError("Expecting stddevs to have same shape as mixture weights.")

  # Reshape the distribution parameters for batched vectorized dot products.
  pi_for_dot_prod = tf.expand_dims(mixture_weight_vector, axis=1)
  mu_for_dot_prod = tf.expand_dims(mean_vector, axis=2)
  sigma_for_dot_prod = tf.expand_dims(stddev_vector, axis=2)

  # weighted average of component means under mixture distribution.
  mean_wa = tf.matmul(pi_for_dot_prod, mu_for_dot_prod)
  mean_wa = tf.reshape(mean_wa, (-1,))
  # weighted average of component variances under mixture distribution.
  var_wa = tf.matmul(pi_for_dot_prod, tf.square(sigma_for_dot_prod))
  var_wa = tf.reshape(var_wa, (-1,))
  # weighted average of component squared means under mixture distribution.
  sq_mean_wa = tf.matmul(pi_for_dot_prod, tf.square(mu_for_dot_prod))
  sq_mean_wa = tf.reshape(sq_mean_wa, (-1,))
  mixture_variance = var_wa + sq_mean_wa - tf.square(mean_wa)
  return tf.sqrt(mixture_variance)


def make_tril_scale(loc=None,
                    scale_tril=None,
                    scale_diag=None,
                    scale_identity_multiplier=None,
                    shape_hint=None,
                    validate_args=False,
                    assert_positive=False,
                    name=None):
  """Creates a LinearOperator representing a lower triangular matrix.

  Args:
    loc: Floating-point `Tensor`. This is used for inferring shape in the case
      where only `scale_identity_multiplier` is set.
    scale_tril: Floating-point `Tensor` representing the diagonal matrix.
      `scale_diag` has shape [N1, N2, ...  k, k], which represents a k x k lower
      triangular matrix. When `None` no `scale_tril` term is added to the
      LinearOperator. The upper triangular elements above the diagonal are
      ignored.
    scale_diag: Floating-point `Tensor` representing the diagonal matrix.
      `scale_diag` has shape [N1, N2, ...  k], which represents a k x k diagonal
      matrix. When `None` no diagonal term is added to the LinearOperator.
    scale_identity_multiplier: floating point rank 0 `Tensor` representing a
      scaling done to the identity matrix. When `scale_identity_multiplier =
      scale_diag = scale_tril = None` then `scale += IdentityMatrix`. Otherwise
      no scaled-identity-matrix is added to `scale`.
    shape_hint: scalar integer `Tensor` representing a hint at the dimension of
      the identity matrix when only `scale_identity_multiplier` is set.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness.
    assert_positive: Python `bool` indicating whether LinearOperator should be
      checked for being positive definite.
    name: Python `str` name given to ops managed by this object.

  Returns:
    `LinearOperator` representing a lower triangular matrix.

  Raises:
    ValueError:  If only `scale_identity_multiplier` is set and `loc` and
      `shape_hint` are both None.
  """

  def _maybe_attach_assertion(x):
    if not validate_args:
      return x
    if assert_positive:
      return with_dependencies([
          assert_util.assert_positive(
              tf.linalg.diag_part(x), message="diagonal part must be positive"),
      ], x)
    return with_dependencies([
        assert_util.assert_none_equal(
            tf.linalg.diag_part(x),
            tf.zeros([], x.dtype),
            message="diagonal part must be non-zero"),
    ], x)

  with tf.name_scope(name or "make_tril_scale"):

    dtype = dtype_util.common_dtype(
        [loc, scale_tril, scale_diag, scale_identity_multiplier],
        preferred_dtype=tf.float32)
    loc = _convert_to_tensor(loc, name="loc", dtype=dtype)
    scale_tril = _convert_to_tensor(scale_tril, name="scale_tril", dtype=dtype)
    scale_diag = _convert_to_tensor(scale_diag, name="scale_diag", dtype=dtype)
    scale_identity_multiplier = _convert_to_tensor(
        scale_identity_multiplier,
        name="scale_identity_multiplier",
        dtype=dtype)

  if scale_tril is not None:
    scale_tril = tf.linalg.band_part(scale_tril, -1, 0)  # Zero out TriU.
    tril_diag = tf.linalg.diag_part(scale_tril)
    if scale_diag is not None:
      tril_diag += scale_diag
    if scale_identity_multiplier is not None:
      tril_diag += scale_identity_multiplier[..., tf.newaxis]

    scale_tril = tf.linalg.set_diag(scale_tril, tril_diag)

    return tf.linalg.LinearOperatorLowerTriangular(
        tril=_maybe_attach_assertion(scale_tril),
        is_non_singular=True,
        is_self_adjoint=False,
        is_positive_definite=assert_positive)

  return make_diag_scale(
      loc=loc,
      scale_diag=scale_diag,
      scale_identity_multiplier=scale_identity_multiplier,
      shape_hint=shape_hint,
      validate_args=validate_args,
      assert_positive=assert_positive,
      name=name)


def make_diag_scale(loc=None,
                    scale_diag=None,
                    scale_identity_multiplier=None,
                    shape_hint=None,
                    validate_args=False,
                    assert_positive=False,
                    name=None,
                    dtype=None):
  """Creates a LinearOperator representing a diagonal matrix.

  Args:
    loc: Floating-point `Tensor`. This is used for inferring shape in the case
      where only `scale_identity_multiplier` is set.
    scale_diag: Floating-point `Tensor` representing the diagonal matrix.
      `scale_diag` has shape [N1, N2, ...  k], which represents a k x k diagonal
      matrix. When `None` no diagonal term is added to the LinearOperator.
    scale_identity_multiplier: floating point rank 0 `Tensor` representing a
      scaling done to the identity matrix. When `scale_identity_multiplier =
      scale_diag = scale_tril = None` then `scale += IdentityMatrix`. Otherwise
      no scaled-identity-matrix is added to `scale`.
    shape_hint: scalar integer `Tensor` representing a hint at the dimension of
      the identity matrix when only `scale_identity_multiplier` is set.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness.
    assert_positive: Python `bool` indicating whether LinearOperator should be
      checked for being positive definite.
    name: Python `str` name given to ops managed by this object.
    dtype: TF `DType` to prefer when converting args to `Tensor`s. Else, we fall
      back to a compatible dtype across all of `loc`, `scale_diag`, and
      `scale_identity_multiplier`.

  Returns:
    `LinearOperator` representing a lower triangular matrix.

  Raises:
    ValueError:  If only `scale_identity_multiplier` is set and `loc` and
      `shape_hint` are both None.
  """

  def _maybe_attach_assertion(x):
    if not validate_args:
      return x
    if assert_positive:
      return with_dependencies([
          assert_util.assert_positive(
              x, message="diagonal part must be positive"),
      ], x)
    return with_dependencies([
        assert_util.assert_none_equal(
            x, tf.zeros([], x.dtype), message="diagonal part must be non-zero")
    ], x)

  with tf.name_scope(name or "make_diag_scale"):
    if dtype is None:
      dtype = dtype_util.common_dtype(
          [loc, scale_diag, scale_identity_multiplier],
          preferred_dtype=tf.float32)
    loc = _convert_to_tensor(loc, name="loc", dtype=dtype)
    scale_diag = _convert_to_tensor(scale_diag, name="scale_diag", dtype=dtype)
    scale_identity_multiplier = _convert_to_tensor(
        scale_identity_multiplier,
        name="scale_identity_multiplier",
        dtype=dtype)

    if scale_diag is not None:
      if scale_identity_multiplier is not None:
        scale_diag += scale_identity_multiplier[..., tf.newaxis]
      return tf.linalg.LinearOperatorDiag(
          diag=_maybe_attach_assertion(scale_diag),
          is_non_singular=True,
          is_self_adjoint=True,
          is_positive_definite=assert_positive)

    if loc is None and shape_hint is None:
      raise ValueError("Cannot infer `event_shape` unless `loc` or "
                       "`shape_hint` is specified.")

    num_rows = shape_hint
    del shape_hint
    if num_rows is None:
      num_rows = tf.compat.dimension_value(loc.shape[-1])
      if num_rows is None:
        num_rows = tf.shape(input=loc)[-1]

    if scale_identity_multiplier is None:
      return tf.linalg.LinearOperatorIdentity(
          num_rows=num_rows,
          dtype=dtype,
          is_self_adjoint=True,
          is_positive_definite=True,
          assert_proper_shapes=validate_args)

    return tf.linalg.LinearOperatorScaledIdentity(
        num_rows=num_rows,
        multiplier=_maybe_attach_assertion(scale_identity_multiplier),
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=assert_positive,
        assert_proper_shapes=validate_args)


def shapes_from_loc_and_scale(loc, scale, name="shapes_from_loc_and_scale"):
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
    event_size = scale.range_dimension_tensor()
    event_size_ = tf.get_static_value(event_size)
    loc_event_size_ = (None if loc is None
                       else tf.compat.dimension_value(loc.shape[-1]))

    if event_size_ is not None and loc_event_size_ is not None:
      # Static check that event shapes match.
      if loc_event_size_ != 1 and loc_event_size_ != event_size_:
        raise ValueError(
            "Event size of 'scale' ({}) could not be broadcast up to that "
            "of 'loc' ({}).".format(event_size_, loc_event_size_))
    elif loc_event_size_ is not None and loc_event_size_ != 1:
      event_size_ = loc_event_size_

    if event_size_ is None:
      event_shape = event_size[tf.newaxis]
    else:
      event_shape = tf.convert_to_tensor(
          value=np.reshape(event_size_, [1]),
          dtype=tf.int32,
          name="event_shape")

    # Get batch shape.
    batch_shape = scale.batch_shape_tensor()
    if loc is not None:
      loc_batch_shape = tensorshape_util.with_rank_at_least(loc.shape, 1)[:-1]
      if tensorshape_util.rank(
          loc.shape) is None or not tensorshape_util.is_fully_defined(
              loc_batch_shape):
        loc_batch_shape = tf.shape(input=loc)[:-1]
      else:
        loc_batch_shape = tf.convert_to_tensor(
            value=loc_batch_shape, dtype=tf.int32, name="loc_batch_shape")
      # This is defined in the core util module.
      batch_shape = prefer_static_broadcast_shape(batch_shape, loc_batch_shape)  # pylint: disable=undefined-variable
      batch_shape = tf.convert_to_tensor(
          value=batch_shape, dtype=tf.int32, name="batch_shape")

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
  d_shape = tf.shape(input=tensors[0])
  for t in tensors[1:]:
    d_shape = tf.broadcast_dynamic_shape(d_shape, tf.shape(input=t))
  return d_shape


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
    raise TypeError("Expected argument 'scale' to be instance of LinearOperator"
                    ". Found: %s" % scale)
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
    raise TypeError("dtype mismatch; "
                    "distribution.dtype=\"{}\" is not \"{}\"".format(
                        dtype_util.name(distribution.dtype),
                        dtype_util.name(expected_base_dtype)))

  # Although `reparameterization_type` is a static property, we guard it by
  # `validate_args`. This allows users to use a `distribution` which is not
  # reparameterized itself. However, we tacitly assume that although the
  # distribution is not reparameterized, it only depends on non-trainable
  # variables.
  if validate_args and (distribution.reparameterization_type !=
                        reparameterization.FULLY_REPARAMETERIZED):
    raise ValueError("Base distribution should be reparameterized or be "
                     "a function of non-trainable variables; "
                     "distribution.reparameterization_type = \"{}\" "
                     "!= \"FULLY_REPARAMETERIZED\".".format(
                         distribution.reparameterization_type))
  with tf.name_scope("check_distribution"):
    assertions = []

    def check_is_scalar(is_scalar, name):
      is_scalar_ = tf.get_static_value(is_scalar)
      if is_scalar_ is not None:
        if not is_scalar_:
          raise ValueError("distribution must be scalar; "
                           "distribution.{}=False is not True".format(name))
      elif validate_args:
        assertions.append(
            assert_util.assert_equal(
                is_scalar,
                True,
                message=("distribution must be scalar; "
                         "distribution.{}=False is not True".format(name))))

    check_is_scalar(distribution.is_scalar_event(), "is_scalar_event")
    check_is_scalar(distribution.is_scalar_batch(), "is_scalar_batch")
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
  with tf.name_scope("pad_mix_dims"):

    def _get_ndims(d):
      if tensorshape_util.rank(d.batch_shape) is not None:
        return tensorshape_util.rank(d.batch_shape)
      return tf.shape(input=d.batch_shape_tensor())[0]

    dist_batch_ndims = _get_ndims(mixture_distribution)
    cat_batch_ndims = _get_ndims(categorical_distribution)
    pad_ndims = tf.where(categorical_distribution.is_scalar_batch(),
                         dist_batch_ndims, dist_batch_ndims - cat_batch_ndims)
    s = tf.shape(input=x)
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
  with tf.name_scope(name or "pick_scalar_condition"):
    pred = tf.convert_to_tensor(
        value=pred, dtype_hint=tf.bool, name="pred")
    true_value = tf.convert_to_tensor(value=true_value, name="true_value")
    false_value = tf.convert_to_tensor(value=false_value, name="false_value")
    pred_ = tf.get_static_value(pred)
    if pred_ is None:
      return tf.where(pred, true_value, false_value)
    return true_value if pred_ else false_value


def make_non_negative_axis(axis, rank):
  """Make (possibly negatively indexed) `axis` argument non-negative."""
  axis = tf.convert_to_tensor(value=axis, name="axis")
  rank = tf.convert_to_tensor(value=rank, name="rank")
  axis_ = tf.get_static_value(axis)
  rank_ = tf.get_static_value(rank)

  # Static case.
  if axis_ is not None and rank_ is not None:
    is_scalar = axis_.ndim == 0
    if is_scalar:
      axis_ = [axis_]
    positive_axis = []
    for a_ in axis_:
      if a_ < 0:
        positive_axis.append(rank_ + a_)
      else:
        positive_axis.append(a_)
    if is_scalar:
      positive_axis = positive_axis[0]
    return tf.convert_to_tensor(value=positive_axis, dtype=axis.dtype)

  # Dynamic case.
  # Unfortunately static values are lost by this tf.where.
  return tf.where(axis < 0, rank + axis, axis)


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
  ndims = prefer_static_rank(x)
  dtype = dtype_util.common_dtype([source_idx, dest_idx],
                                  preferred_dtype=tf.int32)
  source_idx = tf.convert_to_tensor(value=source_idx, dtype=dtype)
  dest_idx = tf.convert_to_tensor(value=dest_idx, dtype=dtype)

  # Handle negative indexing.
  source_idx = pick_scalar_condition(source_idx < 0, ndims + source_idx,
                                     source_idx)
  dest_idx = pick_scalar_condition(dest_idx < 0, ndims + dest_idx, dest_idx)

  # Construct the appropriate permutation of dimensions, depending
  # whether the source is before or after the destination.
  def move_left_permutation():
    return prefer_static_value(
        tf.concat([
            tf.range(0, dest_idx, dtype=dtype), [source_idx],
            tf.range(dest_idx, source_idx, dtype=dtype),
            tf.range(source_idx + 1, ndims, dtype=dtype)
        ],
                  axis=0))

  def move_right_permutation():
    return prefer_static_value(
        tf.concat([
            tf.range(0, source_idx, dtype=dtype),
            tf.range(source_idx + 1, dest_idx + 1, dtype=dtype), [source_idx],
            tf.range(dest_idx + 1, ndims, dtype=dtype)
        ],
                  axis=0))

  def x_permuted():
    return tf.transpose(
        a=x,
        perm=prefer_static.cond(source_idx < dest_idx,
                                move_right_permutation,
                                move_left_permutation))

  # One final conditional to handle the special case where source
  # and destination indices are equal.
  return prefer_static.cond(tf.equal(source_idx, dest_idx),
                            lambda: x, x_permuted)


def assert_integer_form(x,
                        data=None,
                        summarize=None,
                        message=None,
                        int_dtype=None,
                        name="assert_integer_form"):
  """Assert that x has integer components (or floats equal to integers).

  Args:
    x: Floating-point `Tensor`
    data: The tensors to print out if the condition is `False`. Defaults to
      error message and first few entries of `x` and `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    int_dtype: A `tf.dtype` used to cast the float to. The default (`None`)
      implies the smallest possible signed int will be used for casting.
    name: A name for this operation (optional).

  Returns:
    Op raising `InvalidArgumentError` if `cast(x, int_dtype) != x`.
  """
  with tf.name_scope(name):
    x = tf.convert_to_tensor(value=x, name="x")
    if dtype_util.is_integer(x.dtype):
      return tf.no_op()
    message = message or "{} has non-integer components".format(x)
    if int_dtype is None:
      try:
        int_dtype = {
            tf.float16: tf.int16,
            tf.float32: tf.int32,
            tf.float64: tf.int64,
        }[dtype_util.base_dtype(x.dtype)]
      except KeyError:
        raise TypeError("Unrecognized type {}".format(dtype_util.name(x.dtype)))
    return assert_util.assert_equal(
        x,
        tf.cast(tf.cast(x, int_dtype), x.dtype),
        data=data,
        summarize=summarize,
        message=message,
        name=name)


def assert_symmetric(matrix):
  matrix_t = tf.linalg.matrix_transpose(matrix)
  return with_dependencies(
      [assert_util.assert_near(matrix, matrix_t)], matrix)


def embed_check_nonnegative_integer_form(
    x, name="embed_check_nonnegative_integer_form"):
  """Assert x is a non-negative tensor, and optionally of integers."""
  with tf.name_scope(name):
    x = tf.convert_to_tensor(value=x, name="x")
    assertions = [
        assert_util.assert_non_negative(
            x, message="'{}' must be non-negative.".format(x)),
    ]
    if not dtype_util.is_integer(x.dtype):
      assertions += [
          assert_integer_form(
              x,
              message="'{}' cannot contain fractional components.".format(x)),
      ]
    return with_dependencies(assertions, x)


def same_dynamic_shape(a, b):
  """Returns whether a and b have the same dynamic shape.

  Args:
    a: `Tensor`
    b: `Tensor`

  Returns:
    `bool` `Tensor` representing if both tensors have the same shape.
  """
  a = tf.convert_to_tensor(value=a, name="a")
  b = tf.convert_to_tensor(value=b, name="b")

  # Here we can't just do tf.equal(a.shape, b.shape), since
  # static shape inference may break the equality comparison between
  # shape(a) and shape(b) in tf.equal.
  def all_shapes_equal():
    return tf.reduce_all(
        input_tensor=tf.equal(
            tf.concat([tf.shape(input=a), tf.shape(input=b)], 0),
            tf.concat([tf.shape(input=b), tf.shape(input=a)], 0)))

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


def get_logits_and_probs(logits=None,
                         probs=None,
                         multidimensional=False,
                         validate_args=False,
                         name="get_logits_and_probs",
                         dtype=None):
  """Converts logit to probabilities (or vice-versa), and returns both.

  Args:
    logits: Floating-point `Tensor` representing log-odds.
    probs: Floating-point `Tensor` representing probabilities.
    multidimensional: Python `bool`, default `False`. If `True`, represents
      whether the last dimension of `logits` or `probs`, a `[N1, N2, ...  k]`
      dimensional tensor, representing the logit or probability of `shape[-1]`
      classes.
    validate_args: Python `bool`, default `False`. When `True`, either assert `0
      <= probs <= 1` (if not `multidimensional`) or that the last dimension of
      `probs` sums to one.
    name: A name for this operation (optional).
    dtype: `tf.DType` to prefer when converting args to `Tensor`s.

  Returns:
    logits, probs: Tuple of `Tensor`s. If `probs` has an entry that is `0` or
      `1`, then the corresponding entry in the returned logit will be `-Inf` and
      `Inf` respectively.

  Raises:
    ValueError: if neither `probs` nor `logits` were passed in, or both were.
  """
  if dtype is None:
    dtype = dtype_util.common_dtype([probs, logits], preferred_dtype=tf.float32)
  with tf.name_scope(name):
    if (probs is None) == (logits is None):
      raise ValueError("Must pass probs or logits, but not both.")

    if probs is None:
      logits = tf.convert_to_tensor(value=logits, name="logits", dtype=dtype)
      if not dtype_util.is_floating(logits.dtype):
        raise TypeError("logits must having floating type.")
      # We can early return since we constructed probs and therefore know
      # they're valid.
      if multidimensional:
        if validate_args:
          logits = embed_check_categorical_event_shape(logits)
        return logits, tf.nn.softmax(logits, name="probs")
      return logits, tf.sigmoid(logits, name="probs")

    probs = tf.convert_to_tensor(value=probs, name="probs", dtype=dtype)
    if not dtype_util.is_floating(probs.dtype):
      raise TypeError("probs must having floating type.")

    if validate_args:
      with tf.name_scope("validate_probs"):
        one = tf.constant(1., probs.dtype)
        dependencies = [assert_util.assert_non_negative(probs)]
        if multidimensional:
          probs = embed_check_categorical_event_shape(probs)
          dependencies += [
              assert_util.assert_near(
                  tf.reduce_sum(input_tensor=probs, axis=-1),
                  one,
                  message="probs does not sum to 1.")
          ]
        else:
          dependencies += [
              assert_util.assert_less_equal(
                  probs, one, message="probs has components greater than 1.")
          ]
        probs = with_dependencies(dependencies, probs)

    with tf.name_scope("logits"):
      if multidimensional:
        # Here we don't compute the multidimensional case, in a manner
        # consistent with respect to the unidimensional case. We do so
        # following the TF convention. Typically, you might expect to see
        # logits = log(probs) - log(probs[pivot]). A side-effect of
        # being consistent with the TF approach is that the unidimensional case
        # implicitly handles the second dimension but the multidimensional case
        # explicitly keeps the pivot dimension.
        return tf.math.log(probs), probs
      return tf.math.log(probs) - tf.math.log1p(-1. * probs), probs


def _is_known_unsigned_by_dtype(dt):
  """Helper returning True if dtype is known to be unsigned."""
  return {
      tf.bool: True,
      tf.uint8: True,
      tf.uint16: True,
  }.get(dt.base_dtype, False)


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
  }.get(dt.base_dtype, False)


def _is_known_dtype(dt):
  """Helper returning True if dtype is known."""
  return _is_known_unsigned_by_dtype(dt) or _is_known_signed_by_dtype(dt)


def _largest_integer_by_dtype(dt):
  """Helper returning the largest integer exactly representable by dtype."""
  if not _is_known_dtype(dt):
    raise TypeError("Unrecognized dtype: {}".format(dt.name))
  if dt.is_floating:
    return int(2**(np.finfo(dt.as_numpy_dtype).nmant + 1))
  if dt.is_integer:
    return np.iinfo(dt.as_numpy_dtype).max
  if dt.base_dtype == tf.bool:
    return int(1)
  # We actually can't land here but keep the case for completeness.
  raise TypeError("Unrecognized dtype: {}".format(dt.name))


def _smallest_integer_by_dtype(dt):
  """Helper returning the smallest integer exactly representable by dtype."""
  if not _is_known_dtype(dt):
    raise TypeError("Unrecognized dtype: {}".format(dt.name))
  if _is_known_unsigned_by_dtype(dt):
    return 0
  return -1 * _largest_integer_by_dtype(dt)


def _is_integer_like_by_dtype(dt):
  """Helper returning True if dtype.is_integer or is `bool`."""
  if not _is_known_dtype(dt):
    raise TypeError("Unrecognized dtype: {}".format(dt.name))
  return dt.is_integer or dt.base_dtype == tf.bool


def embed_check_categorical_event_shape(
    categorical_param, name="embed_check_categorical_event_shape"):
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
    x = tf.convert_to_tensor(value=categorical_param, name="categorical_param")
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
    if max_event_size is 0:
      raise TypeError("Unable to validate size of unrecognized dtype "
                      "({}).".format(dtype_util.name(x_dtype)))
    try:
      x_shape_static = tensorshape_util.with_rank_at_least(x.shape, 1)
    except ValueError:
      raise ValueError("A categorical-distribution parameter must have "
                       "at least 1 dimension.")
    event_size = tf.compat.dimension_value(x_shape_static[-1])
    if event_size is not None:
      if event_size < 2:
        raise ValueError("A categorical-distribution parameter must have at "
                         "least 2 events.")
      if event_size > max_event_size:
        raise ValueError("Number of classes exceeds `dtype` precision, i.e., "
                         "{} implies shape ({}) cannot exceed {}.".format(
                             dtype_util.name(x_dtype), event_size,
                             max_event_size))
      return x
    else:
      event_size = tf.shape(input=x, out_type=tf.int64, name="x_shape")[-1]
      return with_dependencies([
          assert_util.assert_rank_at_least(
              x,
              1,
              message=("A categorical-distribution parameter must have "
                       "at least 1 dimension.")),
          assert_util.assert_greater_equal(
              tf.shape(input=x)[-1],
              2,
              message=("A categorical-distribution parameter must have at "
                       "least 2 events.")),
          assert_util.assert_less_equal(
              event_size,
              tf.convert_to_tensor(max_event_size, dtype=tf.int64),
              message="Number of classes exceeds `dtype` precision, "
              "i.e., {} dtype cannot exceed {} shape.".format(
                  dtype_util.name(x_dtype), max_event_size)),
      ], x)


def embed_check_integer_casting_closed(x,
                                       target_dtype,
                                       assert_nonnegative=True,
                                       assert_positive=False,
                                       name="embed_check_casting_closed"):
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
    x = tf.convert_to_tensor(value=x, name="x")
    if (not _is_integer_like_by_dtype(x.dtype) and
        not dtype_util.is_floating(x.dtype)):
      raise TypeError("{}.dtype must be floating- or "
                      "integer-type.".format(dtype_util.name(x.dtype)))
    if (not _is_integer_like_by_dtype(target_dtype) and
        not dtype_util.is_floating(target_dtype)):
      raise TypeError("target_dtype ({}) must be floating- or "
                      "integer-type.".format(dtype_util.name(target_dtype)))
    if (not _is_integer_like_by_dtype(x.dtype) and
        not _is_integer_like_by_dtype(target_dtype)):
      raise TypeError("At least one of {}.dtype ({}) and target_dtype ({}) "
                      "must be integer-type.".format(
                          x, dtype_util.name(x.dtype),
                          dtype_util.name(target_dtype)))

    assertions = []
    if assert_positive:
      assertions += [
          assert_util.assert_positive(x, message="Elements must be positive."),
      ]
    elif assert_nonnegative:
      assertions += [
          assert_util.assert_non_negative(
              x, message="Elements must be non-negative."),
      ]

    if dtype_util.is_floating(x.dtype):
      # Being here means _is_integer_like_by_dtype(target_dtype) = True.
      # Since this check implies the magnitude check below, we need only it.
      assertions += [
          assert_integer_form(
              x,
              int_dtype=target_dtype,
              message="Elements must be {}-equivalent.".format(
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
                message=("Elements cannot exceed {}.".format(
                    _largest_integer_by_dtype(target_dtype)))),
        ]
      if (not assert_nonnegative and (_smallest_integer_by_dtype(
          x.dtype) < _smallest_integer_by_dtype(target_dtype))):
        assertions += [
            assert_util.assert_greater_equal(
                x,
                _smallest_integer_by_dtype(target_dtype),
                message=("Elements cannot be smaller than {}.".format(
                    _smallest_integer_by_dtype(target_dtype)))),
        ]

    if not assertions:
      return x
    return with_dependencies(assertions, x)


def log_combinations(n, counts, name="log_combinations"):
  """Multinomial coefficient.

  Given `n` and `counts`, where `counts` has last dimension `k`, we compute
  the multinomial coefficient as:

  ```n! / sum_i n_i!```

  where `i` runs over all `k` classes.

  Args:
    n: Floating-point `Tensor` broadcastable with `counts`. This represents `n`
      outcomes.
    counts: Floating-point `Tensor` broadcastable with `n`. This represents
      counts in `k` classes, where `k` is the last dimension of the tensor.
    name: A name for this operation (optional).

  Returns:
    `Tensor` representing the multinomial coefficient between `n` and `counts`.
  """
  # First a bit about the number of ways counts could have come in:
  # E.g. if counts = [1, 2], then this is 3 choose 2.
  # In general, this is (sum counts)! / sum(counts!)
  # The sum should be along the last dimension of counts. This is the
  # "distribution" dimension. Here n a priori represents the sum of counts.
  with tf.name_scope(name):
    n = tf.convert_to_tensor(value=n, name="n")
    counts = tf.convert_to_tensor(value=counts, name="counts")
    total_permutations = tf.math.lgamma(n + 1)
    counts_factorial = tf.math.lgamma(counts + 1)
    redundant_permutations = tf.reduce_sum(
        input_tensor=counts_factorial, axis=[-1])
    return total_permutations - redundant_permutations


def matrix_diag_transform(matrix, transform=None, name=None):
  """Transform diagonal of [batch-]matrix, leave rest of matrix unchanged.

  Create a trainable covariance defined by a Cholesky factor:

  ```python
  # Transform network layer into 2 x 2 array.
  matrix_values = tf.contrib.layers.fully_connected(activations, 4)
  matrix = tf.reshape(matrix_values, (batch_size, 2, 2))

  # Make the diagonal positive. If the upper triangle was zero, this would be a
  # valid Cholesky factor.
  chol = matrix_diag_transform(matrix, transform=tf.nn.softplus)

  # LinearOperatorLowerTriangular ignores the upper triangle.
  operator = LinearOperatorLowerTriangular(chol)
  ```

  Example of heteroskedastic 2-D linear regression.

  ```python
  tfd = tfp.distributions

  # Get a trainable Cholesky factor.
  matrix_values = tf.contrib.layers.fully_connected(activations, 4)
  matrix = tf.reshape(matrix_values, (batch_size, 2, 2))
  chol = matrix_diag_transform(matrix, transform=tf.nn.softplus)

  # Get a trainable mean.
  mu = tf.contrib.layers.fully_connected(activations, 2)

  # This is a fully trainable multivariate normal!
  dist = tfd.MultivariateNormalTriL(mu, chol)

  # Standard log loss. Minimizing this will "train" mu and chol, and then dist
  # will be a distribution predicting labels as multivariate Gaussians.
  loss = -1 * tf.reduce_mean(dist.log_prob(labels))
  ```

  Args:
    matrix:  Rank `R` `Tensor`, `R >= 2`, where the last two dimensions are
      equal.
    transform:  Element-wise function mapping `Tensors` to `Tensors`. To be
      applied to the diagonal of `matrix`. If `None`, `matrix` is returned
      unchanged. Defaults to `None`.
    name:  A name to give created ops. Defaults to "matrix_diag_transform".

  Returns:
    A `Tensor` with same shape and `dtype` as `matrix`.
  """
  with tf.name_scope(name or "matrix_diag_transform"):
    matrix = tf.convert_to_tensor(value=matrix, name="matrix")
    if transform is None:
      return matrix
    # Replace the diag with transformed diag.
    diag = tf.linalg.diag_part(matrix)
    transformed_diag = transform(diag)
    transformed_mat = tf.linalg.set_diag(matrix, transformed_diag)

  return transformed_mat


def rotate_transpose(x, shift, name="rotate_transpose"):
  """Circularly moves dims left or right.

  Effectively identical to:

  ```python
  numpy.transpose(x, numpy.roll(numpy.arange(len(x.shape)), shift))
  ```

  When `validate_args=False` additional graph-runtime checks are
  performed. These checks entail moving data from to GPU to CPU.

  Example:

  ```python
  x = tf.random_normal([1, 2, 3, 4])  # Tensor of shape [1, 2, 3, 4].
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
    x = tf.convert_to_tensor(value=x, name="x")
    shift = tf.convert_to_tensor(value=shift, name="shift")
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
          tf.less(shift, 0), -shift % ndims,
          ndims - shift % ndims)
      first = tf.range(0, shift)
      last = tf.range(shift, ndims)
      perm = tf.concat([last, first], 0)
      return tf.transpose(a=x, perm=perm)


def pick_vector(cond, true_vector, false_vector, name="pick_vector"):
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
  Example:  ```python pick_vector(tf.less(0, 5), tf.range(10, 12), tf.range(15,
    18))  # [10, 11] pick_vector(tf.less(5, 0), tf.range(10, 12), tf.range(15,
    18))  # [15, 16, 17] ```

  Returns:
    true_or_false_vector: `Tensor`.

  Raises:
    TypeError: if `cond.dtype != tf.bool`
    TypeError: if `cond` is not a constant and
      `true_vector.dtype != false_vector.dtype`
  """
  with tf.name_scope(name):
    cond = tf.convert_to_tensor(
        value=cond, dtype_hint=tf.bool, name="cond")
    if cond.dtype != tf.bool:
      raise TypeError(
          "{}.dtype={} which is not {}".format(cond, cond.dtype, tf.bool))

    true_vector = tf.convert_to_tensor(value=true_vector, name="true_vector")
    false_vector = tf.convert_to_tensor(value=false_vector, name="false_vector")
    if true_vector.dtype != false_vector.dtype:
      raise TypeError(
          "{}.dtype={} does not match {}.dtype={}".format(
              true_vector, true_vector.dtype, false_vector, false_vector.dtype))

    cond_value_static = tf.get_static_value(cond)
    if cond_value_static is not None:
      return true_vector if cond_value_static else false_vector
    n = tf.shape(input=true_vector)[0]
    return tf.slice(
        tf.concat([true_vector, false_vector], 0), [tf.where(cond, 0, n)],
        [tf.where(cond, n, -1)])


def prefer_static_broadcast_shape(shape1,
                                  shape2,
                                  name="prefer_static_broadcast_shape"):
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
      return tf.convert_to_tensor(value=x, name="shape", dtype=tf.int32)

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
      raise ValueError("Cannot broadcast from partially "
                       "defined `TensorShape`.")

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
  return prefer_static_value(tf.rank(x))


def prefer_static_shape(x):
  """Return static shape of tensor `x` if available, else `tf.shape(x)`.

  Args:
    x: `Tensor` (already converted).

  Returns:
    Numpy array (if static shape is obtainable), else `Tensor`.
  """
  return prefer_static_value(tf.shape(input=x))


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
  string = (str(seed) + salt).encode("utf-8")
  return int(hashlib.md5(string).hexdigest()[:8], 16) & 0x7FFFFFFF


def fill_triangular(x, upper=False, name=None):
  r"""Creates a (batch of) triangular matrix from a vector of inputs.

  Created matrix can be lower- or upper-triangular. (It is more efficient to
  create the matrix as upper or lower, rather than transpose.)

  Triangular matrix elements are filled in a clockwise spiral. See example,
  below.

  If `x.shape` is `[b1, b2, ..., bB, d]` then the output shape is
  `[b1, b2, ..., bB, n, n]` where `n` is such that `d = n(n+1)/2`, i.e.,
  `n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.

  Example:

  ```python
  fill_triangular([1, 2, 3, 4, 5, 6])
  # ==> [[4, 0, 0],
  #      [6, 5, 0],
  #      [3, 2, 1]]

  fill_triangular([1, 2, 3, 4, 5, 6], upper=True)
  # ==> [[1, 2, 3],
  #      [0, 5, 6],
  #      [0, 0, 4]]
  ```

  The key trick is to create an upper triangular matrix by concatenating `x`
  and a tail of itself, then reshaping.

  Suppose that we are filling the upper triangle of an `n`-by-`n` matrix `M`
  from a vector `x`. The matrix `M` contains n**2 entries total. The vector `x`
  contains `n * (n+1) / 2` entries. For concreteness, we'll consider `n = 5`
  (so `x` has `15` entries and `M` has `25`). We'll concatenate `x` and `x` with
  the first (`n = 5`) elements removed and reversed:

  ```python
  x = np.arange(15) + 1
  xc = np.concatenate([x, x[5:][::-1]])
  # ==> array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13,
  #            12, 11, 10, 9, 8, 7, 6])

  # (We add one to the arange result to disambiguate the zeros below the
  # diagonal of our upper-triangular matrix from the first entry in `x`.)

  # Now, when reshapedlay this out as a matrix:
  y = np.reshape(xc, [5, 5])
  # ==> array([[ 1,  2,  3,  4,  5],
  #            [ 6,  7,  8,  9, 10],
  #            [11, 12, 13, 14, 15],
  #            [15, 14, 13, 12, 11],
  #            [10,  9,  8,  7,  6]])

  # Finally, zero the elements below the diagonal:
  y = np.triu(y, k=0)
  # ==> array([[ 1,  2,  3,  4,  5],
  #            [ 0,  7,  8,  9, 10],
  #            [ 0,  0, 13, 14, 15],
  #            [ 0,  0,  0, 12, 11],
  #            [ 0,  0,  0,  0,  6]])
  ```

  From this example we see that the resuting matrix is upper-triangular, and
  contains all the entries of x, as desired. The rest is details:
  - If `n` is even, `x` doesn't exactly fill an even number of rows (it fills
    `n / 2` rows and half of an additional row), but the whole scheme still
    works.
  - If we want a lower triangular matrix instead of an upper triangular,
    we remove the first `n` elements from `x` rather than from the reversed
    `x`.

  For additional comparisons, a pure numpy version of this function can be found
  in `distribution_util_test.py`, function `_fill_triangular`.

  Args:
    x: `Tensor` representing lower (or upper) triangular elements.
    upper: Python `bool` representing whether output matrix should be upper
      triangular (`True`) or lower triangular (`False`, default).
    name: Python `str`. The name to give this op.

  Returns:
    tril: `Tensor` with lower (or upper) triangular elements filled from `x`.

  Raises:
    ValueError: if `x` cannot be mapped to a triangular matrix.
  """

  with tf.name_scope(name or "fill_triangular"):
    x = tf.convert_to_tensor(value=x, name="x")
    m = tf.compat.dimension_value(
        tensorshape_util.with_rank_at_least(x.shape, 1)[-1])
    if m is not None:
      # Formula derived by solving for n: m = n(n+1)/2.
      m = np.int32(m)
      n = np.sqrt(0.25 + 2. * m) - 0.5
      if n != np.floor(n):
        raise ValueError("Input right-most shape ({}) does not "
                         "correspond to a triangular matrix.".format(m))
      n = np.int32(n)
      static_final_shape = x.shape[:-1].concatenate([n, n])
    else:
      m = tf.shape(input=x)[-1]
      # For derivation, see above. Casting automatically lops off the 0.5, so we
      # omit it.  We don't validate n is an integer because this has
      # graph-execution cost; an error will be thrown from the reshape, below.
      n = tf.cast(
          tf.sqrt(0.25 + tf.cast(2 * m, dtype=tf.float32)), dtype=tf.int32)
      static_final_shape = tensorshape_util.with_rank_at_least(
          x.shape, 1)[:-1].concatenate([None, None])

    # Try it out in numpy:
    #  n = 3
    #  x = np.arange(n * (n + 1) / 2)
    #  m = x.shape[0]
    #  n = np.int32(np.sqrt(.25 + 2 * m) - .5)
    #  x_tail = x[(m - (n**2 - m)):]
    #  np.concatenate([x_tail, x[::-1]], 0).reshape(n, n)  # lower
    #  # ==> array([[3, 4, 5],
    #               [5, 4, 3],
    #               [2, 1, 0]])
    #  np.concatenate([x, x_tail[::-1]], 0).reshape(n, n)  # upper
    #  # ==> array([[0, 1, 2],
    #               [3, 4, 5],
    #               [5, 4, 3]])
    #
    # Note that we can't simply do `x[..., -(n**2 - m):]` because this doesn't
    # correctly handle `m == n == 1`. Hence, we do nonnegative indexing.
    # Furthermore observe that:
    #   m - (n**2 - m)
    #   = n**2 / 2 + n / 2 - (n**2 - n**2 / 2 + n / 2)
    #   = 2 (n**2 / 2 + n / 2) - n**2
    #   = n**2 + n - n**2
    #   = n
    ndims = prefer_static_rank(x)
    if upper:
      x_list = [x, tf.reverse(x[..., n:], axis=[ndims - 1])]
    else:
      x_list = [x[..., n:], tf.reverse(x, axis=[ndims - 1])]
    new_shape = (
        tensorshape_util.as_list(static_final_shape)
        if tensorshape_util.is_fully_defined(static_final_shape) else tf.concat(
            [tf.shape(input=x)[:-1], [n, n]], axis=0))
    x = tf.reshape(tf.concat(x_list, axis=-1), new_shape)
    x = tf.linalg.band_part(
        x, num_lower=(0 if upper else -1), num_upper=(-1 if upper else 0))
    tensorshape_util.set_shape(x, static_final_shape)
    return x


def fill_triangular_inverse(x, upper=False, name=None):
  """Creates a vector from a (batch of) triangular matrix.

  The vector is created from the lower-triangular or upper-triangular portion
  depending on the value of the parameter `upper`.

  If `x.shape` is `[b1, b2, ..., bB, n, n]` then the output shape is
  `[b1, b2, ..., bB, d]` where `d = n (n + 1) / 2`.

  Example:

  ```python
  fill_triangular_inverse(
    [[4, 0, 0],
     [6, 5, 0],
     [3, 2, 1]])

  # ==> [1, 2, 3, 4, 5, 6]

  fill_triangular_inverse(
    [[1, 2, 3],
     [0, 5, 6],
     [0, 0, 4]], upper=True)

  # ==> [1, 2, 3, 4, 5, 6]
  ```

  Args:
    x: `Tensor` representing lower (or upper) triangular elements.
    upper: Python `bool` representing whether output matrix should be upper
      triangular (`True`) or lower triangular (`False`, default).
    name: Python `str`. The name to give this op.

  Returns:
    flat_tril: (Batch of) vector-shaped `Tensor` representing vectorized lower
      (or upper) triangular elements from `x`.
  """

  with tf.name_scope(name or "fill_triangular_inverse"):
    x = tf.convert_to_tensor(value=x, name="x")
    n = tf.compat.dimension_value(
        tensorshape_util.with_rank_at_least(x.shape, 2)[-1])
    if n is not None:
      n = np.int32(n)
      m = np.int32((n * (n + 1)) // 2)
      static_final_shape = x.shape[:-2].concatenate([m])
    else:
      n = tf.shape(input=x)[-1]
      m = (n * (n + 1)) // 2
      static_final_shape = tensorshape_util.with_rank_at_least(
          x.shape, 2)[:-2].concatenate([None])
    ndims = prefer_static_rank(x)
    if upper:
      initial_elements = x[..., 0, :]
      triangular_portion = x[..., 1:, :]
    else:
      initial_elements = tf.reverse(x[..., -1, :], axis=[ndims - 2])
      triangular_portion = x[..., :-1, :]
    rotated_triangular_portion = tf.reverse(
        tf.reverse(triangular_portion, axis=[ndims - 1]), axis=[ndims - 2])
    consolidated_matrix = triangular_portion + rotated_triangular_portion
    end_sequence = tf.reshape(
        consolidated_matrix,
        tf.concat([tf.shape(input=x)[:-2], [n * (n - 1)]], axis=0))
    y = tf.concat([initial_elements, end_sequence[..., :m - n]], axis=-1)
    tensorshape_util.set_shape(y, static_final_shape)
    return y


def tridiag(below=None, diag=None, above=None, name=None):
  """Creates a matrix with values set above, below, and on the diagonal.

  Example:

  ```python
  tridiag(below=[1., 2., 3.],
          diag=[4., 5., 6., 7.],
          above=[8., 9., 10.])
  # ==> array([[  4.,   8.,   0.,   0.],
  #            [  1.,   5.,   9.,   0.],
  #            [  0.,   2.,   6.,  10.],
  #            [  0.,   0.,   3.,   7.]], dtype=float32)
  ```

  Warning: This Op is intended for convenience, not efficiency.

  Args:
    below: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the below
      diagonal part. `None` is logically equivalent to `below = 0`.
    diag: `Tensor` of shape `[B1, ..., Bb, d]` corresponding to the diagonal
      part.  `None` is logically equivalent to `diag = 0`.
    above: `Tensor` of shape `[B1, ..., Bb, d-1]` corresponding to the above
      diagonal part.  `None` is logically equivalent to `above = 0`.
    name: Python `str`. The name to give this op.

  Returns:
    tridiag: `Tensor` with values set above, below and on the diagonal.

  Raises:
    ValueError: if all inputs are `None`.
  """

  def _pad(x):
    """Prepends and appends a zero to every vector in a batch of vectors."""
    shape = tf.concat([tf.shape(input=x)[:-1], [1]], axis=0)
    z = tf.zeros(shape, dtype=x.dtype)
    return tf.concat([z, x, z], axis=-1)

  def _add(*x):
    """Adds list of Tensors, ignoring `None`."""
    s = None
    for y in x:
      if y is None:
        continue
      elif s is None:
        s = y
      else:
        s += y
    if s is None:
      raise ValueError("Must specify at least one of `below`, `diag`, `above`.")
    return s

  with tf.name_scope(name or "tridiag"):
    if below is not None:
      below = tf.convert_to_tensor(value=below, name="below")
      below = tf.linalg.diag(_pad(below))[..., :-1, 1:]
    if diag is not None:
      diag = tf.convert_to_tensor(value=diag, name="diag")
      diag = tf.linalg.diag(diag)
    if above is not None:
      above = tf.convert_to_tensor(value=above, name="above")
      above = tf.linalg.diag(_pad(above))[..., 1:, :-1]
    # TODO(jvdillon): Consider using scatter_nd instead of creating three full
    # matrices.
    return _add(below, diag, above)


def reduce_weighted_logsumexp(logx,
                              w=None,
                              axis=None,
                              keep_dims=False,
                              return_sign=False,
                              name=None):
  """Computes `log(abs(sum(weight * exp(elements across tensor dimensions))))`.

  If all weights `w` are known to be positive, it is more efficient to directly
  use `reduce_logsumexp`, i.e., `tf.reduce_logsumexp(logx + tf.log(w))` is more
  efficient than `du.reduce_weighted_logsumexp(logx, w)`.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  This function is more numerically stable than log(sum(w * exp(input))). It
  avoids overflows caused by taking the exp of large inputs and underflows
  caused by taking the log of small inputs.

  For example:

  ```python
  x = tf.constant([[0., 0, 0],
                   [0, 0, 0]])

  w = tf.constant([[-1., 1, 1],
                   [1, 1, 1]])

  du.reduce_weighted_logsumexp(x, w)
  # ==> log(-1*1 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1) = log(4)

  du.reduce_weighted_logsumexp(x, w, axis=0)
  # ==> [log(-1+1), log(1+1), log(1+1)]

  du.reduce_weighted_logsumexp(x, w, axis=1)
  # ==> [log(-1+1+1), log(1+1+1)]

  du.reduce_weighted_logsumexp(x, w, axis=1, keep_dims=True)
  # ==> [[log(-1+1+1)], [log(1+1+1)]]

  du.reduce_weighted_logsumexp(x, w, axis=[0, 1])
  # ==> log(-1+5)
  ```

  Args:
    logx: The tensor to reduce. Should have numeric type.
    w: The weight tensor. Should have numeric type identical to `logx`.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keep_dims: If true, retains reduced dimensions with length 1.
    return_sign: If `True`, returns the sign of the result.
    name: A name for the operation (optional).

  Returns:
    lswe: The `log(abs(sum(weight * exp(x))))` reduced tensor.
    sign: (Optional) The sign of `sum(weight * exp(x))`.
  """
  with tf.name_scope(name or "reduce_weighted_logsumexp"):
    logx = tf.convert_to_tensor(value=logx, name="logx")
    if w is None:
      lswe = tf.reduce_logsumexp(
          input_tensor=logx, axis=axis, keepdims=keep_dims)
      if return_sign:
        sgn = tf.ones_like(lswe)
        return lswe, sgn
      return lswe
    w = tf.convert_to_tensor(value=w, dtype=logx.dtype, name="w")
    log_absw_x = logx + tf.math.log(tf.abs(w))
    max_log_absw_x = tf.reduce_max(
        input_tensor=log_absw_x, axis=axis, keepdims=True)
    # If the largest element is `-inf` or `inf` then we don't bother subtracting
    # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
    # this is ok follows from the fact that we're actually free to subtract any
    # value we like, so long as we add it back after taking the `log(sum(...))`.
    max_log_absw_x = tf.where(
        tf.math.is_inf(max_log_absw_x), tf.zeros_like(max_log_absw_x),
        max_log_absw_x)
    wx_over_max_absw_x = (tf.sign(w) * tf.exp(log_absw_x - max_log_absw_x))
    sum_wx_over_max_absw_x = tf.reduce_sum(
        input_tensor=wx_over_max_absw_x, axis=axis, keepdims=keep_dims)
    if not keep_dims:
      max_log_absw_x = tf.squeeze(max_log_absw_x, axis)
    sgn = tf.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + tf.math.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
      return lswe, sgn
    return lswe


# TODO(jvdillon): Merge this test back into:
# tensorflow/python/ops/softplus_op_test.py
# once TF core is accepting new ops.
def softplus_inverse(x, name=None):
  """Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

  Mathematically this op is equivalent to:

  ```none
  softplus_inverse = log(exp(x) - 1.)
  ```

  Args:
    x: `Tensor`. Non-negative (not enforced), floating-point.
    name: A name for the operation (optional).

  Returns:
    `Tensor`. Has the same type/shape as input `x`.
  """
  with tf.name_scope(name or "softplus_inverse"):
    x = tf.convert_to_tensor(value=x, name="x")
    # We begin by deriving a more numerically stable softplus_inverse:
    # x = softplus(y) = Log[1 + exp{y}], (which means x > 0).
    # ==> exp{x} = 1 + exp{y}                                (1)
    # ==> y = Log[exp{x} - 1]                                (2)
    #       = Log[(exp{x} - 1) / exp{x}] + Log[exp{x}]
    #       = Log[(1 - exp{-x}) / 1] + Log[exp{x}]
    #       = Log[1 - exp{-x}] + x                           (3)
    # (2) is the "obvious" inverse, but (3) is more stable than (2) for large x.
    # For small x (e.g. x = 1e-10), (3) will become -inf since 1 - exp{-x} will
    # be zero. To fix this, we use 1 - exp{-x} approx x for small x > 0.
    #
    # In addition to the numerically stable derivation above, we clamp
    # small/large values to be congruent with the logic in:
    # tensorflow/core/kernels/softplus_op.h
    #
    # Finally, we set the input to one whenever the input is too large or too
    # small. This ensures that no unchosen codepath is +/- inf. This is
    # necessary to ensure the gradient doesn't get NaNs. Recall that the
    # gradient of `where` behaves like `pred*pred_true + (1-pred)*pred_false`
    # thus an `inf` in an unselected path results in `0*inf=nan`. We are careful
    # to overwrite `x` with ones only when we will never actually use this
    # value. Note that we use ones and not zeros since `log(expm1(0.)) = -inf`.
    threshold = np.log(np.finfo(dtype_util.as_numpy_dtype(x.dtype)).eps) + 2.
    is_too_small = tf.less(x, np.exp(threshold))
    is_too_large = tf.greater(x, -threshold)
    too_small_value = tf.math.log(x)
    too_large_value = x
    # This `where` will ultimately be a NOP because we won't select this
    # codepath whenever we used the surrogate `ones_like`.
    x = tf.where(tf.logical_or(is_too_small, is_too_large), tf.ones_like(x), x)
    y = x + tf.math.log(-tf.math.expm1(-x))  # == log(expm1(x))
    return tf.where(is_too_small, too_small_value,
                    tf.where(is_too_large, too_large_value, y))


# TODO(b/35290280): Add unit-tests.
def dimension_size(x, axis):
  """Returns the size of a specific dimension."""
  # Since tf.gather isn't "constant-in, constant-out", we must first check the
  # static shape or fallback to dynamic shape.
  s = tf.compat.dimension_value(
      tensorshape_util.with_rank_at_least(x.shape, np.abs(axis))[axis])
  if s is not None:
    return s
  return tf.shape(input=x)[axis]


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
  with tf.name_scope(name or "process_quadrature_grid_and_probs"):
    if quadrature_grid_and_probs is None:
      grid, probs = np.polynomial.hermite.hermgauss(deg=8)
      grid = grid.astype(dtype_util.as_numpy_dtype(dtype))
      probs = probs.astype(dtype_util.as_numpy_dtype(dtype))
      probs /= np.linalg.norm(probs, ord=1, keepdims=True)
      grid = tf.convert_to_tensor(value=grid, name="grid", dtype=dtype)
      probs = tf.convert_to_tensor(value=probs, name="probs", dtype=dtype)
      return grid, probs

    grid, probs = tuple(quadrature_grid_and_probs)
    grid = tf.convert_to_tensor(value=grid, name="grid", dtype=dtype)
    probs = tf.convert_to_tensor(
        value=probs, name="unnormalized_probs", dtype=dtype)
    probs /= tf.norm(tensor=probs, ord=1, axis=-1, keepdims=True, name="probs")

    def _static_event_size(x):
      """Returns the static size of a specific dimension or `None`."""
      return tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(x.shape, 1)[-1])

    m, n = _static_event_size(probs), _static_event_size(grid)
    if m is not None and n is not None:
      if m != n:
        raise ValueError("`quadrature_grid_and_probs` must be a `tuple` of "
                         "same-length zero-th-dimension `Tensor`s "
                         "(saw lengths {}, {})".format(m, n))
    elif validate_args:
      assertions = [
          assert_util.assert_equal(
              dimension_size(probs, axis=-1),
              dimension_size(grid, axis=-1),
              message=("`quadrature_grid_and_probs` must be a `tuple` of "
                       "same-length zero-th-dimension `Tensor`s")),
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
  with tf.name_scope(name or "pad"):
    x = tf.convert_to_tensor(value=x, name="x")
    value = tf.convert_to_tensor(value=value, dtype=x.dtype, name="value")
    count = tf.convert_to_tensor(value=count, name="count")
    if not dtype_util.is_integer(count.dtype):
      raise TypeError("`count.dtype` (`{}`) must be `int`-like.".format(
          dtype_util.name(count.dtype)))
    if not front and not back:
      raise ValueError("At least one of `front`, `back` must be `True`.")
    ndims = (
        tensorshape_util.rank(x.shape)
        if tensorshape_util.rank(x.shape) is not None else tf.rank(
            x, name="ndims"))
    axis = tf.convert_to_tensor(value=axis, name="axis")
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
        final_shape = head.concatenate(middle.concatenate(tail))
      else:
        final_shape = None
    else:
      axis = tf.where(axis < 0, ndims + axis, axis)
      final_shape = None
    x = tf.pad(
        tensor=x,
        paddings=tf.one_hot(
            indices=tf.stack([axis if front else -1, axis if back else -1]),
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
      additional_note="A special note!",
      kwargs_dict={"foo": "An extra arg."})
    def _prob(self, y, foo=None):
      pass
  ```

  In this case, the `AppendDocstring` decorator appends the `additional_note` to
  the docstring of `prob` (not `_prob`) and adds a new `kwargs`
  section with each dictionary item as a bullet-point.

  For a more detailed example, see `TransformedDistribution`.
  """

  def __init__(self, additional_note="", kwargs_dict=None):
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
          raise ValueError("Parameter name \"%s\" contains whitespace." % key)
        value = value.lstrip()
        if "\n" in value:
          raise ValueError(
              "Parameter description for \"%s\" contains newlines." % key)
        bullets.append("*  `%s`: %s" % (key, value))
      self._additional_note += ("\n\n##### `kwargs`:\n\n" + "\n".join(bullets))

  def __call__(self, fn):

    @functools.wraps(fn)
    def _fn(*args, **kwargs):
      return fn(*args, **kwargs)

    if _fn.__doc__ is None:
      _fn.__doc__ = self._additional_note
    else:
      _fn.__doc__ += "\n%s" % self._additional_note
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
  with tf.name_scope(op_name or "expand_to_vector"):
    x = tf.convert_to_tensor(value=x, name="x")
    ndims = tensorshape_util.rank(x.shape)

    if ndims is None:
      # Maybe expand ndims from 0 to 1.
      if validate_args:
        x = with_dependencies([
            assert_util.assert_rank_at_most(
                x, 1, message="Input is neither scalar nor vector.")
        ], x)
      ndims = tf.rank(x)
      expanded_shape = pick_vector(
          tf.equal(ndims, 0), np.array([1], dtype=np.int32), tf.shape(input=x))
      return tf.reshape(x, expanded_shape)

    elif ndims == 0:
      # Definitely expand ndims from 0 to 1.
      x_const = tf.get_static_value(x)
      if x_const is not None:
        return tf.convert_to_tensor(
            value=dtype_util.as_numpy_dtype(x.dtype)([x_const]),
            name=tensor_name)

      else:
        return tf.reshape(x, [1])

    elif ndims != 1:
      raise ValueError("Input is neither scalar nor vector.")

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
  with tf.name_scope(name or "control_dependency") as name:
    with tf.control_dependencies(d for d in dependencies if d is not None):
      output_tensor = tf.convert_to_tensor(value=output_tensor)
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
          hasattr(d, "log_prob") and
          hasattr(d, "sample"))
