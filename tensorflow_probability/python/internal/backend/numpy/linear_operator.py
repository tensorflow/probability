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
"""Numpy implementations of `tf.linalg.LinearOperator` class and subclasses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import logging

# Dependency imports
import numpy as np

from tensorflow_probability.python.internal.backend.numpy.internal import utils


__all__ = [
    "LinearOperator",
    "LinearOperatorBlockDiag",
    "LinearOperatorDiag",
    "LinearOperatorFullMatrix",
    "LinearOperatorIdentity",
    "LinearOperatorScaledIdentity",
    "LinearOperatorLowRankUpdate",
    "LinearOperatorLowerTriangular",
]


def _to_ndarray(x):
  if isinstance(x, (np.ndarray, np.generic)):
    return x
  return np.array(x)


def _adjoint(x):
  return np.conj(np.transpose(x, axes=[-1, -2]))


def _matmul_with_broadcast(a,
                           b,
                           transpose_a=False,  # pylint: disable=unused-argument
                           transpose_b=False,  # pylint: disable=unused-argument
                           adjoint_a=False,
                           adjoint_b=False,
                           a_is_sparse=False,  # pylint: disable=unused-argument
                           b_is_sparse=False,  # pylint: disable=unused-argument
                           name=None):  # pylint: disable=unused-argument
  """Multiplies matrix `a` by matrix `b`, producing `a @ b`.

  Works identically to `tf.matmul`, but broadcasts batch dims
  of `a` and `b` if they are determined statically to be different, or if static
  shapes are not fully defined. Attempts are made to avoid unnecessary
  replication of data, but this is not always possible.

  The inputs must be matrices (or tensors of rank > 2, representing batches of
  matrices).

  Both matrices must be of the same type. The supported types are:
  `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.

  Either matrix can be transposed or adjointed (conjugated and transposed) on
  the fly by setting one of the corresponding flag to `True`. These are `False`
  by default.

  If one or both of the matrices contain a lot of zeros, a more efficient
  multiplication algorithm can be used by setting the corresponding
  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
  This optimization is only available for plain matrices (rank-2 tensors) with
  datatypes `bfloat16` or `float32`.

  For example:

  ```python
  # A 2-batch of 3x4 matrices
  a = tf.random.normal(shape=(2, 3, 4))

  # A single 4x5 matrix
  b = tf.random.normal(shape=(4, 5))

  result = matmul_with_broadcast(a, b)

  result.shape
  ==> (2, 3, 5)

  result[0,...]
  ==> tf.matmul(a[0,...], b)

  result[1,...]
  ==> tf.matmul(a[1,...], b)
  ```

  Args:
    a: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
      `complex128` and `rank > 1`.
    b: `Tensor` with same type as `a` having compatible matrix dimensions and
      broadcastable batch dimensions.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    adjoint_a: If `True`, `a` is conjugated and transposed before
      multiplication.
    adjoint_b: If `True`, `b` is conjugated and transposed before
      multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix.
    name: Name for the operation (optional).

  Returns:
    A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
    the product of the corresponding matrices in `a` and `b`, e.g. if all
    transpose or adjoint attributes are `False`:

    The leading shape of `output` is the result of broadcasting the leading
    dimensions of `a` and `b`.

    `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
    for all indices i, j.

    Note: This is matrix product, not element-wise product.


  Raises:
    ValueError: If transpose_a and adjoint_a, or transpose_b and adjoint_b
      are both set to True.
  """
  a = np.array(a)
  b = np.array(b, dtype=a.dtype)

  if adjoint_a:
    a = _adjoint(a)
  if adjoint_b:
    b = _adjoint(b)

  return np.matmul(a, b)


def _matrix_solve_with_broadcast(matrix, rhs, adjoint=False):
  """Solve systems of linear equations."""
  matrix = np.array(matrix)
  rhs = np.array(rhs, dtype=matrix.dtype)

  if adjoint:
    matrix = _adjoint(matrix)

  # matrix, rhs = np.broadcast_arrays(matrix, rhs)

  return np.linalg.solve(matrix, rhs)


def _reshape_for_efficiency(a,
                            b,
                            transpose_a=False,
                            transpose_b=False,
                            adjoint_a=False,
                            adjoint_b=False):
  """Maybe reshape a, b, and return an inverse map.  For matmul/solve."""
  def identity(x):
    return x

  # At this point, we have not taken transpose/adjoint of a/b.
  still_need_to_transpose = True

  if a.shape.ndims is None or b.shape.ndims is None:
    return a, b, identity, still_need_to_transpose

  # This could be handled in the future, but seems less common.
  if a.shape.ndims >= b.shape.ndims:
    return a, b, identity, still_need_to_transpose

  # From now on, we might modify b, but will not modify a.

  # Suppose:
  #   a.shape =     C + [m, n], b.shape =
  #   b.shape = S + C + [n, r]
  b_extra_ndims = b.shape.ndims - a.shape.ndims

  # b_extra_sh = S, b_main_sh = C + [n, r]
  b_extra_sh = b.shape[:b_extra_ndims]
  b_main_sh = b.shape[b_extra_ndims:]

  # No reason to flip unless the extra dims of b are big enough.  Why?
  # Assume adjoint/transpose = False.  Then...
  # By not flipping, we have to replicate a to shape
  #   b_extra_sh + a.shape,
  # which could use extra memory.  But in all cases, the final output has shape
  #   b_extra_sh + a.shape[:-1] + [b.shape[-1]]
  # So we only end up creating a larger object if the end dim of b is smaller
  # than the end dim of a.  This often happens, e.g. if b was a vector that was
  # expanded to a matrix (by appending a singleton).

  # Since adjoint/transpose may not be False, we must make adjustments here.
  # The dim of b that holds the multiple equations.
  a_domain_sz_ = a.shape[-2 if adjoint_a or transpose_a else -1]
  b_eq_sz_ = b.shape[-2 if adjoint_b or transpose_b else -1]
  b_extra_sz_ = (
      np.prod(b.shape[:b_extra_ndims].as_list())
      if b.shape[:b_extra_ndims].is_fully_defined() else None)
  if (a_domain_sz_ is not None and b_eq_sz_ is not None and
      b_extra_sz_ is not None):
    if b_extra_sz_ < 2 or a_domain_sz_ <= b_eq_sz_:
      return a, b, identity, still_need_to_transpose

  # At this point, we're flipping for sure!
  # Any transposes/adjoints will happen here explicitly, rather than in calling
  # code.  Why?  To avoid having to write separate complex code for each case.
  if adjoint_a:
    a = _adjoint(a)
  elif transpose_a:
    a = np.transpose(a, axes=[-2, -1])
  if adjoint_b:
    b = _adjoint(b)
  elif transpose_b:
    b = np.transpose(b, axes=[-2, -1])
  still_need_to_transpose = False

  # Recompute shapes, since the transpose/adjoint may have changed them.
  b_extra_sh = b.shape[:b_extra_ndims]
  b_main_sh = b.shape[b_extra_ndims:]

  # Permutation to put the extra dims at the end.
  perm = (
      np.concatenate(
          (np.arange(b_extra_ndims, b.shape.ndims),
           np.arange(0, b_extra_ndims)), 0))
  b_extra_on_end = np.transpose(b, axes=perm)

  # Now squash this end into one long dim.
  b_squashed_end = np.reshape(
      b_extra_on_end, np.concatenate((b_main_sh[:-1], [-1]), 0))

  def reshape_inv(y):
    # Expand the extra dims hanging off the end, "b_extra_sh".
    # Note we use y_sh[:-1] + [b_main_sh[-1]] rather than b_main_sh, because y
    # Could have different batch dims than a and b, because of broadcasting.
    y_extra_shape = np.concatenate(
        (y.shape[:-1], [b_main_sh[-1]], b_extra_sh), 0)
    y_extra_on_end = np.reshape(y, y_extra_shape)
    inverse_perm = np.argsort(perm)
    return np.transpose(y_extra_on_end, axes=inverse_perm)

  return a, b_squashed_end, reshape_inv, still_need_to_transpose


class LinearOperator(object):
  """Reimplementation of tf.linalg.LinearOperator."""

  def __init__(self,
               dtype,
               graph_parents=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None):
    r"""Initialize the `LinearOperator`.

    **This is a private method for subclass use.**
    **Subclasses should copy-paste this `__init__` documentation.**

    Args:
      dtype: The type of the this `LinearOperator`.  Arguments to `matmul` and
        `solve` will have to be this type.
      graph_parents: Ignored.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `dtype` is real, this is equivalent to being symmetric.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: Ignored.

    Raises:
      ValueError if flags are inconsistent.
    """
    # Check and auto-set flags.
    if is_positive_definite:
      if is_non_singular is False:  # pylint: disable=g-bool-id-comparison
        raise ValueError("A positive definite matrix is always non-singular.")
      is_non_singular = True

    if is_non_singular is True:  # pylint: disable=g-bool-id-comparison
      if is_square is False:  # pylint: disable=g-bool-id-comparison
        raise ValueError("A non-singular matrix is always square.")
      is_square = True

    if is_self_adjoint:
      if is_square is False:  # pylint: disable=g-bool-id-comparison
        raise ValueError("A self-adjoint matrix is always square.")
      is_square = True

    self._is_square_set_or_implied_by_hints = is_square

    self._dtype = dtype
    self._is_non_singular = is_non_singular
    self._is_self_adjoint = is_self_adjoint
    self._is_positive_definite = is_positive_definite
    self._graph_parents = graph_parents
    self._name = name + "/"

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this `LinearOperator`."""
    return self._dtype

  @property
  def name(self):
    """Name prepended to all ops created by this `LinearOperator`."""
    return self._name

  @property
  def graph_parents(self):
    """List of graph dependencies of this `LinearOperator`."""
    return self._graph_parents

  @property
  def is_non_singular(self):
    return self._is_non_singular

  @property
  def is_self_adjoint(self):
    return self._is_self_adjoint

  @property
  def is_positive_definite(self):
    return self._is_positive_definite

  @property
  def is_square(self):
    """Return `True/False` depending on if this operator is square."""
    # Static checks done after __init__.  Why?  Because domain/range dimension
    # sometimes requires lots of work done in the derived class after init.
    auto_square_check = self.domain_dimension == self.range_dimension
    if self._is_square_set_or_implied_by_hints is False and auto_square_check:  # pylint: disable=g-bool-id-comparison
      raise ValueError(
          "User set is_square hint to False, but the operator was square.")
    if self._is_square_set_or_implied_by_hints is None:
      return auto_square_check

    return self._is_square_set_or_implied_by_hints

  @abc.abstractmethod
  def _shape(self):
    # Write this in derived class to enable all static shape methods.
    raise NotImplementedError("_shape is not implemented.")

  @property
  def shape(self):
    """`TensorShape` of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    return self._shape()

  def shape_tensor(self, name="shape_tensor"):  # pylint: disable=unused-argument
    """Shape of this `LinearOperator`, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    return self.shape

  @property
  def batch_shape(self):
    """`TensorShape` of batch dimensions of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    # Derived classes get this "for free" once .shape is implemented.
    return self.shape[:-2]

  def batch_shape_tensor(self, name="batch_shape_tensor"):  # pylint: disable=unused-argument
    """Shape of batch dimensions of this operator, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb]`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    return self.batch_shape

  @property
  def tensor_rank(self, name="tensor_rank"):  # pylint: disable=unused-argument
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      Python integer, or None if the tensor rank is undefined.
    """
    return len(self.shape) or None

  def tensor_rank_tensor(self, name="tensor_rank_tensor"):  # pylint: disable=unused-argument
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`, determined at runtime.
    """
    return self.tensor_rank(name)

  @property
  def domain_dimension(self):
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Returns:
      int or None.
    """
    if len(self.shape):  # pylint: disable=g-explicit-length-test
      return self.shape[-1]
    return None

  def domain_dimension_tensor(self, name="domain_dimension_tensor"):  # pylint: disable=unused-argument
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    return self.domain_dimension

  @property
  def range_dimension(self):
    """Dimension (in the sense of vector spaces) of the range of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Returns:
      `Dimension` object.
    """
    # Derived classes get this "for free" once .shape is implemented.
    if len(self.shape) > 1:
      return self.shape[-2]
    else:
      return None

  def range_dimension_tensor(self, name="range_dimension_tensor"):  # pylint: disable=unused-argument
    """Dimension (in the sense of vector spaces) of the range of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    return self.range_dimension

  def assert_non_singular(self, name="assert_non_singular"):  # pylint: disable=unused-argument
    """Returns an `Op` that asserts this operator is non singular.

    This operator is considered non-singular if

    ```
    ConditionNumber < max{100, range_dimension, domain_dimension} * eps,
    eps := np.finfo(self.dtype.as_numpy_dtype).eps
    ```

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is singular.
    """
    return None

  def assert_positive_definite(self, name="assert_positive_definite"):  # pylint: disable=unused-argument
    """Returns an `Op` that asserts this operator is positive definite.

    Here, positive definite means that the quadratic form `x^H A x` has positive
    real part for all nonzero `x`.  Note that we do not require the operator to
    be self-adjoint to be positive definite.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not positive definite.
    """
    return None

  def assert_self_adjoint(self, name="assert_self_adjoint"):  # pylint: disable=unused-argument
    """Returns an `Op` that asserts this operator is self-adjoint.

    Here we check that this operator is *exactly* equal to its hermitian
    transpose.

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not self-adjoint.
    """
    return None

  @abc.abstractmethod
  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    raise NotImplementedError("_matmul is not implemented.")

  def matmul(self, x, adjoint=False, adjoint_arg=False, name="matmul"):  # pylint: disable=unused-argument
    """Transform [batch] matrix `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    X = ... # shape [..., N, R], batch matrix, R > 0.

    Y = operator.matmul(X)
    Y.shape
    ==> [..., M, R]

    Y[..., :, r] = sum_j A[..., :, j] X[j, r]
    ```

    Args:
      x: `LinearOperator` or `Tensor` with compatible shape and same `dtype` as
        `self`. See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      adjoint_arg:  Python `bool`.  If `True`, compute `A x^H` where `x^H` is
        the hermitian transpose (transposition and complex conjugation).
      name:  A name for this `Op`.

    Returns:
      A `LinearOperator` or `Tensor` with shape `[..., M, R]` and same `dtype`
        as `self`.
    """
    if isinstance(x, LinearOperator):
      if adjoint or adjoint_arg:
        raise ValueError(".matmul not supported with adjoints.")
      if (x.range_dimension is not None and
          self.domain_dimension is not None and
          x.range_dimension != self.domain_dimension):
        raise ValueError(
            "Operators are incompatible. Expected `x` to have dimension"
            " {} but got {}.".format(self.domain_dimension, x.range_dimension))
      return np.matmul(self.to_dense(), x.to_dense())

    return self._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _matvec(self, x, adjoint=False):
    x_mat = x[..., None]
    y_mat = self.matmul(x_mat, adjoint=adjoint)
    return np.squeeze(y_mat, axis=-1)

  def matvec(self, x, adjoint=False, name="matvec"):  # pylint: disable=unused-argument
    """Transform [batch] vector `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matric A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)

    X = ... # shape [..., N], batch vector

    Y = operator.matvec(X)
    Y.shape
    ==> [..., M]

    Y[..., :] = sum_j A[..., :, j] X[..., j]
    ```

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`.
        `x` is treated as a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      name:  A name for this `Op`.

    Returns:
      A `Tensor` with shape `[..., M]` and same `dtype` as `self`.
    """
    x = np.array(x)
    return self._matvec(x, adjoint=adjoint)

  def _determinant(self):
    logging.warn(
        "Using (possibly slow) default implementation of determinant."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      return np.exp(self.log_abs_determinant())
    return np.linalg.det(self.to_dense())

  def determinant(self, name="det"):  # pylint: disable=unused-argument
    """Determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    if self.is_square is False:  # pylint: disable=g-bool-id-comparison
      raise NotImplementedError(
          "Determinant not implemented for an operator that is expected to "
          "not be square.")
    return self._determinant()

  def _log_abs_determinant(self):
    logging.warn(
        "Using (possibly slow) default implementation of determinant."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      diag = np.diagonal(
          np.linalg.cholesky(self.to_dense()), axis1=-2, axis2=-1)
      return 2 * np.sum(np.log(diag), axis=-1)
    _, log_abs_det = np.linalg.slogdet(self.to_dense())
    return log_abs_det

  def log_abs_determinant(self, name="log_abs_det"):  # pylint: disable=unused-argument
    """Log absolute value of determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    if self.is_square is False:  # pylint: disable=g-bool-id-comparison
      raise NotImplementedError(
          "Determinant not implemented for an operator that is expected to "
          "not be square.")
    return self._log_abs_determinant()

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    """Default implementation of _solve."""
    if self.is_square is False:  # pylint: disable=g-bool-id-comparison
      raise NotImplementedError(
          "Solve is not yet implemented for non-square operators.")
    logging.warn(
        "Using (possibly slow) default implementation of solve."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    rhs = _adjoint(rhs) if adjoint_arg else rhs
    # if self._can_use_cholesky():
    #   return _matrix_solve_with_broadcast(  # TODO(iansf): Use cholesky_solve.
    #       np.linalg.cholesky(self.to_dense()), rhs)
    return _matrix_solve_with_broadcast(
        self.to_dense(), rhs, adjoint=adjoint)

  def solve(self, rhs, adjoint=False, adjoint_arg=False, name="solve"):  # pylint: disable=unused-argument
    """Solve (exact or approx) `R` (batch) systems of equations: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve R > 0 linear systems for every member of the batch.
    RHS = ... # shape [..., M, R]

    X = operator.solve(RHS)
    # X[..., :, r] is the solution to the r'th linear system
    # sum_j A[..., :, j] X[..., j, r] = RHS[..., :, r]

    operator.matmul(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape.
        `rhs` is treated like a [batch] matrix meaning for every set of leading
        dimensions, the last two dimensions defines a matrix.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      adjoint_arg:  Python `bool`.  If `True`, solve `A X = rhs^H` where `rhs^H`
        is the hermitian transpose (transposition and complex conjugation).
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    if self.is_non_singular is False:  # pylint: disable=g-bool-id-comparison
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "be singular.")
    if self.is_square is False:  # pylint: disable=g-bool-id-comparison
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "not be square.")
    rhs = np.array(rhs)

    return self._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _solvevec(self, rhs, adjoint=False):
    """Default implementation of _solvevec."""
    rhs_mat = rhs[..., None]
    solution_mat = self.solve(rhs_mat, adjoint=adjoint)
    return np.squeeze(solution_mat, axis=-1)

  def solvevec(self, rhs, adjoint=False, name="solve"):  # pylint: disable=unused-argument
    """Solve single equation with best effort: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve one linear system for every member of the batch.
    RHS = ... # shape [..., M]

    X = operator.solvevec(RHS)
    # X is the solution to the linear system
    # sum_j A[..., :, j] X[..., j] = RHS[..., :]

    operator.matvec(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator.
        `rhs` is treated like a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.  See class docstring
        for definition of compatibility regarding batch dimensions.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    rhs = np.array(rhs)

    return self._solvevec(rhs, adjoint=adjoint)

  def adjoint(self, name="adjoint"):  # pylint: disable=unused-argument
    """Returns the adjoint of the current `LinearOperator`.

    Given `A` representing this `LinearOperator`, return `A*`.
    Note that calling `self.adjoint()` and `self.H` are equivalent.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the adjoint of this `LinearOperator`.
    """
    if self.is_self_adjoint is True:  # pylint: disable=g-bool-id-comparison
      return self
    return _adjoint(self)

  # self.H is equivalent to self.adjoint().
  H = property(adjoint, None)

  def inverse(self, name="inverse"):  # pylint: disable=unused-argument
    """Returns the Inverse of this `LinearOperator`.

    Given `A` representing this `LinearOperator`, return a `LinearOperator`
    representing `A^-1`.

    Args:
      name: A name scope to use for ops added by this method.

    Returns:
      `LinearOperator` representing inverse of this matrix.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be `non_singular`.
    """
    if self.is_square is False:  # pylint: disable=g-bool-id-comparison
      raise ValueError("Cannot take the Inverse: This operator represents "
                       "a non square matrix.")
    if self.is_non_singular is False:  # pylint: disable=g-bool-id-comparison
      raise ValueError("Cannot take the Inverse: This operator represents "
                       "a singular matrix.")

    return np.linalg.inv(self.to_dense())

  def cholesky(self, name="cholesky"):  # pylint: disable=unused-argument
    """Returns a Cholesky factor as a `LinearOperator`.

    Given `A` representing this `LinearOperator`, if `A` is positive definite
    self-adjoint, return `L`, where `A = L L^T`, i.e. the cholesky
    decomposition.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the lower triangular matrix
      in the Cholesky decomposition.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be positive
        definite and self adjoint.
    """

    if not self._can_use_cholesky():
      raise ValueError("Cannot take the Cholesky decomposition: "
                       "Not a positive definite self adjoint matrix.")
    return np.linalg.cholesky(self.to_dense())

  def _to_dense(self):
    """Generic and often inefficient implementation.  Override often."""
    logging.warn("Using (possibly slow) default implementation of to_dense."
                 "  Converts by self.matmul(identity).")
    batch_shape = self.batch_shape

    n = self.domain_dimension

    eye = np.eye(n, dtype=self.dtype)
    eye = eye * np.ones(batch_shape, dtype=self.dtype)[..., None]

    return self.matmul(eye)

  def to_dense(self, name="to_dense"):  # pylint: disable=unused-argument
    """Return a dense (batch) matrix representing this operator."""
    return self._to_dense()

  def _diag_part(self):
    """Generic and often inefficient implementation.  Override often."""
    return np.diagonal(self.to_dense(), axis1=-2, axis2=-1)

  def diag_part(self, name="diag_part"):  # pylint: disable=unused-argument
    """Efficiently get the [batch] diagonal part of this operator.

    If this operator has shape `[B1,...,Bb, M, N]`, this returns a
    `Tensor` `diagonal`, of shape `[B1,...,Bb, min(M, N)]`, where
    `diagonal[b1,...,bb, i] = self.to_dense()[b1,...,bb, i, i]`.

    ```
    my_operator = LinearOperatorDiag([1., 2.])

    # Efficiently get the diagonal
    my_operator.diag_part()
    ==> [1., 2.]

    # Equivalent, but inefficient method
    tf.matrix_diag_part(my_operator.to_dense())
    ==> [1., 2.]
    ```

    Args:
      name:  A name for this `Op`.

    Returns:
      diag_part:  A `Tensor` of same `dtype` as self.
    """
    return self._diag_part()

  def _trace(self):
    return np.sum(self.diag_part(), axis=-1)

  def trace(self, name="trace"):  # pylint: disable=unused-argument
    """Trace of the linear operator, equal to sum of `self.diag_part()`.

    If the operator is square, this is also the sum of the eigenvalues.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.
    """
    return self._trace()

  def _add_to_tensor(self, x):
    # Override if a more efficient implementation is available.
    return self.to_dense() + x

  def add_to_tensor(self, x, name="add_to_tensor"):  # pylint: disable=unused-argument
    """Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

    Args:
      x:  `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    return self._add_to_tensor(x)

  def _can_use_cholesky(self):
    return self.is_self_adjoint and self.is_positive_definite


class LinearOperatorFullMatrix(LinearOperator):
  """LinearOperatorFullMatrix numpy implementation."""

  def __init__(self,
               matrix,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorFullMatrix"):
    r"""Initialize a `LinearOperatorFullMatrix`.

    Args:
      matrix:  Shape `[B1,...,Bb, M, N]` with `b >= 0`, `M, N >= 0`.
        Allowed dtypes: `float16`, `float32`, `float64`, `complex64`,
        `complex128`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If `diag.dtype` is not an allowed type.
    """
    self._matrix = _to_ndarray(matrix)

    super(LinearOperatorFullMatrix, self).__init__(
        dtype=self._matrix.dtype,
        graph_parents=[self._matrix],
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)

  def _shape(self):
    return self._matrix.shape

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    return _matmul_with_broadcast(
        self._matrix, x, adjoint_a=adjoint, adjoint_b=adjoint_arg)

  def _to_dense(self):
    return self._matrix


class LinearOperatorIdentity(LinearOperatorFullMatrix):
  """LinearOperatorIdentity numpy implementation."""

  def __init__(self,
               num_rows,
               batch_shape=None,
               dtype=None,
               is_non_singular=True,
               is_self_adjoint=True,
               is_positive_definite=True,
               is_square=True,
               assert_proper_shapes=False,  # pylint: disable=unused-argument
               name="LinearOperatorIdentity"):
    r"""Initialize a `LinearOperatorIdentity`.

    The `LinearOperatorIdentity` is initialized with arguments defining `dtype`
    and shape.

    This operator is able to broadcast the leading (batch) dimensions, which
    sometimes requires copying data.  If `batch_shape` is `None`, the operator
    can take arguments of any batch shape without copying.  See examples.

    Args:
      num_rows:  Scalar non-negative integer `Tensor`.  Number of rows in the
        corresponding identity matrix.
      batch_shape:  Optional `1-D` integer `Tensor`.  The shape of the leading
        dimensions.  If `None`, this operator has no leading dimensions.
      dtype:  Data type of the matrix that this operator represents.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      assert_proper_shapes:  Python `bool`.  Ignored
      name: A name for this `LinearOperator`

    Raises:
      ValueError:  If `num_rows` is determined statically to be non-scalar, or
        negative.
      ValueError:  If `batch_shape` is determined statically to not be 1-D, or
        negative.
      ValueError:  If any of the following is not `True`:
        `{is_self_adjoint, is_non_singular, is_positive_definite}`.
    """
    dtype = dtype or np.float32
    if not is_self_adjoint:
      raise ValueError("An identity operator is always self adjoint.")
    if not is_non_singular:
      raise ValueError("An identity operator is always non-singular.")
    if not is_positive_definite:
      raise ValueError("An identity operator is always positive-definite.")
    if not is_square:
      raise ValueError("An identity operator is always square.")

    matrix = np.eye(num_rows, dtype=dtype)
    if batch_shape is not None:
      matrix *= np.ones(batch_shape, dtype=dtype)[..., None]

    super(LinearOperatorIdentity, self).__init__(
        matrix=matrix,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)


class LinearOperatorDiag(LinearOperatorFullMatrix):
  """LinearOperatorDiag numpy implementation."""

  def __new__(cls,
              diag,
              is_non_singular=None,
              is_self_adjoint=None,
              is_positive_definite=None,
              is_square=None,
              name="LinearOperatorDiag"):
    r"""Initialize a `LinearOperatorDiag`.

    Args:
      diag:  Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.
        The diagonal of the operator.  Allowed dtypes: `float16`, `float32`,
          `float64`, `complex64`, `complex128`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `diag.dtype` is real, this is auto-set to `True`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If `diag.dtype` is not an allowed type.
      ValueError:  If `diag.dtype` is real, and `is_self_adjoint` is not `True`.

    Returns:
      LinearOperatorFullMatrix.
    """
    diag = _to_ndarray(diag)

    # Check and auto-set hints.
    if not utils.is_complex(diag.dtype):
      if is_self_adjoint is False:  # pylint: disable=g-bool-id-comparison
        raise ValueError("A real diagonal operator is always self adjoint.")
      else:
        is_self_adjoint = True

    if is_square is False:  # pylint: disable=g-bool-id-comparison
      raise ValueError("Only square diagonal operators currently supported.")
    is_square = True

    if not np.array(diag).shape:
      raise ValueError("Diagonal must have at least 1 dimension")

    matrix = np.tile(
        np.eye(diag.shape[-1]), [1] * len(diag.shape) + [1]) * diag[..., None]
    return LinearOperatorFullMatrix(
        matrix=matrix,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)


class LinearOperatorScaledIdentity(LinearOperatorFullMatrix):
  """LinearOperatorScaledIdentity numpy implementation."""

  def __new__(cls,
              num_rows,
              multiplier,
              is_non_singular=None,
              is_self_adjoint=None,
              is_positive_definite=None,
              is_square=True,
              assert_proper_shapes=False,  # pylint: disable=unused-argument
              name="LinearOperatorScaledIdentity"):
    r"""Initialize a `LinearOperatorScaledIdentity`.

    The `LinearOperatorScaledIdentity` is initialized with `num_rows`, which
    determines the size of each identity matrix, and a `multiplier`,
    which defines `dtype`, batch shape, and scale of each matrix.

    This operator is able to broadcast the leading (batch) dimensions.

    Args:
      num_rows:  Scalar non-negative integer `Tensor`.  Number of rows in the
        corresponding identity matrix.
      multiplier:  `Tensor` of shape `[B1,...,Bb]`, or `[]` (a scalar).
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      assert_proper_shapes:  Python `bool`.  If `False`, only perform static
        checks that initialization and method arguments have proper shape.
        If `True`, and static checks are inconclusive, add asserts to the graph.
      name: A name for this `LinearOperator`

    Raises:
      ValueError:  If `num_rows` is determined statically to be non-scalar, or
        negative.

    Returns:
      LinearOperatorFullMatrix.
    """
    multiplier = _to_ndarray(multiplier)

    # Check and auto-set hints.
    if not utils.is_complex(multiplier.dtype):
      if is_self_adjoint is False:  # pylint: disable=g-bool-id-comparison
        raise ValueError("A real diagonal operator is always self adjoint.")
      else:
        is_self_adjoint = True

    if not is_square:
      raise ValueError("A ScaledIdentity operator is always square.")

    matrix = np.eye(num_rows, dtype=multiplier.dtype)
    matrix = multiplier[..., None, None] * matrix

    return LinearOperatorFullMatrix(
        matrix=matrix,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)


class LinearOperatorLowerTriangular(LinearOperatorFullMatrix):
  """LinearOperatorLowerTriangular numpy implementation."""

  def __new__(cls,
              tril,
              is_non_singular=None,
              is_self_adjoint=None,
              is_positive_definite=None,
              is_square=None,
              name="LinearOperatorLowerTriangular"):
    r"""Initialize a `LinearOperatorLowerTriangular`.

    Args:
      tril:  Shape `[B1,...,Bb, N, N]` with `b >= 0`, `N >= 0`.
        The lower triangular part of `tril` defines this operator.  The strictly
        upper triangle is ignored.
      is_non_singular:  Expect that this operator is non-singular.
        This operator is non-singular if and only if its diagonal elements are
        all non-zero.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  This operator is self-adjoint only if it is diagonal with
        real-valued diagonal entries.  In this case it is advised to use
        `LinearOperatorDiag`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      ValueError:  If `is_square` is `False`.

    Returns:
      LinearOperatorFullMatrix.
    """
    if is_square is False:  # pylint: disable=g-bool-id-comparison
      raise ValueError(
          "Only square lower triangular operators supported at this time.")
    is_square = True

    return LinearOperatorFullMatrix(
        matrix=tril,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)


################################################################################
# LinearOperators that definitely don't work are below.
################################################################################
class LinearOperatorBlockDiag(LinearOperatorFullMatrix):

  def __new__(cls, *args, **kwargs):
    raise NotImplementedError


class LinearOperatorLowRankUpdate(LinearOperatorFullMatrix):

  def __new__(cls, *args, **kwargs):
    raise NotImplementedError

