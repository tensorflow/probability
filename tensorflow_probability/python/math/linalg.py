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
"""Functions for common linear algebra operations.

Note: Many of these functions will eventually be migrated to core TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow.python.ops.linalg import linear_operator_util


__all__ = [
    'cholesky_concat',
    'lu_matrix_inverse',
    'lu_reconstruct',
    'lu_solve',
    'matrix_rank',
    'pinv',
    'sparse_or_dense_matmul',
    'sparse_or_dense_matvecmul',
]


def matrix_rank(a, tol=None, validate_args=False, name=None):
  """Compute the matrix rank; the number of non-zero SVD singular values.

  Arguments:
    a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
      pseudo-inverted.
    tol: Threshold below which the singular value is counted as "zero".
      Default value: `None` (i.e., `eps * max(rows, cols) * max(singular_val)`).
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: "matrix_rank".

  Returns:
    matrix_rank: (Batch of) `int32` scalars representing the number of non-zero
      singular values.
  """
  with tf.compat.v1.name_scope(name, 'matrix_rank', [a, tol]):
    a = tf.convert_to_tensor(value=a, dtype_hint=tf.float32, name='a')
    assertions = _maybe_validate_matrix(a, validate_args)
    if assertions:
      with tf.control_dependencies(assertions):
        a = tf.identity(a)
    s = tf.linalg.svd(a, compute_uv=False)
    if tol is None:
      if a.shape[-2:].is_fully_defined():
        m = np.max(a.shape[-2:].as_list())
      else:
        m = tf.reduce_max(input_tensor=tf.shape(input=a)[-2:])
      eps = np.finfo(a.dtype.as_numpy_dtype).eps
      tol = (eps * tf.cast(m, a.dtype) *
             tf.reduce_max(input_tensor=s, axis=-1, keepdims=True))
    return tf.reduce_sum(input_tensor=tf.cast(s > tol, tf.int32), axis=-1)


def cholesky_concat(chol, cols, name=None):
  """Concatenates `chol @ chol.T` with additional rows and columns.

  This operation is conceptually identical to:
  ```python
  def cholesky_concat_slow(chol, cols):  # cols shaped (n + m) x m = z x m
    mat = tf.matmul(chol, chol, adjoint_b=True)  # batch of n x n
    # Concat columns.
    mat = tf.concat([mat, cols[..., :tf.shape(mat)[-2], :]], axis=-1)  # n x z
    # Concat rows.
    mat = tf.concat([mat, tf.linalg.matrix_transpose(cols)], axis=-2)  # z x z
    return tf.linalg.cholesky(mat)
  ```
  but whereas `cholesky_concat_slow` would cost `O(z**3)` work,
  `cholesky_concat` only costs `O(z**2 + m**3)` work.

  The resulting (implicit) matrix must be symmetric and positive definite.
  Thus, the bottom right `m x m` must be self-adjoint, and we do not require a
  separate `rows` argument (which can be inferred from `conj(cols.T)`).

  Args:
    chol: Cholesky decomposition of `mat = chol @ chol.T`.
    cols: The new columns whose first `n` rows we would like concatenated to the
      right of `mat = chol @ chol.T`, and whose conjugate transpose we would
      like concatenated to the bottom of `concat(mat, cols[:n,:])`. A `Tensor`
      with final dims `(n+m, m)`. The first `n` rows are the top right rectangle
      (their conjugate transpose forms the bottom left), and the bottom `m x m`
      is self-adjoint.
    name: Optional name for this op.

  Returns:
    chol_concat: The Cholesky decomposition of:
      ```
      [ [ mat  cols[:n, :] ]
        [   conj(cols.T)   ] ]
      ```
  """
  with tf.compat.v2.name_scope(name or 'cholesky_extend'):
    dtype = dtype_util.common_dtype([chol, cols], preferred_dtype=tf.float32)
    chol = tf.convert_to_tensor(value=chol, name='chol', dtype=dtype)
    cols = tf.convert_to_tensor(value=cols, name='cols', dtype=dtype)
    n = prefer_static.shape(chol)[-1]
    mat_nm, mat_mm = cols[..., :n, :], cols[..., n:, :]
    solved_nm = linear_operator_util.matrix_triangular_solve_with_broadcast(
        chol, mat_nm)
    lower_right_mm = tf.linalg.cholesky(
        mat_mm - tf.matmul(solved_nm, solved_nm, adjoint_a=True))
    lower_left_mn = tf.math.conj(tf.linalg.matrix_transpose(solved_nm))
    out_batch = prefer_static.shape(solved_nm)[:-2]
    chol = tf.broadcast_to(
        chol,
        tf.concat([out_batch, prefer_static.shape(chol)[-2:]], axis=0))
    top_right_zeros_nm = tf.zeros_like(solved_nm)
    return tf.concat([tf.concat([chol, top_right_zeros_nm], axis=-1),
                      tf.concat([lower_left_mn, lower_right_mm], axis=-1)],
                     axis=-2)


def pinv(a, rcond=None, validate_args=False, name=None):
  """Compute the Moore-Penrose pseudo-inverse of a matrix.

  Calculate the [generalized inverse of a matrix](
  https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) using its
  singular-value decomposition (SVD) and including all large singular values.

  The pseudo-inverse of a matrix `A`, is defined as: "the matrix that 'solves'
  [the least-squares problem] `A @ x = b`," i.e., if `x_hat` is a solution, then
  `A_pinv` is the matrix such that `x_hat = A_pinv @ b`. It can be shown that if
  `U @ Sigma @ V.T = A` is the singular value decomposition of `A`, then
  `A_pinv = V @ inv(Sigma) U^T`. [(Strang, 1980)][1]

  This function is analogous to [`numpy.linalg.pinv`](
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html).
  It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
  default `rcond` is `1e-15`. Here the default is
  `10. * max(num_rows, num_cols) * np.finfo(dtype).eps`.

  Args:
    a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
      pseudo-inverted.
    rcond: `Tensor` of small singular value cutoffs.  Singular values smaller
      (in modulus) than `rcond` * largest_singular_value (again, in modulus) are
      set to zero. Must broadcast against `tf.shape(a)[:-2]`.
      Default value: `10. * max(num_rows, num_cols) * np.finfo(a.dtype).eps`.
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: "pinv".

  Returns:
    a_pinv: The pseudo-inverse of input `a`. Has same shape as `a` except
      rightmost two dimensions are transposed.

  Raises:
    TypeError: if input `a` does not have `float`-like `dtype`.
    ValueError: if input `a` has fewer than 2 dimensions.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  a = tf.constant([[1.,  0.4,  0.5],
                   [0.4, 0.2,  0.25],
                   [0.5, 0.25, 0.35]])
  tf.matmul(tfp.math.pinv(a), a)
  # ==> array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)

  a = tf.constant([[1.,  0.4,  0.5,  1.],
                   [0.4, 0.2,  0.25, 2.],
                   [0.5, 0.25, 0.35, 3.]])
  tf.matmul(tfp.math.pinv(a), a)
  # ==> array([[ 0.76,  0.37,  0.21, -0.02],
               [ 0.37,  0.43, -0.33,  0.02],
               [ 0.21, -0.33,  0.81,  0.01],
               [-0.02,  0.02,  0.01,  1.  ]], dtype=float32)
  ```

  #### References

  [1]: G. Strang. "Linear Algebra and Its Applications, 2nd Ed." Academic Press,
       Inc., 1980, pp. 139-142.
  """
  with tf.compat.v1.name_scope(name, 'pinv', [a, rcond]):
    a = tf.convert_to_tensor(value=a, name='a')

    assertions = _maybe_validate_matrix(a, validate_args)
    if assertions:
      with tf.control_dependencies(assertions):
        a = tf.identity(a)

    dtype = a.dtype.as_numpy_dtype

    if rcond is None:
      def get_dim_size(dim):
        if tf.compat.dimension_value(a.shape[dim]) is not None:
          return tf.compat.dimension_value(a.shape[dim])
        return tf.shape(input=a)[dim]

      num_rows = get_dim_size(-2)
      num_cols = get_dim_size(-1)
      if isinstance(num_rows, int) and isinstance(num_cols, int):
        max_rows_cols = float(max(num_rows, num_cols))
      else:
        max_rows_cols = tf.cast(tf.maximum(num_rows, num_cols), dtype)
      rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = tf.convert_to_tensor(value=rcond, dtype=dtype, name='rcond')

    # Calculate pseudo inverse via SVD.
    # Note: if a is symmetric then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,         # Sigma
        left_singular_vectors,   # U
        right_singular_vectors,  # V
    ] = tf.linalg.svd(a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = rcond * tf.reduce_max(input_tensor=singular_values, axis=-1)
    singular_values = tf.where(
        singular_values > cutoff[..., tf.newaxis], singular_values,
        tf.fill(tf.shape(input=singular_values), np.array(np.inf, dtype)))

    # Although `a == tf.matmul(u, s * v, transpose_b=True)` we swap
    # `u` and `v` here so that `tf.matmul(pinv(A), A) = tf.eye()`, i.e.,
    # a matrix inverse has "transposed" semantics.
    a_pinv = tf.matmul(
        right_singular_vectors / singular_values[..., tf.newaxis, :],
        left_singular_vectors,
        adjoint_b=True)

    if a.shape.ndims is not None:
      a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv


def lu_solve(lower_upper, perm, rhs,
             validate_args=False,
             name=None):
  """Solves systems of linear eqns `A X = RHS`, given LU factorizations.

  Note: this function does not verify the implied matrix is actually invertible
  nor is this condition checked even when `validate_args=True`.

  Args:
    lower_upper: `lu` as returned by `tf.linalg.lu`, i.e., if
      `matmul(P, matmul(L, U)) = X` then `lower_upper = L + U - eye`.
    perm: `p` as returned by `tf.linag.lu`, i.e., if
      `matmul(P, matmul(L, U)) = X` then `perm = argmax(P)`.
    rhs: Matrix-shaped float `Tensor` representing targets for which to solve;
      `A X = RHS`. To handle vector cases, use:
      `lu_solve(..., rhs[..., tf.newaxis])[..., 0]`.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness. Note: this function does not verify the implied matrix is
      actually invertible, even when `validate_args=True`.
      Default value: `False` (i.e., don't validate arguments).
    name: Python `str` name given to ops managed by this object.
      Default value: `None` (i.e., "lu_solve").

  Returns:
    x: The `X` in `A @ X = RHS`.

  #### Examples

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp

  x = [[[1., 2],
        [3, 4]],
       [[7, 8],
        [3, 4]]]
  inv_x = tfp.math.lu_solve(*tf.linalg.lu(x), rhs=tf.eye(2))
  tf.assert_near(tf.matrix_inverse(x), inv_x)
  # ==> True
  ```

  """

  with tf.compat.v1.name_scope(name, 'lu_solve', [lower_upper, perm, rhs]):
    lower_upper = tf.convert_to_tensor(
        value=lower_upper, dtype_hint=tf.float32, name='lower_upper')
    perm = tf.convert_to_tensor(value=perm, dtype_hint=tf.int32, name='perm')
    rhs = tf.convert_to_tensor(
        value=rhs, dtype_hint=lower_upper.dtype, name='rhs')

    assertions = _lu_solve_assertions(lower_upper, perm, rhs, validate_args)
    if assertions:
      with tf.control_dependencies(assertions):
        lower_upper = tf.identity(lower_upper)
        perm = tf.identity(perm)
        rhs = tf.identity(rhs)

    if rhs.shape.ndims == 2 and perm.shape.ndims == 1:
      # Both rhs and perm have scalar batch_shape.
      permuted_rhs = tf.gather(rhs, perm, axis=-2)
    else:
      # Either rhs or perm have non-scalar batch_shape or we can't determine
      # this information statically.
      rhs_shape = tf.shape(input=rhs)
      broadcast_batch_shape = tf.broadcast_dynamic_shape(
          rhs_shape[:-2],
          tf.shape(input=perm)[:-1])
      d, m = rhs_shape[-2], rhs_shape[-1]
      rhs_broadcast_shape = tf.concat([broadcast_batch_shape, [d, m]], axis=0)

      # Tile out rhs.
      broadcast_rhs = tf.broadcast_to(rhs, rhs_broadcast_shape)
      broadcast_rhs = tf.reshape(broadcast_rhs, [-1, d, m])

      # Tile out perm and add batch indices.
      broadcast_perm = tf.broadcast_to(perm, rhs_broadcast_shape[:-1])
      broadcast_perm = tf.reshape(broadcast_perm, [-1, d])
      broadcast_batch_size = tf.reduce_prod(input_tensor=broadcast_batch_shape)
      broadcast_batch_indices = tf.broadcast_to(
          tf.range(broadcast_batch_size)[:, tf.newaxis],
          [broadcast_batch_size, d])
      broadcast_perm = tf.stack([broadcast_batch_indices, broadcast_perm],
                                axis=-1)

      permuted_rhs = tf.gather_nd(broadcast_rhs, broadcast_perm)
      permuted_rhs = tf.reshape(permuted_rhs, rhs_broadcast_shape)

    lower = tf.linalg.set_diag(
        tf.linalg.band_part(lower_upper, num_lower=-1, num_upper=0),
        tf.ones(tf.shape(input=lower_upper)[:-1], dtype=lower_upper.dtype))
    return linear_operator_util.matrix_triangular_solve_with_broadcast(
        lower_upper,  # Only upper is accessed.
        linear_operator_util.matrix_triangular_solve_with_broadcast(
            lower, permuted_rhs),
        lower=False)


def lu_matrix_inverse(lower_upper, perm, validate_args=False, name=None):
  """Computes a matrix inverse given the matrix's LU decomposition.

  This op is conceptually identical to,

  ````python
  inv_X = tf.lu_matrix_inverse(*tf.linalg.lu(X))
  tf.assert_near(tf.matrix_inverse(X), inv_X)
  # ==> True
  ```

  Note: this function does not verify the implied matrix is actually invertible
  nor is this condition checked even when `validate_args=True`.

  Args:
    lower_upper: `lu` as returned by `tf.linalg.lu`, i.e., if
      `matmul(P, matmul(L, U)) = X` then `lower_upper = L + U - eye`.
    perm: `p` as returned by `tf.linag.lu`, i.e., if
      `matmul(P, matmul(L, U)) = X` then `perm = argmax(P)`.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness. Note: this function does not verify the implied matrix is
      actually invertible, even when `validate_args=True`.
      Default value: `False` (i.e., don't validate arguments).
    name: Python `str` name given to ops managed by this object.
      Default value: `None` (i.e., "lu_matrix_inverse").

  Returns:
    inv_x: The matrix_inv, i.e.,
      `tf.matrix_inverse(tfp.math.lu_reconstruct(lu, perm))`.

  #### Examples

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp

  x = [[[3., 4], [1, 2]],
       [[7., 8], [3, 4]]]
  inv_x = tfp.math.lu_matrix_inverse(*tf.linalg.lu(x))
  tf.assert_near(tf.matrix_inverse(x), inv_x)
  # ==> True
  ```

  """

  with tf.compat.v1.name_scope(name, 'lu_matrix_inverse', [lower_upper, perm]):
    lower_upper = tf.convert_to_tensor(
        value=lower_upper, dtype_hint=tf.float32, name='lower_upper')
    perm = tf.convert_to_tensor(value=perm, dtype_hint=tf.int32, name='perm')
    assertions = _lu_reconstruct_assertions(lower_upper, perm, validate_args)
    if assertions:
      with tf.control_dependencies(assertions):
        lower_upper = tf.identity(lower_upper)
        perm = tf.identity(perm)
    shape = tf.shape(input=lower_upper)
    return lu_solve(
        lower_upper, perm,
        rhs=tf.eye(shape[-1], batch_shape=shape[:-2], dtype=lower_upper.dtype),
        validate_args=False)


def lu_reconstruct(lower_upper, perm, validate_args=False, name=None):
  """The inverse LU decomposition, `X == lu_reconstruct(*tf.linalg.lu(X))`.

  Args:
    lower_upper: `lu` as returned by `tf.linalg.lu`, i.e., if
      `matmul(P, matmul(L, U)) = X` then `lower_upper = L + U - eye`.
    perm: `p` as returned by `tf.linag.lu`, i.e., if
      `matmul(P, matmul(L, U)) = X` then `perm = argmax(P)`.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness.
      Default value: `False` (i.e., don't validate arguments).
    name: Python `str` name given to ops managed by this object.
      Default value: `None` (i.e., "lu_reconstruct").

  Returns:
    x: The original input to `tf.linalg.lu`, i.e., `x` as in,
      `lu_reconstruct(*tf.linalg.lu(x))`.

  #### Examples

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp

  x = [[[3., 4], [1, 2]],
       [[7., 8], [3, 4]]]
  x_reconstructed = tfp.math.lu_reconstruct(*tf.linalg.lu(x))
  tf.assert_near(x, x_reconstructed)
  # ==> True
  ```

  """
  with tf.compat.v1.name_scope(name, 'lu_reconstruct', [lower_upper, perm]):
    lower_upper = tf.convert_to_tensor(
        value=lower_upper, dtype_hint=tf.float32, name='lower_upper')
    perm = tf.convert_to_tensor(value=perm, dtype_hint=tf.int32, name='perm')

    assertions = _lu_reconstruct_assertions(lower_upper, perm, validate_args)
    if assertions:
      with tf.control_dependencies(assertions):
        lower_upper = tf.identity(lower_upper)
        perm = tf.identity(perm)

    shape = tf.shape(input=lower_upper)

    lower = tf.linalg.set_diag(
        tf.linalg.band_part(lower_upper, num_lower=-1, num_upper=0),
        tf.ones(shape[:-1], dtype=lower_upper.dtype))
    upper = tf.linalg.band_part(lower_upper, num_lower=0, num_upper=-1)
    x = tf.matmul(lower, upper)

    if lower_upper.shape.ndims is None or lower_upper.shape.ndims != 2:
      # We either don't know the batch rank or there are >0 batch dims.
      batch_size = tf.reduce_prod(input_tensor=shape[:-2])
      d = shape[-1]
      x = tf.reshape(x, [batch_size, d, d])
      perm = tf.reshape(perm, [batch_size, d])
      perm = tf.map_fn(tf.math.invert_permutation, perm)
      batch_indices = tf.broadcast_to(
          tf.range(batch_size)[:, tf.newaxis],
          [batch_size, d])
      x = tf.gather_nd(x, tf.stack([batch_indices, perm], axis=-1))
      x = tf.reshape(x, shape)
    else:
      x = tf.gather(x, tf.math.invert_permutation(perm))

    x.set_shape(lower_upper.shape)
    return x


def _lu_reconstruct_assertions(lower_upper, perm, validate_args):
  """Returns list of assertions related to `lu_reconstruct` assumptions."""
  assertions = []

  message = 'Input `lower_upper` must have at least 2 dimensions.'
  if lower_upper.shape.ndims is not None:
    if lower_upper.shape.ndims < 2:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        tf.compat.v1.assert_rank_at_least(lower_upper, rank=2, message=message))

  message = '`rank(lower_upper)` must equal `rank(perm) + 1`'
  if lower_upper.shape.ndims is not None and perm.shape.ndims is not None:
    if lower_upper.shape.ndims != perm.shape.ndims + 1:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        tf.compat.v1.assert_rank(
            lower_upper, rank=tf.rank(perm) + 1, message=message))

  message = '`lower_upper` must be square.'
  if lower_upper.shape[:-2].is_fully_defined():
    if lower_upper.shape[-2] != lower_upper.shape[-1]:
      raise ValueError(message)
  elif validate_args:
    m, n = tf.split(tf.shape(input=lower_upper)[-2:], num_or_size_splits=2)
    assertions.append(tf.compat.v1.assert_equal(m, n, message=message))

  return assertions


def _lu_solve_assertions(lower_upper, perm, rhs, validate_args):
  """Returns list of assertions related to `lu_solve` assumptions."""
  assertions = _lu_reconstruct_assertions(lower_upper, perm, validate_args)

  message = 'Input `rhs` must have at least 2 dimensions.'
  if rhs.shape.ndims is not None:
    if rhs.shape.ndims < 2:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        tf.compat.v1.assert_rank_at_least(rhs, rank=2, message=message))

  message = '`lower_upper.shape[-1]` must equal `rhs.shape[-1]`.'
  if (tf.compat.dimension_value(lower_upper.shape[-1]) is not None and
      tf.compat.dimension_value(rhs.shape[-2]) is not None):
    if lower_upper.shape[-1] != rhs.shape[-2]:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        tf.compat.v1.assert_equal(
            tf.shape(input=lower_upper)[-1],
            tf.shape(input=rhs)[-2],
            message=message))

  return assertions


def sparse_or_dense_matmul(sparse_or_dense_a,
                           dense_b,
                           validate_args=False,
                           name=None,
                           **kwargs):
  """Returns (batched) matmul of a SparseTensor (or Tensor) with a Tensor.

  Args:
    sparse_or_dense_a: `SparseTensor` or `Tensor` representing a (batch of)
      matrices.
    dense_b: `Tensor` representing a (batch of) matrices, with the same batch
      shape as `sparse_or_dense_a`. The shape must be compatible with the shape
      of `sparse_or_dense_a` and kwargs.
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: "sparse_or_dense_matmul".
    **kwargs: Keyword arguments to `tf.sparse_tensor_dense_matmul` or
      `tf.matmul`.

  Returns:
    product: A dense (batch of) matrix-shaped Tensor of the same batch shape and
    dtype as `sparse_or_dense_a` and `dense_b`. If `sparse_or_dense_a` or
    `dense_b` is adjointed through `kwargs` then the shape is adjusted
    accordingly.
  """
  with tf.compat.v1.name_scope(name, 'sparse_or_dense_matmul',
                               [sparse_or_dense_a, dense_b]):
    dense_b = tf.convert_to_tensor(
        value=dense_b, dtype_hint=tf.float32, name='dense_b')

    if validate_args:
      assert_a_rank_at_least_2 = tf.compat.v1.assert_rank_at_least(
          sparse_or_dense_a,
          rank=2,
          message='Input `sparse_or_dense_a` must have at least 2 dimensions.')
      assert_b_rank_at_least_2 = tf.compat.v1.assert_rank_at_least(
          dense_b,
          rank=2,
          message='Input `dense_b` must have at least 2 dimensions.')
      with tf.control_dependencies(
          [assert_a_rank_at_least_2, assert_b_rank_at_least_2]):
        sparse_or_dense_a = tf.identity(sparse_or_dense_a)
        dense_b = tf.identity(dense_b)

    if isinstance(sparse_or_dense_a,
                  (tf.SparseTensor, tf.compat.v1.SparseTensorValue)):
      return _sparse_tensor_dense_matmul(sparse_or_dense_a, dense_b, **kwargs)
    else:
      return tf.matmul(sparse_or_dense_a, dense_b, **kwargs)


def sparse_or_dense_matvecmul(sparse_or_dense_matrix,
                              dense_vector,
                              validate_args=False,
                              name=None,
                              **kwargs):
  """Returns (batched) matmul of a (sparse) matrix with a column vector.

  Args:
    sparse_or_dense_matrix: `SparseTensor` or `Tensor` representing a (batch of)
      matrices.
    dense_vector: `Tensor` representing a (batch of) vectors, with the same
      batch shape as `sparse_or_dense_matrix`. The shape must be compatible with
      the shape of `sparse_or_dense_matrix` and kwargs.
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: "sparse_or_dense_matvecmul".
    **kwargs: Keyword arguments to `tf.sparse_tensor_dense_matmul` or
      `tf.matmul`.

  Returns:
    product: A dense (batch of) vector-shaped Tensor of the same batch shape and
    dtype as `sparse_or_dense_matrix` and `dense_vector`.
  """
  with tf.compat.v1.name_scope(name, 'sparse_or_dense_matvecmul',
                               [sparse_or_dense_matrix, dense_vector]):
    dense_vector = tf.convert_to_tensor(
        value=dense_vector, dtype_hint=tf.float32, name='dense_vector')
    return tf.squeeze(
        sparse_or_dense_matmul(
            sparse_or_dense_matrix,
            dense_vector[..., tf.newaxis],
            validate_args=validate_args,
            **kwargs),
        axis=[-1])


def _get_shape(x, out_type=tf.int32):
  # Return the shape of a Tensor or a SparseTensor as an np.array if its shape
  # is known statically. Otherwise return a Tensor representing the shape.
  if x.shape.is_fully_defined():
    return np.array(x.shape.as_list(), dtype=out_type.as_numpy_dtype)
  return tf.shape(input=x, out_type=out_type)


def _sparse_tensor_dense_matmul(sp_a, b, **kwargs):
  """Returns (batched) matmul of a SparseTensor with a Tensor.

  Args:
    sp_a: `SparseTensor` representing a (batch of) matrices.
    b: `Tensor` representing a (batch of) matrices, with the same batch shape of
      `sp_a`. The shape must be compatible with the shape of `sp_a` and kwargs.
    **kwargs: Keyword arguments to `tf.sparse_tensor_dense_matmul`.

  Returns:
    product: A dense (batch of) matrix-shaped Tensor of the same batch shape and
    dtype as `sp_a` and `b`. If `sp_a` or `b` is adjointed through `kwargs` then
    the shape is adjusted accordingly.
  """
  batch_shape = _get_shape(sp_a)[:-2]

  # Reshape the SparseTensor into a rank 3 SparseTensors, with the
  # batch shape flattened to a single dimension. If the batch rank is 0, then
  # we add a batch dimension of rank 1.
  sp_a = tf.sparse.reshape(sp_a, tf.concat([[-1], _get_shape(sp_a)[-2:]],
                                           axis=0))
  # Reshape b to stack the batch dimension along the rows.
  b = tf.reshape(b, tf.concat([[-1], _get_shape(b)[-1:]], axis=0))

  # Convert the SparseTensor to a matrix in block diagonal form with blocks of
  # matrices [M, N]. This allow us to use tf.sparse_tensor_dense_matmul which
  # only accepts rank 2 (Sparse)Tensors.
  out = tf.sparse.sparse_dense_matmul(_sparse_block_diag(sp_a), b, **kwargs)

  # Finally retrieve the original batch shape from the resulting rank 2 Tensor.
  # Note that we avoid inferring the final shape from `sp_a` or `b` because we
  # might have transposed one or both of them.
  return tf.reshape(
      out,
      tf.concat([batch_shape, [-1], _get_shape(out)[-1:]], axis=0))


def _sparse_block_diag(sp_a):
  """Returns a block diagonal rank 2 SparseTensor from a batch of SparseTensors.

  Args:
    sp_a: A rank 3 `SparseTensor` representing a batch of matrices.

  Returns:
    sp_block_diag_a: matrix-shaped, `float` `SparseTensor` with the same dtype
    as `sparse_or_matrix`, of shape [B * M, B * N] where `sp_a` has shape
    [B, M, N]. Each [M, N] batch of `sp_a` is lined up along the diagonal.
  """
  # Construct the matrix [[M, N], [1, 0], [0, 1]] which would map the index
  # (b, i, j) to (Mb + i, Nb + j). This effectively creates a block-diagonal
  # matrix of dense shape [B * M, B * N].
  # Note that this transformation doesn't increase the number of non-zero
  # entries in the SparseTensor.
  sp_a_shape = tf.convert_to_tensor(value=_get_shape(sp_a, tf.int64))
  ind_mat = tf.concat([[sp_a_shape[-2:]], tf.eye(2, dtype=tf.int64)], axis=0)
  indices = tf.matmul(sp_a.indices, ind_mat)
  dense_shape = sp_a_shape[0] * sp_a_shape[1:]
  return tf.SparseTensor(
      indices=indices, values=sp_a.values, dense_shape=dense_shape)


def _maybe_validate_matrix(a, validate_args):
  """Checks that input is a `float` matrix."""
  assertions = []
  if not a.dtype.is_floating:
    raise TypeError('Input `a` must have `float`-like `dtype` '
                    '(saw {}).'.format(a.dtype.name))
  if a.shape.ndims is not None:
    if a.shape.ndims < 2:
      raise ValueError('Input `a` must have at least 2 dimensions '
                       '(saw: {}).'.format(a.shape.ndims))
  elif validate_args:
    assertions.append(tf.compat.v1.assert_rank_at_least(
        a, rank=2, message='Input `a` must have at least 2 dimensions.'))
  return assertions
