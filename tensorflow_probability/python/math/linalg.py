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

import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import generic


__all__ = [
    'cholesky_concat',
    'cholesky_update',
    'fill_triangular',
    'fill_triangular_inverse',
    'hpsd_logdet',
    'hpsd_quadratic_form_solve',
    'hpsd_quadratic_form_solvevec',
    'hpsd_solve',
    'hpsd_solvevec',
    'lu_matrix_inverse',
    'lu_reconstruct',
    'lu_reconstruct_assertions',  # Internally visible for MatvecLU.
    'lu_solve',
    'pivoted_cholesky',
    'sparse_or_dense_matmul',
    'sparse_or_dense_matvecmul',
]


def cholesky_concat(chol, cols, name=None):
  """Concatenates `chol @ chol.T` with additional rows and columns.

  This operation is conceptually identical to:
  ```python
  def cholesky_concat_slow(chol, cols):  # cols shaped (n + m) x m = z x m
    mat = tf.matmul(chol, chol, adjoint_b=True)  # shape of n x n
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
  with tf.name_scope(name or 'cholesky_extend'):
    dtype = dtype_util.common_dtype([chol, cols], dtype_hint=tf.float32)
    chol = tf.convert_to_tensor(chol, name='chol', dtype=dtype)
    cols = tf.convert_to_tensor(cols, name='cols', dtype=dtype)
    n = ps.shape(chol)[-1]
    mat_nm, mat_mm = cols[..., :n, :], cols[..., n:, :]
    solved_nm = tf.linalg.triangular_solve(chol, mat_nm)
    lower_right_mm = tf.linalg.cholesky(
        mat_mm - tf.matmul(solved_nm, solved_nm, adjoint_a=True))
    lower_left_mn = tf.math.conj(tf.linalg.matrix_transpose(solved_nm))
    out_batch = ps.shape(solved_nm)[:-2]
    chol = tf.broadcast_to(
        chol, ps.concat([out_batch, ps.shape(chol)[-2:]], axis=0))
    top_right_zeros_nm = tf.zeros_like(solved_nm)
    return tf.concat([tf.concat([chol, top_right_zeros_nm], axis=-1),
                      tf.concat([lower_left_mn, lower_right_mm], axis=-1)],
                     axis=-2)


def cholesky_update(chol, update_vector, multiplier=1., name=None):
  """Returns cholesky of chol @ chol.T + multiplier * u @ u.T.

  Given a (batch of) lower triangular cholesky factor(s) `chol`, along with a
  (batch of) vector(s) `update_vector`, compute the lower triangular cholesky
  factor of the rank-1 update `chol @ chol.T + multiplier * u @ u.T`, where
  `multiplier` is a (batch of) scalar(s).

  If `chol` has shape `[L, L]`, this has complexity `O(L^2)` compared to the
  naive algorithm which has complexity `O(L^3)`.

  Args:
    chol: Floating-point `Tensor` with shape `[B1, ..., Bn, L, L]`.
      Cholesky decomposition of `mat = chol @ chol.T`. Batch dimensions
      must be broadcastable with `update_vector` and `multiplier`.
    update_vector: Floating-point `Tensor` with shape `[B1, ... Bn, L]`. Vector
      defining rank-one update. Batch dimensions must be broadcastable with
      `chol` and `multiplier`.
    multiplier: Floating-point `Tensor` with shape `[B1, ..., Bn]. Scalar
      multiplier to rank-one update. Batch dimensions must be broadcastable
      with `chol` and `update_vector`. Note that updates where `multiplier` is
      positive are numerically stable, while when `multiplier` is negative
      (downdating), the update will only work if the new resulting matrix is
      still positive definite.
    name: Optional name for this op.

  #### References
  [1] Oswin Krause. Christian Igel. A More Efficient Rank-one Covariance
      Matrix Update for Evolution Strategies. 2015 ACM Conference.
      https://www.researchgate.net/publication/300581419_A_More_Efficient_Rank-one_Covariance_Matrix_Update_for_Evolution_Strategies
  """
  # TODO(b/154638092): Move this functionality in to TensorFlow.
  with tf.name_scope(name or 'cholesky_update'):
    dtype = dtype_util.common_dtype(
        [chol, update_vector, multiplier], dtype_hint=tf.float32)
    chol = tf.convert_to_tensor(chol, name='chol', dtype=dtype)
    update_vector = tf.convert_to_tensor(
        update_vector, name='update_vector', dtype=dtype)
    multiplier = tf.convert_to_tensor(
        multiplier, name='multiplier', dtype=dtype)

    batch_shape = ps.broadcast_shape(
        ps.broadcast_shape(
            ps.shape(chol)[:-2],
            ps.shape(update_vector)[:-1]), ps.shape(multiplier))
    chol = tf.broadcast_to(
        chol, ps.concat([batch_shape, ps.shape(chol)[-2:]], axis=0))
    update_vector = tf.broadcast_to(
        update_vector,
        ps.concat([batch_shape, ps.shape(update_vector)[-1:]], axis=0))
    multiplier = tf.broadcast_to(multiplier, batch_shape)

    chol_diag = tf.linalg.diag_part(chol)

    # The algorithm in [1] is implemented as a double for loop. We can treat
    # the inner loop in Algorithm 3.1 as a vector operation, and thus the
    # whole algorithm as a single for loop, and hence can use a `tf.scan`
    # on it.

    # We use for accumulation omega and b as defined in Algorithm 3.1, since
    # these are updated per iteration.

    def compute_new_column(accumulated_quantities, state):
      """Computes the next column of the updated cholesky."""
      _, _, omega, b = accumulated_quantities
      index, diagonal_member, col, col_mask = state
      omega_at_index = omega[..., index]

      # Line 4
      new_diagonal_member = tf.math.sqrt(
          tf.math.square(diagonal_member) + multiplier / b * tf.math.square(
              omega_at_index))
      # `scaling_factor` is the same as `gamma` on Line 5.
      scaling_factor = (tf.math.square(diagonal_member) * b +
                        multiplier * tf.math.square(omega_at_index))

      # The following updates are the same as the for loop in lines 6-8.
      omega = omega - (omega_at_index / diagonal_member)[..., tf.newaxis] * col
      new_col = new_diagonal_member[..., tf.newaxis]  * (
          col / diagonal_member[..., tf.newaxis] +
          (multiplier * omega_at_index / scaling_factor)[
              ..., tf.newaxis] * omega * col_mask)
      b = b + multiplier * tf.math.square(omega_at_index / diagonal_member)
      return new_diagonal_member, new_col, omega, b

    # We will scan over the columns.
    cols_mask = distribution_util.move_dimension(
        tf.linalg.band_part(tf.ones_like(chol), -1, 0),
        source_idx=-1, dest_idx=0)
    chol = distribution_util.move_dimension(chol, source_idx=-1, dest_idx=0)
    chol_diag = distribution_util.move_dimension(
        chol_diag, source_idx=-1, dest_idx=0)

    new_diag, new_chol, _, _ = tf.scan(
        fn=compute_new_column,
        elems=(tf.range(0, ps.shape(chol)[0]), chol_diag, chol, cols_mask),
        initializer=(
            tf.zeros_like(multiplier),
            tf.zeros_like(chol[0, ...]),
            update_vector,
            tf.ones_like(multiplier)))
    new_chol = distribution_util.move_dimension(
        new_chol, source_idx=0, dest_idx=-1)
    new_diag = distribution_util.move_dimension(
        new_diag, source_idx=0, dest_idx=-1)
    new_chol = tf.linalg.set_diag(new_chol, new_diag)
    return new_chol


def _swap_m_with_i(vecs, m, i):
  """Swaps `m` and `i` on axis -1. (Helper for pivoted_cholesky.)

  Given a batch of int64 vectors `vecs`, scalar index `m`, and compatibly shaped
  per-vector indices `i`, this function swaps elements `m` and `i` in each
  vector. For the use-case below, these are permutation vectors.

  Args:
    vecs: Vectors on which we perform the swap, int64 `Tensor`.
    m: Scalar int64 `Tensor`, the index into which the `i`th element is going.
    i: Batch int64 `Tensor`, shaped like vecs.shape[:-1] + [1]; the index into
      which the `m`th element is going.

  Returns:
    vecs: The updated vectors.
  """
  vecs = tf.convert_to_tensor(vecs, dtype=tf.int64, name='vecs')
  m = tf.convert_to_tensor(m, dtype=tf.int64, name='m')
  i = tf.convert_to_tensor(i, dtype=tf.int64, name='i')
  trailing_elts = tf.broadcast_to(
      tf.range(m + 1, ps.shape(vecs, out_type=tf.int64)[-1]),
      ps.shape(vecs[..., m + 1:]))
  trailing_elts = tf.where(
      tf.equal(trailing_elts, i),
      tf.gather(vecs, [m], axis=-1),
      vecs[..., m + 1:])
  # TODO(bjp): Could we use tensor_scatter_nd_update?
  vecs_shape = vecs.shape
  vecs_rank = tensorshape_util.rank(vecs_shape)
  if vecs_rank is None:
    raise NotImplementedError(
        'Input vector to swap must have statically known rank. If you cannot '
        'provide static rank please contact core TensorFlow team and request '
        'that `tf.gather` `batch_dims` argument support `tf.Tensor`-valued '
        'inputs.')
  vecs = tf.concat([
      vecs[..., :m],
      tf.gather(vecs, i, batch_dims=int(vecs_rank) - 1),
      trailing_elts
  ], axis=-1)
  tensorshape_util.set_shape(vecs, vecs_shape)
  return vecs


def _invert_permutation(perm):  # TODO(b/130217510): Remove this function.
  return tf.cast(tf.argsort(perm, axis=-1), perm.dtype)


def pivoted_cholesky(matrix, max_rank, diag_rtol=1e-3, name=None):
  """Computes the (partial) pivoted cholesky decomposition of `matrix`.

  The pivoted Cholesky is a low rank approximation of the Cholesky decomposition
  of `matrix`, i.e. as described in [(Harbrecht et al., 2012)][1]. The
  currently-worst-approximated diagonal element is selected as the pivot at each
  iteration. This yields from a `[B1...Bn, N, N]` shaped `matrix` a `[B1...Bn,
  N, K]` shaped rank-`K` approximation `lr` such that `lr @ lr.T ~= matrix`.
  Note that, unlike the Cholesky decomposition, `lr` is not triangular even in
  a rectangular-matrix sense. However, under a permutation it could be made
  triangular (it has one more zero in each column as you move to the right).

  Such a matrix can be useful as a preconditioner for conjugate gradient
  optimization, i.e. as in [(Wang et al. 2019)][2], as matmuls and solves can be
  cheaply done via the Woodbury matrix identity, as implemented by
  `tf.linalg.LinearOperatorLowRankUpdate`.

  Args:
    matrix: Floating point `Tensor` batch of symmetric, positive definite
      matrices.
    max_rank: Scalar `int` `Tensor`, the rank at which to truncate the
      approximation.
    diag_rtol: Scalar floating point `Tensor` (same dtype as `matrix`). If the
      errors of all diagonal elements of `lr @ lr.T` are each lower than
      `element * diag_rtol`, iteration is permitted to terminate early.
    name: Optional name for the op.

  Returns:
    lr: Low rank pivoted Cholesky approximation of `matrix`.

  #### References

  [1]: H Harbrecht, M Peters, R Schneider. On the low-rank approximation by the
       pivoted Cholesky decomposition. _Applied numerical mathematics_,
       62(4):428-440, 2012.

  [2]: K. A. Wang et al. Exact Gaussian Processes on a Million Data Points.
       _arXiv preprint arXiv:1903.08114_, 2019. https://arxiv.org/abs/1903.08114
  """
  with tf.name_scope(name or 'pivoted_cholesky'):
    dtype = dtype_util.common_dtype([matrix, diag_rtol],
                                    dtype_hint=tf.float32)
    if not isinstance(matrix, tf.linalg.LinearOperator):
      matrix = tf.convert_to_tensor(matrix, name='matrix', dtype=dtype)
    if tensorshape_util.rank(matrix.shape) is None:
      raise NotImplementedError('Rank of `matrix` must be known statically')
    if isinstance(matrix, tf.linalg.LinearOperator):
      matrix_shape = tf.cast(matrix.shape_tensor(), tf.int64)
    else:
      matrix_shape = ps.shape(matrix, out_type=tf.int64)

    max_rank = tf.convert_to_tensor(
        max_rank, name='max_rank', dtype=tf.int64)
    max_rank = tf.minimum(max_rank, matrix_shape[-1])
    diag_rtol = tf.convert_to_tensor(
        diag_rtol, dtype=dtype, name='diag_rtol')
    matrix_diag = tf.linalg.diag_part(matrix)
    # matrix is P.D., therefore all matrix_diag > 0, so we don't need abs.
    orig_error = tf.reduce_max(matrix_diag, axis=-1)

    def cond(m, pchol, perm, matrix_diag):
      """Condition for `tf.while_loop` continuation."""
      del pchol
      del perm
      error = tf.linalg.norm(matrix_diag, ord=1, axis=-1)
      max_err = tf.reduce_max(error / orig_error)
      return (m < max_rank) & (tf.equal(m, 0) | (max_err > diag_rtol))

    batch_dims = tensorshape_util.rank(matrix.shape) - 2
    def batch_gather(params, indices, axis=-1):
      return tf.gather(params, indices, axis=axis, batch_dims=batch_dims)

    def body(m, pchol, perm, matrix_diag):
      """Body of a single `tf.while_loop` iteration."""
      # Here is roughly a numpy, non-batched version of what's going to happen.
      # (See also Algorithm 1 of Harbrecht et al.)
      # 1: maxi = np.argmax(matrix_diag[perm[m:]]) + m
      # 2: maxval = matrix_diag[perm][maxi]
      # 3: perm[m], perm[maxi] = perm[maxi], perm[m]
      # 4: row = matrix[perm[m]][perm[m + 1:]]
      # 5: row -= np.sum(pchol[:m][perm[m + 1:]] * pchol[:m][perm[m]]], axis=-2)
      # 6: pivot = np.sqrt(maxval); row /= pivot
      # 7: row = np.concatenate([[[pivot]], row], -1)
      # 8: matrix_diag[perm[m:]] -= row**2
      # 9: pchol[m, perm[m:]] = row

      # Find the maximal position of the (remaining) permuted diagonal.
      # Steps 1, 2 above.
      permuted_diag = batch_gather(matrix_diag, perm[..., m:])
      maxi = tf.argmax(
          permuted_diag, axis=-1, output_type=tf.int64)[..., tf.newaxis]
      maxval = batch_gather(permuted_diag, maxi)
      maxi = maxi + m
      maxval = maxval[..., 0]
      # Update perm: Swap perm[...,m] with perm[...,maxi]. Step 3 above.
      perm = _swap_m_with_i(perm, m, maxi)
      # Step 4.
      if callable(getattr(matrix, 'row', None)):
        row = matrix.row(perm[..., m])[..., tf.newaxis, :]
      else:
        row = batch_gather(matrix, perm[..., m:m + 1], axis=-2)
      row = batch_gather(row, perm[..., m + 1:])
      # Step 5.
      prev_rows = pchol[..., :m, :]
      prev_rows_perm_m_onward = batch_gather(prev_rows, perm[..., m + 1:])
      prev_rows_pivot_col = batch_gather(prev_rows, perm[..., m:m + 1])
      row -= tf.reduce_sum(
          prev_rows_perm_m_onward * prev_rows_pivot_col,
          axis=-2)[..., tf.newaxis, :]
      # Step 6.
      pivot = tf.sqrt(maxval)[..., tf.newaxis, tf.newaxis]
      # Step 7.
      row = tf.concat([pivot, row / pivot], axis=-1)
      # TODO(b/130899118): Pad grad fails with int64 paddings.
      # Step 8.
      paddings = tf.concat([
          tf.zeros([ps.rank(pchol) - 1, 2], dtype=tf.int32),
          [[tf.cast(m, tf.int32), 0]]], axis=0)
      diag_update = tf.pad(row**2, paddings=paddings)[..., 0, :]
      reverse_perm = _invert_permutation(perm)
      matrix_diag = matrix_diag - batch_gather(diag_update, reverse_perm)
      # Step 9.
      row = tf.pad(row, paddings=paddings)
      # TODO(bjp): Defer the reverse permutation all-at-once at the end?
      row = batch_gather(row, reverse_perm)
      pchol_shape = pchol.shape
      pchol = tf.concat([pchol[..., :m, :], row, pchol[..., m + 1:, :]],
                        axis=-2)
      tensorshape_util.set_shape(pchol, pchol_shape)
      return m + 1, pchol, perm, matrix_diag

    m = np.int64(0)
    pchol = tf.zeros(matrix_shape, dtype=matrix.dtype)[..., :max_rank, :]
    perm = tf.broadcast_to(
        ps.range(matrix_shape[-1]), matrix_shape[:-1])
    _, pchol, _, _ = tf.while_loop(
        cond=cond, body=body, loop_vars=(m, pchol, perm, matrix_diag))
    pchol = tf.linalg.matrix_transpose(pchol)
    tensorshape_util.set_shape(
        pchol, tensorshape_util.concatenate(matrix_diag.shape, [None]))
    return pchol


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
      Default value: `None` (i.e., 'lu_solve').

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

  with tf.name_scope(name or 'lu_solve'):
    lower_upper = tf.convert_to_tensor(
        lower_upper, dtype_hint=tf.float32, name='lower_upper')
    perm = tf.convert_to_tensor(perm, dtype_hint=tf.int32, name='perm')
    rhs = tf.convert_to_tensor(
        rhs, dtype_hint=lower_upper.dtype, name='rhs')

    assertions = _lu_solve_assertions(lower_upper, perm, rhs, validate_args)
    if assertions:
      with tf.control_dependencies(assertions):
        lower_upper = tf.identity(lower_upper)
        perm = tf.identity(perm)
        rhs = tf.identity(rhs)

    if (tensorshape_util.rank(rhs.shape) == 2 and
        tensorshape_util.rank(perm.shape) == 1):
      # Both rhs and perm have scalar batch_shape.
      permuted_rhs = tf.gather(rhs, perm, axis=-2)
    else:
      # Either rhs or perm have non-scalar batch_shape or we can't determine
      # this information statically.
      rhs_shape = tf.shape(rhs)
      broadcast_batch_shape = tf.broadcast_dynamic_shape(
          rhs_shape[:-2],
          tf.shape(perm)[:-1])
      d, m = rhs_shape[-2], rhs_shape[-1]
      rhs_broadcast_shape = tf.concat([broadcast_batch_shape, [d, m]], axis=0)

      # Tile out rhs.
      broadcast_rhs = tf.broadcast_to(rhs, rhs_broadcast_shape)
      broadcast_rhs = tf.reshape(broadcast_rhs, [-1, d, m])

      # Tile out perm and add batch indices.
      broadcast_perm = tf.broadcast_to(perm, rhs_broadcast_shape[:-1])
      broadcast_perm = tf.reshape(broadcast_perm, [-1, d])
      broadcast_batch_size = tf.reduce_prod(broadcast_batch_shape)
      broadcast_batch_indices = tf.broadcast_to(
          tf.range(broadcast_batch_size)[:, tf.newaxis],
          [broadcast_batch_size, d])
      broadcast_perm = tf.stack([broadcast_batch_indices, broadcast_perm],
                                axis=-1)

      permuted_rhs = tf.gather_nd(broadcast_rhs, broadcast_perm)
      permuted_rhs = tf.reshape(permuted_rhs, rhs_broadcast_shape)

    lower = tf.linalg.set_diag(
        tf.linalg.band_part(lower_upper, num_lower=-1, num_upper=0),
        tf.ones(tf.shape(lower_upper)[:-1], dtype=lower_upper.dtype))
    return tf.linalg.triangular_solve(
        lower_upper,  # Only upper is accessed.
        tf.linalg.triangular_solve(lower, permuted_rhs), lower=False)


def lu_matrix_inverse(lower_upper, perm, validate_args=False, name=None):
  """Computes a matrix inverse given the matrix's LU decomposition.

  This op is conceptually identical to,

  ```python
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
      Default value: `None` (i.e., 'lu_matrix_inverse').

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

  with tf.name_scope(name or 'lu_matrix_inverse'):
    lower_upper = tf.convert_to_tensor(
        lower_upper, dtype_hint=tf.float32, name='lower_upper')
    perm = tf.convert_to_tensor(perm, dtype_hint=tf.int32, name='perm')
    assertions = lu_reconstruct_assertions(lower_upper, perm, validate_args)
    if assertions:
      with tf.control_dependencies(assertions):
        lower_upper = tf.identity(lower_upper)
        perm = tf.identity(perm)
    shape = tf.shape(lower_upper)
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
      Default value: `None` (i.e., 'lu_reconstruct').

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
  with tf.name_scope(name or 'lu_reconstruct'):
    lower_upper = tf.convert_to_tensor(
        lower_upper, dtype_hint=tf.float32, name='lower_upper')
    perm = tf.convert_to_tensor(perm, dtype_hint=tf.int32, name='perm')

    assertions = lu_reconstruct_assertions(lower_upper, perm, validate_args)
    if assertions:
      with tf.control_dependencies(assertions):
        lower_upper = tf.identity(lower_upper)
        perm = tf.identity(perm)

    shape = tf.shape(lower_upper)

    lower = tf.linalg.set_diag(
        tf.linalg.band_part(lower_upper, num_lower=-1, num_upper=0),
        tf.ones(shape[:-1], dtype=lower_upper.dtype))
    upper = tf.linalg.band_part(lower_upper, num_lower=0, num_upper=-1)
    x = tf.matmul(lower, upper)

    if (tensorshape_util.rank(lower_upper.shape) is None or
        tensorshape_util.rank(lower_upper.shape) != 2):
      # We either don't know the batch rank or there are >0 batch dims.
      batch_size = tf.reduce_prod(shape[:-2])
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

    tensorshape_util.set_shape(x, lower_upper.shape)
    return x


def lu_reconstruct_assertions(lower_upper, perm, validate_args):
  """Returns list of assertions related to `lu_reconstruct` assumptions."""
  assertions = []

  message = 'Input `lower_upper` must have at least 2 dimensions.'
  if tensorshape_util.rank(lower_upper.shape) is not None:
    if tensorshape_util.rank(lower_upper.shape) < 2:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        assert_util.assert_rank_at_least(lower_upper, rank=2, message=message))

  message = '`rank(lower_upper)` must equal `rank(perm) + 1`'
  if (tensorshape_util.rank(lower_upper.shape) is not None and
      tensorshape_util.rank(perm.shape) is not None):
    if (tensorshape_util.rank(lower_upper.shape) !=
        tensorshape_util.rank(perm.shape) + 1):
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        assert_util.assert_rank(
            lower_upper, rank=tf.rank(perm) + 1, message=message))

  message = '`lower_upper` must be square.'
  if tensorshape_util.is_fully_defined(lower_upper.shape[:-2]):
    if lower_upper.shape[-2] != lower_upper.shape[-1]:
      raise ValueError(message)
  elif validate_args:
    m, n = tf.split(tf.shape(lower_upper)[-2:], num_or_size_splits=2)
    assertions.append(assert_util.assert_equal(m, n, message=message))

  return assertions


def _lu_solve_assertions(lower_upper, perm, rhs, validate_args):
  """Returns list of assertions related to `lu_solve` assumptions."""
  assertions = lu_reconstruct_assertions(lower_upper, perm, validate_args)

  message = 'Input `rhs` must have at least 2 dimensions.'
  if tensorshape_util.rank(rhs.shape) is not None:
    if tensorshape_util.rank(rhs.shape) < 2:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        assert_util.assert_rank_at_least(rhs, rank=2, message=message))

  message = '`lower_upper.shape[-1]` must equal `rhs.shape[-1]`.'
  if (tf.compat.dimension_value(lower_upper.shape[-1]) is not None and
      tf.compat.dimension_value(rhs.shape[-2]) is not None):
    if lower_upper.shape[-1] != rhs.shape[-2]:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        assert_util.assert_equal(
            tf.shape(lower_upper)[-1],
            tf.shape(rhs)[-2],
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
      Default value: 'sparse_or_dense_matmul'.
    **kwargs: Keyword arguments to `tf.sparse_tensor_dense_matmul` or
      `tf.matmul`.

  Returns:
    product: A dense (batch of) matrix-shaped Tensor of the same batch shape and
    dtype as `sparse_or_dense_a` and `dense_b`. If `sparse_or_dense_a` or
    `dense_b` is adjointed through `kwargs` then the shape is adjusted
    accordingly.
  """
  with tf.name_scope(name or 'sparse_or_dense_matmul'):
    dense_b = tf.convert_to_tensor(
        dense_b, dtype_hint=tf.float32, name='dense_b')

    if validate_args:
      assert_a_rank_at_least_2 = assert_util.assert_rank_at_least(
          sparse_or_dense_a,
          rank=2,
          message='Input `sparse_or_dense_a` must have at least 2 dimensions.')
      assert_b_rank_at_least_2 = assert_util.assert_rank_at_least(
          dense_b,
          rank=2,
          message='Input `dense_b` must have at least 2 dimensions.')
      with tf.control_dependencies(
          [assert_a_rank_at_least_2, assert_b_rank_at_least_2]):
        sparse_or_dense_a = tf.identity(sparse_or_dense_a)
        dense_b = tf.identity(dense_b)

    if isinstance(sparse_or_dense_a, (tf.SparseTensor, tf1.SparseTensorValue)):
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
      Default value: 'sparse_or_dense_matvecmul'.
    **kwargs: Keyword arguments to `tf.sparse_tensor_dense_matmul` or
      `tf.matmul`.

  Returns:
    product: A dense (batch of) vector-shaped Tensor of the same batch shape and
    dtype as `sparse_or_dense_matrix` and `dense_vector`.
  """
  with tf.name_scope(name or 'sparse_or_dense_matvecmul'):
    dense_vector = tf.convert_to_tensor(
        dense_vector, dtype_hint=tf.float32, name='dense_vector')
    return tf.squeeze(
        sparse_or_dense_matmul(
            sparse_or_dense_matrix,
            dense_vector[..., tf.newaxis],
            validate_args=validate_args,
            **kwargs),
        axis=[-1])


def fill_triangular(x, upper=False, name=None):
  """Creates a (batch of) triangular matrix from a vector of inputs.

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

  with tf.name_scope(name or 'fill_triangular'):
    x = tf.convert_to_tensor(x, name='x')
    m = tf.compat.dimension_value(
        tensorshape_util.with_rank_at_least(x.shape, 1)[-1])
    if m is not None:
      # Formula derived by solving for n: m = n(n+1)/2.
      m = np.int32(m)
      n = np.sqrt(0.25 + 2. * m) - 0.5
      if n != np.floor(n):
        raise ValueError('Input right-most shape ({}) does not '
                         'correspond to a triangular matrix.'.format(m))
      n = np.int32(n)
      static_final_shape = tensorshape_util.concatenate(x.shape[:-1], [n, n])
    else:
      m = tf.shape(x)[-1]
      # For derivation, see above. Casting automatically lops off the 0.5, so we
      # omit it.  We don't validate n is an integer because this has
      # graph-execution cost; an error will be thrown from the reshape, below.
      n = tf.cast(
          tf.sqrt(0.25 + tf.cast(2 * m, dtype=tf.float32)), dtype=tf.int32)
      static_final_shape = tensorshape_util.concatenate(
          tensorshape_util.with_rank_at_least(x.shape, 1)[:-1], [None, None])

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
    ndims = ps.rank(x)
    if upper:
      x_list = [x, tf.reverse(x[..., n:], axis=[ndims - 1])]
    else:
      x_list = [x[..., n:], tf.reverse(x, axis=[ndims - 1])]
    new_shape = (
        tensorshape_util.as_list(static_final_shape)
        if tensorshape_util.is_fully_defined(static_final_shape) else tf.concat(
            [tf.shape(x)[:-1], [n, n]], axis=0))
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

  with tf.name_scope(name or 'fill_triangular_inverse'):
    x = tf.convert_to_tensor(x, name='x')
    n = tf.compat.dimension_value(
        tensorshape_util.with_rank_at_least(x.shape, 2)[-1])
    if n is not None:
      n = np.int32(n)
      m = np.int32((n * (n + 1)) // 2)
      static_final_shape = tensorshape_util.concatenate(x.shape[:-2], [m])
    else:
      n = tf.shape(x)[-1]
      m = (n * (n + 1)) // 2
      static_final_shape = tensorshape_util.concatenate(
          tensorshape_util.with_rank_at_least(x.shape, 2)[:-2], [None])
    ndims = ps.rank(x)
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
        ps.concat([ps.shape(x)[:-2], [n * (n - 1)]], axis=0))
    y = tf.concat([initial_elements, end_sequence[..., :m - n]], axis=-1)
    tensorshape_util.set_shape(y, static_final_shape)
    return y


def _get_shape(x, out_type=tf.int32):
  # Return the shape of a Tensor or a SparseTensor as an np.array if its shape
  # is known statically. Otherwise return a Tensor representing the shape.
  if tensorshape_util.is_fully_defined(x.shape):
    return np.array(
        tensorshape_util.as_list(x.shape),
        dtype=dtype_util.as_numpy_dtype(out_type))
  return tf.shape(x, out_type=out_type)


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
  sp_a_shape = tf.convert_to_tensor(_get_shape(sp_a, tf.int64))
  ind_mat = tf.concat([[sp_a_shape[-2:]], tf.eye(2, dtype=tf.int64)], axis=0)
  indices = tf.matmul(sp_a.indices, ind_mat)
  dense_shape = sp_a_shape[0] * sp_a_shape[1:]
  return tf.SparseTensor(
      indices=indices, values=sp_a.values, dense_shape=dense_shape)


def _maybe_validate_matrix(a, validate_args):
  """Checks that input is a `float` matrix."""
  assertions = []
  if not dtype_util.is_floating(a.dtype):
    raise TypeError('Input `a` must have `float`-like `dtype` '
                    '(saw {}).'.format(dtype_util.name(a.dtype)))
  if tensorshape_util.rank(a.shape) is not None:
    if tensorshape_util.rank(a.shape) < 2:
      raise ValueError('Input `a` must have at least 2 dimensions '
                       '(saw: {}).'.format(tensorshape_util.rank(a.shape)))
  elif validate_args:
    assertions.append(assert_util.assert_rank_at_least(
        a, rank=2, message='Input `a` must have at least 2 dimensions.'))
  return assertions


def _hpsd_logdet_fwd(matrix, cholesky_matrix):
  if cholesky_matrix is None:
    cholesky_matrix = tf.linalg.cholesky(matrix)
  output = 2. * tf.reduce_sum(
      tf.math.log(tf.linalg.diag_part(cholesky_matrix)), axis=-1)
  return output, (cholesky_matrix,)


def _hpsd_logdet_bwd(aux, g):
  """Reverse mode impl for hpsd_logdet."""
  cholesky_matrix, = aux
  dmatrix = g
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  chol_inverse = chol_linop.solve(tf.linalg.eye(
      num_rows=chol_linop.domain_dimension_tensor(),
      dtype=chol_linop.dtype))
  inverse_matrix = tf.linalg.matmul(
      chol_inverse, chol_inverse, transpose_a=True)

  return dmatrix[..., tf.newaxis, tf.newaxis] * inverse_matrix, None


def _hpsd_logdet_jvp(primals, tangents):
  matrix, cholesky_matrix = primals
  gmatrix, _ = tangents
  output, (cholesky_matrix,) = _hpsd_logdet_fwd(matrix, cholesky_matrix)
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  matrix_jvp = chol_linop.solve(chol_linop.solve(gmatrix), adjoint=True)
  return output, tf.linalg.trace(matrix_jvp)


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_hpsd_logdet_fwd,
    vjp_bwd=_hpsd_logdet_bwd,
    jvp_fn=_hpsd_logdet_jvp)
def _hpsd_logdet_custom_gradient(matrix, cholesky_matrix):
  return _hpsd_logdet_fwd(matrix, cholesky_matrix)[0]


def hpsd_logdet(matrix, cholesky_matrix=None):
  """Computes `log|det(matrix)|`, where `matrix` is a HPSD matrix.

  Given `matrix` computes `log|det(matrix)|`, where `matrix` is Hermitian
  positive Semi-definite matrix.

  Args:
    matrix: Floating-point `Tensor` of shape `[..., N, N]`. Represents
      a Hermitian positive semi-definite matrix.
    cholesky_matrix: (Optional) Floating-point `Tensor` of shape `[..., N, N]`
      that represents a Cholesky factor of `matrix`.
  Returns:
    hpsd_logdet: Scalar `Tensor`, retaining the batch shape of `matrix`.
  """
  with tf.name_scope('hpsd_logdet'):
    dtype = dtype_util.common_dtype([matrix, cholesky_matrix], tf.float32)
    matrix = tf.convert_to_tensor(matrix, dtype=dtype)
    if cholesky_matrix is not None:
      cholesky_matrix = tf.convert_to_tensor(cholesky_matrix, dtype=dtype)
    return _hpsd_logdet_custom_gradient(matrix, cholesky_matrix)


def _hpsd_solve_fwd(matrix, rhs, cholesky_matrix):
  del matrix
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  solve_rhs = chol_linop.solve(chol_linop.solve(rhs), adjoint=True)
  return solve_rhs, (cholesky_matrix, rhs, solve_rhs)


def _hpsd_solve_bwd(cholesky_matrix, aux, g):
  """Reverse mode impl for hpsd_solve."""
  del cholesky_matrix

  cholesky_matrix, rhs, solve_rhs = aux
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)

  chol_inv = chol_linop.solve(
      tf.eye(chol_linop.domain_dimension_tensor(), dtype=chol_linop.dtype))

  rhs_grad = tf.linalg.matmul(
      chol_inv, tf.linalg.matmul(chol_inv, g), transpose_a=True)
  matrix_grad = -0.5 * (
      tf.linalg.matmul(rhs_grad, solve_rhs, adjoint_b=True) +
      tf.linalg.matmul(solve_rhs, rhs_grad, adjoint_b=True))

  matrix_grad, rhs_grad = generic.fix_gradient_for_broadcasting(
      [cholesky_matrix[..., tf.newaxis], rhs[..., tf.newaxis, :]],
      [matrix_grad[..., tf.newaxis], rhs_grad[..., tf.newaxis, :]])
  return tf.squeeze(matrix_grad, axis=-1), tf.squeeze(rhs_grad, axis=-2)


def _hpsd_solve_jvp(cholesky_matrix, primals, tangents):
  """JVP for hpsd_solve."""
  matrix, rhs = primals
  gmatrix, grhs = tangents
  output, (cholesky_matrix, _, solve_rhs) = _hpsd_solve_fwd(
      matrix, rhs, cholesky_matrix)
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  gmatrix_linop = tf.linalg.LinearOperatorFullMatrix(gmatrix)
  grad = grhs - 0.5 * (gmatrix_linop.matmul(solve_rhs) +
                       gmatrix_linop.matmul(solve_rhs, adjoint=True))
  chol_inv = chol_linop.solve(
      tf.eye(chol_linop.domain_dimension_tensor(), dtype=chol_linop.dtype))
  grad = tf.linalg.matmul(
      chol_inv, tf.linalg.matmul(chol_inv, grad), transpose_a=True)
  return output, grad


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_hpsd_solve_fwd,
    vjp_bwd=_hpsd_solve_bwd,
    jvp_fn=_hpsd_solve_jvp,
    nondiff_argnums=(2,))
def _hpsd_solve_custom_gradient(matrix, rhs, cholesky_matrix):
  return _hpsd_solve_fwd(matrix, rhs, cholesky_matrix)[0]


def hpsd_solve(matrix, rhs, cholesky_matrix=None):
  """Computes `matrix^-1 rhs`, where `matrix` is HPSD.

  Given `matrix` and `rhs`, computes `matrix^-1 rhs`, where
  `matrix` is a Hermitian positive semi-definite matrix.

  Args:
    matrix: Floating-point `Tensor` of shape `[..., N, N]`. Represents
      a Hermitian positive semi-definite matrix.
    rhs: Floating-point `Tensor` of shape `[..., N, K]`.
    cholesky_matrix: (Optional) Floating-point `Tensor` of shape `[..., N, N]`
      that represents a Cholesky factor of `matrix`.
  Returns:
    hpsd_solve: `Tensor` of shape `[..., N, K]`.
  """
  with tf.name_scope('hpsd_solve'):
    dtype = dtype_util.common_dtype([matrix, rhs, cholesky_matrix], tf.float32)
    matrix = tf.convert_to_tensor(matrix, dtype=dtype)
    rhs = tf.convert_to_tensor(rhs, dtype=dtype)
    if cholesky_matrix is None:
      cholesky_matrix = tf.linalg.cholesky(matrix)
    else:
      cholesky_matrix = tf.convert_to_tensor(cholesky_matrix, dtype=dtype)
    return _hpsd_solve_custom_gradient(matrix, rhs, cholesky_matrix)


def hpsd_solvevec(matrix, rhs, cholesky_matrix=None):
  """Computes `matrix^-1 rhs`, where `matrix` is HPSD.

  Given `matrix` and `rhs`, computes `matrix^-1 rhs`, where
  `matrix` is a Hermitian positive semi-definite matrix.

  Args:
    matrix: Floating-point `Tensor` of shape `[..., N, N]`. Represents
      a Hermitian positive semi-definite matrix.
    rhs: Floating-point `Tensor` of shape `[..., N]`.
    cholesky_matrix: (Optional) Floating-point `Tensor` of shape `[..., N, N]`
      that represents a Cholesky factor of `matrix`.
  Returns:
    hpsd_solvevec: `Tensor` of shape `[..., N]`.
  """
  with tf.name_scope('hpsd_solvevec'):
    dtype = dtype_util.common_dtype([matrix, rhs, cholesky_matrix], tf.float32)
    matrix = tf.convert_to_tensor(matrix, dtype=dtype)
    rhs = tf.convert_to_tensor(rhs, dtype=dtype)
    if cholesky_matrix is None:
      cholesky_matrix = tf.linalg.cholesky(matrix)
    else:
      cholesky_matrix = tf.convert_to_tensor(cholesky_matrix, dtype=dtype)
    return tf.squeeze(
        _hpsd_solve_custom_gradient(
            matrix, rhs[..., tf.newaxis], cholesky_matrix), axis=-1)


def _hpsd_quadratic_form_solve_fwd(matrix, rhs, cholesky_matrix):
  if cholesky_matrix is None:
    cholesky_matrix = tf.linalg.cholesky(matrix)
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  solve_rhs = chol_linop.solve(rhs)
  output = tf.linalg.matmul(solve_rhs, solve_rhs, transpose_a=True)
  return output, (cholesky_matrix, rhs, solve_rhs)


def _hpsd_quadratic_form_solve_bwd(cholesky_matrix, aux, g):
  """Reverse mode impl for hpsd_quadratic_form_solve."""
  del cholesky_matrix

  cholesky_matrix, rhs, solve_rhs = aux
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  full_solve = chol_linop.solve(solve_rhs, adjoint=True)
  matrix_grad = -tf.linalg.matmul(
      full_solve, tf.linalg.matmul(g, full_solve, transpose_b=True))
  rhs_grad = tf.linalg.matmul(2. * full_solve, g)

  # Given that Matrix has shape [N, N] and RHS has shape [N, M], we
  # need to add extra ones to make the shapes agree.

  matrix_grad, rhs_grad = generic.fix_gradient_for_broadcasting(
      [cholesky_matrix[..., tf.newaxis], rhs[..., tf.newaxis, :]],
      [matrix_grad[..., tf.newaxis], rhs_grad[..., tf.newaxis, :]])
  return tf.squeeze(matrix_grad, axis=-1), tf.squeeze(rhs_grad, axis=-2)


def _hpsd_quadratic_form_solve_jvp(cholesky_matrix, primals, tangents):
  """JVP for hpsd_quadratic_form_solve."""
  matrix, rhs = primals
  gmatrix, grhs = tangents
  output, (cholesky_matrix, _, solve_rhs) = _hpsd_quadratic_form_solve_fwd(
      matrix, rhs, cholesky_matrix)
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  full_solve = chol_linop.solve(solve_rhs, adjoint=True)
  gmatrix_linop = tf.linalg.LinearOperatorFullMatrix(gmatrix)
  jvp = tf.linalg.matmul(
      full_solve,
      2. * grhs - gmatrix_linop.matmul(full_solve),
      transpose_a=True)
  return output, jvp


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_hpsd_quadratic_form_solve_fwd,
    vjp_bwd=_hpsd_quadratic_form_solve_bwd,
    jvp_fn=_hpsd_quadratic_form_solve_jvp,
    nondiff_argnums=(2,))
def _hpsd_quadratic_form_solve_custom_gradient(
    matrix, rhs, cholesky_matrix):
  return _hpsd_quadratic_form_solve_fwd(matrix, rhs, cholesky_matrix)[0]


def hpsd_quadratic_form_solve(matrix, rhs, cholesky_matrix=None):
  """Computes `rhs^T matrix^-1 rhs`, where `matrix` is HPSD.

  Given `matrix` and `rhs`, computes `rhs^T @ matrix^-1 rhs`, where
  `matrix` is a Hermitian positive semi-definite matrix.

  Args:
    matrix: Floating-point `Tensor` of shape `[..., N, N]`. Represents
      a Hermitian positive semi-definite matrix.
    rhs: Floating-point `Tensor` of shape `[..., N, K]`.
    cholesky_matrix: (Optional) Floating-point `Tensor` of shape `[..., N, N]`
      that represents a Cholesky factor of `matrix`.
  Returns:
    hpsd_quadratic_form_solve: `Tensor` of shape `[..., K, K]`.
  """
  with tf.name_scope('hpsd_quadratic_form_solve'):
    dtype = dtype_util.common_dtype([matrix, rhs, cholesky_matrix], tf.float32)
    matrix = tf.convert_to_tensor(matrix, dtype=dtype)
    rhs = tf.convert_to_tensor(rhs, dtype=dtype)
    if cholesky_matrix is not None:
      cholesky_matrix = tf.convert_to_tensor(cholesky_matrix, dtype=dtype)
    return _hpsd_quadratic_form_solve_custom_gradient(
        matrix, rhs, cholesky_matrix)


def _hpsd_quadratic_form_solvevec_fwd(matrix, rhs, cholesky_matrix):
  if cholesky_matrix is None:
    cholesky_matrix = tf.linalg.cholesky(matrix)
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  solve_rhs = chol_linop.solvevec(rhs)
  output = tf.math.reduce_sum(tf.math.square(solve_rhs), axis=-1)
  return output, (cholesky_matrix, rhs, solve_rhs)


def _hpsd_quadratic_form_solvevec_bwd(aux, g):
  """Reverse mode impl for hpsd_quadratic_form_solvevec."""
  cholesky_matrix, rhs, solve_rhs = aux
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  full_solve = chol_linop.solvevec(solve_rhs, adjoint=True)
  matrix_grad = -full_solve[..., tf.newaxis, :] * full_solve[..., tf.newaxis]
  rhs_grad = 2. * full_solve

  matrix_grad, rhs_grad = generic.fix_gradient_for_broadcasting(
      [cholesky_matrix, rhs[..., tf.newaxis]],
      [matrix_grad * g[..., tf.newaxis, tf.newaxis],
       (rhs_grad * g[..., tf.newaxis])[..., tf.newaxis]])
  return matrix_grad, tf.squeeze(rhs_grad, axis=-1), None


def _hpsd_quadratic_form_solvevec_jvp(primals, tangents):
  """JVP for hpsd_quadratic_form_solvevec."""
  matrix, rhs, cholesky_matrix = primals
  gmatrix, grhs, _ = tangents
  output, (cholesky_matrix, _, solve_rhs) = _hpsd_quadratic_form_solvevec_fwd(
      matrix, rhs, cholesky_matrix)
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
  full_solve = chol_linop.solvevec(solve_rhs, adjoint=True)
  gmatrix_linop = tf.linalg.LinearOperatorFullMatrix(gmatrix)
  jvp = tf.math.reduce_sum(
      full_solve * (2. * grhs - gmatrix_linop.matvec(full_solve)), axis=-1)
  return output, jvp


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_hpsd_quadratic_form_solvevec_fwd,
    vjp_bwd=_hpsd_quadratic_form_solvevec_bwd,
    jvp_fn=_hpsd_quadratic_form_solvevec_jvp)
def _hpsd_quadratic_form_solvevec_custom_gradient(
    matrix, rhs, cholesky_matrix):
  return _hpsd_quadratic_form_solvevec_fwd(matrix, rhs, cholesky_matrix)[0]


def hpsd_quadratic_form_solvevec(matrix, rhs, cholesky_matrix=None):
  """Computes `rhs^T matrix^-1 rhs`, where `matrix` is HPSD.

  Given `matrix` and `rhs`, computes `rhs^T @ matrix^-1 rhs`, where
  `matrix` is a Hermitian positive semi-definite matrix.

  Args:
    matrix: Floating-point `Tensor` of shape `[..., N, N]`. Represents
      a Hermitian positive semi-definite matrix.
    rhs: Floating-point `Tensor` of shape `[..., N]`.
    cholesky_matrix: (Optional) Floating-point `Tensor` of shape `[..., N, N]`
      that represents a Cholesky factor of `matrix`.
  Returns:
    hpsd_quadratic_form_solvevec: Scalar `Tensor`.
  """
  with tf.name_scope('hpsd_quadratic_form_solvevec'):
    dtype = dtype_util.common_dtype([matrix, rhs, cholesky_matrix], tf.float32)
    matrix = tf.convert_to_tensor(matrix, dtype=dtype)
    rhs = tf.convert_to_tensor(rhs, dtype=dtype)
    if cholesky_matrix is not None:
      cholesky_matrix = tf.convert_to_tensor(cholesky_matrix, dtype=dtype)
    return _hpsd_quadratic_form_solvevec_custom_gradient(
        matrix, rhs, cholesky_matrix)
