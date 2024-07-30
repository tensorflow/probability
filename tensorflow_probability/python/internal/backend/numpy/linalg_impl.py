# Copyright 2019 The TensorFlow Probability Authors.
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
"""Numpy implementations of `tf.linalg` functions."""

import collections
import functools
# Dependency imports
import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import ops

scipy_linalg = utils.try_import('scipy.linalg')


__all__ = [
    'adjoint',
    'band_part',
    'cholesky',
    'cholesky_solve',
    'det',
    'diag',
    'diag_part',
    'eig',
    'eigh',
    'eigvals',
    'eigvalsh',
    'einsum',
    'eye',
    'inv',
    'logdet',
    'lstsq',
    'lu',
    'matmul',
    'matvec',
    'matrix_rank',
    'matrix_transpose',
    'norm',
    'pinv',
    'qr',
    'set_diag',
    'slogdet',
    'solve',
    'svd',
    'tensor_diag',
    'tensordot',
    'trace',
    'triangular_solve',
    'cross',
    # 'expm',
    # 'global_norm',
    # 'logm',
    # 'l2_normalize'
    # 'sqrtm',
    # 'tensor_diag_part',
    # 'tridiagonal_solve',
]


JAX_MODE = False


def _band_part(input, num_lower, num_upper, name=None):  # pylint: disable=redefined-builtin
  del name
  result = ops.convert_to_tensor(input)
  if num_lower > -1:
    result = np.triu(result, -num_lower)
  if num_upper > -1:
    result = np.tril(result, num_upper)
  return result


def _create_solve_broadcastable_inputs(a, b):
  """Internal method to handle broadcasting / reshaping for solves.

  Args:
    a: `Tensor` of shape `[B1, ..., Bk, N, N]`.
    b: `Tensor` of shape `[C1, ..., Cl, N, M]`.

  Returns:
    a: Reshaped version of `a` suitable for solves.
    b: Reshaped version of `b` suitable for solves.
    b_batch_dims: List of dimensions of `b` that have been reshaped away in `b`.
    result_permutation: Permutation of dimensions of the result of the solve,
      that produces the same shape as naively computing
      `solve(broadcast(a), broadcast(b))`.
  """

  # Returns reshaped inputs along with a list of batch dimensions and where
  # they should be in the result.

  # We mainly take advantage of the fact that we can compute a solve with a
  # matrix `a` and batch of matrices `b` of shapes `[N, N]` and `[B, N, M]` via
  # a solve with matrix `a` and a matrix `b'` of shapes `[N, N]` and `[N, M *
  # B]`, where `b'` is related to `b` via a transpose in reshape.
  # In general we can do this for arbitrary batch dimensions on both matrices by
  # a combination of transposing and broadcasting.

  a = ops.convert_to_tensor(a)
  b = ops.convert_to_tensor(b)
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as onp  # pylint: disable=g-import-not-at-top,reimported

  # Special case batch dimensions are equal, only one tensor has batch
  # dimensions, or batch dimensions are the same.

  # If there are zero dimensions, just broadcast arguments and pass through.
  if 0 in a.shape or 0 in b.shape:
    a = a + np.zeros(b.shape[:-2] + (1, 1), dtype=a.dtype)
    b = b + np.zeros(a.shape[:-2] + (1, 1), dtype=b.dtype)
    return a, b, None, None

  if a.shape[:-2] == b.shape[:-2]:
    return a, b, None, None

  # If a has batch dimensions and b doesn't, we need to explicitly broadcast.
  if len(a.shape) > 2 and len(b.shape) == 2:
    b = jnp.broadcast_to(b, a.shape[:-2] + b.shape[-2:])
    return a, b, None, None

  # If b has batch dimensions and a doesn't
  if len(b.shape) > 2 and len(a.shape) == 2:
    # Move all batch dimensions to the right.
    transpose_dims = [len(b.shape) - 2, len(b.shape) - 1]
    transpose_dims.extend(range(0, len(b.shape) - 2))
    new_b = jnp.transpose(b, axes=transpose_dims)
    new_b = new_b.reshape(new_b.shape[0], -1)

    # Undo the transpose.
    result_permutation = list(range(2, len(b.shape))) + [0, 1]
    return a, new_b, b.shape[:-2], result_permutation

  # Both inputs have batch dimensions that are unequal.
  batch_a = a.shape[:-2]
  batch_b = b.shape[:-2]

  reversed_batch_a = list(reversed(batch_a))
  reversed_batch_b = list(reversed(batch_b))

  keep_a_dims = []
  keep_b_dims = []
  transpose_b_dims = []
  result_permutation = []

  b_batch_dims = []

  # Iterate backwards over dimensions since the batch shapes might be of
  # different sizes.
  for i in range(min(len(batch_a), len(batch_b))):
    if reversed_batch_a[i] == reversed_batch_b[i]:
      keep_a_dims.append(i)
      keep_b_dims.append(i)
    else:
      if reversed_batch_a[i] == 1 and reversed_batch_b[i] != 1:
        transpose_b_dims.append(i)
        b_batch_dims.append(reversed_batch_b[i])
      else:
        keep_a_dims.append(i)
        keep_b_dims.append(i)

  # Add extra dimensions of the larger list
  if len(batch_a) > len(batch_b):
    keep_a_dims.extend(range(len(batch_b), len(batch_a)))
  elif len(batch_b) > len(batch_a):
    transpose_b_dims.extend(range(len(batch_a), len(batch_b)))
    b_batch_dims.extend(reversed_batch_b[len(batch_a):])

  # We can compute the permutation of the broadcasted result that
  # produces the result with the dimensions permuted as we do. We can then
  # invert that permutation via an argsort.
  result_batch_size = max(len(batch_a), len(batch_b))
  result_permutation = [result_batch_size - 1 - x for x in reversed(
      keep_a_dims)]
  result_permutation.extend([result_batch_size + 1, result_batch_size + 2])
  result_permutation.extend(
      [result_batch_size - 1 - x for x in reversed(transpose_b_dims)])
  result_permutation = onp.argsort(onp.array(result_permutation), axis=-1)

  keep_a_dims = [len(batch_a) - 1 - x for x in reversed(keep_a_dims)]
  keep_b_dims = [len(batch_b) - 1 - x for x in reversed(keep_b_dims)]
  transpose_b_dims = [len(batch_b) - 1 - x for x in reversed(transpose_b_dims)]

  b_batch_dims = list(reversed(b_batch_dims))

  # Reshape away possible one dimensions.
  new_a_shape = onp.take(
      onp.array(a.shape[:-2]), onp.array(keep_a_dims, dtype=onp.int64))
  new_a_shape = tuple(new_a_shape) + a.shape[-2:]
  new_a = a.reshape(*new_a_shape)

  new_b = jnp.transpose(
      b, keep_b_dims + [len(b.shape) - 2, len(b.shape) - 1] + transpose_b_dims)
  new_b_shape = onp.take(
      onp.array(b.shape[:-2]), onp.array(keep_b_dims, dtype=onp.int64))
  new_b_shape = tuple(new_b_shape) + (b.shape[-2], -1)
  new_b = new_b.reshape(*new_b_shape)
  if new_a.shape[:-2] != new_b.shape[:-2]:
    new_b = jnp.broadcast_to(new_b, new_a.shape[:-2] + new_b.shape[-2:])

  return new_a, new_b, b_batch_dims, result_permutation


def _reshape_solve_result(result, b_batch_dims, result_permutation):
  """Reshapes result of a solve to the original shapes."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as onp  # pylint: disable=g-import-not-at-top,reimported

  if b_batch_dims is None:
    return result

  new_shape = list(result.shape[:-1])
  new_shape.append(result.shape[-1] // int(onp.prod(onp.array(b_batch_dims))))
  new_shape.extend(b_batch_dims)
  new_result = result.reshape(*new_shape)
  return jnp.transpose(new_result, result_permutation)


def _cholesky_solve(chol, rhs, name=None):  # pylint: disable=unused-argument
  """Scipy cho_solve does not broadcast, so we must do so explicitly."""
  chol = ops.convert_to_tensor(chol)
  rhs = ops.convert_to_tensor(rhs)
  if JAX_MODE:
    (chol_broadcast,
     rhs_broadcast,
     rhs_batch_dims,
     result_permutation) = _create_solve_broadcastable_inputs(chol, rhs)
    result = scipy_linalg.cho_solve((chol_broadcast, True), rhs_broadcast)
    return _reshape_solve_result(result, rhs_batch_dims, result_permutation)
  try:
    bcast_shp = np.broadcast_shapes(chol.shape[:-2], rhs.shape[:-2])
  except ValueError as e:
    raise ValueError(
        f'Error with inputs shaped `chol`={chol.shape}, rhs={rhs.shape}') from e
  dim = chol.shape[-1]
  chol = np.broadcast_to(chol, bcast_shp + (dim, dim))
  rhs = np.broadcast_to(rhs, bcast_shp + (dim, rhs.shape[-1],))
  nbatch = int(np.prod(bcast_shp))
  flat_chol = chol.reshape(nbatch, dim, dim)
  flat_rhs = rhs.reshape(nbatch, dim, rhs.shape[-1])
  result = np.empty_like(flat_rhs)
  if np.size(result):
    for i, (ch, rh) in enumerate(zip(flat_chol, flat_rhs)):
      result[i] = scipy_linalg.cho_solve((ch, True), rh)
  return result.reshape(*rhs.shape)


def _diag(diagonal, name=None, k=0, num_rows=-1, num_cols=-1, padding_value=0.,
          align='RIGHT_LEFT'):
  del name
  diagonal = ops.convert_to_tensor(diagonal)
  if (k != 0 or num_rows != -1 or num_cols != -1 or padding_value != 0. or
      align != 'RIGHT_LEFT'):
    raise NotImplementedError((k, num_rows, num_cols, padding_value, align))
  return _set_diag(
      np.zeros(diagonal.shape + (diagonal.shape[-1],), dtype=diagonal.dtype),
      diagonal)


def _einsum(equation, *inputs, **kwargs):
  del kwargs
  return np.einsum(equation, *inputs)


def _eye(num_rows, num_columns=None, batch_shape=None,
         dtype=np.float32, name=None):  # pylint: disable=unused-argument
  dt = utils.numpy_dtype(dtype)
  x = np.eye(num_rows, num_columns).astype(dt)
  if batch_shape is not None:
    x = x * np.ones(tuple(batch_shape) + (1, 1)).astype(dt)
  return x


def _eig(tensor, name=None):
  del name
  tensor = ops.convert_to_tensor(tensor)
  e, v = np.linalg.eig(tensor)
  dt = tensor.dtype
  if dt == np.float32:
    out_dtype = np.complex64
  elif dt == np.float64:
    out_dtype = np.complex128
  return e.astype(out_dtype), v.astype(out_dtype)


def _eigvals(tensor, name=None):
  del name
  tensor = ops.convert_to_tensor(tensor)
  e = np.linalg.eigvals(tensor)
  dt = tensor.dtype
  if dt == np.float32:
    out_dtype = np.complex64
  elif dt == np.float64:
    out_dtype = np.complex128
  return e.astype(out_dtype)


def _lu_pivot_to_permutation(swaps, m):
  """Converts the pivot (row swaps) returned by LU to a permutation.

  Args:
    swaps: an array of shape (k,) of row swaps to perform
    m: the size of the output permutation. m should be >= k.
  Returns:
    An int32 array of shape (m,).
  """
  assert len(swaps.shape) >= 1

  permutation = np.arange(m, dtype=np.int32)
  for i in range(swaps.shape[-1]):
    j = swaps[i]
    permutation[i], permutation[j] = permutation[j], permutation[i]
  return permutation


Lu = collections.namedtuple('Lu', 'lu,p')


def _lu(input, output_idx_type=np.int32, name=None):  # pylint: disable=redefined-builtin
  """Returns Lu(lu, p), as TF does."""
  del name
  input = ops.convert_to_tensor(input)
  if JAX_MODE:  # JAX uses XLA, which can do a batched factorization.
    lu_out, pivots = scipy_linalg.lu_factor(input)
    from jax import lax  # pylint: disable=g-import-not-at-top
    return Lu(lu_out,
              lax.linalg.lu_pivots_to_permutation(pivots, lu_out.shape[-1]))
  # Scipy can't batch, so we must do so manually.
  nbatch = int(np.prod(input.shape[:-2]))
  dim = input.shape[-1]
  flat_mat = input.reshape(nbatch, dim, dim)
  flat_lu = np.empty((nbatch, dim, dim), dtype=input.dtype)
  flat_piv = np.empty((nbatch, dim), dtype=utils.numpy_dtype(output_idx_type))
  if np.size(flat_lu):  # Avoid non-empty batches of empty matrices.
    for i, mat in enumerate(flat_mat):
      lu_out, pivots = scipy_linalg.lu_factor(mat)
      flat_lu[i] = lu_out
      flat_piv[i] = _lu_pivot_to_permutation(pivots, flat_lu.shape[-1])
  return Lu(flat_lu.reshape(*input.shape), flat_piv.reshape(*input.shape[:-1]))


def _lstsq(matrix, rhs, l2_regularizer=0.0, fast=True, name=None):
  """JAX/NumPy variant of tf.linalg.lstsq."""
  del l2_regularizer, name
  if fast:
    raise ValueError('`fast=True` is not supported')
  if matrix.ndim > 2:
    if JAX_MODE:
      import jax  # pylint: disable=g-import-not-at-top
      return jax.vmap(functools.partial(_lstsq, fast=False))(matrix, rhs)
    res = np.array([_lstsq(mat, r, fast=False) for mat, r in zip(matrix, rhs)])
    if matrix.shape[0] == 0:
      res = res.reshape(matrix.shape[:-2] + (matrix.shape[-1], rhs.shape[-1]))
    return res
  rcond = None
  if JAX_MODE and matrix.dtype == np.float32:
    rcond = 0.  # https://github.com/google/jax/issues/15591
  return np.linalg.lstsq(matrix, rhs, rcond=rcond)[0]


def _matrix_transpose(a, name='matrix_transpose', conjugate=False):  # pylint: disable=unused-argument
  a = np.array(a)
  if a.ndim < 2:
    raise ValueError(
        'Input must have rank at least `2`; found {}.'.format(a.ndim))
  x = np.swapaxes(a, -2, -1)
  return np.conj(x) if conjugate else x


def _matmul(a, b,
            transpose_a=False, transpose_b=False,
            adjoint_a=False, adjoint_b=False,
            a_is_sparse=False, b_is_sparse=False,
            name=None):  # pylint: disable=unused-argument
  """Numpy matmul wrapper."""
  a = ops.convert_to_tensor(a)
  b = ops.convert_to_tensor(b)
  if a_is_sparse or b_is_sparse:
    raise NotImplementedError('Numpy backend does not support sparse matmul.')
  if transpose_a or adjoint_a:
    a = _matrix_transpose(a, conjugate=adjoint_a)
  if transpose_b or adjoint_b:
    b = _matrix_transpose(b, conjugate=adjoint_b)
  return np.matmul(a, b)


def _matvec(a, b,
            transpose_a=False, transpose_b=False,
            adjoint_a=False, adjoint_b=False,
            a_is_sparse=False, b_is_sparse=False,
            name=None):  # pylint: disable=unused-argument
  """Numpy matvec wrapper."""
  return np.squeeze(_matmul(
      a,
      b[..., np.newaxis],
      transpose_a=transpose_a,
      transpose_b=transpose_b,
      adjoint_a=adjoint_a,
      adjoint_b=adjoint_b,
      a_is_sparse=a_is_sparse,
      b_is_sparse=b_is_sparse), axis=-1)


def _norm(tensor, ord='euclidean', axis=None, keepdims=None,  # pylint: disable=redefined-builtin
          name=None, keep_dims=None):
  """Numpy norm wrapper."""
  del name
  del keep_dims
  if axis is None:
    tensor = np.reshape(tensor, [-1])
    axis = -1
  if isinstance(axis, (tuple, list)):
    # JAX expects tuples or ints.
    axis = tuple(axis)
  return np.linalg.norm(
      tensor,
      ord=2 if ord == 'euclidean' else ord,
      axis=axis, keepdims=bool(keepdims))


Qr = collections.namedtuple('Qr', 'q,r')


def _qr(input, full_matrices=False, name=None):  # pylint: disable=redefined-builtin,unused-argument
  mode = 'complete' if full_matrices else 'reduced'
  op = functools.partial(np.linalg.qr, mode=mode)
  if not JAX_MODE:
    op = np.vectorize(op, signature='(m,n)->(m,r),(r,n)')
  return Qr(*op(input))


def _set_diag(input, diagonal, name=None, k=0, align='RIGHT_LEFT'):  # pylint: disable=unused-argument,redefined-builtin
  if k != 0 or align != 'RIGHT_LEFT':
    raise NotImplementedError('set_diag not implemented for k != 0')
  return np.where(np.eye(diagonal.shape[-1], dtype=np.bool_),
                  diagonal[..., np.newaxis, :],
                  input)


def _solve(matrix, rhs, adjoint=False, name=None):  # pylint: disable=redefined-outer-name
  """Numpy solve does not broadcast, so we must do so explicitly."""
  del name
  if adjoint:
    matrix = _matrix_transpose(matrix, conjugate=True)
  if JAX_MODE:
    (matrix_broadcast,
     rhs_broadcast,
     rhs_batch_dims,
     result_permutation) = _create_solve_broadcastable_inputs(matrix, rhs)
    result = np.linalg.solve(matrix_broadcast, rhs_broadcast)
    return _reshape_solve_result(
        result, rhs_batch_dims, result_permutation)
  try:
    bcast_shp = np.broadcast_shapes(matrix.shape[:-2], rhs.shape[:-2])
  except ValueError as e:
    raise ValueError(
        f'Error with inputs shaped `matrix`={matrix.shape}, rhs={rhs.shape}'
        ) from e
  dim = matrix.shape[-1]
  matrix = np.broadcast_to(matrix, bcast_shp + (dim, dim))
  rhs = np.broadcast_to(rhs, bcast_shp + (dim, rhs.shape[-1]))
  nbatch = int(np.prod(bcast_shp))
  flat_mat = matrix.reshape(nbatch, dim, dim)
  flat_rhs = rhs.reshape(nbatch, dim, rhs.shape[-1])
  result = np.empty(flat_rhs.shape)
  if np.size(result):
    # ValueError: On entry to STRTRS parameter number 7 had an illegal value.
    for i, (mat, rh) in enumerate(zip(flat_mat, flat_rhs)):
      result[i] = np.linalg.solve(mat, rh)
  return result.reshape(*rhs.shape)


def _svd(tensor, full_matrices=False, compute_uv=True, name=None):
  del name
  ret = np.linalg.svd(
      tensor, full_matrices=full_matrices, compute_uv=compute_uv)
  if compute_uv:
    u, s, vh = ret
    # Batched matrix transpose.
    return s, u, np.conj(np.swapaxes(vh, -2, -1))
  return ret


def _tensordot(a, b, axes, name=None):  # pylint: disable=redefined-outer-name
  del name
  return np.tensordot(a, b, axes=axes)


def _tensor_diag(diagonal, name=None):  # pylint: disable=redefined-outer-name
  """tf.linalg.tensor_diag reimpl."""
  del name
  if diagonal.ndim == 0:
    return diagonal
  out = np.zeros((diagonal.shape[0], diagonal.shape[0]) +
                 diagonal.shape[1:] + diagonal.shape[1:],
                 dtype=diagonal.dtype)
  for i in range(diagonal.shape[0]):
    if JAX_MODE:
      out = out.at[i, i].set(_tensor_diag(diagonal[i]))
    else:
      out[i, i] = _tensor_diag(diagonal[i])
  axes = ([0] + list(range(2, diagonal.ndim + 1)) +
          [1] + list(range(diagonal.ndim + 1, diagonal.ndim * 2)))
  return np.transpose(out, axes)


def _trace(x, name=None):
  del name
  return np.trace(x, axis1=-1, axis2=-2)


def _triangular_solve(matrix, rhs, lower=True, adjoint=False, name=None):  # pylint: disable=redefined-outer-name
  """Scipy solve does not broadcast, so we must do so explicitly."""
  del name
  if JAX_MODE:
    (matrix_broadcast,
     rhs_broadcast,
     rhs_batch_dims,
     result_permutation) = _create_solve_broadcastable_inputs(matrix, rhs)
    result = scipy_linalg.solve_triangular(
        matrix_broadcast, rhs_broadcast, lower=lower,
        trans='C' if adjoint else 'N')
    return _reshape_solve_result(result, rhs_batch_dims, result_permutation)
  try:
    bcast_shp = np.broadcast_shapes(matrix.shape[:-2], rhs.shape[:-2])
  except ValueError as e:
    raise ValueError(
        f'Error with inputs shaped `matrix`={matrix.shape}, rhs={rhs.shape}'
        ) from e
  dim = matrix.shape[-1]
  matrix = np.broadcast_to(matrix, bcast_shp + (dim, dim))
  rhs = np.broadcast_to(rhs, bcast_shp + (dim, rhs.shape[-1]))
  nbatch = int(np.prod(bcast_shp))
  flat_mat = matrix.reshape(nbatch, dim, dim)
  flat_rhs = rhs.reshape(nbatch, dim, rhs.shape[-1])
  result = np.empty(flat_rhs.shape, dtype=flat_rhs.dtype)
  if np.size(result):
    # ValueError: On entry to STRTRS parameter number 7 had an illegal value.
    for i, (mat, rh) in enumerate(zip(flat_mat, flat_rhs)):
      result[i] = scipy_linalg.solve_triangular(mat, rh, lower=lower,
                                                trans='C' if adjoint else 'N')
  return result.reshape(*rhs.shape)


# --- Begin Public Functions --------------------------------------------------

adjoint = utils.copy_docstring(
    'tf.linalg.adjoint',
    lambda matrix, name=None: _matrix_transpose(matrix, conjugate=True))

band_part = utils.copy_docstring(
    'tf.linalg.band_part',
    _band_part)

cholesky = utils.copy_docstring(
    'tf.linalg.cholesky',
    lambda input, name=None: np.linalg.cholesky(ops.convert_to_tensor(input)))

cholesky_solve = utils.copy_docstring(
    'tf.linalg.cholesky_solve',
    _cholesky_solve)

cross = utils.copy_docstring(
    'tf.linalg.cross',
    lambda a, b, name=None: np.cross(a, b))

det = utils.copy_docstring(
    'tf.linalg.det',
    lambda input, name=None: np.linalg.det(input))

diag = utils.copy_docstring(
    'tf.linalg.diag',
    _diag)

diag_part = utils.copy_docstring(
    'tf.linalg.diag_part',
    lambda input, name=None: np.diagonal(  # pylint: disable=g-long-lambda
        ops.convert_to_tensor(input), axis1=-2, axis2=-1))

eig = utils.copy_docstring('tf.linalg.eig', _eig)

eigh = utils.copy_docstring(
    'tf.linalg.eigh',
    lambda tensor, name=None: tuple(np.linalg.eigh(tensor)))

eigvals = utils.copy_docstring('tf.linalg.eigvals', _eigvals)

eigvalsh = utils.copy_docstring(
    'tf.linalg.eigvalsh',
    lambda tensor, name=None: np.linalg.eigvalsh(tensor))

einsum = utils.copy_docstring(
    'tf.linalg.einsum',
    _einsum)

eye = utils.copy_docstring(
    'tf.eye',
    _eye)

inv = utils.copy_docstring(
    'tf.linalg.inv',
    lambda input, name=None: np.linalg.inv(input))

logdet = utils.copy_docstring(
    'tf.linalg.logdet',
    lambda matrix, name=None: np.linalg.slogdet(matrix)[1])

lu = utils.copy_docstring(
    'tf.linalg.lu',
    _lu)

lstsq = utils.copy_docstring(
    'tf.linalg.lstsq',
    _lstsq)

matmul = utils.copy_docstring(
    'tf.linalg.matmul',
    _matmul)

matvec = utils.copy_docstring(
    'tf.linalg.matvec',
    _matvec)

matrix_rank = utils.copy_docstring(
    'tf.linalg.matrix_rank',
    lambda input, name=None: np.linalg.matrix_rank(input))

norm = utils.copy_docstring('tf.norm', _norm)

pinv = utils.copy_docstring(
    'tf.linalg.pinv', lambda input, name=None: np.linalg.pinv(input))

qr = utils.copy_docstring('tf.linalg.qr', _qr)

set_diag = utils.copy_docstring(
    'tf.linalg.set_diag',
    _set_diag)

SLogDet = collections.namedtuple('LogMatrixDeterminant',
                                 'sign,log_abs_determinant')

slogdet = utils.copy_docstring(
    'tf.linalg.slogdet',
    lambda input, name=None: SLogDet(*np.linalg.slogdet(input)))

matrix_transpose = utils.copy_docstring(
    'tf.linalg.matrix_transpose',
    _matrix_transpose)

transpose = matrix_transpose

solve = utils.copy_docstring('tf.linalg.solve', _solve)

svd = utils.copy_docstring(
    'tf.linalg.svd',
    _svd)

tensordot = utils.copy_docstring(
    'tf.linalg.tensordot',
    _tensordot)

tensor_diag = utils.copy_docstring(
    'tf.linalg.tensor_diag',
    _tensor_diag)

trace = utils.copy_docstring(
    'tf.linalg.trace',
    _trace)

triangular_solve = utils.copy_docstring(
    'tf.linalg.triangular_solve',
    _triangular_solve)
