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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    'einsum',
    'eye',
    'inv',
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
    'tensordot',
    'trace',
    'triangular_solve',
    # 'cross',
    # 'eigh',
    # 'eigvalsh',
    # 'expm',
    # 'global_norm',
    # 'logdet',
    # 'logm',
    # 'lstsq',
    # 'l2_normalize'
    # 'sqrtm',
    # 'tensor_diag',
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


def _cholesky_solve(chol, rhs, name=None):  # pylint: disable=unused-argument
  """Scipy cho_solve does not broadcast, so we must do so explicitly."""
  if JAX_MODE:  # But JAX uses XLA, which can do a batched solve.
    chol = chol + np.zeros(rhs.shape[:-2] + (1, 1), dtype=chol.dtype)
    rhs = rhs + np.zeros(chol.shape[:-2] + (1, 1), dtype=rhs.dtype)
    return scipy_linalg.cho_solve((chol, True), rhs)
  try:
    bcast = np.broadcast(chol[..., :1], rhs)
  except ValueError as e:
    raise ValueError('Error with inputs shaped `chol`={}, rhs={}:\n{}'.format(
        chol.shape, rhs.shape, str(e)))
  dim = chol.shape[-1]
  chol = np.broadcast_to(chol, bcast.shape[:-1] + (dim,))
  rhs = np.broadcast_to(rhs, bcast.shape)
  nbatch = int(np.prod(chol.shape[:-2]))
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
  if JAX_MODE:  # JAX uses XLA, which can do a batched factorization.
    lu_out, pivots = scipy_linalg.lu_factor(input)
    from jax import lax_linalg  # pylint: disable=g-import-not-at-top
    return Lu(lu_out,
              lax_linalg.lu_pivots_to_permutation(pivots, lu_out.shape[-1]))
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
  return np.where(np.eye(diagonal.shape[-1], dtype=np.bool),
                  diagonal[..., np.newaxis, :],
                  input)


def _solve(matrix, rhs, adjoint=False, name=None):  # pylint: disable=redefined-outer-name
  """Numpy solve does not broadcast, so we must do so explicitly."""
  del name
  if adjoint:
    matrix = _matrix_transpose(matrix, conjugate=True)
  if JAX_MODE:  # But JAX uses XLA, which can do a batched solve.
    matrix = matrix + np.zeros(rhs.shape[:-2] + (1, 1), dtype=matrix.dtype)
    rhs = rhs + np.zeros(matrix.shape[:-2] + (1, 1), dtype=rhs.dtype)
    return np.linalg.solve(matrix, rhs)
  try:
    bcast = np.broadcast(matrix[..., :1], rhs)
  except ValueError as e:
    raise ValueError('Error with inputs shaped `matrix`={}, rhs={}:\n{}'.format(
        matrix.shape, rhs.shape, str(e)))
  dim = matrix.shape[-1]
  matrix = np.broadcast_to(matrix, bcast.shape[:-1] + (dim,))
  rhs = np.broadcast_to(rhs, bcast.shape)
  nbatch = int(np.prod(matrix.shape[:-2]))
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


def _trace(x, name=None):
  del name
  return np.trace(x, axis1=-1, axis2=-2)


def _triangular_solve(matrix, rhs, lower=True, adjoint=False, name=None):  # pylint: disable=redefined-outer-name
  """Scipy solve does not broadcast, so we must do so explicitly."""
  del name
  if JAX_MODE:  # But JAX uses XLA, which can do a batched solve.
    matrix = matrix + np.zeros(rhs.shape[:-2] + (1, 1), dtype=matrix.dtype)
    rhs = rhs + np.zeros(matrix.shape[:-2] + (1, 1), dtype=rhs.dtype)
    return scipy_linalg.solve_triangular(matrix, rhs, lower=lower,
                                         trans='C' if adjoint else 'N')
  try:
    bcast = np.broadcast(matrix[..., :1], rhs)
  except ValueError as e:
    raise ValueError('Error with inputs shaped `matrix`={}, rhs={}:\n{}'.format(
        matrix.shape, rhs.shape, str(e)))
  dim = matrix.shape[-1]
  matrix = np.broadcast_to(matrix, bcast.shape[:-1] + (dim,))
  rhs = np.broadcast_to(rhs, bcast.shape)
  nbatch = int(np.prod(matrix.shape[:-2]))
  flat_mat = matrix.reshape(nbatch, dim, dim)
  flat_rhs = rhs.reshape(nbatch, dim, rhs.shape[-1])
  result = np.empty(flat_rhs.shape)
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
    lambda input, name=None: np.linalg.cholesky(input))

cholesky_solve = utils.copy_docstring(
    'tf.linalg.cholesky_solve',
    _cholesky_solve)

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

einsum = utils.copy_docstring(
    'tf.linalg.einsum',
    _einsum)

eye = utils.copy_docstring(
    'tf.eye',
    _eye)

inv = utils.copy_docstring(
    'tf.linalg.inv',
    lambda input, name=None: np.linalg.inv(input))

lu = utils.copy_docstring(
    'tf.linalg.lu',
    _lu)

matmul = utils.copy_docstring(
    'tf.linalg.matmul',
    _matmul)

matvec = utils.copy_docstring(
    'tf.linalg.matvec',
    _matvec)

# TODO(b/140157055): Remove the try/except.
matrix_rank = lambda input, name=None: np.linalg.matrix_rank(input)

try:
  matrix_rank = utils.copy_docstring(
      'tf.linalg.matrix_rank',
      lambda input, name=None: np.linalg.matrix_rank(input))
except AttributeError:
  pass

norm = utils.copy_docstring('tf.norm', _norm)

# TODO(b/140157055): Remove the try/except.
pinv = lambda input, name=None: np.linalg.pinv(input)

try:
  pinv = utils.copy_docstring(
      'tf.linalg.pinv', lambda input, name=None: np.linalg.pinv(input))
except AttributeError:
  pass

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

solve = utils.copy_docstring('tf.linalg.solve', _solve)

svd = utils.copy_docstring(
    'tf.linalg.svd',
    _svd)

tensordot = utils.copy_docstring(
    'tf.linalg.tensordot',
    _tensordot)

trace = utils.copy_docstring(
    'tf.linalg.trace',
    _trace)

triangular_solve = utils.copy_docstring(
    'tf.linalg.triangular_solve',
    _triangular_solve)
