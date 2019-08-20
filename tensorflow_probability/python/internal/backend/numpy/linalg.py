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
"""Numpy implementations of `tf.linalg` functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils

from tensorflow_probability.python.internal.backend.numpy.linear_operator import *  # pylint: disable=wildcard-import

scipy_linalg = utils.try_import('scipy.linalg')


__all__ = [
    'band_part',
    'cholesky',
    'cholesky_solve',
    'det',
    'diag',
    'diag_part',
    'eye',
    'matmul',
    'matrix_transpose',
    'norm',
    'set_diag',
    'slogdet',
    'triangular_solve',
    # 'adjoint',
    # 'cross',
    # 'eigh',
    # 'eigvalsh',
    # 'einsum',
    # 'expm',
    # 'global_norm',
    # 'inv',
    # 'l2_normalize',
    # 'logdet',
    # 'logm',
    # 'lstsq',
    # 'lu',
    # 'matvec',
    # 'norm',
    # 'qr',
    # 'solve',
    # 'sqrtm',
    # 'svd',
    # 'tensor_diag',
    # 'tensor_diag_part',
    # 'tensordot',
    # 'trace',
    # 'tridiagonal_solve',
]


JAX_MODE = False


def _band_part(input, num_lower, num_upper, name=None):  # pylint: disable=redefined-builtin
  del name
  result = input
  if num_lower > -1:
    result = np.triu(result, -num_lower)
  if num_upper > -1:
    result = np.tril(result, num_upper)
  return result


def _cholesky_solve(chol, rhs, name=None):
  """Scipy cho_solve does not broadcast, so we must do so explicitly."""
  del name
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


def _diag(diagonal, name=None):
  del name
  return _set_diag(
      np.zeros(diagonal.shape + (diagonal.shape[-1],), dtype=diagonal.dtype),
      diagonal)


def _eye(num_rows, num_columns=None, batch_shape=None,
         dtype=tf.float32, name=None):  # pylint: disable=unused-argument
  dt = utils.numpy_dtype(dtype)
  x = np.eye(num_rows, num_columns).astype(dt)
  if batch_shape is not None:
    x = x * np.ones(tuple(batch_shape) + (1, 1)).astype(dt)
  return x


def _set_diag(input, diagonal, name=None):  # pylint: disable=unused-argument,redefined-builtin
  return np.where(np.eye(diagonal.shape[-1], dtype=np.bool),
                  diagonal[..., np.newaxis, :],
                  input)


def _matrix_transpose(a, name='matrix_transpose', conjugate=False):  # pylint: disable=unused-argument
  a = np.array(a)
  if a.ndim < 2:
    raise ValueError(
        'Input must have rank at least `2`; found {}.'.format(a.ndim))
  x = np.swapaxes(a, -2, -1)
  return np.conjugate(x) if conjugate else x


def _matmul(a, b,
            transpose_a=False, transpose_b=False,
            adjoint_a=False, adjoint_b=False,
            a_is_sparse=False, b_is_sparse=False,
            name=None):  # pylint: disable=unused-argument
  """Numpy matmul wrapper."""
  if a_is_sparse or b_is_sparse:
    raise NotImplementedError('Numpy backend does not support sparse matmul.')
  if transpose_a or adjoint_a:
    a = _matrix_transpose(a, conjugate=adjoint_a)
  if transpose_b or adjoint_b:
    b = _matrix_transpose(b, conjugate=adjoint_b)
  return np.matmul(a, b)


def _triangular_solve(matrix, rhs, lower=True, adjoint=False, name=None):
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

band_part = utils.copy_docstring(
    tf.linalg.band_part,
    _band_part)

cholesky = utils.copy_docstring(
    tf.linalg.cholesky,
    lambda input, name=None: np.linalg.cholesky(input))

cholesky_solve = utils.copy_docstring(
    tf.linalg.cholesky_solve,
    _cholesky_solve)

det = utils.copy_docstring(
    tf.linalg.det,
    lambda input, name=None: np.linalg.det(input))

diag = utils.copy_docstring(
    tf.linalg.diag,
    _diag)

diag_part = utils.copy_docstring(
    tf.linalg.diag_part,
    lambda input, name=None: np.diagonal(input, axis1=-2, axis2=-1))

eye = utils.copy_docstring(
    tf.eye,
    _eye)

matmul = utils.copy_docstring(
    tf.linalg.matmul,
    _matmul)

norm = utils.copy_docstring(
    tf.norm,
    lambda tensor, ord='euclidean', axis=None, keepdims=None, name=None,  # pylint: disable=g-long-lambda
           keep_dims=None: np.linalg.norm(
               np.reshape(tensor, [-1]) if axis is None else tensor,
               ord=2 if ord == 'euclidean' else ord,
               axis=-1 if axis is None else axis, keepdims=bool(keepdims))
)

set_diag = utils.copy_docstring(
    tf.linalg.set_diag,
    _set_diag)


SLogDet = collections.namedtuple('slogdet', 'sign,log_abs_determinant')


slogdet = utils.copy_docstring(
    tf.linalg.slogdet,
    lambda input, name=None: SLogDet(*np.linalg.slogdet(input)))

matrix_transpose = utils.copy_docstring(
    tf.linalg.matrix_transpose,
    _matrix_transpose)


triangular_solve = utils.copy_docstring(
    tf.linalg.triangular_solve,
    _triangular_solve)
