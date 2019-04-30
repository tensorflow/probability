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

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy.internal import utils

scipy_linalg = utils.try_import('scipy.linalg')


__all__ = [
    'band_part',
    'cholesky',
    'cholesky_solve',
    'diag',
    'diag_part',
    'eye',
    'matmul',
    'matrix_transpose',
    'set_diag',
    'triangular_solve',
    # 'adjoint',
    # 'cross',
    # 'det',
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
    # 'slogdet',
    # 'solve',
    # 'sqrtm',
    # 'svd',
    # 'tensor_diag',
    # 'tensor_diag_part',
    # 'tensordot',
    # 'trace',
    # 'tridiagonal_solve',
]


def _eye(num_rows, num_columns=None, batch_shape=None,
         dtype=tf.float32, name=None):  # pylint: disable=unused-argument
  dt = utils.numpy_dtype(dtype)
  x = np.eye(num_rows, num_columns).astype(dt)
  if batch_shape is not None:
    x *= np.ones(np.concatenate([batch_shape, [1, 1]], axis=0)).astype(dt)
  return x


def _fill_diagonal(input, diagonal, name=None):  # pylint: disable=unused-argument,redefined-builtin
  x = np.array(input).copy()
  np.fill_diagonal(x, diagonal)
  return x


def _matrix_transpose(a, name='matrix_transpose', conjugate=False):  # pylint: disable=unused-argument
  a = np.array(a)
  if a.ndim < 2:
    raise ValueError(
        'Input must have rank at least `2`; found {}.'.format(a.ndim))
  perm = np.concatenate([np.arange(a.ndim - 2), [a.ndim - 1, a.ndim - 2]],
                        axis=0)
  x = np.transpose(a, perm)
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


# --- Begin Public Functions --------------------------------------------------

band_part = utils.copy_docstring(
    tf.linalg.band_part,
    lambda input, num_lower, num_upper, name=None: (  # pylint: disable=g-long-lambda
        np.tril(np.triu(input, -num_lower), num_upper)))

cholesky = utils.copy_docstring(
    tf.linalg.cholesky,
    lambda input, name=None: np.linalg.cholesky(input))

cholesky_solve = utils.copy_docstring(
    tf.linalg.cholesky_solve,
    lambda chol, rhs, name=None: scipy_linalg.cho_solve((chol, True), rhs))

diag = utils.copy_docstring(
    tf.linalg.diag,
    lambda diagonal, name=None: np.diag(diagonal))

diag_part = utils.copy_docstring(
    tf.linalg.diag_part,
    lambda input, name=None: np.diagonal(input))

eye = utils.copy_docstring(
    tf.eye,
    _eye)

matmul = utils.copy_docstring(
    tf.linalg.matmul,
    _matmul)

set_diag = utils.copy_docstring(
    tf.linalg.set_diag,
    _fill_diagonal)

matrix_transpose = utils.copy_docstring(
    tf.linalg.matrix_transpose,
    _matrix_transpose)

triangular_solve = utils.copy_docstring(
    tf.linalg.triangular_solve,
    lambda matrix, rhs, lower=True, adjoint=False, name=None: (  # pylint: disable=g-long-lambda
        scipy_linalg.solve_triangular(matrix, rhs)))
