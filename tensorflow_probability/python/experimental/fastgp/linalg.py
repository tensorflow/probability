# Copyright 2024 The TensorFlow Probability Authors.
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
"""Linear algebra routines to use for preconditioners."""

import functools
import jax
import jax.experimental.sparse
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.fastgp import partial_lanczos
from tensorflow_probability.python.internal.backend import jax as tf2jax

Array = jnp.ndarray

# pylint: disable=invalid-name


def _matvec(M, x) -> jax.Array:
  if isinstance(M, tf2jax.linalg.LinearOperator):
    return M.matvec(x)
  return M @ x


def largest_eigenvector(
    M: tf2jax.linalg.LinearOperator, key: jax.Array, num_iters: int = 10
):
  """Returns the largest (eigenvalue, eigenvector) of M."""
  n = M.shape[-1]
  v = jax.random.uniform(key, shape=(n,), dtype=M.dtype)
  for _ in range(num_iters):
    v = _matvec(M, v)
    v = v / jnp.linalg.norm(v)

  nv = _matvec(M, v)
  eigenvalue = jnp.linalg.norm(nv)
  return eigenvalue, v


def make_randomized_truncated_svd(
    key: jax.Array,
    M: tf2jax.linalg.LinearOperator,
    rank: int = 20,
    oversampling: int = 10,
    num_iters: int = 4,
):
  """Returns approximate SVD for symmetric `M`."""
  # This is based on:
  # N. Halko, P.G. Martinsson, J. A. Tropp
  # Finding Structure with Randomness: Probabilistic Algorithms for Constucting
  # Approximate Matrix Decompositions
  # We use the recommended oversampling parameter of 10 by default.
  # https://arxiv.org/pdf/0909.4061.pdf
  # It's also recommended to run for 2-4 iterations.
  max_rank = min(M.shape[-2:])
  if rank > max_rank + 1:
    print(
        'Warning, make_randomized_truncated_svd called '
        f'with {rank=} and {max_rank=}')
    rank = max_rank + 1
  p = jax.random.uniform(
      key,
      shape=M.shape[:-1] + (rank + oversampling,),
      dtype=M.dtype,
      minval=-1.,
      maxval=1.)
  for _ in range(num_iters):
    # We will assume that M is symmetric to avoid a transpose.
    q = M @ p
    q, _ = jnp.linalg.qr(q)
    p = M @ q
    p, _ = jnp.linalg.qr(p)

  # SVD of Q*MQ
  u, s, _ = jnp.linalg.svd(
      jnp.swapaxes(p, -1, -2) @ (M @ p), hermitian=True)

  return (p @ u * jnp.sqrt(s))[..., :rank]


def make_partial_lanczos(
    key: jax.Array, M: tf2jax.linalg.LinearOperator, rank: int
) -> Array:
  """Return low rank approximation to M based on the partial Lancozs alg."""
  n = M.shape[-1]
  key1, key2 = jax.random.split(key)
  v = jax.random.uniform(
      key1, shape=(n, 1), minval=-1.0, maxval=1.0, dtype=M.dtype)
  Q, T = partial_lanczos.partial_lanczos(lambda x: M @ x, v, key2, rank)

  # Now diagonalize T as R^t D R.
  full_T = (
      jnp.diag(T.diag[0, :])
      + jnp.diag(T.off_diag[0, :], 1)
      + jnp.diag(T.off_diag[0, :], -1)
  )
  # TODO(thomaswc): When jnp.linalg includes eigh_tridiagonal, replace this
  # with that.
  evalues, evectors = jnp.linalg.eigh(full_T)
  sqrt_evalues = jnp.sqrt(evalues)

  # M ~ F^t F, where F = sqrt(D) R Q.
  F = jnp.einsum('i,ij->ij', sqrt_evalues, evectors @ Q[0])
  low_rank = jnp.transpose(F)

  return low_rank


def make_truncated_svd(
    key, M: tf2jax.linalg.LinearOperator, rank: int, num_iters: int
) -> Array:
  """Return low rank approximation to M based on the partial SVD alg."""
  n = M.shape[-1]
  if 5 * rank >= n:
    print(
        f'Warning, make_truncated_svd called with {rank=} and {n=}')
    rank = int((n-1) / 5)
  if rank > 0:
    X = jax.random.uniform(
        key, shape=(n, rank), minval=-1.0, maxval=1.0, dtype=M.dtype)
    evalues, evectors, _ = jax.experimental.sparse.linalg.lobpcg_standard(
        M, X, num_iters, None)
    low_rank = evectors * jnp.sqrt(evalues)
  else:
    low_rank = jnp.zeros((n, 0), dtype=M.dtype)
  return low_rank


@functools.partial(jax.jit, static_argnums=1)
def make_partial_pivoted_cholesky(
    M: tf2jax.linalg.LinearOperator, rank: int
) -> Array:
  """Return low rank approximation to M based on partial pivoted Cholesky."""
  n = M.shape[-1]

  def swap_row(a, index1, index2):
    temp = a[index1]
    a = jax.lax.dynamic_update_index_in_dim(a, a[index2], index1, axis=0)
    a = jax.lax.dynamic_update_index_in_dim(a, temp, index2, axis=0)
    return a

  def body_fn(i, val):
    diag, transpositions, permutation, low_rank = val
    largest_index = jnp.argmax(diag)
    transpositions = jax.lax.dynamic_update_index_in_dim(
        transpositions, jnp.array([i, largest_index]), i, axis=0)
    diag = swap_row(diag, largest_index, i)
    low_rank = swap_row(low_rank.T, largest_index, i).T
    permutation = swap_row(permutation, largest_index, i)

    pivot = jnp.sqrt(diag[i])
    row = M[permutation[i], :]

    def reswap_row(index, row):
      index, transposition_i = jax.lax.dynamic_index_in_dim(
          transpositions, index, 0, keepdims=False)
      return swap_row(row.T, index, transposition_i).T
    row = jax.lax.fori_loop(0, i + 1, reswap_row, row)

    low_rank_i = jax.lax.dynamic_index_in_dim(low_rank, i, 1, keepdims=False)
    low_rank_i = jnp.where(
        jnp.arange(n) > i,
        (row - jnp.dot(low_rank_i, low_rank)) / pivot, 0.)
    low_rank_i = jax.lax.dynamic_update_index_in_dim(
        low_rank_i, pivot, i, axis=0)
    low_rank = jax.lax.dynamic_update_index_in_dim(
        low_rank, low_rank_i, i, axis=0)
    diag -= jnp.where(jnp.arange(n) >= i, low_rank_i**2, 0.)
    return diag, transpositions, permutation, low_rank

  diag = jnp.diag(M)
  _, _, permutation, low_rank = jax.lax.fori_loop(
      0, rank, body_fn, (
          diag,
          -jnp.ones([rank, 2], dtype=np.int64),
          jnp.arange(n, dtype=np.int64),
          jnp.zeros([rank, n], dtype=M.dtype)))
  # Invert the permutation
  permutation = jnp.argsort(permutation, axis=-1)
  return low_rank.T[..., permutation, :]
