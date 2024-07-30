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
"""Run Lanczos for m iterations to get a good preconditioner."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import scipy
from tensorflow_probability.python.experimental.fastgp import mbcg
from tensorflow_probability.python.internal.backend import jax as tf2jax


Array = jnp.ndarray

# pylint: disable=invalid-name


def gram_schmidt(Q, v):
  """Return a vector after applying the Gram-Schmidt process to it.

  Args:
    Q: A tensor of shape (batch_dims, m, n) representing a batch of
      orthonormal m-by-n matrices. The `m` vectors to perform Gram-Schmidt
      with respect to.
    v: Initial starting vector of shape (n, batch_dims).

  Returns:
    A tensor t of shape (n, batch_dims) such that for every i,
    t[:, i] is orthogonal to Q[i, j, :] for every j.
  """
  # This will be used several times to do twice is enough reorthogonalization,
  # after initial orthogonalization.
  # See:
  # [1] L. Giruad, J. Langou, M. Rozloznik, On the round-offf error analysis of
  # the Gram-Schmidt Algorithm with reorthogonalization.
  correction = Q @ v.T[..., jnp.newaxis]
  correction = jnp.squeeze(jnp.swapaxes(Q, -1, -2) @ correction, axis=-1)
  return v - correction.T


def reorthogonalize(Q, v):
  for _ in range(3):
    v = gram_schmidt(Q, v)
  return v


@jax.named_call
def partial_lanczos(
    multiplier: Callable[[Array], Array],
    v: Array,
    key: jax.Array,
    num_iters: int = 20) -> Tuple[Array, mbcg.SymmetricTridiagonalMatrix]:
  """Returns orthonormal Q and tridiagonal T such that A ~ Q^t T Q.

  Similar to modified_batched_conjugate_gradients, but this returns the
  Q matrix that performs the tridiagonalization.  (But unlike mbcg, this
  doesn't support preconditioning.)

  Args:
    multiplier: A function for left matrix multiplying by an n by n
      symmetric positive definite matrix A.
    v: A tensor of shape (n, k) representing a batch of k n-dimensional
      vectors.  Each vector v[:, i] will be used to form a Krylov subspace
      spanned by A^j v[:, i] for j in [0, num_iters).
    key: RNG generator used to initialize the model's parameters.
    num_iters: The number of iterations to run the partial Lanczos algorithm
      for.

  Returns:
    A pair (Q, T) where Q is a shape (k, num_iters, n) batch of k
    num_iters-by-n orthonormal matrices and T is a size k batch of
    num_iters-by-num_iters symmetric tridiagonal matrices.
  """
  n = v.shape[0]
  k = v.shape[1]
  num_iters = int(min(n, num_iters))

  def scan_func(loop_info, unused_x):
    i, old_Q, old_diags, old_off_diags, old_v, key = loop_info
    key1, key2 = jax.random.split(key)

    beta = jnp.linalg.norm(old_v, axis=0, keepdims=True)
    norm = beta

    # If beta is too small, that means that we have / are close to finding a
    # full basis for the Krylov subspace. We need to rejuvenate the subspace
    # with a new random vector that is orthogonal to what has come so far.
    old_v_too_small = jax.random.uniform(
        key1, shape=old_v.shape, minval=-1.0, maxval=1.0, dtype=old_v.dtype)
    old_v_too_small = reorthogonalize(old_Q, old_v_too_small)

    norm_too_small = jnp.linalg.norm(old_v_too_small, axis=0, keepdims=True)

    eps = 10 * jnp.finfo(old_Q.dtype).eps

    old_v = jnp.where(beta < eps, old_v_too_small, old_v)
    norm = jnp.where(beta < eps, norm_too_small, norm)
    beta = jnp.where(beta < eps, 0., beta)

    w = old_v / norm

    Aw = multiplier(w)
    alpha = jnp.einsum('ij,ij->j', w, Aw)

    # Compute full reorthogonalization every time, using "twice is enough"
    # reorthogonalization.
    v = Aw - alpha[jnp.newaxis, :] * w
    v = reorthogonalize(old_Q, v)

    diags = old_diags.at[:, i].set(alpha)
    Q = old_Q.at[:, i, :].set(jnp.transpose(w))
    # Using an if here is okay, because num_iters will be statically known.
    # (Either because it will be set explicitly, or because n will be
    # statically known.  Note that we can't use a jax.lax.cond here because
    # the at/set branch will raise an error when num_iters == 1.
    if num_iters > 1:
      off_diags = old_off_diags.at[:, i - 1].set(beta[0])
    else:
      off_diags = old_off_diags

    new_loop_info = (i + 1, Q, diags, off_diags, v, key2)

    return new_loop_info, None

  init_Q = jnp.zeros(shape=(k, num_iters, n), dtype=v.dtype)
  init_diags = jnp.ones(shape=(k, num_iters), dtype=v.dtype)
  init_off_diags = jnp.zeros(shape=(k, num_iters - 1), dtype=v.dtype)
  # Normalize v beforehand. This is because v could have a small norm, and
  # trigger rejuvenating the Krylov subspace, which we don't want.
  v = v / jnp.linalg.norm(v, axis=0, keepdims=True)
  initial_state = (0, init_Q, init_diags, init_off_diags, v, key)

  scan_out, _ = jax.lax.scan(
      scan_func,
      initial_state,
      None,
      num_iters)

  _, Q, diags, off_diags, _, _ = scan_out

  return Q, mbcg.SymmetricTridiagonalMatrix(diag=diags, off_diag=off_diags)


def make_lanczos_preconditioner(
    kernel: tf2jax.linalg.LinearOperator, key: jax.Array, num_iters: int = 20
):
  """Return a preconditioner as a linear operator."""
  n = kernel.shape[-1]
  key1, key2 = jax.random.split(key)
  v = jax.random.uniform(
      key1, shape=(n, 1), minval=-1.0, maxval=1.0, dtype=kernel.dtype)
  Q, T = partial_lanczos(lambda x: kernel @ x, v, key2, num_iters)

  # Now diagonalize T as Q^t D Q
  # TODO(thomaswc): Once jax.scipy.linalg.eigh_tridiagonal supports
  # eigenvectors (https://github.com/google/jax/issues/14019), replace
  # this with that so that it can be jit-ed.
  evalues, evectors = scipy.linalg.eigh_tridiagonal(
      T.diag[0, :], T.off_diag[0, :])
  sqrt_evalues = jnp.sqrt(evalues)

  F = jnp.einsum('i,ij->ij', sqrt_evalues, evectors @ Q[0])

  # diag(F^t F)_i = sum_k (F^t)_{i, k} F_{k, i} = sum_k F_{k, i}^2
  diag_Ft_F = jnp.sum(F * F, axis=0)
  residual_diag = tf2jax.linalg.diag_part(kernel) - diag_Ft_F

  eps = jnp.finfo(kernel.dtype).eps

  # TODO(srvasude): Modify this when residual_diag is near zero. This means that
  # we captured the diagonal appropriately, and modifying with a shift of eps
  # can alter the preconditioner greatly.
  diag_linop = tf2jax.linalg.LinearOperatorDiag(
      jnp.maximum(residual_diag, 0.0) + 10.0 * eps, is_positive_definite=True
  )
  return tf2jax.linalg.LinearOperatorLowRankUpdate(
      diag_linop, jnp.transpose(F), is_positive_definite=True
  )


def my_tridiagonal_solve(
    lower_diagonal: Array,
    middle_diagonal: Array,
    upper_diagonal: Array,
    b: Array) -> Array:
  """Like jax.linalg.tridiagonal_solve, but works for all sizes."""
  m, = middle_diagonal.shape

  if m >= 3:
    return jax.lax.linalg.tridiagonal_solve(
        lower_diagonal, middle_diagonal, upper_diagonal, b)

  if m == 1:
    return b / middle_diagonal[0]

  if m == 2:
    return jnp.linalg.solve(
        jnp.array([[middle_diagonal[0], upper_diagonal[0]],
                   [lower_diagonal[1], middle_diagonal[1]]]),
        b)

  if m == 0:
    return b

  raise ValueError(f'Logic error; unanticipated {m=}')


def tridiagonal_solve_multishift(
    T: mbcg.SymmetricTridiagonalMatrix,
    shifts: Array,
    v_norm: Array) -> Array:
  """Solve (T - shift_k I) x_k = v_norm e_1 for batch of sym. tridiagonal T.

  Args:
    T: A batched SymmetricTridiagonalMatrix.  T.diag should be of shape
      (k, n).
    shifts: A tensor of shape (s) representing s distinct scalar shifts.
    v_norm: A size (k) batch of vector lengths.

  Returns:
    A tensor x of shape (s, k, n) that approximately satisfies
      (T[k] - shifts[i] I) x[i, j, :] = v_norm[k] e_1
  """
  n = T.diag.shape[-1]
  lower_diagonal = jnp.pad(T.off_diag, ((0, 0), (1, 0)))
  upper_diagonal = jnp.pad(T.off_diag, ((0, 0), (0, 1)))
  target = v_norm[..., jnp.newaxis, jnp.newaxis] * jnp.eye(
      n, 1, dtype=T.diag.dtype)
  # Batch over the Tridiagonal matrix batch.
  batch_solve = jax.vmap(my_tridiagonal_solve, in_axes=(0, 0, 0, 0))
  # Batch over the shift dimension.
  multishift_batch_solve = jax.vmap(batch_solve, in_axes=(None, 0, None, None))
  solutions = multishift_batch_solve(
      lower_diagonal,
      T.diag - shifts[..., jnp.newaxis, jnp.newaxis],
      upper_diagonal,
      target)
  return jnp.squeeze(solutions, axis=-1)


@jax.named_call
def psd_solve_multishift(
    multiplier: Callable[[Array], Array],
    v: Array,
    shifts: Array,
    key: jax.Array,
    num_iters: int = 20) -> Array:
  """Solve (A - shift_k I) x_k = v for PSD A.

  Args:
    multiplier: A function for left matrix multiplying by an n by n
      symmetric positive definite matrix A.
    v:  A tensor of shape (n, t) representing t n-dim vectors.
    shifts: A tensor of shape (s) representing s distinct scalar shifts.
    key: A random seed.
    num_iters: The number of iterations to run the Lanczos tridiagonalization
      algorithm for.

  Returns:
    A tensor x of shape (s, t, n) that approximately satisfies
      (A - shift[i] I) x[i, j, :] = v[:, j]
  """
  Q, T = partial_lanczos(multiplier, v, key, num_iters)
  v_norm = jnp.linalg.norm(v, axis=0, keepdims=False)
  ys = tridiagonal_solve_multishift(T, shifts, v_norm)
  # Q is of shape (t, num_iters, n) and ys is of shape
  # (num_shifts, t, num_iters).
  return jnp.einsum('tin,sti->stn', Q, ys)
