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
"""Modified Batched Conjugate Gradients (mBCG) for JAX."""

from typing import Callable, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray


class SymmetricTridiagonalMatrix(NamedTuple):
  """Holds a batch of symmetric, tridiagonal matrices."""
  diag: Array
  off_diag: Array


def safe_stack(list_of_arrays: List[Array], empty_size: int) -> Array:
  """Like jnp.stack, but handles len == 0 or 1."""
  l = len(list_of_arrays)
  if l == 0:
    return jnp.empty(shape=(empty_size, 0))
  if l == 1:
    return list_of_arrays[0][:, jnp.newaxis]
  return jnp.stack(list_of_arrays, axis=-1)


# pylint: disable=invalid-name


@jax.named_call
def modified_batched_conjugate_gradients(
    matrix_matrix_multiplier: Callable[[Array], Array],
    B: Array,
    preconditioner_fn: Callable[[Array], Array],
    max_iters: int = 20,
    tolerance: float = 1e-6,
) -> Tuple[Array, SymmetricTridiagonalMatrix]:
  """Return A^(-1)B and Lanczos tridiagonal matrices.

  Based on Algorithm 2 on page 14 of https://arxiv.org/pdf/1809.11165.pdf

  Args:
    matrix_matrix_multiplier: A function for left-matrix multiplying by an n x n
      matrix A, which should be symmetric and positive definite.
    B: An n x t matrix containing the vectors v_i for which we want A^(-1) v_i.
    preconditioner_fn: A function that applies an invertible linear
      transformation to its input, designed to increase the rate of convergence
      by decreasing the condition number.  The preconditioner_fn should Act like
      left application of an n by n linear operator, i.e. preconditioner_fn(n x
      m) should have shape n x m.  Somewhat amazingly, this algorithm doesn't
      care if the passed preconditioner is a left, right, or split
      preconditioner -- the same output will result.  Just be sure to pass in
      full_preconditioner.solve when using a SplitPreconditioner.
    max_iters: Run conjugate gradients for at most this many iterations.  Note
      that no matter the value of max_iters, the loop will run for at most n
      iterations.
    tolerance: Stop early if all the errors have less than this magnitude.

  Returns:
    A pair (C, T) where C is the n x t matrix A^(-1) B and T is a
    SymmetricTridiagonalMatrix where the diag part is of shape (t, max_iters)
    and the off_diag part is of shape (t, max_iters -1).
  """
  n, t = B.shape
  init_solutions = jnp.zeros_like(B)
  # Algorithm 2 has
  # current_errors = matrix_matrix_multiplier(current_solutions) - B
  # but this leads to the first update being towards -B instead of B,
  # which is wrong.
  init_errors = B
  init_search_directions = jnp.zeros_like(B)
  # init_preconditioned_errors doesn't get used because
  # init_search_directions is zero, but it needs to be non-zero to avoid
  # divide by zero nans.
  init_preconditioned_errors = B
  max_iters = min(max_iters, n)

  diags = jnp.ones(shape=(t, max_iters), dtype=B.dtype)
  # When n = 1, we still need a location to write a dummy value.
  off_diags = jnp.zeros(shape=(t, max(1, max_iters - 1)), dtype=B.dtype)

  def loop_body(carry, _):
    """Body for jax.lax.while_loop."""
    (j, old_errors, old_solutions,
     old_preconditioned_errors, old_search_directions,
     old_alpha, beta_factor, diags, off_diags) = carry
    preconditioned_errors = preconditioner_fn(old_errors)

    converged = jnp.all(jnp.abs(old_errors) < tolerance, axis=0)
    # We check convergence per batch member.
    assert converged.shape == (t,)

    # beta is a size t vector, i-th entry =
    # preconditioned_errors[:, i] dot preconditioned_errors[:, i]
    # -----------------------------------------------------------
    # old_preconditioned_errors[:, i] dot old_preconditioned_errors[:, i]
    beta_numerator = jnp.einsum('ij,ij->j',
                                preconditioned_errors, preconditioned_errors)
    beta_denominator = jnp.einsum(
        'ij,ij->j', old_preconditioned_errors, old_preconditioned_errors)
    beta = beta_factor * beta_numerator / beta_denominator
    safe_beta = jnp.where(converged, 1., beta)

    search_directions = jnp.where(
        converged, old_search_directions, preconditioned_errors +
        safe_beta[jnp.newaxis] * old_search_directions)

    v = matrix_matrix_multiplier(search_directions)

    # alpha is a size t vector, i-th entry =
    # current_errors[:, i] dot preconditioned_errors[:, i]
    # ----------------------------------------------------
    # search_directions[:, i] dot v[:, i]
    alpha_num = jnp.einsum('ij,ij->j', old_errors, preconditioned_errors)
    alpha_denom = jnp.einsum('ij,ij->j', search_directions, v)
    alpha = alpha_num / alpha_denom
    safe_alpha = jnp.where(converged, 1., alpha)

    new_solutions = jnp.where(
        converged,
        old_solutions,
        old_solutions + safe_alpha[jnp.newaxis] * search_directions)
    # TODO(srvasude): Test out the following change:
    # new_errors = B - matrix_matrix_multiplier(new_solutions)
    # While requiring one more matrix multiplication, this is a more numerically
    # stable expression and can be used more reliably as a stopping criterion.
    new_errors = jnp.where(
        converged,
        old_errors,
        old_errors - safe_alpha[jnp.newaxis] * v)

    # When j = 0, beta = 0 (because of beta_factor), so old_alpha doesn't
    # matter.
    # Here and below, use the double-where trick to avoid NaN gradients.
    diag_update = jnp.where(
        converged, 1., 1. / safe_alpha + safe_beta / old_alpha)
    new_diags = diags.at[:, j].set(diag_update)
    # When j = 0, beta = 0 (because of beta_factor), so this ends up writing
    # a zero vector at the end of off_diags, which is a no-op.
    off_diag_update = jnp.where(converged, 0., jnp.sqrt(safe_beta) / old_alpha)
    new_off_diags = off_diags.at[:, j - 1].set(off_diag_update)

    # Only update if we are not within tolerance.
    (preconditioned_errors, search_directions, alpha) = (jax.tree.map(
        lambda o, n: jnp.where(converged, o, n),
        (old_preconditioned_errors, old_search_directions, old_alpha),
        (preconditioned_errors, search_directions, safe_alpha)))

    new_beta_factor = jnp.array([1.0], dtype=B.dtype)

    return (j + 1, new_errors, new_solutions,
            preconditioned_errors, search_directions, alpha, new_beta_factor,
            new_diags, new_off_diags), ()

  init_alpha = jnp.ones(shape=(t,), dtype=B.dtype)
  beta_factor = jnp.array([0.0], dtype=B.dtype)

  scan_out, _ = jax.lax.scan(
      loop_body,
      (0, init_errors, init_solutions,
       init_preconditioned_errors, init_search_directions,
       init_alpha, beta_factor,
       diags, off_diags),
      None, max_iters)
  _, _, solutions, _, _, _, _, diags, off_diags = scan_out

  return solutions, SymmetricTridiagonalMatrix(diag=diags, off_diag=off_diags)


@jax.jit
def tridiagonal_det(diag: Array, off_diag: Array) -> float:
  """Return the determinant of a tridiagonal matrix."""
  # From https://en.wikipedia.org/wiki/Tridiagonal_matrix#Determinant
  # TODO(thomaswc): Turn this into a method of SymmetricTridiagonalMatrix.
  # Using a scan reduce the number of ops in the graph, therefore reducing
  # lowering and compile times.
  def scan_body(carry, xs):
    d = xs[0]
    o = xs[1]
    pv = carry[0]
    v = carry[1]
    new_value = d * v - o**2 * pv
    return (v, new_value), new_value

  initial_d = diag[0]
  # Return the last value of the determinant recursion.
  return jax.lax.scan(
      scan_body,
      init=(jnp.ones_like(initial_d), initial_d),
      xs=(diag[1:], off_diag))[1][-1]
