# Copyright 2021 The TensorFlow Probability Authors.
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
"""Implementation of pathfinder algorithm from [1].

[1] Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2021). Pathfinder:
Parallel quasi-Newton variational inference. arXiv preprint arXiv:2108.03782.
"""
from typing import Tuple, List, Sequence, Callable
import jax
import jax.numpy as jnp
import numpy as np


def pathfinder(
    *,
    target_density: Callable[[jnp.ndarray], jnp.ndarray],
    initial_value: jnp.ndarray,
    lbfgs_max_iters: int,  # L
    key,
    lbfgs_relative_tolerance: float = 1e-13,  # tau_rel
    lbfgs_inverse_hessian_history: int = 6,  # J
    monte_carlo_elbo_draws: int = 10,  # K
    num_draws: int = 10,  # M
):
  """Single pathfinder algorithm."""

  optimization_path, grad_optimization_path = lbfgs(
      log_target_density=target_density,
      initial_value=initial_value,
      inverse_hessian_history=lbfgs_inverse_hessian_history,
      relative_tolerance=lbfgs_relative_tolerance,
      max_iters=lbfgs_max_iters,
  )
  cov_factors = cov_estimate(
      optimization_path=optimization_path,
      optimization_path_grads=grad_optimization_path,
      history=lbfgs_inverse_hessian_history,
  )
  key, *keys = jax.random.split(key, num=len(optimization_path))
  elbos = np.empty(len(optimization_path) - 1)
  for idx, (value, factors,
            sample_key) in enumerate(zip(optimization_path, cov_factors, keys)):
    diagonal_estimate, thin_factors, scaling_outer_product = factors
    draws, draw_probs = bfgs_sample(
        value=value,
        grad_density=grad_optimization_path[idx],
        diagonal_estimate=diagonal_estimate,
        thin_factors=thin_factors,
        scaling_outer_product=scaling_outer_product,
        num_draws=monte_carlo_elbo_draws,
        key=sample_key,
    )
    target_probs = target_density(draws)

    elbos[idx] = elbo(target_probs, draw_probs)
  best_idx = jnp.argmax(elbos)
  diagonal_estimate, thin_factors, scaling_outer_product = cov_factors[best_idx]
  return bfgs_sample(
      value=optimization_path[best_idx],
      grad_density=grad_optimization_path[best_idx],
      diagonal_estimate=diagonal_estimate,
      thin_factors=thin_factors,
      scaling_outer_product=scaling_outer_product,
      num_draws=num_draws,
      key=key,
  )


def multipath_pathfinder(
    *,
    target_density: Callable[[jnp.ndarray], jnp.ndarray],
    initial_values: jnp.ndarray,
    lbfgs_max_iters: int,  # L
    key,
    lbfgs_relative_tolerance: float = 1e-13,  # tau_rel
    lbfgs_inverse_hessian_history: int = 6,  # J
    monte_carlo_elbo_draws: int = 10,  # K
    num_pathfinder_draws: int = 10,  # M
    num_draws: int,  # R
):
  """Multi-pathfinder algorithm."""
  ps_ir_key, *keys = jax.random.split(key, len(initial_values) + 1)
  draws, q_probs, p_probs = [], [], []
  for key, initial_value in zip(keys, initial_values):
    draw, prob = pathfinder(
        target_density=target_density,
        initial_value=initial_value,
        lbfgs_max_iters=lbfgs_max_iters,
        key=key,
        lbfgs_relative_tolerance=lbfgs_relative_tolerance,
        lbfgs_inverse_hessian_history=lbfgs_inverse_hessian_history,
        monte_carlo_elbo_draws=monte_carlo_elbo_draws,
        num_draws=num_pathfinder_draws,
    )
    draws.append(draw)
    q_probs.append(prob)
    p_probs.append(target_density(draw))
  constant = jnp.log(len(initial_values))
  return ps_ir(
      draws=jnp.concatenate(draws),
      proposal_densities=jnp.concatenate(q_probs) - constant,
      target_densities=jnp.concatenate(p_probs) - constant,
      num_resampled=num_draws,
      key=ps_ir_key,
  )


def ps_ir(*, draws, proposal_densities, target_densities, num_resampled, key):
  logits = target_densities - proposal_densities
  logits = jnp.nan_to_num(logits, nan=-jnp.inf, neginf=-jnp.inf)
  idxs = jax.random.categorical(key=key, logits=logits, shape=(num_resampled,))
  return draws[idxs]


def elbo(log_p, log_q):
  return jnp.mean(log_p - log_q)


def bfgs_sample(
    *,
    value,
    grad_density,
    diagonal_estimate,
    thin_factors,
    scaling_outer_product,
    num_draws,
    key,
):
  """Sample from BFGS factorized estimates."""
  q, r = jnp.linalg.qr(
      jnp.diag(jax.lax.rsqrt(diagonal_estimate)) @ thin_factors)
  lower_chol = jnp.linalg.cholesky(
      jnp.eye(r.shape[0]) + r @ scaling_outer_product @ r.T)
  log_sigma = (
      jnp.log(jnp.prod(jnp.abs(diagonal_estimate))) +
      2.0 * jnp.linalg.slogdet(lower_chol)[1])
  mu = (
      value + diagonal_estimate * grad_density + thin_factors
      @ (scaling_outer_product @ (jnp.transpose(thin_factors) @ grad_density)))
  draws = jax.random.multivariate_normal(
      key=key,
      mean=jnp.zeros_like(value),
      cov=jnp.eye(value.shape[0]),
      shape=(num_draws,),
  ).T
  transformed = q.T @ draws
  transformed_draws = (
      mu + (jnp.diag(jnp.sqrt(diagonal_estimate))
            @ (q @ lower_chol @ transformed + draws - q @ transformed)).T)
  draw_probs = -0.5 * (
      log_sigma + jnp.sum(draws**2, axis=0) +
      value.shape[0] * jnp.log(2 * jnp.pi))

  return transformed_draws, draw_probs


def lbfgs(
    *,
    log_target_density: Callable[[jnp.ndarray], jnp.ndarray],
    initial_value: jnp.ndarray,  # theta_init
    inverse_hessian_history: int = 6,  # J
    relative_tolerance: float = 1e-13,  # tau_rel
    max_iters: int = 1000,  # L
    wolfe_bounds: Tuple[float, float] = (1e-4, 0.9),
    positivity_threshold: float = 2.2e-16,
):
  """LBFGS implementation which returns the optimization path and gradients."""
  dim = initial_value.shape[0]
  grad_log_density = jax.grad(log_target_density)
  optimization_path = [initial_value]
  current_lp = log_target_density(initial_value)
  grad_optimization_path = [grad_log_density(initial_value)]
  position_diffs = jnp.empty((dim, 0))
  gradient_diffs = jnp.empty((dim, 0))
  for _ in range(max_iters):
    diagonal_estimate, thin_factors, scaling_outer_product = bfgs_inverse_hessian(
        updates_of_position_differences=position_diffs,
        updates_of_gradient_differences=gradient_diffs,
    )
    grad_lp = grad_optimization_path[-1]
    search_direction = diagonal_estimate * grad_lp + thin_factors @ (
        scaling_outer_product @ (jnp.transpose(thin_factors) @ grad_lp))
    step_size = 1.0
    while step_size > 1e-8:
      proposed = optimization_path[-1] + step_size * search_direction
      proposed_lp = log_target_density(proposed)
      if proposed_lp >= current_lp + (wolfe_bounds[0] *
                                      grad_lp) @ (step_size * search_direction):
        proposed_grad = grad_log_density(proposed)
        if (proposed_grad @ search_direction <=
            wolfe_bounds[1] * grad_lp @ search_direction):
          break
      step_size = 0.5 * step_size
    optimization_path.append(proposed)
    grad_optimization_path.append(proposed_grad)
    if (proposed_lp - current_lp) / jnp.abs(current_lp) < relative_tolerance:
      return optimization_path, grad_optimization_path
    current_lp = proposed_lp

    position_diff: jnp.ndarray = optimization_path[-1] - optimization_path[-2]
    grad_diff = -grad_optimization_path[-1] + grad_optimization_path[-2]
    if position_diff @ grad_diff > positivity_threshold * jnp.sum(grad_diff**2):
      position_diffs = jnp.column_stack(
          (position_diffs[:, -inverse_hessian_history + 1:], position_diff))
      gradient_diffs = jnp.column_stack(
          (gradient_diffs[:, -inverse_hessian_history + 1:], grad_diff))
  return optimization_path, grad_optimization_path


def bfgs_inverse_hessian(*, updates_of_position_differences,
                         updates_of_gradient_differences):
  """Estimate the inverse hessian from the position updates."""
  dim, history = jnp.shape(updates_of_position_differences)
  inverse_outer = jnp.linalg.inv(
      jnp.tril(
          jnp.einsum(
              "ji,jk->ik",
              updates_of_position_differences,
              updates_of_gradient_differences,
          )))
  if history == 0:
    diagonal_scale = 1.0
  else:
    diagonal_scale = jnp.linalg.norm(
        updates_of_gradient_differences[:, -1], ord=2) / (
            updates_of_position_differences[:, -1]
            @ updates_of_gradient_differences[:, -1])
  diagonal_estimate = diagonal_scale * jnp.ones(dim)
  eta = jnp.einsum("ij,ij->j", updates_of_position_differences,
                   updates_of_gradient_differences)
  thin_factors = jnp.concatenate(
      (
          diagonal_scale * updates_of_gradient_differences,
          updates_of_position_differences,
      ),
      axis=-1,
  )
  scaling_outer_product = jnp.block([
      [jnp.zeros((history, history)), -inverse_outer],
      [
          -jnp.transpose(inverse_outer),
          jnp.transpose(inverse_outer)
          @ (jnp.diag(eta) +
             diagonal_scale * jnp.transpose(updates_of_gradient_differences)
             @ updates_of_gradient_differences) @ inverse_outer,
      ],
  ])
  return diagonal_estimate, thin_factors, scaling_outer_product


def cov_estimate(*, optimization_path: Sequence[jnp.ndarray],
                 optimization_path_grads: Sequence[jnp.ndarray], history: int):
  """Estimate covariance from an optimization path."""
  dim = optimization_path[0].shape[0]
  position_diffs = jnp.empty((dim, 0))
  gradient_diffs = jnp.empty((dim, 0))
  approximations: List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
  diagonal_estimate = jnp.ones(dim)
  for j in range(len(optimization_path) - 1):
    _, thin_factors, scaling_outer_product = bfgs_inverse_hessian(
        updates_of_position_differences=position_diffs,
        updates_of_gradient_differences=gradient_diffs,
    )
    position_diff = optimization_path[j + 1] - optimization_path[j]
    gradient_diff = optimization_path_grads[j] - optimization_path_grads[j + 1]
    b = position_diff @ gradient_diff
    gradient_diff_norm = gradient_diff**2
    new_diagonal_estimate = diagonal_estimate
    if b < 1e-12 * jnp.sum(gradient_diff_norm):
      position_diffs = jnp.column_stack(
          (position_diffs[:, -history + 1:], position_diff))
      gradient_diffs = jnp.column_stack(
          (gradient_diffs[:, -history + 1:], gradient_diff))
      a = gradient_diff @ (diagonal_estimate * gradient_diff)
      c = position_diff @ (position_diff / diagonal_estimate)
      new_diagonal_estimate = 1.0 / (
          a / (b * diagonal_estimate) + gradient_diff_norm / b -
          (a * position_diff**2) / (b * c * diagonal_estimate**2))
    approximations.append(
        (diagonal_estimate, thin_factors, scaling_outer_product))
    diagonal_estimate = new_diagonal_estimate
  return approximations
