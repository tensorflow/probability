# Copyright 2022 The TensorFlow Probability Authors.
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
"""Adaptive MALT implementation."""

from typing import Any, Dict, Optional, Mapping, NamedTuple, Tuple, Union

import gin
import immutabledict
import jax
import jax.numpy as jnp
from fun_mc import using_jax as fun_mc
from inference_gym import using_jax as gym
import tensorflow_probability.substrates.jax as tfp

try:
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  from fun_mc.dynamic.backend_jax import malt as malt_lib
except ImportError:
  pass


def ccipca(sample: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
  """Run a step of CCIPCA[1].

  Args:
    sample: A batch of samples. Shape: [B, D]
    r: Current guess at the unnormalized principal component. Shape: [D]

  Returns:
    New guess at the unnormalized principal component. Shape: [D]

  #### References:

  [1]: Weng et al. (2003). Candid Covariance-free Incremental Principal
       Component Analysis. IEEE Trans. Pattern Analysis and Machine
       Intelligence.
  """
  r = r / (1e-20 + jnp.linalg.norm(r))
  act = jnp.einsum('d,nd->n', r, sample)
  return (act[:, jnp.newaxis] * sample).mean(0)


def snaper_criterion(
    previous_state: jnp.ndarray,
    proposed_state: jnp.ndarray,
    accept_prob: jnp.ndarray,
    trajectory_length: jnp.ndarray,
    principal: jnp.ndarray,
    power: jnp.ndarray = 1.,
    state_mean: jnp.ndarray = Optional[jnp.ndarray],
    state_mean_weight: jnp.ndarray = 0.,
):
  """SNAPER criterion[1].

  Args:
    previous_state: Previous MCMC state. Shape: [B, D]
    proposed_state: Proposed MCMC state. Shape: [B, D]
    accept_prob: Acceptance probability. Shape: [B]
    trajectory_length: Trajectory length. Shape: []
    principal: Principal component. Shape: [D]
    power: Power to raise the trajectory length to when normalizing. Shape: []
    state_mean: Optional mean of the state. Computed across the B dimension
      otherwise.
    state_mean_weight: Weight in [0, 1] to use when mixing `state_mean` and the
      empirical average across the B dimension.

  Returns:
    A tuple of (criterion, ()).

  #### References:

  [1]: Sountsov, P., & Hoffman, M. D. (2021). Focusing on Difficult Directions
       for Learning HMC Trajectory Lengths. In arXiv [stat.CO]. arXiv.
       http://arxiv.org/abs/2110.11576
  """

  batch_ndims = len(accept_prob.shape)
  batch_axes = tuple(range(batch_ndims))

  no_state_mean = object()
  if state_mean is None:
    state_mean = fun_mc.maybe_broadcast_structure(no_state_mean, previous_state)
  state_mean_weight = fun_mc.maybe_broadcast_structure(state_mean_weight,
                                                       previous_state)
  mx = state_mean
  mw = state_mean_weight

  x_center = previous_state.mean(0)
  if mx is not no_state_mean:
    x_center = x_center * (1 - mw) + mx * mw
  previous_state = previous_state - x_center

  x = proposed_state
  expand_shape = list(accept_prob.shape) + [1] * (
      len(x.shape) - len(accept_prob.shape))
  expanded_accept_prob = jnp.reshape(accept_prob, expand_shape)

  x_safe = jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x))
  # If all accept_prob's are zero, the x_center will have a nonsense value,
  # but we'll set the overall criterion to zero in this case, so it's fine.
  x_center = (
      jnp.sum(
          expanded_accept_prob * x_safe,
          axis=batch_axes,
      ) / (jnp.sum(expanded_accept_prob, axis=batch_axes) + 1e-20))
  if mx is not no_state_mean:
    x_center = x_center * (1 - mw) + mx * mw
  proposed_state = x - jax.lax.stop_gradient(x_center)

  previous_state = jnp.einsum('d,nd->n', principal, previous_state)
  proposed_state = jnp.einsum('d,nd->n', principal, proposed_state)

  esjd = ((previous_state**2 - proposed_state**2)**2)

  esjd = jnp.where(accept_prob > 1e-4, esjd, 0.)
  accept_prob = accept_prob / jnp.sum(accept_prob + 1e-20)
  esjd = esjd * accept_prob

  return esjd.mean() / trajectory_length**power, ()


class AdaptiveMCMCState(NamedTuple):
  """Adaptive MCMC state.

  Attributes:
    mcmc_state: MCMC state.
    rvar_state: Running variance of the state.
    principal_rmean_state: Running mean of the unnormalized principal
      components.
    log_step_size_opt_state: Optimizer state for the log step size.
    log_trajectory_length_opt_state: Optimizer state for the log trajectory
      length.
    step_size_rmean_state: Iterate averaging for step size.
    trajectory_length_rmean_state: Iterate averaging for trajectory length.
    step: Current step.
  """
  mcmc_state: Union[fun_mc.HamiltonianMonteCarloState,
                    fun_mc.prefab.MetropolisAdjustedLangevinTrajectoriesState]
  rvar_state: fun_mc.RunningVarianceState
  principal_rmean_state: fun_mc.RunningMeanState
  log_step_size_opt_state: fun_mc.AdamState
  log_trajectory_length_opt_state: fun_mc.AdamState
  step_size_rmean_state: fun_mc.RunningMeanState
  trajectory_length_rmean_state: fun_mc.RunningMeanState
  step: jnp.ndarray


class AdaptiveMCMCExtra(NamedTuple):
  """Adapive MCMC extra outputs.

  Attributes:
    mcmc_extra: MCMC extra outputs.
    scalar_step_size: Scalar step size.
    vector_step_size: Vector step size.
    principal: Principal component.
    max_eigenvalue: Maximum eigenvalue.
    mean_trajectory_length: Mean trajectory length.
    log_trajectory_length_opt_extra: Extra outputs from log_trajectory_length
      optimizer.
    num_integrator_steps: Number of integrator steps actually taken.
    damping: Damping that was used.
  """
  mcmc_extra: Union[fun_mc.HamiltonianMonteCarloExtra,
                    fun_mc.prefab.MetropolisAdjustedLangevinTrajectoriesExtra]
  scalar_step_size: jnp.ndarray
  vector_step_size: jnp.ndarray
  principal: jnp.ndarray
  max_eigenvalue: jnp.ndarray
  mean_trajectory_length: jnp.ndarray
  log_trajectory_length_opt_extra: fun_mc.AdamExtra
  num_integrator_steps: jnp.ndarray
  damping: jnp.ndarray


def halton(float_index: jnp.ndarray, max_bits: int = 10) -> jnp.ndarray:
  """Generates an element from the halton sequence."""
  float_index = jnp.asarray(float_index)
  bit_masks = 2**jnp.arange(max_bits, dtype=float_index.dtype)
  return jnp.einsum('i,i->', jnp.mod((float_index + 1) // bit_masks, 2),
                    0.5 / bit_masks)


def adaptive_mcmc_init(state: jnp.ndarray,
                       target_log_prob_fn: fun_mc.PotentialFn,
                       init_step_size: jnp.ndarray,
                       init_trajectory_length: jnp.ndarray,
                       method='hmc') -> AdaptiveMCMCState:
  """Initializes the adaptive MCMC algorithm.

  Args:
    state: Initial state of the MCMC chain.
    target_log_prob_fn: Target log prob function.
    init_step_size: Initial scalar step size.
    init_trajectory_length: Initial trajectory length.
    method: MCMC method. Either 'hmc' or 'malt'.

  Returns:
    Adaptive MCMC state.
  """
  if method == 'hmc':
    mcmc_state = fun_mc.hamiltonian_monte_carlo_init(
        state=state,
        target_log_prob_fn=target_log_prob_fn,
    )
  elif method == 'malt':
    mcmc_state = fun_mc.prefab.metropolis_adjusted_langevin_trajectories_init(
        state=state,
        target_log_prob_fn=target_log_prob_fn,
    )

  return AdaptiveMCMCState(
      mcmc_state=mcmc_state,
      principal_rmean_state=fun_mc.running_mean_init(
          state.shape[1:], state.dtype)._replace(
              mean=jax.random.normal(
                  jax.random.PRNGKey(0), state.shape[1:], state.dtype)),
      rvar_state=fun_mc.running_variance_init(state.shape[1:], state.dtype),
      log_step_size_opt_state=fun_mc.adam_init(jnp.log(init_step_size)),
      log_trajectory_length_opt_state=fun_mc.adam_init(
          jnp.log(init_trajectory_length)),
      step_size_rmean_state=fun_mc.running_mean_init([], jnp.float32),
      trajectory_length_rmean_state=fun_mc.running_mean_init([], jnp.float32),
      step=jnp.array(0, jnp.int32),
  )


def adaptive_mcmc_step(
    amcmc_state: AdaptiveMCMCState,
    target_log_prob_fn: fun_mc.PotentialFn,
    num_mala_steps: int,
    num_adaptation_steps: int,
    seed: jax.random.KeyArray,
    method: str = 'hmc',
    damping: Optional[jnp.ndarray] = None,
    scalar_step_size: Optional[jnp.ndarray] = None,
    vector_step_size: Optional[jnp.ndarray] = None,
    mean_trajectory_length: Optional[jnp.ndarray] = None,
    principal: Optional[jnp.ndarray] = None,
    max_num_integrator_steps: int = 500,
    rvar_factor: int = 8,
    iterate_factor: int = 16,
    principal_factor: int = 8,
    target_accept_prob: float = 0.8,
    step_size_adaptation_rate: float = 0.05,
    trajectory_length_adaptation_rate: float = 0.05,
    damping_factor: float = 1.,
    principal_mean_method: str = 'running_mean',
    min_preconditioning_points: int = 64,
    state_grad_estimator: str = 'two_dir',
    trajectory_opt_kwargs: Mapping[str, Any] = immutabledict.immutabledict({}),
    step_size_opt_kwargs: Mapping[str, Any] = immutabledict.immutabledict({}),
):
  """Applies a step of the adaptive MCMC algorithm.

  Args:
    amcmc_state: Adaptive MCMC state.
    target_log_prob_fn: Target log prob function.
    num_mala_steps: Number of initial steps to be MALA.
    num_adaptation_steps: Number of steps to adapt hyperparameters.
    seed: Random seed.
    method: MCMC method. Either 'hmc' or 'malt'.
    damping: If not None, the fixed damping to use.
    scalar_step_size: If not None, the fixed scalar step size to use.
    vector_step_size: If not None, the fixed vector step size to use.
    mean_trajectory_length: If not None, the fixed mean trajectory length to
      use.
    principal: If not None, the fixed unnormalized principal component to use.
    max_num_integrator_steps: Maximum number of integrator steps.
    rvar_factor: Factor for running variance adaptation rate.
    iterate_factor: Factor for iterate averaging rate.
    principal_factor: Factor for principal component adaptation rate.
    target_accept_prob: Target acceptance probability.
    step_size_adaptation_rate: Scalar step size adaptation rate.
    trajectory_length_adaptation_rate: Trajectory length adaptation rate.
    damping_factor: Factor for computing the damping coefficient.
    principal_mean_method: Method for computing principal mean. Either
      'running_mean' or 'state_mean'.
    min_preconditioning_points: Minimum number of points to intake before using
      the running mean for preconditioning:
    state_grad_estimator: State grad estimator to use. Can be 'one_dir' or
      'two_dir'.
    trajectory_opt_kwargs: Extra arguments to the trajectory length optimizer.
    step_size_opt_kwargs: Extra arguments to the step size optimizer.

  Returns:
    A tuple of new adaptive MCMC state and extra.
  """
  seed, mcmc_seed = jax.random.split(seed)

  # ===========================
  # Compute common hyper-params
  # ===========================
  adapt = amcmc_state.step < num_adaptation_steps
  mala = amcmc_state.step < num_mala_steps
  num_chains = amcmc_state.mcmc_state.state.shape[0]

  if scalar_step_size is None:
    scalar_step_size = jnp.where(
        adapt, jnp.exp(amcmc_state.log_step_size_opt_state.state),
        amcmc_state.step_size_rmean_state.mean)
  if vector_step_size is None:
    vector_step_size = jnp.sqrt(amcmc_state.rvar_state.variance)
    vector_step_size = vector_step_size / vector_step_size.max()
    vector_step_size = jnp.where(
        amcmc_state.rvar_state.num_points > min_preconditioning_points,
        vector_step_size, jnp.ones_like(vector_step_size))

  if principal is None:
    max_eigenvalue = jnp.linalg.norm(amcmc_state.principal_rmean_state.mean)
    principal = amcmc_state.principal_rmean_state.mean / max_eigenvalue
  else:
    max_eigenvalue = jnp.linalg.norm(principal)
    principal = principal / max_eigenvalue

  if method == 'hmc':
    traj_factor = halton(amcmc_state.step.astype(jnp.float32)) * 2.
  elif method == 'malt':
    traj_factor = 1.

  if mean_trajectory_length is None:
    mean_trajectory_length = jnp.where(
        adapt, jnp.exp(amcmc_state.log_trajectory_length_opt_state.state),
        amcmc_state.trajectory_length_rmean_state.mean)
  mean_trajectory_length = jnp.where(mala, scalar_step_size,
                                     mean_trajectory_length)

  trajectory_length = traj_factor * mean_trajectory_length
  num_integrator_steps = jnp.ceil(trajectory_length / scalar_step_size)
  num_integrator_steps = jnp.where(
      jnp.isfinite(num_integrator_steps), num_integrator_steps, 1)
  num_integrator_steps = num_integrator_steps.astype(jnp.int32)

  # ====================
  # Apply a step of MCMC
  # ====================
  if method == 'hmc':
    mcmc_state, mcmc_extra = fun_mc.hamiltonian_monte_carlo_step(
        amcmc_state.mcmc_state,
        target_log_prob_fn=target_log_prob_fn,
        step_size=scalar_step_size * vector_step_size,
        num_integrator_steps=num_integrator_steps,
        seed=mcmc_seed,
    )
    proposed_state = mcmc_extra.proposed_hmc_state.state
  elif method == 'malt':
    if damping is None:
      damping = damping_factor / max_eigenvalue
    mcmc_state, mcmc_extra = fun_mc.prefab.metropolis_adjusted_langevin_trajectories_step(
        amcmc_state.mcmc_state,
        target_log_prob_fn=target_log_prob_fn,
        step_size=scalar_step_size * vector_step_size,
        momentum_refresh_fn=lambda m, seed:  # pylint: disable=g-long-lambda
        (
            malt_lib._gaussian_momentum_refresh_fn(  # pylint: disable=protected-access
                m,
                damping=damping,
                step_size=scalar_step_size,
                seed=seed,
            )),
        num_integrator_steps=num_integrator_steps,
        damping=damping,
        seed=mcmc_seed,
    )
    proposed_state = mcmc_extra.proposed_malt_state.state
  # This assumes a N(0, I) momentum distribution
  initial_momentum = mcmc_extra.initial_momentum
  final_momentum = mcmc_extra.integrator_extra.momentum_grads

  # ==========
  # Adaptation
  # ==========

  # Adjust running-variance estimate.
  cand_rvar_state, _ = fun_mc.running_variance_step(
      amcmc_state.rvar_state,
      mcmc_state.state,
      axis=0,
      window_size=jnp.maximum(1, num_chains * amcmc_state.step // rvar_factor))
  rvar_state = fun_mc.choose(adapt, cand_rvar_state, amcmc_state.rvar_state)

  # Adjust trajectory length.
  def log_trajectory_length_surrogate_loss_fn(log_trajectory_length):
    log_trajectory_length = fun_mc.clip_grads(log_trajectory_length, 1e6)
    mean_trajectory_length = jnp.exp(log_trajectory_length)
    mean_trajectory_length = jnp.where(mala, scalar_step_size,
                                       traj_factor * mean_trajectory_length)
    trajectory_length = mean_trajectory_length

    if state_grad_estimator == 'one_dir':
      start_end_grads = [
          (amcmc_state.mcmc_state.state, proposed_state, final_momentum),
      ]
    elif state_grad_estimator == 'two_dir':
      start_end_grads = [
          (amcmc_state.mcmc_state.state, proposed_state, final_momentum),
          (
              proposed_state,
              amcmc_state.mcmc_state.state,
              -initial_momentum,
          ),
      ]

    criteria = []
    extras = []
    for start_state, end_state, state_grads in start_end_grads:
      action = trajectory_length * vector_step_size * state_grads
      end_state_with_action = (
          end_state + action - jax.lax.stop_gradient(action))

      criterion, extra = snaper_criterion(
          previous_state=start_state,
          proposed_state=end_state_with_action,
          accept_prob=jnp.exp(jnp.minimum(0., -mcmc_extra.log_accept_ratio)),
          trajectory_length=trajectory_length + scalar_step_size,
          principal=principal,
          power=1.,
          # These two expressions are a bit weird for the reverse direction...
          state_mean=amcmc_state.rvar_state.mean,
          state_mean_weight=(amcmc_state.rvar_state.num_points) /
          (amcmc_state.rvar_state.num_points + num_chains),
      )
      criteria.append(criterion)
      extras.append(extra)
    return -jnp.array(criteria).mean(), extras

  (log_trajectory_length_opt_state, log_trajectory_length_opt_extra) = (
      fun_mc.adam_step(
          amcmc_state.log_trajectory_length_opt_state,
          log_trajectory_length_surrogate_loss_fn,
          jnp.where(adapt, trajectory_length_adaptation_rate, 0.),
          **trajectory_opt_kwargs,
      ))

  # Adjust step size.
  log_accept_ratio = mcmc_extra.log_accept_ratio
  min_log_accept_prob = jnp.full(log_accept_ratio.shape, jnp.log(1e-5))

  log_accept_prob = jnp.minimum(log_accept_ratio, jnp.zeros([]))
  log_accept_prob = jnp.maximum(log_accept_prob, min_log_accept_prob)
  log_accept_prob = jnp.where(
      jnp.isfinite(log_accept_prob), log_accept_prob, min_log_accept_prob)
  accept_prob = jnp.exp(tfp.math.reduce_log_harmonic_mean_exp(log_accept_prob))

  log_step_size_surrogate_loss_fn = fun_mc.make_surrogate_loss_fn(
      lambda _: (target_accept_prob - accept_prob, ()))

  (log_step_size_opt_state, _) = (
      fun_mc.adam_step(
          amcmc_state.log_step_size_opt_state,
          log_step_size_surrogate_loss_fn,
          jnp.where(adapt, step_size_adaptation_rate, 0.),
          **step_size_opt_kwargs,
      ))

  # Clip the trajectory length (need new step size estimate)
  log_trajectory_length_opt_state = log_trajectory_length_opt_state._replace(
      state=jnp.minimum(
          jnp.log(scalar_step_size * max_num_integrator_steps),
          log_trajectory_length_opt_state.state))

  # Adjust the principal component + max_eigval estimate.
  if principal_mean_method == 'state_mean':
    mean = mcmc_state.state.mean(0)
  elif principal_mean_method == 'running_mean':
    mean = rvar_state.mean

  cand_principal_rmean_state, _ = fun_mc.running_mean_step(
      amcmc_state.principal_rmean_state,
      ccipca(mcmc_state.state - mean, amcmc_state.principal_rmean_state.mean),
      window_size=amcmc_state.step // principal_factor,
  )
  principal_rmean_state = fun_mc.choose(adapt, cand_principal_rmean_state,
                                        amcmc_state.principal_rmean_state)

  # =================
  # Iterate averaging
  # =================
  cand_step_size_rmean_state, _ = fun_mc.running_mean_step(
      amcmc_state.step_size_rmean_state,
      scalar_step_size,
      window_size=amcmc_state.step // iterate_factor)
  step_size_rmean_state = fun_mc.choose(adapt, cand_step_size_rmean_state,
                                        amcmc_state.step_size_rmean_state)

  cand_trajectory_length_rmean_state, _ = fun_mc.running_mean_step(
      amcmc_state.trajectory_length_rmean_state,
      mean_trajectory_length,
      window_size=amcmc_state.step // iterate_factor)
  trajectory_length_rmean_state = fun_mc.choose(
      adapt, cand_trajectory_length_rmean_state,
      amcmc_state.trajectory_length_rmean_state)

  amcmc_state = amcmc_state._replace(
      mcmc_state=mcmc_state,
      rvar_state=rvar_state,
      principal_rmean_state=principal_rmean_state,
      log_step_size_opt_state=log_step_size_opt_state,
      log_trajectory_length_opt_state=log_trajectory_length_opt_state,
      step_size_rmean_state=step_size_rmean_state,
      trajectory_length_rmean_state=trajectory_length_rmean_state,
      step=amcmc_state.step + 1,
  )
  amcmc_extra = AdaptiveMCMCExtra(
      mcmc_extra=mcmc_extra,
      scalar_step_size=scalar_step_size,
      vector_step_size=vector_step_size,
      principal=principal,
      max_eigenvalue=max_eigenvalue,
      damping=damping,
      mean_trajectory_length=mean_trajectory_length,
      log_trajectory_length_opt_extra=log_trajectory_length_opt_extra,
      num_integrator_steps=num_integrator_steps,
  )
  return amcmc_state, amcmc_extra


def compute_stats(state: jnp.ndarray, num_grads: jnp.ndarray, mean: jnp.ndarray,
                  principal: jnp.ndarray) -> Dict[str, Dict[str, jnp.ndarray]]:
  """Computes MCMC statistics.

  Args:
    state: MCMC chain.
    num_grads: Number of gradient evaluations used to compute the chain.
    mean: State mean.
    principal: State principal component.

  Returns:
     A dict of statistics dicts.
  """

  transforms = {
      'm1':
          lambda x: x,
      'm2':
          lambda x: (x - mean)**2,
      'norm_sq':
          lambda x: (x - mean)**2,
      'p_m2':
          lambda x:  # pylint: disable=g-long-lambda
          (jnp.einsum('...d,d->...', x - mean, principal)[..., jnp.newaxis]**2),
  }

  res = {}
  for name, transform_fn in transforms.items():
    transformed_state = transform_fn(state)
    ess = tfp.mcmc.effective_sample_size(transformed_state).mean(0)
    ess_per_grad = ess / num_grads
    rhat = tfp.mcmc.potential_scale_reduction(
        transformed_state, split_chains=True)

    white_state = transformed_state - transformed_state.mean(
        (0, 1)) / transformed_state.std((0, 1))
    lag_1_autocorr = (white_state[:-1] * white_state[1:]).sum((0, 1))

    stats = {}
    stats['ess'] = ess
    stats['ess_per_grad'] = ess_per_grad
    stats['rhat'] = rhat
    stats['lag_1_autocorr'] = lag_1_autocorr
    res[name] = stats

  return res


@gin.configurable
def run_adaptive_mcmc_on_target(
    target: gym.targets.VectorModel,
    method: str,
    num_chains: int,
    init_step_size: jnp.ndarray,
    num_adaptation_steps: int,
    num_results: int,
    seed: jax.random.KeyArray,
    num_mala_steps: int = 100,
    **amcmc_kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Does a run of adaptive MCMC.

  Args:
    target: Target model.
    method: MCMC method. Either 'hmc' or 'malt'.
    num_chains: Number of chains.
    init_step_size: Initial scalar step size.
    num_adaptation_steps: Number of adaptation steps.
    num_results: Number of post-adaptation results.
    seed: Random seed.
    num_mala_steps: Number of initial steps to be MALA.
    **amcmc_kwargs: Extra kwargs to pass to Adaptive MCMC.

  Returns:
     A tuple of final and traced results.
  """
  init_z = target.default_event_space_bijector.inverse(
      jnp.tile(
          target.structured_event_to_vector(
              target.model.prior_distribution().sample(
                  256, seed=jax.random.PRNGKey(0))).mean(0, keepdims=True),
          [num_chains, 1]))

  def target_log_prob_fn(x):
    return target.unnormalized_log_prob(x), ()

  target_log_prob_fn = fun_mc.transform_log_prob_fn(
      target_log_prob_fn, target.default_event_space_bijector)

  def kernel(amcmc_state, seed):
    seed, mcmc_seed = jax.random.split(seed)

    amcmc_state, amcmc_extra = adaptive_mcmc_step(
        amcmc_state,
        target_log_prob_fn=target_log_prob_fn,
        num_mala_steps=num_mala_steps,
        num_adaptation_steps=num_adaptation_steps,
        seed=mcmc_seed,
        min_preconditioning_points=64,
        method=method,
        trajectory_opt_kwargs={
            'beta_1': 0.,
            'beta_2': 0.95
        },
        **amcmc_kwargs,
    )

    traced = {
        'state': amcmc_state.mcmc_state.state,
        'is_accepted': amcmc_extra.mcmc_extra.is_accepted,
        'log_accept_ratio': amcmc_extra.mcmc_extra.log_accept_ratio,
        'scalar_step_size': amcmc_extra.scalar_step_size,
        'vector_step_size': amcmc_extra.vector_step_size,
        'principal': amcmc_extra.principal,
        'max_eigenvalue': amcmc_extra.max_eigenvalue,
        'mean_trajectory_length': amcmc_extra.mean_trajectory_length,
        'num_integrator_steps': amcmc_extra.num_integrator_steps,
        'traj_loss': amcmc_extra.log_trajectory_length_opt_extra.loss,
        'damping': amcmc_extra.damping,
    }

    return (amcmc_state, seed), traced

  amcmc_state = adaptive_mcmc_init(
      init_z,
      target_log_prob_fn,
      init_step_size=init_step_size,
      init_trajectory_length=init_step_size,
      method=method,
  )

  _, trace = fun_mc.trace((amcmc_state, seed), kernel,
                          num_results + num_adaptation_steps)

  warmed_up_steps = int(num_results * 0.8)
  warmed_up_state = trace['state'][-warmed_up_steps:]
  cov = jnp.cov(
      warmed_up_state.reshape([-1, warmed_up_state.shape[-1]]), rowvar=False)
  principal = jnp.linalg.eigh(cov)[1][:, -1]

  final = {
      'stats':
          compute_stats(
              state=warmed_up_state,
              num_grads=trace['num_integrator_steps'][-warmed_up_steps:].sum(),
              mean=warmed_up_state.mean((0, 1)),
              principal=principal,
          )
  }
  return trace, final
