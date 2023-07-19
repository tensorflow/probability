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

import functools
import os
from typing import Any, Callable, Dict, Optional, Mapping, NamedTuple, Tuple, Union

from etils import epath
import gin
import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
from discussion.adaptive_malt import utils
from fun_mc import using_jax as fun_mc
from inference_gym import using_jax as gym
import tensorflow_probability.substrates.jax as tfp

try:
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  from fun_mc.dynamic.backend_jax import malt as malt_lib
except ImportError:
  pass

tfd = tfp.distributions
tfb = tfp.bijectors


class TestGaussian(gym.targets.Model):
  """Test Gaussian."""

  def __init__(
      self,
      ndims=200,
      name='test_gaussian',
      pretty_name='Test-Gaussian',
  ):
    sigma = np.sqrt(np.linspace(1., 1 / ndims, ndims)).astype(np.float32)

    gaussian = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(ndims), scale_diag=sigma)

    sample_transformations = {
        'identity':
            gym.targets.Model.SampleTransformation(
                fn=lambda params: params,
                pretty_name='Identity',
                ground_truth_mean=np.zeros(ndims),
                ground_truth_standard_deviation=sigma,
            )
    }

    self._gaussian = gaussian

    super().__init__(
        default_event_space_bijector=tfb.Identity(),
        event_shape=gaussian.event_shape,
        dtype=gaussian.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _unnormalized_log_prob(self, value):
    return self._gaussian.log_prob(value)

  def sample(self, sample_shape=(), seed=None, name='sample'):
    return self._gaussian.sample(sample_shape, seed=seed, name=name)


@functools.lru_cache(maxsize=None)
def get_target(name: str) -> gym.targets.Model:
  """Return the target name."""
  if name == 'german_credit_numeric_logistic_regression':
    return gym.targets.VectorModel(
        gym.targets.GermanCreditNumericLogisticRegression())
  elif name == 'german_credit_numeric_sparse_logistic_regression':
    return gym.targets.VectorModel(
        gym.targets.GermanCreditNumericSparseLogisticRegression(
            positive_constraint_fn='softplus'))
  elif name == 'brownian_motion':
    return gym.targets.VectorModel(
        gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations(
            use_markov_chain=True))
  elif name == 'item_response_theory':
    return gym.targets.VectorModel(gym.targets.SyntheticItemResponseTheory())
  elif name == 'radon_indiana':
    return gym.targets.VectorModel(gym.targets.RadonContextualEffectsIndiana())
  elif name == 'stochastic_volatility':
    return gym.targets.VectorModel(
        gym.targets.VectorizedStochasticVolatilitySP500())
  elif name == 'test_gaussian_1':
    return TestGaussian()
  elif name == 'test_gaussian_2':
    return TestGaussian(ndims=50)
  elif name == 'banana':
    return gym.targets.Banana()
  else:
    raise ValueError(f'Unknown target. {name}')


def load_inits(name: str, directory: str) -> Dict[str, np.ndarray]:
  with epath.Path(os.path.join(directory, name + '.npz')).open('rb') as f:
    return dict(np.load(f))


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


def snaper_criterion(  # pytype: disable=annotation-type-mismatch  # jax-ndarray
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

  previous_projection = jnp.einsum('d,nd->n', principal, previous_state)
  proposed_projection = jnp.einsum('d,nd->n', principal, proposed_state)

  esjd = ((previous_projection**2 - proposed_projection**2)**2)

  esjd = jnp.where(accept_prob > 1e-4, esjd, 0.)
  accept_prob = accept_prob / jnp.sum(accept_prob + 1e-20)
  esjd = esjd * accept_prob

  return esjd.mean() / trajectory_length**power, {
      'previous_projection': previous_projection,
      'proposed_projection': proposed_projection,
  }


class AdaptiveMCMCState(NamedTuple):
  """Adaptive MCMC state.

  Attributes:
    mcmc_state: MCMC state.
    rvar_state: Running variance of the state.
    proj_rautocov_state: Running lag-1 auto covariance of the projections.
    principal_rmean_state: Running mean of the unnormalized principal
      components.
    precond_principal_rmean_state: Running mean of the preconditioned
      unnormalized principal components.
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
  proj_rautocov_state: fun_mc.RunningCovarianceState
  principal_rmean_state: fun_mc.RunningMeanState
  precond_principal_rmean_state: fun_mc.RunningMeanState
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
    power: Power used in the SNAPER criterion.
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
  power: jnp.ndarray
  principal: jnp.ndarray
  max_eigenvalue: jnp.ndarray
  mean_trajectory_length: jnp.ndarray
  log_trajectory_length_opt_extra: fun_mc.AdamExtra
  num_integrator_steps: jnp.ndarray
  damping: jnp.ndarray
  extra: Any


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
                       rvar_smoothing: int,
                       method='hmc') -> AdaptiveMCMCState:
  """Initializes the adaptive MCMC algorithm.

  Args:
    state: Initial state of the MCMC chain.
    target_log_prob_fn: Target log prob function.
    init_step_size: Initial scalar step size.
    init_trajectory_length: Initial trajectory length.
    rvar_smoothing: Smoothing points.
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
                  jax.random.PRNGKey(0), state.shape[1:], state.dtype),
              num_points=rvar_smoothing),
      precond_principal_rmean_state=fun_mc.running_mean_init(
          state.shape[1:], state.dtype)._replace(
              mean=jax.random.normal(
                  jax.random.PRNGKey(0), state.shape[1:], state.dtype),
              num_points=rvar_smoothing),
      rvar_state=fun_mc.running_variance_init(
          state.shape[1:], state.dtype)._replace(num_points=rvar_smoothing),
      proj_rautocov_state=fun_mc.running_covariance_init(
          [2], state.dtype)._replace(num_points=rvar_smoothing),
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
    power: Optional[jnp.ndarray] = None,
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
    state_grad_estimator: str = 'two_dir',
    adapt_normalization_power: bool = False,
    mix_state_mean_in_snaper: bool = False,
    step_size_adaptation_rate_decay: str = 'none',
    trajectory_length_adaptation_rate_decay: str = 'none',
    rvar_smoothing: int = 0,
    jitter_style: str = 'halton',
    use_precond_eigenvalue: bool = False,
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
    method: MCMC method. Either 'hmc', 'malt'.
    damping: If not None, the fixed damping to use.
    scalar_step_size: If not None, the fixed scalar step size to use.
    vector_step_size: If not None, the fixed vector step size to use.
    mean_trajectory_length: If not None, the fixed mean trajectory length to
      use.
    power: Power used in the SNAPER criterion.
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
      'running_mean' or 'state_mean'. the running mean for preconditioning:
    state_grad_estimator: State grad estimator to use. Can be 'one_dir' or
      'two_dir'.
    adapt_normalization_power: Whether to adapt the power used for trajectory
      length normalization term in the snaper criterion.
    mix_state_mean_in_snaper: Whether to mix in the current state mean when
      computing the criterion.
    step_size_adaptation_rate_decay: How to decay the adaptation rate. One of
      'none' or 'linear'.
    trajectory_length_adaptation_rate_decay: How to decay the adaptation rate.
      One of 'none' or 'linear'.
    rvar_smoothing: Smoothing points.
    jitter_style: HMC jitter style. One of 'halton' or 'exponential'.
    use_precond_eigenvalue: Whether to use the preconditioned eigenvalue when
      computing damping.
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

  if power is None:
    if adapt_normalization_power:
      var = 0.5 * (
          amcmc_state.proj_rautocov_state.covariance[0, 0] +
          amcmc_state.proj_rautocov_state.covariance[1, 1])
      cov = amcmc_state.proj_rautocov_state.covariance[1, 0]
      power = 0.5 * (1. + cov / var)
    else:
      power = 1.

  if principal is None:
    max_eigenvalue = jnp.linalg.norm(amcmc_state.principal_rmean_state.mean)
    principal = amcmc_state.principal_rmean_state.mean / max_eigenvalue
  else:
    max_eigenvalue = jnp.linalg.norm(principal)
    principal = principal / max_eigenvalue

  if use_precond_eigenvalue:
    max_eigenvalue = jnp.linalg.norm(
        amcmc_state.precond_principal_rmean_state.mean)

  if method == 'hmc':
    if jitter_style == 'halton':
      traj_factor = halton(amcmc_state.step.astype(jnp.float32)) * 2.
    elif jitter_style == 'exponential':
      traj_factor = tfd.Exponential(1.).sample(
          seed=jax.random.PRNGKey(amcmc_state.step))  # pytype: disable=wrong-arg-types  # jax-ndarray
    elif jitter_style == 'halton_exponential':
      traj_factor = -jnp.log(halton(amcmc_state.step.astype(jnp.float32)))
  elif method == 'malt':
    traj_factor = 1.

  if mean_trajectory_length is None:
    mean_trajectory_length = jnp.where(
        adapt, jnp.exp(amcmc_state.log_trajectory_length_opt_state.state),
        amcmc_state.trajectory_length_rmean_state.mean)
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
      damping = damping_factor / (1e-10 + jnp.sqrt(max_eigenvalue))
    mcmc_state, mcmc_extra = fun_mc.prefab.metropolis_adjusted_langevin_trajectories_step(
        amcmc_state.mcmc_state,
        target_log_prob_fn=target_log_prob_fn,
        step_size=scalar_step_size * vector_step_size,
        momentum_refresh_fn=lambda m, seed:  # pylint: disable=g-long-lambda
        (
            malt_lib._gaussian_momentum_refresh_fn(  # pylint: disable=protected-access
                m,
                damping=damping,
                step_size=scalar_step_size / 2.,
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
      window_size=rvar_smoothing + num_chains * amcmc_state.step // rvar_factor)
  rvar_state = fun_mc.choose(adapt, cand_rvar_state, amcmc_state.rvar_state)

  # Adjust trajectory length.
  def log_trajectory_length_surrogate_loss_fn(log_trajectory_length):
    log_trajectory_length = fun_mc.clip_grads(log_trajectory_length, 1e6)
    mean_trajectory_length = jnp.exp(log_trajectory_length)
    trajectory_length = traj_factor * mean_trajectory_length

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

      if mix_state_mean_in_snaper:
        state_mean_weight = (amcmc_state.rvar_state.num_points) / (
            amcmc_state.rvar_state.num_points + num_chains)
      else:
        state_mean_weight = 1.

      criterion, extra = snaper_criterion(
          previous_state=start_state,
          proposed_state=end_state_with_action,
          accept_prob=jnp.exp(jnp.minimum(0., -mcmc_extra.log_accept_ratio)),
          trajectory_length=trajectory_length + scalar_step_size,
          principal=principal,
          power=power,
          # These two expressions are a bit weird for the reverse direction...
          state_mean=amcmc_state.rvar_state.mean,
          state_mean_weight=state_mean_weight,
      )
      criteria.append(criterion)
      extras.append(extra)
    return -jnp.array(criteria).mean(), extras

  cur_adaptation_rate = jnp.where(adapt, trajectory_length_adaptation_rate, 0.)
  if trajectory_length_adaptation_rate_decay == 'linear':
    cur_adaptation_rate = (
        (1. - 0.95 * amcmc_state.step / num_adaptation_steps) *
        cur_adaptation_rate)
  (log_trajectory_length_opt_state, log_trajectory_length_opt_extra) = (
      fun_mc.adam_step(
          amcmc_state.log_trajectory_length_opt_state,
          log_trajectory_length_surrogate_loss_fn,
          cur_adaptation_rate,
          **trajectory_opt_kwargs,
      ))
  log_trajectory_length_opt_state = log_trajectory_length_opt_state._replace(
      state=jnp.where(mala, jnp.log(scalar_step_size),
                      log_trajectory_length_opt_state.state))

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

  cur_adaptation_rate = jnp.where(adapt, step_size_adaptation_rate, 0.)
  if step_size_adaptation_rate_decay == 'linear':
    cur_adaptation_rate = (
        (1. - 0.95 * amcmc_state.step / num_adaptation_steps) *
        cur_adaptation_rate)
  (log_step_size_opt_state, _) = (
      fun_mc.adam_step(
          amcmc_state.log_step_size_opt_state,
          log_step_size_surrogate_loss_fn,
          cur_adaptation_rate,
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
      window_size=rvar_smoothing + amcmc_state.step // principal_factor,
  )
  principal_rmean_state = fun_mc.choose(adapt, cand_principal_rmean_state,
                                        amcmc_state.principal_rmean_state)

  cand_precond_principal_rmean_state, _ = fun_mc.running_mean_step(
      amcmc_state.precond_principal_rmean_state,
      ccipca((mcmc_state.state - mean) / vector_step_size,
             amcmc_state.precond_principal_rmean_state.mean),
      window_size=rvar_smoothing + amcmc_state.step // principal_factor,
  )
  precond_principal_rmean_state = fun_mc.choose(
      adapt, cand_precond_principal_rmean_state,
      amcmc_state.precond_principal_rmean_state)

  # Adjust auto-covariance of the squared projections.
  cand_proj_rautocov_state, _ = fun_mc.running_covariance_step(
      amcmc_state.proj_rautocov_state,
      jnp.stack([
          log_trajectory_length_opt_extra.loss_extra[0]['proposed_projection']**
          2,
          log_trajectory_length_opt_extra.loss_extra[0]['previous_projection']**
          2,
      ], -1),
      axis=0,
      window_size=rvar_smoothing + num_chains * amcmc_state.step // rvar_factor)
  proj_rautocov_state = fun_mc.choose(adapt, cand_proj_rautocov_state,
                                      amcmc_state.proj_rautocov_state)

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

  proposed_projection2 = ((proposed_state - mean) * principal).sum(-1)
  previous_projection2 = ((amcmc_state.mcmc_state.state - mean) *
                          principal).sum(-1)
  amcmc_state = amcmc_state._replace(
      mcmc_state=mcmc_state,
      rvar_state=rvar_state,
      proj_rautocov_state=proj_rautocov_state,
      principal_rmean_state=principal_rmean_state,
      precond_principal_rmean_state=precond_principal_rmean_state,
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
      power=power,
      max_eigenvalue=max_eigenvalue,
      damping=damping,
      mean_trajectory_length=mean_trajectory_length,
      log_trajectory_length_opt_extra=log_trajectory_length_opt_extra,
      num_integrator_steps=num_integrator_steps,
      extra={
          'rautocov':
              proj_rautocov_state.covariance,
          'proposed_projection':
              log_trajectory_length_opt_extra.loss_extra[0]
              ['proposed_projection'],
          'previous_projection':
              log_trajectory_length_opt_extra.loss_extra[0]
              ['previous_projection'],
          'proposed_projection2':
              proposed_projection2,
          'previous_projection2':
              previous_projection2,
          'mean':
              mean,
          'principal_rmean':
              principal_rmean_state.mean,
      },
  )
  return amcmc_state, amcmc_extra


class AdaptiveNUTSState(NamedTuple):
  nuts_state: Tuple[jnp.ndarray, Any]
  rvar_state: fun_mc.RunningVarianceState
  log_step_size_opt_state: fun_mc.AdamState
  step_size_rmean_state: fun_mc.RunningMeanState
  step: jnp.ndarray


class AdaptiveNUTSExtra(NamedTuple):
  scalar_step_size: jnp.ndarray
  vector_step_size: jnp.ndarray
  num_integrator_steps: jnp.ndarray
  mean_num_integrator_steps: jnp.ndarray
  log_accept_ratio: jnp.ndarray
  extra: Any


def adaptive_nuts_init(state: jnp.ndarray,
                       target_log_prob_fn: fun_mc.PotentialFn,
                       init_step_size: jnp.ndarray,
                       rvar_smoothing: int) -> AdaptiveNUTSState:
  """Initializes the Adaptive NUTS algorithm."""

  def tfp_target_log_prob_fn(x):
    return target_log_prob_fn(x)[0]

  trans_kernel = tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn=tfp_target_log_prob_fn,
      step_size=init_step_size * jnp.ones([state.shape[-1]]))

  nuts_state = (state, trans_kernel.bootstrap_results(state))

  return AdaptiveNUTSState(
      nuts_state=nuts_state,
      rvar_state=fun_mc.running_variance_init(
          state.shape[1:], state.dtype)._replace(num_points=rvar_smoothing),
      log_step_size_opt_state=fun_mc.adam_init(jnp.log(init_step_size)),
      step_size_rmean_state=fun_mc.running_mean_init([], jnp.float32),
      step=jnp.array(0, jnp.int32),
  )


def adaptive_nuts_step(
    anuts_state: AdaptiveNUTSState,
    target_log_prob_fn: fun_mc.PotentialFn,
    num_mala_steps: int,
    num_adaptation_steps: int,
    seed: jax.random.KeyArray,
    scalar_step_size: Optional[jnp.ndarray] = None,
    vector_step_size: Optional[jnp.ndarray] = None,
    rvar_factor: int = 8,
    iterate_factor: int = 16,
    target_accept_prob: float = 0.8,
    step_size_adaptation_rate: float = 0.05,
    rvar_smoothing: int = 0,
    step_size_adaptation_rate_decay: str = 'none',
    step_size_opt_kwargs: Mapping[str, Any] = immutabledict.immutabledict({}),
):
  """One step of Adaptive NUTS."""
  seed, mcmc_seed = jax.random.split(seed)

  def tfp_target_log_prob_fn(x):
    return target_log_prob_fn(x)[0]

  # ===========================
  # Compute common hyper-params
  # ===========================
  adapt = anuts_state.step < num_adaptation_steps
  num_chains = anuts_state.nuts_state[0].shape[0]

  if scalar_step_size is None:
    scalar_step_size = jnp.where(
        adapt, jnp.exp(anuts_state.log_step_size_opt_state.state),
        anuts_state.step_size_rmean_state.mean)
  if vector_step_size is None:
    vector_step_size = jnp.sqrt(anuts_state.rvar_state.variance)
    vector_step_size = vector_step_size / vector_step_size.max()

  trans_kernel = tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn=tfp_target_log_prob_fn,
      step_size=scalar_step_size * vector_step_size)
  trans_kernel._max_tree_depth = jnp.where(anuts_state.step < num_mala_steps, 1,  # pylint: disable=protected-access
                                           10)

  # ====================
  # Apply a step of MCMC
  # ====================
  state, kr = anuts_state.nuts_state
  kr = kr._replace(step_size=scalar_step_size * vector_step_size)
  nuts_state = trans_kernel.one_step(state, kr, seed=mcmc_seed)

  # ==========
  # Adaptation
  # ==========

  # Adjust running-variance estimate.
  cand_rvar_state, _ = fun_mc.running_variance_step(
      anuts_state.rvar_state,
      nuts_state[0],
      axis=0,
      window_size=rvar_smoothing + num_chains * anuts_state.step // rvar_factor)
  rvar_state = fun_mc.choose(adapt, cand_rvar_state, anuts_state.rvar_state)

  # Adjust step size.
  log_accept_ratio = nuts_state[1].log_accept_ratio
  min_log_accept_prob = jnp.full(log_accept_ratio.shape, jnp.log(1e-5))

  log_accept_prob = jnp.minimum(log_accept_ratio, jnp.zeros([]))
  log_accept_prob = jnp.maximum(log_accept_prob, min_log_accept_prob)
  log_accept_prob = jnp.where(
      jnp.isfinite(log_accept_prob), log_accept_prob, min_log_accept_prob)
  accept_prob = jnp.exp(tfp.math.reduce_log_harmonic_mean_exp(log_accept_prob))

  log_step_size_surrogate_loss_fn = fun_mc.make_surrogate_loss_fn(
      lambda _: (target_accept_prob - accept_prob, ()))

  cur_adaptation_rate = jnp.where(adapt, step_size_adaptation_rate, 0.)
  if step_size_adaptation_rate_decay == 'linear':
    cur_adaptation_rate = (
        (1. - 0.95 * anuts_state.step / num_adaptation_steps) *
        cur_adaptation_rate)
  (log_step_size_opt_state, _) = (
      fun_mc.adam_step(
          anuts_state.log_step_size_opt_state,
          log_step_size_surrogate_loss_fn,
          cur_adaptation_rate,
          **step_size_opt_kwargs,
      ))

  # =================
  # Iterate averaging
  # =================
  cand_step_size_rmean_state, _ = fun_mc.running_mean_step(
      anuts_state.step_size_rmean_state,
      scalar_step_size,
      window_size=anuts_state.step // iterate_factor)
  step_size_rmean_state = fun_mc.choose(adapt, cand_step_size_rmean_state,
                                        anuts_state.step_size_rmean_state)

  anuts_state = anuts_state._replace(
      nuts_state=nuts_state,
      rvar_state=rvar_state,
      log_step_size_opt_state=log_step_size_opt_state,
      step_size_rmean_state=step_size_rmean_state,
      step=anuts_state.step + 1,
  )
  anuts_extra = AdaptiveNUTSExtra(
      scalar_step_size=scalar_step_size,
      vector_step_size=vector_step_size,
      num_integrator_steps=nuts_state[1].leapfrogs_taken.max(),
      mean_num_integrator_steps=nuts_state[1].leapfrogs_taken.mean(),
      log_accept_ratio=log_accept_ratio,
      extra={},
  )
  return anuts_state, anuts_extra


def _get_transforms(
    mean: jnp.ndarray,
    principal: jnp.ndarray) -> Dict[str, Callable[[jnp.ndarray], jnp.ndarray]]:
  return {
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
  transforms = _get_transforms(mean, principal)

  res = {}
  for name, transform_fn in transforms.items():
    transformed_state = transform_fn(state)
    ess = jax.lax.map(tfp.mcmc.effective_sample_size,
                      jnp.transpose(transformed_state, [1, 0, 2])).mean(0)
    ess_per_step = ess / len(state)
    ess_per_grad = ess / num_grads
    rhat = tfp.mcmc.potential_scale_reduction(
        transformed_state, split_chains=True)

    white_state = transformed_state - transformed_state.mean(
        (0, 1)) / transformed_state.std((0, 1))
    lag_1_autocorr = (white_state[:-1] * white_state[1:]).sum((0, 1))

    stats = {}
    stats['mean'] = transformed_state.mean((0, 1))
    stats['ess'] = ess
    stats['ess_per_grad'] = ess_per_grad
    stats['ess_per_step'] = ess_per_step
    stats['rhat'] = rhat
    stats['lag_1_autocorr'] = lag_1_autocorr
    res[name] = stats

  return res


def compute_bias(
    state: jnp.ndarray,
    ground_truth: Dict[str, Dict[str, jnp.ndarray]],
    mean: jnp.ndarray,
    principal: jnp.ndarray,
) -> Dict[str, Dict[str, jnp.ndarray]]:
  """Compute the bias over time of an MCMC chain."""
  transforms = _get_transforms(mean, principal)

  res = {}
  for name, transform_fn in transforms.items():
    transformed_state = transform_fn(state)

    avg_state = tfp.stats.windowed_mean(transformed_state.mean(1))
    bias_sq = ((avg_state - ground_truth['stats'][name]['mean'])**2)

    stats = {}
    stats['bias_sq'] = bias_sq
    res[name] = stats

  return res


def estimate_largest_eigenvalue_of_covariance(x, remove_mean=True):
  """Estimate the largest eigenvalue of a covariance matrix given a sample.

  Implements Algorithm 2 from [1].

  We assume the rows of the input matrix `x` are i.i.d. draws from some
  distribution with (unknown) covariance matrix `sigma`. We want an estimate of
  the largest eigenvalue of `sigma`, but we cannot directly observe `sigma`.
  The naive estimator
  ```
  sigma_hat = x.T.dot(x) / (x.shape[0] - 1)
  max_eig = np.linalg.eigvalsh(sigma_hat).max()
  ```
  can be quite biased if the number of rows in x is not quite large, even
  though `sigma_hat` is unbiased. Instead, we compute the ratio of the sum
  of `sigma_hat`'s squared eigenvalues of to the sum of its eigenvalues.
  Although this estimator is asymptotically biased, it tends to be much more
  accurate than the naive estimator above unless `x` has a very large number
  of rows.

  [1] M.D. Hoffman and P. Sountsov, "Tuning-Free Generalized Hamiltonian Monte
    Carlo," AISTATS 2022.

  Args:
    x: A matrix whose rows are independent draws from the distribution whose
      covariance we are interested in.
    remove_mean: A boolean flag indicating whether or not to replace `x` with `x
      - x.mean(0)`.

  Returns:
    max_eig: An estimate of the largest eigenvalue of the covariance of the
      distribution that generated `x`.
  """
  x = x - remove_mean * x.mean(0)
  trace_est = (x**2).sum() / x.shape[0]
  # Note that this has a cost that's quadratic in num_chains. This should only
  # be an issue if cost_per_gradient < num_chains*num_dimensions. In this
  # presumably uncommon event, we could potentially use the Hutchinson
  # trace estimator.
  trace_sq_est = (x.dot(x.T)**2).sum() / x.shape[0]**2
  return trace_sq_est / trace_est


class MeadsState(NamedTuple):
  phmc_state: fun_mc.prefab.PersistentHamiltonianMonteCarloState
  fold_to_skip: jnp.ndarray
  step: jnp.ndarray


class MeadsExtra(NamedTuple):
  """Adapive MCMC extra outputs.

  Attributes:
    phmc_extra: MCMC extra outputs.
    scalar_step_size: Scalar step size.
    vector_step_size: Vector step size.
    max_eigenvalue: Maximum eigenvalue.
    num_integrator_steps: Number of integrator steps actually taken.
    damping: Damping that was used.
  """
  phmc_extra: fun_mc.prefab.PersistentHamiltonianMonteCarloExtra
  scalar_step_size: jnp.ndarray
  vector_step_size: jnp.ndarray
  max_eigenvalue: jnp.ndarray
  num_integrator_steps: jnp.ndarray
  damping: jnp.ndarray
  num_integrator_steps: jnp.ndarray
  level: jnp.ndarray
  extra: Any


def meads_init(state: jnp.ndarray, target_log_prob_fn: fun_mc.PotentialFn,
               num_folds: int, seed: jax.random.KeyArray):
  """Initializes MEADS."""
  num_dimensions = state.shape[-1]
  num_chains = state.shape[0]
  chains_per_fold = num_chains // num_folds
  state = state.reshape([num_folds, chains_per_fold, num_dimensions])

  m_seed, slice_seed = jax.random.split(seed)
  m = jax.random.normal(m_seed, state.shape)
  u = 2 * jax.random.uniform(slice_seed, state.shape[:-1]) - 1
  phmc_state = fun_mc.prefab.persistent_hamiltonian_monte_carlo_init(
      state, target_log_prob_fn, m, u)

  return MeadsState(
      phmc_state=phmc_state,
      fold_to_skip=jnp.array(0, jnp.int32),
      step=jnp.array(0, jnp.int32),
  )


def meads_step(meads_state: MeadsState,
               target_log_prob_fn: fun_mc.PotentialFn,
               seed: jax.random.KeyArray,
               vector_step_size: Optional[jnp.ndarray] = None,
               damping: Optional[jnp.ndarray] = None,
               step_size_multiplier: float = 0.5,
               damping_slowdown: float = 1.):
  """One step of MEADS."""
  phmc_seed, seed = jax.random.split(seed)

  num_folds = meads_state.phmc_state.state.shape[0]
  chains_per_fold = meads_state.phmc_state.state.shape[1]
  num_chains = num_folds * chains_per_fold
  step = meads_state.step
  fold_to_skip = meads_state.fold_to_skip
  phmc_state = meads_state.phmc_state

  # Randomly refold the walkers.
  perm = jax.random.permutation(jax.random.PRNGKey(step // 4), num_chains)  # pytype: disable=wrong-arg-types  # jax-ndarray
  # TODO(mhoffman): This should really done with a scatter.
  unperm = jnp.eye(num_chains)[perm].argmax(0)

  def refold(x, perm):
    return x.reshape((num_chains,) + x.shape[2:])[perm].reshape(x.shape)

  phmc_state = jax.tree_map(functools.partial(refold, perm=perm), phmc_state)

  if vector_step_size is None:
    vector_step_size = phmc_state.state.std(1, keepdims=True)
  else:
    vector_step_size = jnp.tile(vector_step_size[jnp.newaxis, jnp.newaxis],
                                [num_folds, 1, 1])
  # Apply preconditioning within-fold to estimate step size and damping.
  self_preconditioned_state = phmc_state.state / vector_step_size
  self_preconditioned_grads = phmc_state.state_grads * vector_step_size
  # Apply preconditioning across folds to do the actual update.
  rolled_vector_step_size = jnp.roll(vector_step_size, 1, 0)

  # Set step size for each fold based on the fold to its left.
  scalar_step_size = step_size_multiplier / jnp.sqrt(
      jax.vmap(
          functools.partial(
              estimate_largest_eigenvalue_of_covariance,
              remove_mean=False))(self_preconditioned_grads))
  scalar_step_size = jnp.minimum(1., scalar_step_size)
  scalar_step_size = jnp.roll(scalar_step_size, 1)

  # Set damping.
  max_eigenvalue = jax.vmap(estimate_largest_eigenvalue_of_covariance)(
      self_preconditioned_state)
  if damping is None:
    damping = scalar_step_size / jnp.sqrt(max_eigenvalue)
  else:
    damping = jnp.broadcast_to(damping, num_folds)

  # Put a floor on the amount of damping in early iterations.
  damping = jnp.maximum(damping_slowdown / step, damping)

  noise_fraction = (1 - jnp.exp(-2 * damping))**0.5
  mh_drift = 0.5 * noise_fraction**2

  # TODO(mhoffman): Consider using lax.gather instead of jnp.roll logic.
  # An advantage of roll is that it makes it clear to XLA that there's not
  # actually any dynamic sizing going on here.
  def select_folds(x):
    return jnp.roll(jnp.roll(x, -fold_to_skip, 0)[1:], fold_to_skip, 0)

  def rejoin_folds(updated, original):
    return jnp.roll(
        jnp.concatenate([
            original[fold_to_skip][jnp.newaxis],
            jnp.roll(updated, -fold_to_skip, 0)
        ], 0), fold_to_skip, 0)

  active_fold_state, phmc_extra = fun_mc.prefab.persistent_hamiltonian_monte_carlo_step(
      jax.tree_map(select_folds, phmc_state),
      target_log_prob_fn=target_log_prob_fn,
      step_size=select_folds(scalar_step_size[:, jnp.newaxis, jnp.newaxis] *
                             rolled_vector_step_size),
      num_integrator_steps=1,
      noise_fraction=select_folds(noise_fraction)[:, jnp.newaxis, jnp.newaxis],
      mh_drift=select_folds(mh_drift)[:, jnp.newaxis],
      seed=phmc_seed)
  phmc_state = jax.tree_map(rejoin_folds, active_fold_state, phmc_state)

  # Revert the ordering of the walkers.
  phmc_state = jax.tree_map(functools.partial(refold, perm=unperm), phmc_state)

  meads_state = MeadsState(
      phmc_state=phmc_state,
      fold_to_skip=(fold_to_skip + 1) % num_folds,
      step=step + 1,
  )

  extra = MeadsExtra(  # pytype: disable=wrong-arg-types  # jax-ndarray
      phmc_extra=phmc_extra,
      scalar_step_size=scalar_step_size,
      vector_step_size=vector_step_size,
      damping=damping,
      max_eigenvalue=max_eigenvalue,
      num_integrator_steps=1,
      level=phmc_state.pmh_state.level,
      extra={},
  )

  return meads_state, extra


def get_init_x(target: gym.targets.Model,
               num_chains: int,
               num_prior_samples: int = 256,
               method: str = 'prior_mean') -> jnp.ndarray:
  """Returns a 'good' initializer for MCMC chains.

  Args:
    target: Target model.
    num_chains: Number of chains to return.
    num_prior_samples: Number of prior samples to use when computing the prior
      mean.
    method: Method to use. Can be either 'prior_mean' or 'z_zero'.

  Returns:
    Initial position of the MCMC chain.
  """
  if method == 'prior_mean':
    if (isinstance(target, gym.targets.VectorModel) and
        hasattr(target.model, 'prior_distribution')):
      prior = target.model.prior_distribution()
      try:
        init_point = target.structured_event_to_vector(prior.mean())
      except (ValueError, NotImplementedError):
        prior_samples = target.structured_event_to_vector(
            prior.sample(num_prior_samples, seed=jax.random.PRNGKey(0)))
        init_point = prior_samples.mean(0)
    elif hasattr(target, 'prior_distribution'):
      prior = target.prior_distribution()
      try:
        init_point = prior.mean()
      except (ValueError, NotImplementedError):
        prior_samples = prior.sample(
            num_prior_samples, seed=jax.random.PRNGKey(0))
        init_point = prior_samples.mean(0)
    else:
      b = target.default_event_space_bijector
      init_point = b(
          jnp.zeros(b.inverse_event_shape(target.event_shape), target.dtype))
  elif method == 'z_zero':
    b = target.default_event_space_bijector
    init_point = b(
        jnp.zeros(b.inverse_event_shape(target.event_shape), target.dtype))

  return jnp.tile(init_point[jnp.newaxis], [num_chains, 1])


@gin.configurable
def run_adaptive_mcmc_on_target(
    target: gym.targets.Model,
    method: str,
    init_step_size: jnp.ndarray,
    num_adaptation_steps: int,
    num_results: int,
    seed: jax.random.KeyArray,
    num_mala_steps: int = 100,
    rvar_smoothing: int = 0,
    trajectory_opt_kwargs: Mapping[str, Any] = immutabledict.immutabledict({
        'beta_1': 0.,
        'beta_2': 0.95
    }),
    num_chains: Optional[int] = None,
    init_x: Optional[jnp.ndarray] = None,
    ground_truth: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
    save_warmup: bool = True,
    **amcmc_kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Does a run of adaptive MCMC.

  Args:
    target: Target model.
    method: MCMC method. Either 'hmc' or 'malt'.
    init_step_size: Initial scalar step size.
    num_adaptation_steps: Number of adaptation steps.
    num_results: Number of post-adaptation results.
    seed: Random seed.
    num_mala_steps: Number of initial steps to be MALA.
    rvar_smoothing: Smoothing points.
    trajectory_opt_kwargs: Kwargs sent to the trajectory length optimizer.
    num_chains: Number of chains. Optional if `init_x` is specified.
    init_x: Initial constrained chain position. Optional if `num_chains` is
      specified.
    ground_truth: Ground truth means.
    save_warmup: Whether to save the warmup iterations.
    **amcmc_kwargs: Extra kwargs to pass to Adaptive MCMC.

  Returns:
     A tuple of final and traced results.
  """
  if init_x is None:
    init_x = get_init_x(target, num_chains)
  init_z = target.default_event_space_bijector.inverse(init_x)

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
        method=method,
        trajectory_opt_kwargs=trajectory_opt_kwargs,
        rvar_smoothing=rvar_smoothing,
        **amcmc_kwargs,
    )

    traced = {
        'state': amcmc_state.mcmc_state.state,
        'is_accepted': amcmc_extra.mcmc_extra.is_accepted,
        'log_accept_ratio': amcmc_extra.mcmc_extra.log_accept_ratio,
        'scalar_step_size': amcmc_extra.scalar_step_size,
        'vector_step_size': amcmc_extra.vector_step_size,
        'principal': amcmc_extra.principal,
        'power': amcmc_extra.power,
        'max_eigenvalue': amcmc_extra.max_eigenvalue,
        'mean_trajectory_length': amcmc_extra.mean_trajectory_length,
        'num_integrator_steps': amcmc_extra.num_integrator_steps,
        'traj_loss': amcmc_extra.log_trajectory_length_opt_extra.loss,
        'damping': amcmc_extra.damping,
        'extra': amcmc_extra.extra,
    }

    return (amcmc_state, seed), traced

  amcmc_state = adaptive_mcmc_init(
      init_z,
      target_log_prob_fn,
      init_step_size=init_step_size,
      init_trajectory_length=init_step_size,
      rvar_smoothing=rvar_smoothing,
      method=method,
  )

  if save_warmup:
    _, trace = fun_mc.trace((amcmc_state, seed), kernel,
                            num_results + num_adaptation_steps)
  else:
    state, _ = fun_mc.trace((amcmc_state, seed),
                            kernel,
                            num_adaptation_steps,
                            trace_fn=lambda *_: ())
    _, trace = fun_mc.trace(state, kernel, num_results)

  warmed_up_steps = int(num_results * 0.8)
  warmed_up_state = trace['state'][-warmed_up_steps:]

  if ground_truth is None:
    cov = jnp.cov(
        warmed_up_state.reshape([-1, warmed_up_state.shape[-1]]), rowvar=False)
    principal = jnp.linalg.eigh(cov)[1][:, -1]
    mean = warmed_up_state.mean((0, 1))
    bias = {}
  else:
    principal = ground_truth['principal']
    mean = ground_truth['stats']['m1']['mean']
    if save_warmup:
      bias = {
          'bias': compute_bias(trace['state'], ground_truth, mean, principal)  # pytype: disable=wrong-arg-types  # jax-ndarray
      }
    else:
      bias = {}

  final = {
      'stats':
          compute_stats(
              state=warmed_up_state,
              num_grads=trace['num_integrator_steps'][-warmed_up_steps:].sum(),
              mean=mean,
              principal=principal,
          ),
      'final_x':
          target.default_event_space_bijector(trace['state'][-1]),
      **bias
  }
  return trace, final


@gin.configurable
def run_adaptive_nuts_on_target(
    target: gym.targets.Model,
    init_step_size: jnp.ndarray,
    num_adaptation_steps: int,
    num_results: int,
    seed: jax.random.KeyArray,
    num_mala_steps: int = 100,
    rvar_smoothing: int = 0,
    num_chains: Optional[int] = None,
    init_x: Optional[jnp.ndarray] = None,
    ground_truth: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
    save_warmup: bool = True,
    **anuts_kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Does a run of adaptive MCMC.

  Args:
    target: Target model.
    init_step_size: Initial scalar step size.
    num_adaptation_steps: Number of adaptation steps.
    num_results: Number of post-adaptation results.
    seed: Random seed.
    num_mala_steps: Number of initial steps to be MALA.
    rvar_smoothing: Smoothing points.
    num_chains: Number of chains. Optional if `init_x` is specified.
    init_x: Initial constrained chain position. Optional if `num_chains` is
      specified.
    ground_truth: Ground truth means.
    save_warmup: Whether to save the warmup iterations.
    **anuts_kwargs: Extra kwargs to pass to Adaptive MCMC.

  Returns:
     A tuple of final and traced results.
  """
  if init_x is None:
    init_x = get_init_x(target, num_chains)
  init_z = target.default_event_space_bijector.inverse(init_x)

  def target_log_prob_fn(x):
    return target.unnormalized_log_prob(x), ()

  target_log_prob_fn = fun_mc.transform_log_prob_fn(
      target_log_prob_fn, target.default_event_space_bijector)

  def kernel(anuts_state, seed):
    seed, mcmc_seed = jax.random.split(seed)

    anuts_state, anuts_extra = adaptive_nuts_step(
        anuts_state,
        target_log_prob_fn=target_log_prob_fn,
        num_mala_steps=num_mala_steps,
        num_adaptation_steps=num_adaptation_steps,
        seed=mcmc_seed,
        rvar_smoothing=rvar_smoothing,
        **anuts_kwargs,
    )
    traced = {
        'state': anuts_state.nuts_state[0],
        'log_accept_ratio': anuts_extra.log_accept_ratio,
        'scalar_step_size': anuts_extra.scalar_step_size,
        'vector_step_size': anuts_extra.vector_step_size,
        'num_integrator_steps': anuts_extra.num_integrator_steps,
        'mean_num_integrator_steps': anuts_extra.mean_num_integrator_steps,
        'extra': anuts_extra.extra,
    }

    return (anuts_state, seed), traced

  anuts_state = adaptive_nuts_init(
      init_z,
      target_log_prob_fn,
      init_step_size=init_step_size,
      rvar_smoothing=rvar_smoothing,
  )

  if save_warmup:
    _, trace = fun_mc.trace((anuts_state, seed), kernel,
                            num_results + num_adaptation_steps)
  else:
    state, _ = fun_mc.trace((anuts_state, seed),
                            kernel,
                            num_adaptation_steps,
                            trace_fn=lambda *_: ())
    _, trace = fun_mc.trace(state, kernel, num_results)

  warmed_up_steps = int(num_results * 0.8)
  warmed_up_state = trace['state'][-warmed_up_steps:]

  if ground_truth is None:
    cov = jnp.cov(
        warmed_up_state.reshape([-1, warmed_up_state.shape[-1]]), rowvar=False)
    principal = jnp.linalg.eigh(cov)[1][:, -1]
    mean = warmed_up_state.mean((0, 1))
    bias = {}
  else:
    principal = ground_truth['principal']
    mean = ground_truth['stats']['m1']['mean']
    if save_warmup:
      bias = {
          'bias': compute_bias(trace['state'], ground_truth, mean, principal)  # pytype: disable=wrong-arg-types  # jax-ndarray
      }
    else:
      bias = {}

  final = {
      'stats':
          compute_stats(
              state=warmed_up_state,
              num_grads=trace['num_integrator_steps'][-warmed_up_steps:].sum(),
              mean=mean,
              principal=principal,
          ),
      'final_x':
          target.default_event_space_bijector(trace['state'][-1]),
      **bias
  }
  return trace, final


@gin.configurable
def run_meads_on_target(
    target: gym.targets.Model,
    num_adaptation_steps: int,
    num_results: int,
    thinning: int,
    seed: jax.random.KeyArray,
    num_folds: int,
    num_chains: Optional[int] = None,
    init_x: Optional[jnp.ndarray] = None,
    ground_truth: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
    save_warmup: bool = True,
    **meads_kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Does a run of adaptive MCMC.

  Args:
    target: Target model.
    num_adaptation_steps: Number of adaptation steps.
    num_results: Number of post-adaptation results.
    thinning: How much to thin the chain by.
    seed: Random seed.
    num_folds: Number of folds.
    num_chains: Number of chains. Optional if `init_x` is specified.
    init_x: Initial constrained chain position. Optional if `num_chains` is
      specified.
    ground_truth: Ground truth means.
    save_warmup: Whether to save the warmup iterations.
    **meads_kwargs: Extra kwargs to pass to Adaptive MCMC.

  Returns:
     A tuple of final and traced results.
  """
  if init_x is None:
    init_x = get_init_x(target, num_chains)
  init_z = target.default_event_space_bijector.inverse(init_x)

  def target_log_prob_fn(x):
    return target.unnormalized_log_prob(x), ()

  target_log_prob_fn = fun_mc.transform_log_prob_fn(
      target_log_prob_fn, target.default_event_space_bijector)

  def kernel(meads_state, seed):
    seed, mcmc_seed = jax.random.split(seed)

    meads_state, meads_extra = meads_step(
        meads_state,
        target_log_prob_fn=target_log_prob_fn,
        seed=mcmc_seed,
        **meads_kwargs,
    )

    traced = {
        'state': meads_state.phmc_state.state.reshape([-1, init_z.shape[-1]]),
        'is_accepted': meads_extra.phmc_extra.is_accepted.reshape([-1]),
        'scalar_step_size': meads_extra.scalar_step_size.mean(0),
        'vector_step_size': meads_extra.vector_step_size.mean((0, 1)),
        'max_eigenvalue': meads_extra.max_eigenvalue.mean(0),
        'damping': meads_extra.damping.mean(0),
        'extra': meads_extra.extra,
    }

    return (meads_state, seed), traced

  def chunked_kernel(*state):
    state, traced = fun_mc.trace(state, kernel, thinning, trace_mask=False)
    traced = traced.copy()
    traced['num_integrator_steps'] = thinning
    return state, traced

  seed, init_seed = jax.random.split(seed)

  meads_state = meads_init(
      init_z,
      target_log_prob_fn,
      num_folds=num_folds,
      seed=init_seed,
  )

  if save_warmup:
    _, trace = fun_mc.trace((meads_state, seed), chunked_kernel,
                            num_results + num_adaptation_steps)
  else:
    state, _ = fun_mc.trace((meads_state, seed),
                            chunked_kernel,
                            num_adaptation_steps,
                            trace_fn=lambda *_: ())
    _, trace = fun_mc.trace(state, chunked_kernel, num_results)

  warmed_up_steps = int(num_results * 0.8)
  warmed_up_state = trace['state'][-warmed_up_steps:]

  if ground_truth is None:
    cov = jnp.cov(
        warmed_up_state.reshape([-1, warmed_up_state.shape[-1]]), rowvar=False)
    principal = jnp.linalg.eigh(cov)[1][:, -1]
    mean = warmed_up_state.mean((0, 1))
    bias = {}
  else:
    principal = ground_truth['principal']
    mean = ground_truth['stats']['m1']['mean']
    bias = {'bias': compute_bias(trace['state'], ground_truth, mean, principal)}  # pytype: disable=wrong-arg-types  # jax-ndarray

  final = {
      'stats':
          compute_stats(
              state=warmed_up_state,
              num_grads=trace['num_integrator_steps'][-warmed_up_steps:].sum(),
              mean=mean,
              principal=principal,
          ),
      'final_x':
          target.default_event_space_bijector(trace['state'][-1]),
      **bias
  }
  return trace, final


@gin.configurable
def run_fixed_mcmc_on_target(
    target: gym.targets.Model,
    init_x: jnp.ndarray,
    method: str,
    seed: jax.random.KeyArray,
    num_warmup_steps: int,
    num_results: int,
    scalar_step_size: jnp.ndarray,
    vector_step_size: jnp.ndarray,
    num_integrator_steps: jnp.ndarray,
    damping: Optional[jnp.ndarray] = None,
):
  """Run a fixed-hyperparameter MCMC.

  Args:
    target: Target model.
    init_x: Initial constrained chain state.
    method: MCMC method. Either 'hmc' or 'malt'.
    seed: Random seed.
    num_warmup_steps: Number of warmup steps.
    num_results: Number of post-warmup results.
    scalar_step_size: Scalar step size.
    vector_step_size: Vector step size.
    num_integrator_steps: Number of integrator steps.
    damping: Damping.

  Returns:
     A tuple of final and traced results.
  """
  init_z = target.default_event_space_bijector.inverse(init_x)

  def target_log_prob_fn(x):
    return target.unnormalized_log_prob(x), ()

  target_log_prob_fn = fun_mc.transform_log_prob_fn(
      target_log_prob_fn, target.default_event_space_bijector)

  def kernel(mcmc_state, seed):
    seed, jitter_seed, mcmc_seed = jax.random.split(seed, 3)

    if method == 'hmc':
      cur_num_integrator_steps = jax.random.randint(jitter_seed, [], 1,
                                                    num_integrator_steps + 1)
      mcmc_state, mcmc_extra = fun_mc.hamiltonian_monte_carlo_step(
          mcmc_state,
          target_log_prob_fn=target_log_prob_fn,
          step_size=scalar_step_size * vector_step_size,
          num_integrator_steps=cur_num_integrator_steps,
          seed=mcmc_seed,
      )
    elif method == 'malt':
      cur_num_integrator_steps = num_integrator_steps
      mcmc_state, mcmc_extra = fun_mc.prefab.metropolis_adjusted_langevin_trajectories_step(
          mcmc_state,
          target_log_prob_fn=target_log_prob_fn,
          step_size=scalar_step_size * vector_step_size,
          momentum_refresh_fn=lambda m, seed:  # pylint: disable=g-long-lambda
          (
              malt_lib._gaussian_momentum_refresh_fn(  # pylint: disable=protected-access
                  m,
                  damping=damping,
                  step_size=scalar_step_size / 2.,
                  seed=seed,
              )),
          num_integrator_steps=cur_num_integrator_steps,
          damping=damping,
          seed=mcmc_seed,
      )

    traced = {
        'state': mcmc_state.state,
        'is_accepted': mcmc_extra.is_accepted,
        'log_accept_ratio': mcmc_extra.log_accept_ratio,
        'num_integrator_steps': cur_num_integrator_steps,
    }

    return (mcmc_state, seed), traced

  if method == 'hmc':
    mcmc_state = fun_mc.hamiltonian_monte_carlo_init(
        state=init_z,
        target_log_prob_fn=target_log_prob_fn,
    )
  elif method == 'malt':
    mcmc_state = fun_mc.prefab.metropolis_adjusted_langevin_trajectories_init(
        state=init_z,
        target_log_prob_fn=target_log_prob_fn,
    )

  _, trace = fun_mc.trace((mcmc_state, seed), kernel,
                          num_warmup_steps + num_results)

  warmed_up_state = trace['state'][num_warmup_steps:]
  cov = jnp.cov(
      warmed_up_state.reshape([-1, warmed_up_state.shape[-1]]), rowvar=False)
  principal = jnp.linalg.eigh(cov)[1][:, -1]

  final = {
      'stats':
          compute_stats(
              state=warmed_up_state,
              num_grads=trace['num_integrator_steps'][-num_warmup_steps:].sum(),
              mean=warmed_up_state.mean((0, 1)),
              principal=principal,
          )
  }
  return trace, final


def run_vi_on_target(
    target: gym.targets.Model,
    init_x: jnp.ndarray,
    num_steps: int,
    learning_rate: float,
    seed: jax.random.KeyArray,
):
  """Run VI on a target.

  Args:
    target: Target model.
    init_x: Initial constrained chain state.
    num_steps: Number of steps to take.
    learning_rate: Learning rate for adam.
    seed: Random seed.

  Returns:
     A tuple of final and traced results.
  """
  init_z = target.default_event_space_bijector.inverse(init_x)

  def loss_fn(loc, isp_scale, seed):
    q = tfd.Independent(tfd.Normal(loc, jax.nn.softplus(isp_scale)), 1)
    q_sg = tfd.Independent(
        tfd.Normal(
            jax.lax.stop_gradient(loc),
            jax.nn.softplus(jax.lax.stop_gradient(isp_scale))), 1)

    z = q.sample(seed=seed)

    x = target.default_event_space_bijector(z)
    tlp = target.unnormalized_log_prob(x)
    elbo = tlp - q_sg.log_prob(z)

    return -elbo, ()

  def kernel(opt_state, seed):
    seed, vi_seed = jax.random.split(seed)
    opt_state, opt_extra = fun_mc.adam_step(
        opt_state, functools.partial(loss_fn, seed=vi_seed), learning_rate)

    traced = {
        'loc': opt_state.state[0],
        'isp_scale': opt_state.state[1],
        'loss': opt_extra.loss,
    }

    return (opt_state, seed), traced

  opt_state = fun_mc.adam_init(
      (init_z, tfp.math.softplus_inverse(jnp.full_like(init_z, 1e-3))))

  _, trace = fun_mc.trace((opt_state, seed), kernel, num_steps)

  final = {
      'final_x': target.default_event_space_bijector.forward(trace['loc'][-1]),
  }
  return trace, final


@functools.partial(
    jax.jit,
    static_argnames=('target', 'method', 'num_adaptation_steps', 'num_results',
                     'num_chains', 'jitter_style'))
def _run_grid_element_impl(seed, inits, target, method, num_adaptation_steps,
                           num_results, mean_trajectory_length, damping,
                           num_chains, jitter_style, target_accept_prob):
  """Implementation of run_grid_element."""
  init_z = inits['init_z']
  if num_chains is not None:
    if num_chains <= init_z.shape[0]:
      init_z = init_z[:num_chains]
    else:
      init_z = jnp.repeat(init_z, int(np.ceil(num_chains / init_z.shape[0])), 0)
      init_z = init_z[:num_chains]
  init_x = target.default_event_space_bijector.forward(init_z)

  if method == 'meads':
    trace, final = run_meads_on_target(
        target=target,
        init_x=init_x,
        seed=seed,
        num_adaptation_steps=num_adaptation_steps,
        num_results=num_results,
        thinning=10,
        num_folds=4,
        step_size_multiplier=0.5,
        damping_slowdown=1.,
        vector_step_size=inits['vector_step_size'],
        damping=damping,
    )
  else:
    trace, final = run_adaptive_mcmc_on_target(
        target=target,
        init_x=init_x,
        method=method,
        seed=seed,
        num_adaptation_steps=num_adaptation_steps,
        num_results=num_results,
        step_size_adaptation_rate_decay='linear',
        target_accept_prob=target_accept_prob,
        init_step_size=inits['scalar_step_size'],
        vector_step_size=inits['vector_step_size'],
        mean_trajectory_length=mean_trajectory_length,
        jitter_style=jitter_style,
        damping=damping,
    )

  res = {
      'stats': final['stats'],
      'scalar_step_size': trace['scalar_step_size'][-1]
  }
  return res


@gin.configurable
def run_grid_element(mean_trajectory_length: jnp.ndarray,
                     damping: jnp.ndarray,
                     target_name: str,
                     method: str,
                     num_replicas: int,
                     seed: int = 0,
                     num_adaptation_steps: int = 1000,
                     num_results: int = 1000,
                     num_chains: Optional[int] = None,
                     jitter_style: str = 'halton',
                     target_accept_prob: float = 0.7,
                     inits_dir: str = '') -> Dict[str, Any]:
  """Runs a grid search element."""
  target = get_target(target_name)
  inits = load_inits(target_name, inits_dir)

  seed = jax.random.PRNGKey(seed)
  res = []
  for i in range(num_replicas):
    with utils.delete_device_buffers():
      res.append(
          jax.tree_map(
              np.array,
              _run_grid_element_impl(
                  seed=jax.random.fold_in(seed, i),
                  inits=inits,
                  target=target,
                  method=method,
                  num_adaptation_steps=num_adaptation_steps,
                  num_results=num_results,
                  mean_trajectory_length=mean_trajectory_length,
                  damping=damping,
                  num_chains=num_chains,
                  jitter_style=jitter_style,
                  target_accept_prob=target_accept_prob,
              )))
  res = jax.tree_map(lambda *x: np.stack(x, 0), *res)
  res['mean_trajectory_length'] = mean_trajectory_length
  res['damping'] = damping

  return res


@functools.partial(
    jax.jit,
    static_argnames=('target', 'method', 'num_adaptation_steps', 'num_results',
                     'adapt_normalization_power', 'num_chains', 'jitter_style',
                     'step_size_adaptation_rate_decay',
                     'trajectory_length_adaptation_rate_decay', 'save_warmup'))
def _run_trial_impl(seed, inits, ground_truth, target, method,
                    num_adaptation_steps, num_results, num_chains, jitter_style,
                    adapt_normalization_power, target_accept_prob,
                    step_size_adaptation_rate_decay,
                    trajectory_length_adaptation_rate_decay, save_warmup):
  """Implementation of run_trial."""
  init_z = inits['init_z']
  if num_chains is not None:
    if num_chains <= init_z.shape[0]:
      init_z = init_z[:num_chains]
    else:
      init_z = jnp.repeat(init_z, int(np.ceil(num_chains / init_z.shape[0])), 0)
      init_z = init_z[:num_chains]
  init_x = target.default_event_space_bijector.forward(init_z)

  if method == 'meads':
    trace, final = run_meads_on_target(
        target=target,
        init_x=init_x,
        seed=seed,
        num_adaptation_steps=num_adaptation_steps,
        num_results=num_results,
        thinning=10,
        num_folds=4,
        step_size_multiplier=0.5,
        damping_slowdown=1.,
        ground_truth=ground_truth,
        save_warmup=save_warmup,
    )
  elif method == 'nuts':
    trace, final = run_adaptive_nuts_on_target(
        target=target,
        init_x=init_x,
        seed=seed,
        num_adaptation_steps=num_adaptation_steps,
        num_results=num_results,
        rvar_factor=8,
        rvar_smoothing=1,
        init_step_size=1e-2,
        step_size_adaptation_rate_decay=step_size_adaptation_rate_decay,
        target_accept_prob=0.7,
        ground_truth=ground_truth,
        save_warmup=save_warmup,
    )
  else:
    trace, final = run_adaptive_mcmc_on_target(
        target=target,
        init_x=init_x,
        method=method,
        seed=seed,
        num_adaptation_steps=num_adaptation_steps,
        num_results=num_results,
        init_step_size=1e-2,
        trajectory_length_adaptation_rate=0.05,
        trajectory_length_adaptation_rate_decay=trajectory_length_adaptation_rate_decay,
        step_size_adaptation_rate_decay=step_size_adaptation_rate_decay,
        target_accept_prob=target_accept_prob,
        adapt_normalization_power=adapt_normalization_power,
        use_precond_eigenvalue=True,
        trajectory_opt_kwargs={
            'beta_1': 0.,
            'beta_2': 0.95,
            'epsilon': 1e-8
        },
        rvar_factor=8,
        principal_factor=3,
        rvar_smoothing=1,
        jitter_style=jitter_style,
        ground_truth=ground_truth,
        save_warmup=save_warmup,
    )

  res = {
      'stats': final['stats'],
      'num_integrator_steps': trace['num_integrator_steps'],
      'scalar_step_size': trace['scalar_step_size'],
      'vector_step_size': trace['vector_step_size'][-1],
  }
  if save_warmup:
    rel_max_m2_bias_sq = (final['bias']['m2']['bias_sq'] /
                          ground_truth['stats']['m2']['mean']**2).max(-1)
    res['rel_max_m2_bias_sq'] = rel_max_m2_bias_sq
  if 'max_eigenvalue' in trace:
    res['max_eigenvalue'] = trace['max_eigenvalue']
  if 'mean_trajectory_length' in trace:
    res['mean_trajectory_length'] = trace['mean_trajectory_length']
  if 'damping' in trace and trace['damping'] is not None:
    res['damping'] = trace['damping']
  if 'mean_num_integrator_steps' in trace:
    res['mean_num_integrator_steps'] = trace['mean_num_integrator_steps']
  return res


@gin.configurable
def run_trial(
    target_name: str,
    method: str,
    num_replicas: int,
    seed: int = 0,
    num_adaptation_steps: int = 1000,
    num_results: int = 1000,
    num_chains: Optional[int] = None,
    jitter_style: str = 'halton',
    target_accept_prob: float = 0.7,
    adapt_normalization_power: bool = False,
    step_size_adaptation_rate_decay: str = 'linear',
    trajectory_length_adaptation_rate_decay: str = 'linear',
    inits_dir: str = '',
    ground_truth_dir: str = '',
    save_warmup: bool = True,
) -> Dict[str, Any]:
  """Runs a grid search element."""
  target = get_target(target_name)
  inits = load_inits(target_name, inits_dir)
  ground_truth = utils.h5_to_dict(
      utils.load_h5py(os.path.join(ground_truth_dir, f'{target_name}.h5')))

  seed = jax.random.PRNGKey(seed)
  res = []
  for i in range(num_replicas):
    with utils.delete_device_buffers():
      res.append(
          jax.tree_map(
              np.array,
              _run_trial_impl(
                  seed=jax.random.fold_in(seed, i),
                  inits=inits,
                  ground_truth=ground_truth,
                  target=target,
                  method=method,
                  adapt_normalization_power=adapt_normalization_power,
                  num_adaptation_steps=num_adaptation_steps,
                  num_results=num_results,
                  num_chains=num_chains,
                  jitter_style=jitter_style,
                  target_accept_prob=target_accept_prob,
                  step_size_adaptation_rate_decay=step_size_adaptation_rate_decay,
                  trajectory_length_adaptation_rate_decay=trajectory_length_adaptation_rate_decay,
                  save_warmup=save_warmup,
              )))
  res = jax.tree_map(lambda *x: np.stack(x, 0), *res)
  return res
