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
"""Implementation of MALT [1].

#### References

[1]: Riou-Durand, L., & Vogrinc, J. (2022). Metropolis Adjusted Langevin
     Trajectories: a robust alternative to Hamiltonian Monte Carlo.
     http://arxiv.org/abs/2202.13230
"""

import functools
from typing import Any, Callable, NamedTuple, Optional, Tuple

from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc

jax = backend.jax
jnp = backend.jnp
tfp = backend.tfp
util = backend.util
distribute_lib = backend.distribute_lib

__all__ = [
    'metropolis_adjusted_langevin_trajectories_init',
    'metropolis_adjusted_langevin_trajectories_step',
    'MetropolisAdjustedLangevinTrajectoriesExtra',
    'MetropolisAdjustedLangevinTrajectoriesState',
]


def _gaussian_momentum_refresh_fn(
    old_momentum: fun_mc.State,
    damping: Optional[float | fun_mc.FloatArray] = 0.0,
    step_size: Optional[float | fun_mc.FloatArray] = 1.0,
    named_axis: Optional[fun_mc.StringNest] = None,
    seed: Optional[Any] = None,
) -> fun_mc.State:
  """Momentum refresh function for Gaussian momentum distribution."""
  if named_axis is None:
    named_axis = util.map_tree(lambda _: [], old_momentum)
  damping = fun_mc.maybe_broadcast_structure(damping, old_momentum)
  step_size = fun_mc.maybe_broadcast_structure(step_size, old_momentum)
  decay_fraction = util.map_tree(
      lambda d, s: jnp.exp(-d * s), damping, step_size
  )
  noise_fraction = util.map_tree(
      lambda df: jnp.sqrt(1.0 - jnp.square(df)), decay_fraction
  )

  def _sample_part(
      old_momentum, seed, named_axis, decay_fraction, noise_fraction
  ):
    seed = backend.distribute_lib.fold_in_axis_index(seed, named_axis)
    return decay_fraction * old_momentum + noise_fraction * util.random_normal(
        old_momentum.shape, old_momentum.dtype, seed
    )

  seeds = util.unflatten_tree(
      old_momentum, util.split_seed(seed, len(util.flatten_tree(old_momentum)))
  )
  new_momentum = util.map_tree_up_to(
      old_momentum,
      _sample_part,
      old_momentum,
      seeds,
      named_axis,
      decay_fraction,
      noise_fraction,
  )
  return new_momentum


def _default_energy_change_fn(
    old_int_state: fun_mc.IntegratorState,
    new_int_state: fun_mc.IntegratorState,
    kinetic_energy_fn: Optional[fun_mc.PotentialFn],
) -> Tuple[fun_mc.FloatArray, Tuple[Any, Any]]:
  """Default energy change function."""
  old_kinetic_energy, old_kinetic_energy_extra = fun_mc.call_potential_fn(
      kinetic_energy_fn, old_int_state.momentum
  )
  new_kinetic_energy, new_kinetic_energy_extra = fun_mc.call_potential_fn(
      kinetic_energy_fn, new_int_state.momentum
  )

  old_energy = -old_int_state.target_log_prob + old_kinetic_energy
  new_energy = -new_int_state.target_log_prob + new_kinetic_energy

  return new_energy - old_energy, (
      old_kinetic_energy_extra,
      new_kinetic_energy_extra,
  )


class MetropolisAdjustedLangevinTrajectoriesState(NamedTuple):
  """Integrator state."""

  state: fun_mc.State
  state_extra: Any
  state_grads: fun_mc.State
  target_log_prob: fun_mc.FloatArray


class MetropolisAdjustedLangevinTrajectoriesExtra(NamedTuple):
  """Hamiltonian Monte Carlo extra outputs."""

  is_accepted: fun_mc.BooleanArray
  log_accept_ratio: fun_mc.FloatArray
  proposed_malt_state: fun_mc.State
  integrator_state: fun_mc.IntegratorState
  integrator_extra: fun_mc.IntegratorExtras
  initial_momentum: fun_mc.State


def metropolis_adjusted_langevin_trajectories_init(
    state: fun_mc.State, target_log_prob_fn: fun_mc.PotentialFn
) -> MetropolisAdjustedLangevinTrajectoriesState:
  """Initializes the `MetropolisAdjustedLangevinTrajectoriesState`.

  Args:
    state: State of the chain.
    target_log_prob_fn: Target log prob fn.

  Returns:
    malt_state: State of the `metropolis_adjusted_langevin_trajectories_step`
      `TransitionOperator`.
  """
  state = util.map_tree(jnp.array, state)
  target_log_prob, state_extra, state_grads = util.map_tree(
      jnp.array,
      fun_mc.call_potential_fn_with_grads(target_log_prob_fn, state),
  )
  return MetropolisAdjustedLangevinTrajectoriesState(
      state=state,
      state_grads=state_grads,
      target_log_prob=target_log_prob,
      state_extra=state_extra,
  )


def metropolis_adjusted_langevin_trajectories_step(
    malt_state: MetropolisAdjustedLangevinTrajectoriesState,
    target_log_prob_fn: fun_mc.PotentialFn,
    step_size: Optional[Any] = None,
    num_integrator_steps: Optional[fun_mc.IntArray] = None,
    damping: Optional[fun_mc.FloatArray] = None,
    momentum: Optional[fun_mc.State] = None,
    integrator_trace_fn: Optional[
        Callable[
            [
                fun_mc.IntegratorState,
                fun_mc.IntegratorStepState,
                fun_mc.IntegratorStepExtras,
            ],
            fun_mc.ArrayNest,
        ]
    ] = None,
    unroll_integrator: bool = False,
    log_uniform: Optional[fun_mc.FloatArray] = None,
    kinetic_energy_fn: Optional[fun_mc.PotentialFn] = None,
    momentum_sample_fn: Optional[fun_mc.MomentumSampleFn] = None,
    momentum_refresh_fn: Optional[
        Callable[[fun_mc.State, Any], fun_mc.State]
    ] = None,
    energy_change_fn: Optional[
        Callable[
            [fun_mc.IntegratorState, fun_mc.IntegratorState],
            Tuple[fun_mc.FloatArray, Any],
        ]
    ] = None,
    integrator_fn: Optional[
        Callable[
            [fun_mc.IntegratorState, Any],
            Tuple[fun_mc.IntegratorState, fun_mc.IntegratorExtras],
        ]
    ] = None,
    named_axis: Optional[fun_mc.StringNest] = None,
    seed: Any = None,
) -> Tuple[
    MetropolisAdjustedLangevinTrajectoriesState,
    MetropolisAdjustedLangevinTrajectoriesExtra,
]:
  """MALT `TransitionOperator`.

  This implements the Metropolis Adjusted Langevin Trajectories (MALT) algorithm
  from [1]. By default, an isotropic Gaussian distribution is used for the
  kinetic energy. If changed, make sure to change `momentum_sample_fn` and
  `momentum_refresh_fn` callbacks to ensure they describe the same distribution.

  #### Example

  ```python
  step_size = 0.2
  num_steps = 2000
  num_integrator_steps = 10
  damping = 0.5
  state = tf.ones([16, 2])

  base_mean = [1., 0]
  base_cov = [[1, 0.5], [0.5, 1]]

  bijector = tfb.Softplus()
  base_dist = tfd.MultivariateNormalFullCovariance(
      loc=base_mean, covariance_matrix=base_cov)
  target_dist = bijector(base_dist)

  def orig_target_log_prob_fn(x):
    return target_dist.log_prob(x), ()

  target_log_prob_fn, state = fun_mc.transform_log_prob_fn(
      orig_target_log_prob_fn, bijector, state)

  kernel = tf.function(lambda state: (
          fun_mc.prefab.metropolis_adjusted_langevin_trajectories_step(
              state,
              step_size=step_size,
              num_integrator_steps=num_integrator_steps,
              damping=damping,
              target_log_prob_fn=target_log_prob_fn)))

  _, chain = fun_mc.trace(
      state=fun_mc.prefab.metropolis_adjusted_langevin_trajectories_init(
          state, target_log_prob_fn),
      fn=kernel,
      num_steps=num_steps,
      trace_fn=lambda state, extra: state.state_extra[0])
  ```

  Args:
    malt_state: HamiltonianMonteCarloState.
    target_log_prob_fn: Target log prob fn.
    step_size: Step size, structure broadcastable to the `target_log_prob_fn`
      state. Optional if `integrator_fn` is specified.
    num_integrator_steps: Number of integrator steps to take. Optional if
      `integrator_fn` is specified.
    damping: Dampening parameter for the momentum refreshment. Optional if
      `momentum_refresh_fn` is specified.
    momentum: Initial momentum, by default sampled from a standard gaussian.
    integrator_trace_fn: Trace function for the integrator.
    unroll_integrator: Whether to unroll the loop in the integrator. Only works
      if `num_integrator_steps` is statically known. Ignored if `integrator_fn`
      is specified.
    log_uniform: Optional logarithm of a uniformly distributed random sample in
      [0, 1], used for the MH accept/reject step.
    kinetic_energy_fn: Kinetic energy function.
    momentum_sample_fn: Sampler for the momentum.
    momentum_refresh_fn: Refreshment step for the momentum.
    energy_change_fn: Callable with signature: `(current_integrator_state,
      proposed_integrator_state,) -> (energy_change, energy_change_extra)`.
      Computes the change in energy between current and proposed states.
    integrator_fn: Integrator to use for the Langevin dynamics.
    named_axis: Named axes of the state, same structure as `malt_state.state`.
    seed: For reproducibility.

  Returns:
    malt_state: MetropolisAdjustedLangevinTrajectoriesState
    malt_extra: MetropolisAdjustedLangevinTrajectoriesExtra


  #### References

  [1]: Riou-Durand, L., & Vogrinc, J. (2022). Metropolis Adjusted Langevin
       Trajectories: a robust alternative to Hamiltonian Monte Carlo.
       http://arxiv.org/abs/2202.13230
  """
  target_log_prob = malt_state.target_log_prob

  if integrator_fn is None:
    if kinetic_energy_fn is None:
      kinetic_energy_fn = fun_mc.make_gaussian_kinetic_energy_fn(
          (
              len(target_log_prob.shape)
              if target_log_prob.shape is not None
              else len(target_log_prob.shape)
          ),  # pytype: disable=attribute-error
          named_axis=named_axis,
      )
    if energy_change_fn is None:
      energy_change_fn = lambda old_is, new_is: _default_energy_change_fn(  # pylint: disable=g-long-lambda
          old_is, new_is, kinetic_energy_fn
      )
    if momentum_sample_fn is None:
      momentum_sample_fn = lambda seed: fun_mc.gaussian_momentum_sample(  # pylint: disable=g-long-lambda
          state=malt_state.state, seed=seed, named_axis=named_axis
      )
    if momentum_refresh_fn is None:
      momentum_refresh_fn = lambda m, seed: _gaussian_momentum_refresh_fn(  # pylint: disable=g-long-lambda
          m,
          seed=seed,
          step_size=step_size / 2.0,
          damping=damping,
          named_axis=named_axis,
      )
    integrator_fn = lambda int_state, seed: fun_mc.obabo_langevin_integrator(  # pylint: disable=g-long-lambda
        int_state=int_state,
        num_steps=num_integrator_steps,
        integrator_step_fn=functools.partial(
            fun_mc.leapfrog_step,
            step_size=step_size,
            target_log_prob_fn=target_log_prob_fn,
            kinetic_energy_fn=kinetic_energy_fn,
        ),
        momentum_refresh_fn=momentum_refresh_fn,
        integrator_trace_fn=integrator_trace_fn,
        energy_change_fn=energy_change_fn,
        unroll=unroll_integrator,
        seed=seed,
    )

  mh_seed, sample_seed, integrator_seed = util.split_seed(seed, 3)
  if momentum is None:
    momentum = momentum_sample_fn(sample_seed)

  initial_integrator_state = fun_mc.IntegratorState(
      state=malt_state.state,
      state_extra=malt_state.state_extra,
      state_grads=malt_state.state_grads,
      target_log_prob=malt_state.target_log_prob,
      momentum=momentum,
  )

  integrator_state, integrator_extra = integrator_fn(
      initial_integrator_state, integrator_seed
  )

  proposed_state = MetropolisAdjustedLangevinTrajectoriesState(
      state=integrator_state.state,
      state_grads=integrator_state.state_grads,
      target_log_prob=integrator_state.target_log_prob,
      state_extra=integrator_state.state_extra,
  )

  malt_state, mh_extra = fun_mc.metropolis_hastings_step(
      malt_state,
      proposed_state,
      integrator_extra.energy_change,
      log_uniform=log_uniform,
      seed=mh_seed,
  )

  return malt_state, MetropolisAdjustedLangevinTrajectoriesExtra(
      is_accepted=mh_extra.is_accepted,
      proposed_malt_state=proposed_state,
      log_accept_ratio=-integrator_extra.energy_change,
      integrator_state=integrator_state,
      integrator_extra=integrator_extra,
      initial_momentum=momentum,
  )
