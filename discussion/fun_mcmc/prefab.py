# Copyright 2020 The TensorFlow Probability Authors.
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
"""Prefabricated transition operators.

These are compositions of FunMC API into slightly more robust, 'turn-key',
opinionated inference and optimization algorithms. They are not themselves meant
to be infinitely composable, or as flexible as the base FunMC API. When
flexibility is required, the users are encouraged to copy paste the
implementations and adjust the pieces that they want. The code in this module
strives to be easy to copy paste and modify in this manner.
"""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import collections
import functools

import numpy as np

from discussion.fun_mcmc import backend
from discussion.fun_mcmc import fun_mcmc_lib as fun_mc
from typing import Any, Optional, Tuple

tf = backend.tf
tfp = backend.tfp
util = backend.util
tfb = tfp.bijectors

__all__ = [
    'AdaptiveHamiltonianMonteCarloState',
    'adaptive_hamiltonian_monte_carlo_init',
    'adaptive_hamiltonian_monte_carlo_step',
]


def _polynomial_decay(step: 'fun_mc.AnyTensor',
                      step_size: 'fun_mc.FloatTensor',
                      decay_steps: 'fun_mc.AnyTensor',
                      final_step_size: 'fun_mc.FloatTensor',
                      power: 'fun_mc.FloatTensor' = 1.) -> 'fun_mc.FloatTensor':
  """Polynomial decay step size schedule."""
  step_size = tf.convert_to_tensor(step_size)
  step_f = tf.cast(step, step_size.dtype)
  decay_steps_f = tf.cast(decay_steps, step_size.dtype)
  step_mult = (1. - step_f / decay_steps_f)**power
  step_mult = tf.where(step >= decay_steps, tf.zeros_like(step_mult), step_mult)
  return step_mult * (step_size - final_step_size) + final_step_size


AdaptiveHamiltonianMonteCarloState = collections.namedtuple(
    'AdaptiveHamiltonianMonteCarloState',
    'hmc_state, running_var_state, log_step_size_opt_state, step')


class AdaptiveHamiltonianMonteCarloExtra(
    collections.namedtuple(
        'AdaptiveHamiltonianMonteCarloExtra',
        'hmc_state, hmc_extra, step_size, num_leapfrog_steps')):
  """Extra outputs for Adaptive HMC `TransitionOperator`."""
  __slots__ = ()

  @property
  def state(self) -> 'fun_mc.TensorNest':
    """Returns the chain state.

    Note that this assumes that `target_log_prob_fn` has the `state_extra` be a
    tuple, with the first element thereof containing the state in the original
    space.
    """
    return self.hmc_state.state_extra[0]

  @property
  def is_accepted(self) -> 'fun_mc.BooleanTensor':
    return self.hmc_extra.is_accepted


def adaptive_hamiltonian_monte_carlo_init(
    state: 'fun_mc.TensorNest',
    target_log_prob_fn: 'fun_mc.PotentialFn',
    step_size: 'fun_mc.FloatTensor' = 1e-2,
    initial_mean: 'fun_mc.FloatNest' = 0.,
    initial_scale: 'fun_mc.FloatNest' = 1.,
    scale_smoothing_steps: 'fun_mc.IntTensor' = 10,
) -> 'AdaptiveHamiltonianMonteCarloState':
  """Initializes `AdaptiveHamiltonianMonteCarloState`.

  Args:
    state: Initial state of the chain.
    target_log_prob_fn: Target log prob fn.
    step_size: Initial scalar step size.
    initial_mean: Initial mean for computing the running variance estimate. Must
      broadcast structurally and tensor-wise with state.
    initial_scale: Initial scale for computing the running variance estimate.
      Must broadcast structurally and tensor-wise with state.
    scale_smoothing_steps: How much weight to assign to the `initial_mean` and
      `initial_scale`. Increase this to stabilize early adaptation.

  Returns:
    adaptive_hmc_state: State of the `adaptive_hamiltonian_monte_carlo_step`
      `TransitionOperator`.
  """
  hmc_state = fun_mc.hamiltonian_monte_carlo_init(state, target_log_prob_fn)
  dtype = util.flatten_tree(hmc_state.state)[0].dtype
  chain_ndims = len(hmc_state.target_log_prob.shape)
  running_var_state = fun_mc.running_variance_init(
      shape=util.map_tree(lambda s: s.shape[chain_ndims:], hmc_state.state),
      dtype=util.map_tree(lambda s: s.dtype, hmc_state.state),
  )
  initial_mean = fun_mc.maybe_broadcast_structure(initial_mean, state)
  initial_scale = fun_mc.maybe_broadcast_structure(initial_scale, state)

  # It's important to add some smoothing here, as initial updates can be very
  # different than the stationary distribution.
  # TODO(siege): Add a pseudo-update functionality to avoid fiddling with the
  # internals here.
  running_var_state = running_var_state._replace(
      num_points=util.map_tree(
          lambda p: (  # pylint: disable=g-long-lambda
              int(np.prod(hmc_state.target_log_prob.shape)) * tf.cast(
                  scale_smoothing_steps, p.dtype)),
          running_var_state.num_points),
      mean=util.map_tree(  # pylint: disable=g-long-lambda
          lambda m, init_m: tf.ones_like(m) * init_m, running_var_state.mean,
          initial_mean),
      variance=util.map_tree(  # pylint: disable=g-long-lambda
          lambda v, init_s: tf.ones_like(v) * init_s**2,
          running_var_state.variance, initial_scale),
  )
  log_step_size_opt_state = fun_mc.adam_init(
      tf.math.log(tf.convert_to_tensor(step_size, dtype=dtype)))

  return AdaptiveHamiltonianMonteCarloState(
      hmc_state=hmc_state,
      running_var_state=running_var_state,
      log_step_size_opt_state=log_step_size_opt_state,
      step=tf.zeros([], tf.int32))


def adaptive_hamiltonian_monte_carlo_step(
    adaptive_hmc_state: 'AdaptiveHamiltonianMonteCarloState',
    target_log_prob_fn: 'fun_mc.PotentialFn',
    num_adaptation_steps: 'Optional[fun_mc.IntTensor]',
    variance_window_steps: 'fun_mc.IntTensor' = 100,
    trajectory_length_factor: 'fun_mc.FloatTensor' = 1.,
    num_trajectory_ramp_steps: 'Optional[int]' = 100,
    trajectory_warmup_power: 'fun_mc.FloatTensor' = 1.,
    max_num_leapfrog_steps: 'Optional[int]' = 100,
    step_size_adaptation_rate: 'fun_mc.FloatTensor' = 0.05,
    step_size_adaptation_rate_decay_power: 'fun_mc.FloatTensor' = 0.1,
    target_accept_prob: 'fun_mc.FloatTensor' = 0.8,
    seed: 'Any' = None,
) -> ('Tuple[AdaptiveHamiltonianMonteCarloState, '
      'AdaptiveHamiltonianMonteCarloExtra]'):
  """Adaptive Hamiltonian Monte Carlo `TransitionOperator`.

  This implements a relatively straighforward adaptive HMC algorithm with
  diagonal mass-matrix adaptation and step size adaptation. The algorithm also
  estimates the trajectory length based on the variance of the chain.

  All adaptation stops after `num_adaptation_steps`, after which this algorithm
  becomes regular HMC with fixed number of leapfrog steps, and fixed
  per-component step size. Typically, chain samples after `num_adaptation_steps
  + num_warmup_steps` are discarded, where `num_warmup_steps` is typically
  heuristically chosen to be `0.25 * num_adaptation_steps`. For maximum
  efficiency, however, it's recommended to actually use
  `fun_mc.hamiltonian_monte_carlo` initialized with the relevant
  hyperparameters, as that `TransitionOperator` won't have the overhead of the
  adaptation logic. This can be set to `None`, in which case adaptation never
  stops (and the algorihthm ceases to be calibrated).

  The mass matrix is adapted by computing an exponential moving variance of the
  chain. The averaging window is controlled by the `variance_window_steps`, with
  larger values leading to a smoother estimate.

  Trajectory length is computed as `max(sqrt(chain_variance)) *
  trajectory_length_factor`. To be resilient to poor initialization,
  `trajectory_length_factor` can be increased from 0 based on a polynomial
  schedule, controlled by `num_trajectory_ramp_steps` and
  `trajectory_warmup_power`.

  The step size is adapted to make the acceptance probability close to
  `target_accept_prob`, using the `Adam` optimizer. This is controlled by the
  `step_size_adaptation_rate`. If `num_adaptation_steps` is not `None`, this
  rate is decayed using a polynomial schedule controlled by
  `step_size_adaptation_rate`.

  Args:
    adaptive_hmc_state: `AdaptiveHamiltonianMonteCarloState`
    target_log_prob_fn: Target log prob fn.
    num_adaptation_steps: Number of adaptation steps, can be `None`.
    variance_window_steps: Window to compute the chain variance over.
    trajectory_length_factor: Trajectory length factor.
    num_trajectory_ramp_steps: Number of steps to warmup the
      `trajectory_length_factor`.
    trajectory_warmup_power: Power of the polynomial schedule for
      `trajectory_length_factor` warmup.
    max_num_leapfrog_steps: Maximum number of leapfrog steps to take.
    step_size_adaptation_rate: Step size adaptation rate.,
    step_size_adaptation_rate_decay_power:  Power of the polynomial schedule for
      `trajectory_length_factor` warmup.
    target_accept_prob: Target acceptance probability.
    seed: Random seed to use.

  Returns:
    adaptive_hmc_state: `AdaptiveHamiltonianMonteCarloState`.
    adaptive_hmc_extra: `AdaptiveHamiltonianMonteCarloExtra`.

  #### Examples

  Here's an example using using Adaptive HMC and TensorFlow Probability to
  sample from a simple model.

  ```python
  num_chains = 16
  num_steps = 2000
  num_warmup_steps = num_steps // 2
  num_adapt_steps = int(0.8 * num_warmup_steps)

  # Setup the model and state constraints.
  model = tfp.distributions.JointDistributionSequential([
      tfp.distributions.Normal(loc=0., scale=1.),
      tfp.distributions.Independent(
          tfp.distributions.LogNormal(loc=[1., 1.], scale=0.5), 1),
  ])
  bijector = [tfp.bijectors.Identity(), tfp.bijectors.Exp()]
  transform_fn = fun_mcmc.util_tfp.bijector_to_transform_fn(
      bijector, model.dtype, batch_ndims=1)

  def target_log_prob_fn(*x):
    return model.log_prob(x), ()

  # Start out at zeros (in the unconstrained space).
  state, _ = transform_fn(
      *map(lambda e: tf.zeros([num_chains] + list(e)), model.event_shape))

  reparam_log_prob_fn, reparam_state = fun_mcmc.reparameterize_potential_fn(
      target_log_prob_fn, transform_fn, state)

  # Define the kernel.
  def kernel(adaptive_hmc_state):
    adaptive_hmc_state, adaptive_hmc_extra = (
        fun_mcmc.prefab.adaptive_hamiltonian_monte_carlo_step(
            adaptive_hmc_state,
            target_log_prob_fn=reparam_log_prob_fn,
            num_adaptation_steps=num_adapt_steps))

    return adaptive_hmc_state, (adaptive_hmc_extra.state,
                                adaptive_hmc_extra.is_accepted,
                                adaptive_hmc_extra.step_size)


  _, (state_chain, is_accepted_chain, step_size_chain) = tf.function(
      lambda: fun_mcmc.trace(
          state=fun_mcmc.prefab.adaptive_hamiltonian_monte_carlo_init(
              reparam_state, reparam_log_prob_fn),
          fn=kernel,
          num_steps=num_steps),
      autograph=False)()

  # Discard the warmup samples.
  state_chain = [s[num_warmup_steps:] for s in state_chain]
  is_accepted_chain = is_accepted_chain[num_warmup_steps:]

  # Compute diagnostics.
  accept_rate = tf.reduce_mean(tf.cast(is_accepted_chain, tf.float32))
  ess = tfp.mcmc.effective_sample_size(
      state_chain, filter_beyond_positive_pairs=True, cross_chain_dims=[1, 1])
  rhat = tfp.mcmc.potential_scale_reduction(state_chain)

  # Compute relevant quantities.
  sample_mean = [tf.reduce_mean(s, axis=[0, 1]) for s in state_chain]
  sample_var = [tf.math.reduce_variance(s, axis=[0, 1]) for s in state_chain]

  # It's also important to look at the `step_size_chain` (e.g. via a plot), to
  # verify that adaptation succeeded.
  ```

  """
  dtype = util.flatten_tree(adaptive_hmc_state.hmc_state.state)[0].dtype
  step_size_adaptation_rate = tf.convert_to_tensor(
      step_size_adaptation_rate, dtype=dtype)
  trajectory_length_factor = tf.convert_to_tensor(
      trajectory_length_factor, dtype=dtype)
  target_accept_prob = tf.convert_to_tensor(target_accept_prob, dtype=dtype)
  step_size_adaptation_rate_decay_power = tf.convert_to_tensor(
      step_size_adaptation_rate_decay_power, dtype=dtype)
  trajectory_warmup_power = tf.convert_to_tensor(
      trajectory_warmup_power, dtype=dtype)

  hmc_state = adaptive_hmc_state.hmc_state
  running_var_state = adaptive_hmc_state.running_var_state
  log_step_size_opt_state = adaptive_hmc_state.log_step_size_opt_state
  step = adaptive_hmc_state.step

  # Warmup the trajectory length, if requested.
  if num_trajectory_ramp_steps is not None:
    trajectory_length_factor = _polynomial_decay(
        step=step,
        step_size=tf.constant(0., dtype),
        decay_steps=num_trajectory_ramp_steps,
        final_step_size=trajectory_length_factor,
        power=trajectory_warmup_power,
    )

  # Compute the per-component step_size and num_leapfrog_steps from the variance
  # estimate.
  scale = util.map_tree(tf.math.sqrt, running_var_state.variance)
  max_scale = functools.reduce(
      tf.maximum, util.flatten_tree(util.map_tree(tf.reduce_max, scale)))
  step_size = tf.exp(log_step_size_opt_state.state)
  num_leapfrog_steps = tf.cast(
      tf.math.ceil(max_scale * trajectory_length_factor / step_size), tf.int32)
  if max_num_leapfrog_steps is not None:
    num_leapfrog_steps = tf.minimum(max_num_leapfrog_steps, num_leapfrog_steps)
  # We implement mass-matrix adaptation via step size rescaling, as this is a
  # little bit simpler to code up.
  step_size = util.map_tree(lambda scale: scale / max_scale * step_size, scale)

  # Run a step of HMC.
  hmc_state, hmc_extra = fun_mc.hamiltonian_monte_carlo(
      hmc_state,
      target_log_prob_fn=target_log_prob_fn,
      step_size=step_size,
      num_integrator_steps=num_leapfrog_steps,
      seed=seed,
  )

  # Update the running variance estimate.
  chain_ndims = len(hmc_state.target_log_prob.shape)
  old_running_var_state = running_var_state
  running_var_state, _ = fun_mc.running_variance_step(
      running_var_state,
      hmc_state.state,
      axis=tuple(range(chain_ndims)) if chain_ndims else None,
      window_size=int(np.prod(hmc_state.target_log_prob.shape)) *
      variance_window_steps)

  if num_adaptation_steps is not None:
    # Take care of adaptation for variance and step size.
    running_var_state = util.map_tree(
        lambda n, o: tf.where(step < num_adaptation_steps, n, o),  # pylint: disable=g-long-lambda
        running_var_state,
        old_running_var_state)
    step_size_adaptation_rate = _polynomial_decay(
        step=step,
        step_size=step_size_adaptation_rate,
        decay_steps=num_adaptation_steps,
        final_step_size=0.,
        power=step_size_adaptation_rate_decay_power,
    )

  # Update the scalar step size as a function of acceptance rate.
  p_accept = tf.reduce_mean(
      tf.exp(tf.minimum(hmc_extra.log_accept_ratio, tf.zeros([], dtype))))
  p_accept = tf.where(
      tf.math.is_finite(p_accept), p_accept, tf.zeros_like(p_accept))

  loss_fn = fun_mc.make_surrogate_loss_fn(lambda _:  # pylint: disable=g-long-lambda
                                          (target_accept_prob - p_accept, ()))

  log_step_size_opt_state, _ = fun_mc.adam_step(log_step_size_opt_state,
                                                loss_fn,
                                                step_size_adaptation_rate)

  adaptive_hmc_state = AdaptiveHamiltonianMonteCarloState(
      hmc_state=hmc_state,
      running_var_state=running_var_state,
      log_step_size_opt_state=log_step_size_opt_state,
      step=step + 1,
  )

  extra = AdaptiveHamiltonianMonteCarloExtra(
      hmc_state=hmc_state,
      hmc_extra=hmc_extra,
      step_size=step_size,
      num_leapfrog_steps=num_leapfrog_steps)

  return adaptive_hmc_state, extra
