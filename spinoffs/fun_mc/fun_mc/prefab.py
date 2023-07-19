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
"""Prefabricated transition operators.

These are compositions of FunMC API into slightly more robust, 'turn-key',
opinionated inference and optimization algorithms. They are not themselves meant
to be infinitely composable, or as flexible as the base FunMC API. When
flexibility is required, the users are encouraged to copy paste the
implementations and adjust the pieces that they want. The code in this module
strives to be easy to copy paste and modify in this manner.
"""

import functools
import typing
from typing import Any, Callable, Iterable, NamedTuple, Optional  # pylint: disable=unused-import

import numpy as np

from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc
from fun_mc import malt
from fun_mc import sga_hmc
# Re-export sga_hmc and malt symbols.
from fun_mc.malt import *  # pylint: disable=wildcard-import
from fun_mc.sga_hmc import *  # pylint: disable=wildcard-import

tf = backend.tf
tfp = backend.tfp
util = backend.util

__all__ = [
    'adaptive_hamiltonian_monte_carlo_init',
    'adaptive_hamiltonian_monte_carlo_step',
    'AdaptiveHamiltonianMonteCarloState',
    'interactive_trace',
    'step_size_adaptation_init',
    'step_size_adaptation_step',
    'StepSizeAdaptationExtra',
    'StepSizeAdaptationState',
] + sga_hmc.__all__ + malt.__all__


@util.named_call(name='polynomial_decay')
def _polynomial_decay(step: fun_mc.AnyTensor,
                      step_size: fun_mc.FloatTensor,
                      decay_steps: fun_mc.AnyTensor,
                      final_step_size: fun_mc.FloatTensor,
                      power: fun_mc.FloatTensor = 1.) -> fun_mc.FloatTensor:
  """Polynomial decay step size schedule."""
  step_size = tf.convert_to_tensor(step_size)
  step_f = tf.cast(step, step_size.dtype)
  decay_steps_f = tf.cast(decay_steps, step_size.dtype)
  step_mult = (1. - step_f / decay_steps_f)**power
  step_mult = tf.where(step >= decay_steps, tf.zeros_like(step_mult), step_mult)
  return step_mult * (step_size - final_step_size) + final_step_size


class StepSizeAdaptationState(NamedTuple):
  """Step size adaptation state."""
  step: fun_mc.IntTensor
  opt_state: fun_mc.AdamState
  rms_state: fun_mc.RunningMeanState

  def opt_step_size(self):
    return tf.exp(self.opt_state.state)

  @property
  def rms_step_size(self):
    return self.rms_state.mean

  def step_size(self, num_adaptation_steps=None):
    if num_adaptation_steps is not None:
      return tf.where(self.step < num_adaptation_steps, self.opt_step_size(),
                      self.rms_step_size)
    else:
      return self.opt_step_size()


class StepSizeAdaptationExtra(NamedTuple):
  opt_extra: fun_mc.AdamExtra
  accept_prob: fun_mc.FloatTensor


@util.named_call
def step_size_adaptation_init(
    init_step_size: fun_mc.FloatTensor) -> StepSizeAdaptationState:
  """Initializes `StepSizeAdaptationState`.

  Args:
    init_step_size: Floating point Tensor. Initial step size.

  Returns:
    step_size_adaptation_state: `StepSizeAdaptationState`
  """
  init_step_size = tf.convert_to_tensor(init_step_size)
  rms_state = fun_mc.running_mean_init(init_step_size.shape,
                                       init_step_size.dtype)
  rms_state = rms_state._replace(mean=init_step_size)

  return StepSizeAdaptationState(
      step=tf.constant(0, tf.int32),
      opt_state=fun_mc.adam_init(tf.math.log(init_step_size)),
      rms_state=rms_state,
  )


@util.named_call
def step_size_adaptation_step(
    state: StepSizeAdaptationState,
    log_accept_ratio: fun_mc.FloatTensor,
    num_adaptation_steps: Optional[fun_mc.IntTensor],
    target_accept_prob: fun_mc.FloatTensor = 0.8,
    adaptation_rate: fun_mc.FloatTensor = 0.05,
    adaptation_rate_decay_power: fun_mc.FloatTensor = 0.1,
    averaging_window_steps: fun_mc.IntTensor = 100,
    min_log_accept_prob: fun_mc.FloatTensor = np.log(1e-5),
    reduce_fn: Callable[[fun_mc.FloatTensor], fun_mc.FloatTensor] = (
        tfp.math.reduce_logmeanexp
    ),
    opt_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[StepSizeAdaptationState, StepSizeAdaptationExtra]:
  """Gradient based step size adaptation using ADAM.

  Given the `log_accept_ratio` statistic from an Metropolis-Hastings algorithm,
  this adapts the step size hyperparameter to make that statistic hit some
  `target_accept_prob`. The step size can be extracted using the `step_size`
  method on the state structure.

  Args:
    state: `StepSizeAdaptationState`
    log_accept_ratio: Float tensor. The logarithm of the accept ratio.
    num_adaptation_steps: Number of adaptation steps, can be `None`.
    target_accept_prob: Target acceptance probability.
    adaptation_rate: Step size adaptation rate.
    adaptation_rate_decay_power:  Power of the polynomial schedule for
      `trajectory_length_factor` warmup.
    averaging_window_steps: Number of steps to compute the averaged step size.
    min_log_accept_prob: Clamps acceptance probability to this value for
      numerical stability.
    reduce_fn: A function that reduces `log_accept_ratio` in log-space. By
      default, this computes the log-mean-exp.
    opt_kwargs: Additional arguments to pass to the optimizer.

  Returns:
    step_size_adaptation_state: `StepSizeAdaptationState`
    step_size_adaptation_extra: `StepSizeAdaptationExtra`
  """
  opt_kwargs = {} if opt_kwargs is None else opt_kwargs
  dtype = log_accept_ratio.dtype
  adaptation_rate = tf.convert_to_tensor(adaptation_rate, dtype=dtype)
  target_accept_prob = tf.convert_to_tensor(target_accept_prob, dtype=dtype)
  adaptation_rate_decay_power = tf.convert_to_tensor(
      adaptation_rate_decay_power, dtype=dtype)
  min_log_accept_prob = tf.fill(log_accept_ratio.shape,
                                tf.constant(min_log_accept_prob, dtype))

  log_accept_prob = tf.minimum(log_accept_ratio, tf.zeros([], dtype))
  log_accept_prob = tf.maximum(log_accept_prob, min_log_accept_prob)
  log_accept_prob = tf.where(
      tf.math.is_finite(log_accept_prob), log_accept_prob, min_log_accept_prob)
  accept_prob = tf.exp(reduce_fn(log_accept_prob))

  loss_fn = fun_mc.make_surrogate_loss_fn(lambda _:  # pylint: disable=g-long-lambda
                                          (target_accept_prob - accept_prob, ()
                                          ))

  if num_adaptation_steps is not None:
    adaptation_rate = _polynomial_decay(
        step=state.step,
        step_size=adaptation_rate,
        decay_steps=num_adaptation_steps,
        final_step_size=0.,
        power=adaptation_rate_decay_power,
    )

  # Optimize step size.
  opt_state, opt_extra = fun_mc.adam_step(state.opt_state, loss_fn,
                                          adaptation_rate, **opt_kwargs)

  # Do iterate averaging.
  old_rms_state = state.rms_state
  rms_state, _ = fun_mc.running_mean_step(
      old_rms_state,
      tf.exp(opt_state.state),
      window_size=averaging_window_steps)

  if num_adaptation_steps is not None:
    rms_state = util.map_tree(
        lambda n, o: tf.where(state.step < num_adaptation_steps, n, o),
        rms_state, old_rms_state)

  state = state._replace(
      opt_state=opt_state, rms_state=rms_state, step=state.step + 1)
  extra = StepSizeAdaptationExtra(opt_extra=opt_extra, accept_prob=accept_prob)
  return state, extra


class AdaptiveHamiltonianMonteCarloState(NamedTuple):
  """Adaptive HMC `TransitionOperator` state."""
  hmc_state: fun_mc.HamiltonianMonteCarloState
  running_var_state: fun_mc.RunningVarianceState
  ssa_state: StepSizeAdaptationState
  step: fun_mc.IntTensor


class AdaptiveHamiltonianMonteCarloExtra(NamedTuple):
  """Extra outputs for Adaptive HMC `TransitionOperator`."""
  hmc_state: fun_mc.HamiltonianMonteCarloState
  hmc_extra: fun_mc.HamiltonianMonteCarloExtra
  step_size: fun_mc.FloatTensor
  num_leapfrog_steps: fun_mc.IntTensor
  mean_num_leapfrog_steps: fun_mc.IntTensor

  @property
  def state(self) -> fun_mc.TensorNest:
    """Returns the chain state.

    Note that this assumes that `target_log_prob_fn` has the `state_extra` be a
    tuple, with the first element thereof containing the state in the original
    space.
    """
    return self.hmc_state.state_extra[0]

  @property
  def is_accepted(self) -> fun_mc.BooleanTensor:
    return self.hmc_extra.is_accepted


@util.named_call
def adaptive_hamiltonian_monte_carlo_init(
    state: fun_mc.TensorNest,
    target_log_prob_fn: fun_mc.PotentialFn,
    step_size: fun_mc.FloatTensor = 1e-2,
    initial_mean: fun_mc.FloatNest = 0.,
    initial_scale: fun_mc.FloatNest = 1.,
    scale_smoothing_steps: fun_mc.IntTensor = 10,
) -> AdaptiveHamiltonianMonteCarloState:
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
  ssa_state = step_size_adaptation_init(
      tf.convert_to_tensor(step_size, dtype=dtype))

  return AdaptiveHamiltonianMonteCarloState(
      hmc_state=hmc_state,
      running_var_state=running_var_state,
      ssa_state=ssa_state,
      step=tf.zeros([], tf.int32))


@util.named_call
def _uniform_jitter(mean_num_leapfrog_steps, step, seed):
  del step
  return util.random_integer(
      [],
      dtype=tf.int32,
      minval=1,
      maxval=2 * mean_num_leapfrog_steps,
      seed=seed,
  )


@util.named_call
def adaptive_hamiltonian_monte_carlo_step(
    adaptive_hmc_state: AdaptiveHamiltonianMonteCarloState,
    target_log_prob_fn: fun_mc.PotentialFn,
    num_adaptation_steps: Optional[fun_mc.IntTensor],
    variance_window_steps: fun_mc.IntTensor = 100,
    trajectory_length_factor: fun_mc.FloatTensor = 1.0,
    num_trajectory_ramp_steps: Optional[int] = 100,
    trajectory_warmup_power: fun_mc.FloatTensor = 1.0,
    max_num_leapfrog_steps: Optional[int] = 100,
    step_size_adaptation_rate: fun_mc.FloatTensor = 0.05,
    step_size_adaptation_rate_decay_power: fun_mc.FloatTensor = 0.1,
    target_accept_prob: fun_mc.FloatTensor = 0.8,
    step_size_averaging_window_steps: fun_mc.IntTensor = 100,
    jitter_sample_fn: Callable[
        [fun_mc.IntTensor, fun_mc.IntTensor, Any], fun_mc.IntTensor
    ] = (_uniform_jitter),
    log_accept_ratio_reduce_fn: Callable[
        [fun_mc.FloatTensor], fun_mc.FloatTensor
    ] = (tfp.math.reduce_logmeanexp),
    hmc_kwargs: Optional[dict[str, Any]] = None,
    seed: Any = None,
) -> tuple[
    AdaptiveHamiltonianMonteCarloState, AdaptiveHamiltonianMonteCarloExtra
]:
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
  `fun_mc.hamiltonian_monte_carlo_step` initialized with the relevant
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
    step_size_adaptation_rate: Step size adaptation rate.
    step_size_adaptation_rate_decay_power:  Power of the polynomial schedule for
      `trajectory_length_factor` warmup.
    target_accept_prob: Target acceptance probability.
    step_size_averaging_window_steps: Number of steps to compute the averaged
      step size.
    jitter_sample_fn: Function with signature `(mean_num_leapfrog_steps, step,
      seed) -> num_leapfrog_steps`. By default, this does a uniform jitter
      returning a step size in [1, 2 * mean_num_leapfrog_steps).
    log_accept_ratio_reduce_fn: A function that reduces `log_accept_ratio` in
      log-space. By default, this computes the log-mean-exp.
    hmc_kwargs: Additional keyword arguments to pass to
      `hamiltonian_monte_carlo_step`.
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
  transform_fn = fun_mc.util_tfp.bijector_to_transform_fn(
      bijector, model.dtype, batch_ndims=1)

  def target_log_prob_fn(*x):
    return model.log_prob(x), ()

  # Start out at zeros (in the unconstrained space).
  state, _ = transform_fn(
      *map(lambda e: tf.zeros([num_chains] + list(e)), model.event_shape))

  reparam_log_prob_fn, reparam_state = fun_mc.reparameterize_potential_fn(
      target_log_prob_fn, transform_fn, state)

  # Define the kernel.
  def kernel(adaptive_hmc_state):
    adaptive_hmc_state, adaptive_hmc_extra = (
        fun_mc.prefab.adaptive_hamiltonian_monte_carlo_step(
            adaptive_hmc_state,
            target_log_prob_fn=reparam_log_prob_fn,
            num_adaptation_steps=num_adapt_steps))

    return adaptive_hmc_state, (adaptive_hmc_extra.state,
                                adaptive_hmc_extra.is_accepted,
                                adaptive_hmc_extra.step_size)


  _, (state_chain, is_accepted_chain, step_size_chain) = tf.function(
      lambda: fun_mc.trace(
          state=fun_mc.prefab.adaptive_hamiltonian_monte_carlo_init(
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
  trajectory_length_factor = tf.convert_to_tensor(
      trajectory_length_factor, dtype=dtype)
  trajectory_warmup_power = tf.convert_to_tensor(
      trajectory_warmup_power, dtype=dtype)

  hmc_state = adaptive_hmc_state.hmc_state
  running_var_state = adaptive_hmc_state.running_var_state
  ssa_state = adaptive_hmc_state.ssa_state
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
  step_size = ssa_state.step_size(num_adaptation_steps=num_adaptation_steps)
  num_leapfrog_steps = tf.cast(
      tf.math.ceil(trajectory_length_factor / step_size), tf.int32)
  num_leapfrog_steps = tf.maximum(1, num_leapfrog_steps)
  if max_num_leapfrog_steps is not None:
    num_leapfrog_steps = tf.minimum(max_num_leapfrog_steps, num_leapfrog_steps)
  # We implement mass-matrix adaptation via step size rescaling, as this is a
  # little bit simpler to code up.
  step_size = util.map_tree(lambda scale: scale * step_size, scale)

  hmc_seed, jitter_seed = util.split_seed(seed, 2)
  jittered_num_leapfrog_steps = jitter_sample_fn(num_leapfrog_steps, step,
                                                 jitter_seed)

  # Run a step of HMC.
  hmc_kwargs = hmc_kwargs or {}
  hmc_state, hmc_extra = fun_mc.hamiltonian_monte_carlo_step(
      hmc_state,
      target_log_prob_fn=target_log_prob_fn,
      step_size=step_size,
      num_integrator_steps=jittered_num_leapfrog_steps,
      seed=hmc_seed,
      **hmc_kwargs,
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
    # Take care of adaptation for variance.
    running_var_state = util.map_tree(
        lambda n, o: tf.where(step < num_adaptation_steps, n, o),  # pylint: disable=g-long-lambda
        running_var_state,
        old_running_var_state)

  # Update the scalar step size as a function of acceptance rate.
  ssa_state, _ = step_size_adaptation_step(
      ssa_state,
      log_accept_ratio=hmc_extra.log_accept_ratio,
      num_adaptation_steps=num_adaptation_steps,
      target_accept_prob=target_accept_prob,
      adaptation_rate=step_size_adaptation_rate,
      adaptation_rate_decay_power=step_size_adaptation_rate_decay_power,
      reduce_fn=log_accept_ratio_reduce_fn,
      averaging_window_steps=step_size_averaging_window_steps,
  )

  adaptive_hmc_state = AdaptiveHamiltonianMonteCarloState(
      hmc_state=hmc_state,
      running_var_state=running_var_state,
      ssa_state=ssa_state,
      step=step + 1,
  )

  extra = AdaptiveHamiltonianMonteCarloExtra(
      hmc_state=hmc_state,
      hmc_extra=hmc_extra,
      step_size=step_size,
      mean_num_leapfrog_steps=num_leapfrog_steps,
      num_leapfrog_steps=jittered_num_leapfrog_steps)

  return adaptive_hmc_state, extra


def _tqdm_progress_bar_fn(iterable: Iterable[Any]) -> Iterable[Any]:
  """The TQDM progress bar function."""
  # pytype: disable=import-error
  import tqdm  # pylint: disable=g-import-not-at-top
  # pytype: enable=import-error
  return tqdm.tqdm(iterable, leave=True)


def interactive_trace(
    state: fun_mc.State,
    fn: fun_mc.TransitionOperator,
    num_steps: fun_mc.IntTensor,
    trace_mask: fun_mc.BooleanNest = True,
    iteration_axis: int = 0,
    block_until_ready: bool = True,
    progress_bar_fn: Callable[[Iterable[Any]], Iterable[Any]] = (
        _tqdm_progress_bar_fn
    ),
) -> tuple[fun_mc.State, fun_mc.TensorNest]:
  """Wrapped around fun_mc.trace, suited for interactive work.

  This is accomplished through unrolling fun_mc.trace, as well as optionally
  using a progress bar (TQDM by default).

  Args:
    state: A nest of `Tensor`s or None.
    fn: A `TransitionOperator`.
    num_steps: Number of steps to run the function for. Must be greater than 1.
    trace_mask: A potentially shallow nest with boolean leaves applied to the
      `extra` return value of `fn`. This controls whether or not to actually
      trace the quantities in `extra`. For subtrees of `extra` where the mask
      leaf is `True`, those subtrees are traced (i.e. the corresponding subtrees
      in `traces` will contain an extra leading dimension equalling
      `num_steps`). For subtrees of `extra` where the mask leaf is `False`,
      those subtrees are merely propagated, and their corresponding subtrees in
      `traces` correspond to their final value.
    iteration_axis: Integer. Indicates the axis of the trace outputs that should
      be flattened with the first axis. This is most useful when `fn` is
      `trace`. E.g. if the trace has shape `[num_steps, 2, 5]` and
      `iteration_axis=2`, the trace outputs will be reshaped/transposed to `[2,
      5 * num_steps]`. A value of 0 disables this operation.
    block_until_ready: Whether to wait for the computation to finish between
      steps. This results in smoother progress bars under, e.g., JAX.
    progress_bar_fn: A callable that will be called with an iterable with length
      `num_steps` and which returns another iterable with the same length. This
      will be advanced for every step taken. If None, no progress bar is
      shown. Default: `lambda it: tqdm.tqdm(it, leave=True)`.

  Returns:
    state: The final state returned by `fn`.
    traces: A nest with the same structure as the extra return value of `fn`,
      but with leaves replaced with stacked and unstacked values according to
      the `trace_mask`.
  """
  num_steps = tf.get_static_value(num_steps)
  if num_steps is None:
    raise ValueError(
        'Interactive tracing requires `num_steps` to be statically known.')

  if progress_bar_fn is None:
    pbar = None
  else:
    pbar = iter(progress_bar_fn(range(num_steps)))

  def fn_with_progress(state):
    state, extra = fun_mc.call_transition_operator(fn, state)
    if block_until_ready:
      state, extra = util.block_until_ready((state, extra))
    if pbar is not None:
      try:
        next(pbar)
      except StopIteration:
        pass
    return [state], extra

  [state], trace = fun_mc.trace(
      # Wrap the state in a singleton list to simplify implementation of
      # `fn_with_progress`.
      state=[state],
      fn=fn_with_progress,
      num_steps=num_steps,
      trace_mask=trace_mask,
      unroll=True,
  )

  if iteration_axis != 0:

    def fix_part(x):
      x = util.move_axis(x, 0, iteration_axis - 1)
      x = tf.reshape(
          x,
          tuple(x.shape[:iteration_axis - 1]) + (-1,) +
          tuple(x.shape[iteration_axis + 1:]))
      return x

    trace = util.map_tree(fix_part, trace)
  return state, trace


class PersistentHamiltonianMonteCarloState(NamedTuple):
  """Persistent Hamiltonian Monte Carlo state."""
  state: fun_mc.State
  state_grads: fun_mc.State
  momentum: fun_mc.State
  target_log_prob: fun_mc.FloatTensor
  state_extra: Any
  direction: fun_mc.FloatTensor
  pmh_state: fun_mc.PersistentMetropolistHastingsState


class PersistentHamiltonianMonteCarloExtra(NamedTuple):
  """Persistent Hamiltonian Monte Carlo extra outputs."""
  is_accepted: fun_mc.BooleanTensor
  log_accept_ratio: fun_mc.FloatTensor
  proposed_phmc_state: fun_mc.State
  integrator_state: fun_mc.IntegratorState
  integrator_extra: fun_mc.IntegratorExtras
  initial_momentum: fun_mc.State
  energy_change_extra: Any


@util.named_call
def persistent_hamiltonian_monte_carlo_init(
    state: fun_mc.TensorNest,
    target_log_prob_fn: fun_mc.PotentialFn,
    momentum: Optional[fun_mc.State] = None,
    init_level: fun_mc.FloatTensor = 0.,
) -> PersistentHamiltonianMonteCarloState:
  """Initializes the `PersistentHamiltonianMonteCarloState`.

  Args:
    state: State of the chain.
    target_log_prob_fn: Target log prob fn.
    momentum: Initial momentum. Set to all zeros by default.
    init_level: Initial level for the persistent Metropolis Hastings.

  Returns:
    hmc_state: `PersistentMetropolistHastingsState`.
  """
  state = util.map_tree(tf.convert_to_tensor, state)
  target_log_prob, state_extra, state_grads = util.map_tree(
      tf.convert_to_tensor,
      fun_mc.call_potential_fn_with_grads(target_log_prob_fn, state),
  )
  return PersistentHamiltonianMonteCarloState(
      state=state,
      state_grads=state_grads,
      momentum=momentum if momentum is not None else util.map_tree(
          tf.zeros_like, state),
      target_log_prob=target_log_prob,
      state_extra=state_extra,
      direction=tf.ones_like(target_log_prob),
      pmh_state=fun_mc.persistent_metropolis_hastings_init(
          shape=target_log_prob.shape,
          dtype=target_log_prob.dtype,
          init_level=init_level),
  )


PersistentMomentumSampleFn = Callable[[fun_mc.State, Any], fun_mc.State]


@util.named_call
def persistent_hamiltonian_monte_carlo_step(
    phmc_state: PersistentHamiltonianMonteCarloState,
    target_log_prob_fn: fun_mc.PotentialFn,
    step_size: Optional[Any] = None,
    num_integrator_steps: Optional[fun_mc.IntTensor] = None,
    noise_fraction: Optional[fun_mc.FloatTensor] = None,
    mh_drift: Optional[fun_mc.FloatTensor] = None,
    kinetic_energy_fn: Optional[fun_mc.PotentialFn] = None,
    momentum_sample_fn: Optional[PersistentMomentumSampleFn] = None,
    integrator_trace_fn: Callable[
        [fun_mc.IntegratorStepState, fun_mc.IntegratorStepExtras],
        fun_mc.TensorNest,
    ] = lambda *args: (),
    log_uniform: Optional[fun_mc.FloatTensor] = None,
    integrator_fn: Optional[
        Callable[
            [fun_mc.IntegratorState, fun_mc.FloatTensor],
            tuple[fun_mc.IntegratorState, fun_mc.IntegratorExtras],
        ]
    ] = None,
    unroll_integrator: bool = False,
    max_num_integrator_steps: Optional[fun_mc.IntTensor] = None,
    energy_change_fn: Callable[
        [
            fun_mc.IntegratorState,
            fun_mc.IntegratorState,
            fun_mc.IntegratorExtras,
        ],
        tuple[fun_mc.FloatTensor, Any],
    ] = (
        fun_mc._default_hamiltonian_monte_carlo_energy_change_fn  # pylint: disable=protected-access
    ),
    named_axis: Optional[fun_mc.StringNest] = None,
    seed=None,
) -> tuple[
    PersistentHamiltonianMonteCarloState, PersistentHamiltonianMonteCarloExtra
]:
  """A step of the Persistent Hamiltonian Monte Carlo `TransitionOperator`.

  This is an implementation of the generalized HMC with persistent momentum
  described in [1] (algorithm 15) combined with the persistent Metropolis
  Hastings test from [2]. This generalizes the regular HMC with persistent
  momentum from [3] and the various underdamped langevin dynamics schemes (e.g.
  [4]).

  The generalization lies in the free choice of `momentum_sample_fn` and
  `kinetic_energy_fn`. The former forms a Markov Chain with the stationary
  distribution implied by the `kinetic_energy_fn`. By default, the standard
  quadratic kinetic energy is used and the underdamped update is used for
  `momentum_sample_fn`, namely:

  ```none
  new_momentum = (
      (1 - noise_fraction**2)**0.5 * old_momentum  +
      noise_fraction * eps)
  eps ~ Normal(0, 1)
  ```

  Here are the parameter settings for few special cases:

  1. Persistent Hamiltonian Monte Carlo [1] + persistent MH [2]:

  ```none
  num_integrator_steps >= 1
  step_size > 0
  noise_fraction in [0, 1]
  mh_drift = 0.03
  ```

  Empirical results suggest that if `num_integrator_steps == 1`, then a
  reasonable value for `mh_drift` is `1 - (1 - noise_fraction**2)**0.5`.

  2. Unadjusted Underdamped Langevin Dynamics (see [4]):

  ```none
  num_integrator_steps = 1
  step_size > 0
  noise_fraction = (1 - exp(-2 * step_size * damping))**0.5
  # This disables the MH step for all but most extreme divergences.
  log_uniform = -1000
  ```

  `damping` refers to the parameter in the SDE formulation of the algorithm:

  ```none
  dv_t = -damping * v_t * dt - grad(f)(x_t) * dt + (2 * damping)**0.5 * dB_t
  dx_t = v_t * dt
  ```

  Args:
    phmc_state: `PersistentHamiltonianMonteCarloState`.
    target_log_prob_fn: Target log prob fn.
    step_size: Step size, structure broadcastable to the `target_log_prob_fn`
      state. Optional if `integrator_fn` is specified.
    num_integrator_steps: Number of integrator steps to take. Optional if
      `integrator_fn` is specified.
    noise_fraction: Noise fraction when refreshing momentum. Optional if
      `momentum_sample_fn` is specified.
    mh_drift: Metropolis Hastings drift term. Optional if `log_uniform` is
      specified.
    kinetic_energy_fn: Kinetic energy function.
    momentum_sample_fn: Sampler for the momentum.
    integrator_trace_fn: Trace function for the integrator.
    log_uniform: Optional logarithm of a uniformly distributed random sample in
      [0, 1], used for the MH accept/reject step.
    integrator_fn: Integrator to use for the HMC dynamics. Uses a
      `hamiltonian_integrator` with `leapfrog_step` by default.
    unroll_integrator: Whether to unroll the loop in the integrator. Only works
      if `num_integrator_steps`/`max_num_integrator_steps' is statically known.
      Ignored if `integrator_fn` is specified.
    max_num_integrator_steps: Maximum number of integrator steps to take. Useful
      when `num_integrator_steps` is dynamic, and yet you still want
      gradients/tracing to work. Ignored if `integrator_fn` is specified.
    energy_change_fn: Callable with signature: `(current_integrator_state,
      proposed_integrator_state,) -> (energy_change, energy_change_extra)`.
      Computes the change in energy between current and proposed states. By
      default, it just substracts the current and proposed energies. A typical
      reason to override this is to improve numerical stability.
    named_axis: Named axes of the state, same structure as `hmc_state.state`.
    seed: For reproducibility.

  Returns:
    phmc_state: PersistentHamiltonianMonteCarloState
    phmc_extra: PersistentHamiltonianMonteCarloExtra

  #### References

  [1]: Neklyudov, K., Welling, M., Egorov, E., & Vetrov, D. (2020). Involutive
       MCMC: a Unifying Framework.

  [2]: Neal, R. M. (2020). Non-reversibly updating a uniform [0,1] value for
       Metropolis accept/reject decisions.

  [3]: Horowitz, A. M. (1991). A generalized guided Monte Carlo algorithm.
       Physics Letters. [Part B], 268(2), 247-252.

  [4]: Ma, Y.-A., Chatterji, N., Cheng, X., Flammarion, N., Bartlett, P., &
       Jordan, M. I. (2019). Is There an Analog of Nesterov Acceleration for
       MCMC?
  """
  state = phmc_state.state
  momentum = phmc_state.momentum
  direction = phmc_state.direction
  state_grads = phmc_state.state_grads
  target_log_prob = phmc_state.target_log_prob
  state_extra = phmc_state.state_extra
  pmh_state = phmc_state.pmh_state

  # Impute the optional args.
  if kinetic_energy_fn is None:
    kinetic_energy_fn = fun_mc.make_gaussian_kinetic_energy_fn(
        len(target_log_prob.shape) if target_log_prob.shape is not None else tf
        .rank(target_log_prob), named_axis=named_axis)

  if momentum_sample_fn is None:
    if named_axis is None:
      named_axis = util.map_tree(lambda _: [], state)

    def _momentum_sample_fn(old_momentum: fun_mc.State,
                            seed: Any) -> tuple[fun_mc.State, tuple[()]]:
      seeds = util.unflatten_tree(
          old_momentum,
          util.split_seed(seed, len(util.flatten_tree(old_momentum))))

      def _sample_part(old_momentum, seed, named_axis):
        seed = backend.distribute_lib.fold_in_axis_index(seed, named_axis)
        return (
            tf.math.sqrt(1 - tf.square(noise_fraction)) * old_momentum +
            noise_fraction *
            util.random_normal(old_momentum.shape, old_momentum.dtype, seed))

      new_momentum = util.map_tree_up_to(state, _sample_part, old_momentum,
                                         seeds, named_axis)
      return new_momentum

    momentum_sample_fn = _momentum_sample_fn

  if integrator_fn is None:
    step_size = util.map_tree(tf.convert_to_tensor, step_size)
    step_size = fun_mc.maybe_broadcast_structure(step_size, state)

    def _integrator_fn(
        state: fun_mc.IntegratorState, direction: fun_mc.FloatTensor
    ) -> tuple[fun_mc.IntegratorState, fun_mc.IntegratorExtras]:

      directional_step_size = util.map_tree(
          lambda step_size, state: (  # pylint: disable=g-long-lambda
              step_size * tf.reshape(
                  direction,
                  list(direction.shape) + [1] *
                  (len(state.shape) - len(direction.shape)))),
          step_size,
          state.state)
      # TODO(siege): Ideally we'd pass in the direction here, but the
      # `hamiltonian_integrator` cannot handle dynamic direction switching like
      # that.
      return fun_mc.hamiltonian_integrator(
          state,
          num_steps=num_integrator_steps,
          integrator_step_fn=functools.partial(
              fun_mc.leapfrog_step,
              step_size=directional_step_size,
              target_log_prob_fn=target_log_prob_fn,
              kinetic_energy_fn=kinetic_energy_fn),
          kinetic_energy_fn=kinetic_energy_fn,
          unroll=unroll_integrator,
          max_num_steps=max_num_integrator_steps,
          integrator_trace_fn=integrator_trace_fn)

    integrator_fn = _integrator_fn

  seed, sample_seed = util.split_seed(seed, 2)
  momentum = momentum_sample_fn(momentum, sample_seed)

  initial_integrator_state = fun_mc.IntegratorState(
      target_log_prob=target_log_prob,
      momentum=momentum,
      state=state,
      state_grads=state_grads,
      state_extra=state_extra,
  )

  integrator_state, integrator_extra = integrator_fn(initial_integrator_state,
                                                     direction)

  proposed_state = phmc_state._replace(
      state=integrator_state.state,
      state_grads=integrator_state.state_grads,
      target_log_prob=integrator_state.target_log_prob,
      momentum=integrator_state.momentum,
      state_extra=integrator_state.state_extra,
      # Flip the direction in the proposal, for reversibility.
      direction=util.map_tree(lambda d: -d, direction),
  )

  # Stick the new momentum into phmc_state. We're doing accept/reject purely on
  # the Hamiltonian proposal, not the momentum refreshment kernel.
  phmc_state = phmc_state._replace(momentum=momentum)

  energy_change, energy_change_extra = energy_change_fn(
      initial_integrator_state,
      integrator_state,
      integrator_extra,
  )

  if log_uniform is None:
    pmh_state, pmh_extra = fun_mc.persistent_metropolis_hastings_step(
        pmh_state,
        current_state=phmc_state,
        proposed_state=proposed_state,
        energy_change=energy_change,
        drift=mh_drift)
    is_accepted = pmh_extra.is_accepted
    phmc_state = pmh_extra.accepted_state
  else:
    # We explicitly don't update the PMH state.
    phmc_state, mh_extra = fun_mc.metropolis_hastings_step(
        current_state=phmc_state,
        proposed_state=proposed_state,
        energy_change=energy_change,
        log_uniform=log_uniform)
    is_accepted = mh_extra.is_accepted

  phmc_state = typing.cast(PersistentHamiltonianMonteCarloState, phmc_state)
  phmc_state = phmc_state._replace(
      pmh_state=pmh_state,
      # Flip the direction unconditionally; when the state is accepted, this
      # undoes the flip made in the proposal, maintaining the old momentum
      # direction.
      direction=util.map_tree(lambda d: -d, phmc_state.direction),
  )

  return phmc_state, PersistentHamiltonianMonteCarloExtra(
      is_accepted=is_accepted,
      proposed_phmc_state=proposed_state,
      log_accept_ratio=-energy_change,
      integrator_state=integrator_state,
      integrator_extra=integrator_extra,
      energy_change_extra=energy_change_extra,
      initial_momentum=momentum)
