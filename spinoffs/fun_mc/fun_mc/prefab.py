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

from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Tuple  # pylint: disable=unused-import

import numpy as np

from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc

tf = backend.tf
tfp = backend.tfp
util = backend.util

__all__ = [
    'adaptive_hamiltonian_monte_carlo_init',
    'adaptive_hamiltonian_monte_carlo_step',
    'AdaptiveHamiltonianMonteCarloState',
    'hamiltonian_monte_carlo_with_state_grads_step',
    'HamiltonianMonteCarloWithStateGradsExtra',
    'interactive_trace',
    'step_size_adaptation_init',
    'step_size_adaptation_step',
    'StepSizeAdaptationExtra',
    'StepSizeAdaptationState',
]


@util.named_call(name='polynomial_decay')
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


class AdaptiveHamiltonianMonteCarloState(NamedTuple):
  """Adaptive HMC `TransitionOperator` state."""
  hmc_state: 'fun_mc.HamiltonianMonteCarloState'
  running_var_state: 'fun_mc.RunningVarianceState'
  ssa_state: 'StepSizeAdaptationState'
  step: 'fun_mc.IntTensor'


class AdaptiveHamiltonianMonteCarloExtra(NamedTuple):
  """Extra outputs for Adaptive HMC `TransitionOperator`."""
  hmc_state: 'fun_mc.HamiltonianMonteCarloState'
  hmc_extra: 'fun_mc.HamiltonianMonteCarloExtra'
  step_size: 'fun_mc.FloatTensor'
  num_leapfrog_steps: 'fun_mc.IntTensor'
  mean_num_leapfrog_steps: 'fun_mc.IntTensor'

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


@util.named_call
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
    step_size_averaging_window_steps: 'fun_mc.IntTensor' = 100,
    jitter_sample_fn:
    'Callable[[fun_mc.IntTensor, fun_mc.IntTensor, Any], fun_mc.IntTensor]' = (
        _uniform_jitter),
    log_accept_ratio_reduce_fn:
    'Callable[[fun_mc.FloatTensor], fun_mc.FloatTensor]' = (
        tfp.math.reduce_logmeanexp),
    hmc_kwargs: 'Optional[Dict[str, Any]]' = None,
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


def _tqdm_progress_bar_fn(iterable: 'Iterable[Any]') -> 'Iterable[Any]':
  """The TQDM progress bar function."""
  # pytype: disable=import-error
  import tqdm  # pylint: disable=g-import-not-at-top
  # pytype: enable=import-error
  return tqdm.tqdm(iterable, leave=True)


def interactive_trace(
    state: 'fun_mc.State',
    fn: 'fun_mc.TransitionOperator',
    num_steps: 'fun_mc.IntTensor',
    trace_mask: 'fun_mc.BooleanNest' = True,
    iteration_axis: int = 0,
    block_until_ready: 'bool' = True,
    progress_bar_fn: 'Callable[[Iterable[Any]], Iterable[Any]]' = (
        _tqdm_progress_bar_fn),
) -> 'Tuple[fun_mc.State, fun_mc.TensorNest]':
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


class StepSizeAdaptationState(NamedTuple):
  """Step size adaptation state."""
  step: 'fun_mc.IntTensor'
  opt_state: 'fun_mc.AdamState'
  rms_state: 'fun_mc.RunningMeanState'

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
  opt_extra: 'fun_mc.AdamExtra'
  accept_prob: 'fun_mc.FloatTensor'


@util.named_call
def step_size_adaptation_init(
    init_step_size: 'fun_mc.FloatTensor') -> 'StepSizeAdaptationState':
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
    state: 'StepSizeAdaptationState',
    log_accept_ratio: 'fun_mc.FloatTensor',
    num_adaptation_steps: 'Optional[fun_mc.IntTensor]',
    target_accept_prob: 'fun_mc.FloatTensor' = 0.8,
    adaptation_rate: 'fun_mc.FloatTensor' = 0.05,
    adaptation_rate_decay_power: 'fun_mc.FloatTensor' = 0.1,
    averaging_window_steps: 'fun_mc.IntTensor' = 100,
    min_log_accept_prob: 'fun_mc.FloatTensor' = np.log(1e-5),
    reduce_fn: 'Callable[[fun_mc.FloatTensor], fun_mc.FloatTensor]' = (
        tfp.math.reduce_logmeanexp),
) -> 'Tuple[StepSizeAdaptationState, StepSizeAdaptationExtra]':
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

  Returns:
    step_size_adaptation_state: `StepSizeAdaptationState`
    step_size_adaptation_extra: `StepSizeAdaptationExtra`
  """
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
                                          adaptation_rate)

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


class HamiltonianMonteCarloWithStateGradsExtra(NamedTuple):
  """Extra outputs for 'hamiltonian_monte_carlo_with_state_grads_step'."""
  hmc_extra: 'fun_mc.HamiltonianMonteCarloExtra'
  num_integrator_steps: 'fun_mc.IntTensor'
  proposed_state: 'fun_mc.State'


def hamiltonian_monte_carlo_with_state_grads_step(
    hmc_state: 'fun_mc.HamiltonianMonteCarloState',
    trajectory_length: 'fun_mc.FloatTensor',
    scalar_step_size: 'fun_mc.FloatTensor',
    step_size_scale: 'fun_mc.FloatNest' = 1.,
    shard_axis_names: 'fun_mc.StringNest' = (),
    **hmc_kwargs
) -> ('Tuple[fun_mc.HamiltonianMonteCarloState, '
      'HamiltonianMonteCarloWithStateGradsExtra]'):
  """Hamiltonian Monte Carlo (HMC) step with gradients for proposed state.

  This acts as a `fun_mc.hamiltonian_monte_carlo_step`, where the
  `num_integrator_steps` is defined as `ceil(trajectory_length /
  scalar_step_size)` and `step_size` is defined as `scalar_step_size *
  step_size_scale`. The main feature of this function is that it propagates the
  gradients from `hmc_with_state_grads_extra.proposed_state` to
  `trajectory_length` (these are the only gradients propagated at the moment).
  This feature can be used to do gradient-based optimization of
  `trajectory_length` based on criteria that depend on the `proposed_state`
  (e.g. [1]).

  This function supports SPMD via sharded states in the same sense as TensorFlow
  Probability's `tfp.experimental.distribute.Sharded`. Certain state tensors can
  be annotated as having different values on different devices, with
  cross-device reductions being inserted accordingly.

  Args:
    hmc_state: `fun_mc.HamiltonianMonteCarloState`.
    trajectory_length: Trajectory length used by HMC.
    scalar_step_size: Scalar step size (used to compute the number of leapfrog
      steps).
    step_size_scale: Step size scale, structure broadcastable to the
      `hmc_state.state`.
    shard_axis_names: Shard axes names, used for SPMD.
    **hmc_kwargs: Passed to `fun_mc.hamiltonian_monte_carlo_step`.

  Returns:
    hmc_state: `fun_mc.HamiltonianMonteCarloState`.
    hmc_with_grads_extra: Extra outputs.

  #### References

  [1]: Hoffman, M., Radul, A., & Sountsov, P. (2021). An Adaptive MCMC Scheme
       for Setting Trajectory Lengths in Hamiltonian Monte Carlo.
       http://proceedings.mlr.press/v130/hoffman21a.html
  """

  @tf.custom_gradient
  def hmc(trajectory_length):
    trajectory_length = tf.convert_to_tensor(trajectory_length)
    num_integrator_steps = tf.cast(
        tf.math.ceil(trajectory_length / scalar_step_size), tf.int32)
    # In case something goes negative.
    num_integrator_steps = tf.maximum(1, num_integrator_steps)
    new_hmc_state, hmc_extra = fun_mc.hamiltonian_monte_carlo_step(
        hmc_state,
        num_integrator_steps=num_integrator_steps,
        step_size=util.map_tree(lambda s: s * scalar_step_size,
                                step_size_scale),
        **hmc_kwargs)
    hmc_with_grads_extra = HamiltonianMonteCarloWithStateGradsExtra(
        proposed_state=hmc_extra.proposed_hmc_state.state,
        hmc_extra=hmc_extra,
        num_integrator_steps=num_integrator_steps)
    res = (new_hmc_state, hmc_with_grads_extra)

    def grad(*grads):
      grads = util.unflatten_tree(res, util.flatten_tree(grads))

      step_size_scale_bc = fun_mc.maybe_broadcast_structure(
          step_size_scale, hmc_extra.integrator_extra.momentum_grads)

      # We wish to compute `grads^T @
      # jacobian(proposed_state(trajectory_length))`.
      #
      # The Jacobian is known from from Hamilton's equations:
      #
      # dx / dt = dK(v) / dv
      #
      # where `x` is the state, `v` is the momentum and `K` is the kinetic
      # energy. Since `step_size_scale` rescales momentum, we the right hand
      # side of that expression is `momentum_grads * step_size_scale` by the
      # chain rule. Since the Jacobian in question has 1 row, the
      # vector-Jacobian product is simply the dot product.
      state_grads = util.map_tree(lambda s, m, g: s * m * g, step_size_scale_bc,
                                  hmc_extra.integrator_extra.momentum_grads,
                                  grads[1].proposed_state)

      def do_sum(x, shard_axis_names):
        res = tf.reduce_sum(
            x, list(range(len(trajectory_length.shape), len(x.shape))))
        if shard_axis_names:
          res = backend.distribute_lib.psum(res, shard_axis_names)
        return res

      if shard_axis_names:
        shard_axis_names_bc = shard_axis_names
      else:
        shard_axis_names_bc = util.map_tree(lambda _: [], state_grads)

      return sum(
          util.flatten_tree(
              util.map_tree_up_to(state_grads, do_sum, state_grads,
                                  shard_axis_names_bc)))

    return res, grad

  return hmc(trajectory_length)
