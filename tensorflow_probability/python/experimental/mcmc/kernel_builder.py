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
"""Transition Kernel convenience builders / modifiers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow_probability.python.experimental.mcmc import kernel_outputs
from tensorflow_probability.python.experimental.mcmc import preconditioned_hmc
from tensorflow_probability.python.experimental.mcmc import progress_bar_reducer
from tensorflow_probability.python.experimental.mcmc import sample_discarding_kernel
from tensorflow_probability.python.experimental.mcmc import step
from tensorflow_probability.python.experimental.mcmc import tracing_reducer
from tensorflow_probability.python.experimental.mcmc import with_reductions
from tensorflow_probability.python.mcmc import dual_averaging_step_size_adaptation
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import langevin
from tensorflow_probability.python.mcmc import nuts
from tensorflow_probability.python.mcmc import random_walk_metropolis
from tensorflow_probability.python.mcmc import replica_exchange_mc
from tensorflow_probability.python.mcmc import simple_step_size_adaptation
from tensorflow_probability.python.mcmc import transformed_kernel
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


# Notes #

# things that will never be supported:
# arbitrary nesting of kernels

# things that might be supported in the future:
# annealed importance sampling (convert to kernel)
# slice sampler (+ step size adaptation)
# step size adaptation for RWM
# REMC step parameter weirdness
# MALA doesn't support step size adaptation
# SMC

# TODO(leben): link step_size_getter between core and step adapters
# TODO(leben): builder support for common reducers
# TODO(leben): step_count_smoothing defaults to 1/7 num_adaptation_steps?
# TODO(leben): implement real automatic tracing
# TODO(leben): automatically generate builder args from class defs (metaclass)
# TODO(leben): momentum adapter support/phmc from hmc+momentum


__all__ = [
    'KernelBuilder'
]


CORE_KERNELS_ADAPTABLE_STEPS = [
    hmc.HamiltonianMonteCarlo,
    nuts.NoUTurnSampler,
    preconditioned_hmc.PreconditionedHamiltonianMonteCarlo,
]


CORE_KERNELS = CORE_KERNELS_ADAPTABLE_STEPS + [
    random_walk_metropolis.RandomWalkMetropolis,
    langevin.MetropolisAdjustedLangevinAlgorithm,  # doesn't work says bjp?
    # SliceSampler,  # has step size but no acceptance prob?
]


DEFAULT_STEP_SIZE = 1.  # ??


DEFAULT_TARGET_ACCEPT_PROB = {
    hmc.HamiltonianMonteCarlo: 0.75,
    langevin.MetropolisAdjustedLangevinAlgorithm: 0.75,  # ??
    nuts.NoUTurnSampler: 0.75,  # ??
    preconditioned_hmc.PreconditionedHamiltonianMonteCarlo: 0.75,
    random_walk_metropolis.RandomWalkMetropolis: 0.25,  # for once RWM is fixed?
}


def _trace_all_results(state, results):
  del state
  return results


class KernelBuilder(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'KernelBuilder',
        [
            'target_log_prob_fn',
            'core_class',
            'core_params',
            'default_step_size_on',
            'step_size',
            'step_adapter_class',
            'step_adapter_params',
            # 'momentum_adapter_class',
            # 'momentum_adapter_params',
            'default_target_accept_prob_on',
            'target_accept_prob',
            'replica_exchange_params',
            'transform_params',
            'num_steps_between_results',
            'auto_tracing_on',
            'tracing_params',
            'show_progress',
            'user_reducer',
        ])
):
  """Convenience constructor for common MCMC transition kernels.

  `KernelBuilder` gives an alternative interface for building MCMC transition
  kernels. It wraps the base `tfp.mcmc` library, offering more convenience at
  the cost of some power and flexibility.

  It is designed to work in conjunction with `KernelOutputs` for more
  convenience.

  Example usage:

  ```
  # Initialize builder with `target_log_prob_fn`.
  builder = KernelBuilder(target_log_prob_fn)
  # Configure initial transition kernel.
  builder = (
    builder
    # Use Hamilton Monte Carlo
    .hmc(num_leapfrog_steps=3)
    # with step size adaptation
    .dual_averaging_adaptation()
    .set_num_adaptation_steps(50)
    # and a transformation
    .transform(my_bijector))

  # Do sampling...
  outputs = builder.sample(num_steps, initial_state)

  # Start from the previous `KernelBuilder` configuration.
  builder = (
    builder
    # Continue using HMC...
    # Still use `my_bijector` transform
    # But with no step size adaptation:
    .clear_step_adapter()
    # set a static step size
    .set_step_size(outputs.new_step_size))

  # More sampling starting from where we left off.
  outputs2 = builder.sample(num_steps, outputs.current_state)

  # Etc ...

  ```

  All methods except `build()` and `sample()` return a modified copy of the
  builder for further method-chaining.  The builder itself is immutable
  (namedtuple) for safe use inside of graphs.

  `KernelBuilder` builds kernels with the following parts (in order):

  1. A core transition kernel
  2. Optional step size adaptation or replica exchange
  3. Transformating bijector
  4. Thinning
  5. Streaming reductions

  The core kernels can be either `HamiltonianMonteCarlo`, `NoUTurnSampler`,
  `PreconditionedHamiltonianMonteCarlo`, `MetropolisAdjustedLangevinAlgorithm`,
  or `RandomWalkMetropolis`.  Support for additional core kernels may be added
  in the future.

  Step size adaption is performed by `SimpleStepSizeAdaptation` or
  `DualAveragingStepSizeAdaptation`. Note not all core kernels are currently
  compatible with step size adaptation.

  `KernelBuilder` maintains some state between kernel builds which can be reused
  or overriden:

   1. Target log prob function
   2. Core kernel class
   3. Step size (initial)
   4. Step size adapter class (optional)
   4a. Number of adaptation steps
   4b. Target acceptance probability
   5. Replica exchange parameters (optional)
   6. `TransformedTransitionKernel` bijector/params (optional)
   7. Thinning: number of steps between results (optional)
   8. Tracing parameters for `TracingReducer` / auto-tracing.
   9. Show progress boolean.
  10. Reductions for `WithReductions`

  See instance method documentation for more information.
  """

  __slots__ = ()

  @classmethod
  def make(cls, target_log_prob_fn):
    """Construct a `KernelBuilder` with empty defaults."""
    builder = cls(
        target_log_prob_fn=target_log_prob_fn,
        core_class=None,
        core_params=None,
        default_step_size_on=True,
        step_size=DEFAULT_STEP_SIZE,
        step_adapter_class=None,
        step_adapter_params=None,
        # momentum_adapter_class=None,
        # momentum_adapter_params=None,
        default_target_accept_prob_on=True,
        target_accept_prob=.5,
        replica_exchange_params=None,
        transform_params=None,
        num_steps_between_results=0,
        auto_tracing_on=True,
        tracing_params=None,
        show_progress=False,
        user_reducer=None,
    )
    return builder.set_tracing()

  def build(self, num_steps=None):
    """Build and return the specified kernel.

    Args:
      num_steps: An integer. Some kernel pieces (step adaptation) require
        knowing the number of steps to sample in advance; pass that in here.

    Returns:
      kernel: The configured `TransitionKernel`.
    """

    if num_steps is None:
      if self.step_adapter_class or self.show_progress:
        raise ValueError(
            '`num_steps` is required for step adaptation or progress bars.')

    def build_inner(target_log_prob_fn):
      kernel = self.core_class(**self._build_core_params(target_log_prob_fn))
      if self.step_adapter_class is not None:
        assert self.core_class in CORE_KERNELS_ADAPTABLE_STEPS
        kernel = self.step_adapter_class(
            **self._build_step_adapter_params(kernel, num_steps))
      return kernel

    if self.replica_exchange_params is not None:
      kernel = replica_exchange_mc.ReplicaExchangeMC(
          target_log_prob_fn=self.target_log_prob_fn,
          make_kernel_fn=build_inner,
          **self.replica_exchange_params)
    else:
      kernel = build_inner(self.target_log_prob_fn)

    if self.transform_params is not None:
      kernel = transformed_kernel.TransformedTransitionKernel(
          **dict(self.transform_params, inner_kernel=kernel))

    if self.num_steps_between_results > 0:
      kernel = sample_discarding_kernel.SampleDiscardingKernel(
          kernel, num_steps_between_results=self.num_steps_between_results)

    reducers = self._build_reducers(size=num_steps)
    if reducers:
      kernel = with_reductions.WithReductions(
          inner_kernel=kernel, reducer=reducers)

    return kernel

  def sample(self, num_steps, current_state, previous_kernel_results=None):
    """Sample from the configured kernel.

    Args:
      num_steps: Integer number of Markov chain steps.
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s).
      previous_kernel_results: A `Tensor` or a nested collection of `Tensor`s.
        Warm-start for the auxiliary state needed by the given `kernel`.
        If not supplied, `step_kernel` will cold-start with
        `kernel.bootstrap_results`.

    Returns:
      outputs: A `KernelOutputs` object containing the states, trace, etc.
    """
    kernel = self.build(num_steps)
    state, results = step.step_kernel(
        num_steps=num_steps,
        current_state=current_state,
        previous_kernel_results=previous_kernel_results,
        kernel=kernel,
        return_final_kernel_results=True)
    return kernel_outputs.KernelOutputs(kernel, state, results)

  ## Core kernels

  def get_step_size(self):
    """Return the set or default step size."""
    return DEFAULT_STEP_SIZE if self.default_step_size_on else self.step_size

  def set_step_size(self, step_size):
    """Set the step size (for core kernels with a step size.)"""
    return self._replace(default_step_size_on=False, step_size=step_size)

  def use_default_step_size(self, on=True):
    """Use default step size (or not.)"""
    return self._replace(default_step_size_on=on)

  def hmc(self,
          num_leapfrog_steps,
          state_gradients_are_stopped=False,
          store_parameters_in_results=False,
          name=None):
    """Use the `HamiltonianMonteCarlo` core transition kernel.

    See the `HamiltonianMonteCarlo` docs for more details.

    Args:
      num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
        for. Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
      state_gradients_are_stopped: Python `bool` indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      store_parameters_in_results: If `True`, then `step_size` and
        `num_leapfrog_steps` are written to and read from eponymous fields in
        the kernel results objects returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (e.g., 'hmc_kernel').

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._set_core_kernel(
        hmc.HamiltonianMonteCarlo,
        {'num_leapfrog_steps': num_leapfrog_steps,
         'state_gradients_are_stopped': state_gradients_are_stopped,
         'store_parameters_in_results': store_parameters_in_results,
         'name': name})

  def mala(self,
           volatility_fn=None,
           parallel_iterations=10,
           name=None):
    """Use the `MetropolisAdjustedLangevinAlgorithm` core transition kernel.

    See the `MetropolisAdjustedLangevinAlgorithm` docs for more details.

    Args:
      volatility_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns
        volatility value at `current_state`. Should return a `Tensor` or Python
        `list` of `Tensor`s that must broadcast with the shape of
        `current_state` Defaults to the identity function.
      parallel_iterations: the number of coordinates for which the gradients of
        the volatility matrix `volatility_fn` can be computed in parallel.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'mala_kernel').

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._set_core_kernel(
        langevin.MetropolisAdjustedLangevinAlgorithm,
        {'volatility_fn': volatility_fn,
         'parallel_iterations': parallel_iterations,
         'name': name})

  def nuts(self,
           max_tree_depth=10,
           max_energy_diff=1000.,
           unrolled_leapfrog_steps=1,
           parallel_iterations=10,
           name=None):
    """Use the `NoUTurnSampler` core kernel.

    See the `NoUTurnSampler` docs for more details.

    Args:
      max_tree_depth: Maximum depth of the tree implicitly built by NUTS. The
        maximum number of leapfrog steps is bounded by `2**max_tree_depth` i.e.
        the number of nodes in a binary tree `max_tree_depth` nodes deep. The
        default setting of 10 takes up to 1024 leapfrog steps.
      max_energy_diff: Scaler threshold of energy differences at each leapfrog,
        divergence samples are defined as leapfrog steps that exceed this
        threshold. Default to 1000.
      unrolled_leapfrog_steps: The number of leapfrogs to unroll per tree
        expansion step. Applies a direct linear multipler to the maximum
        trajectory length implied by max_tree_depth. Defaults to 1.
      parallel_iterations: The number of iterations allowed to run in parallel.
        It must be a positive integer. See `tf.while_loop` for more details.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'nuts_kernel').

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._set_core_kernel(
        nuts.NoUTurnSampler,
        {'max_tree_depth': max_tree_depth,
         'max_energy_diff': max_energy_diff,
         'unrolled_leapfrog_steps': unrolled_leapfrog_steps,
         'parallel_iterations': parallel_iterations,
         'name': name})

  def phmc(self,
           num_leapfrog_steps,
           momentum_distribution=None,
           state_gradients_are_stopped=False,
           store_parameters_in_results=False,
           name=None):
    """Use the `PreconditionedHamiltonianMonteCarlo` core transition kernel.

    See the `PreconditionedHamiltonianMonteCarlo` docs for more details.

    Args:
      num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
        for. Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
      momentum_distribution: A `tfp.distributions.Distribution` instance to draw
        momentum from. Defaults to isotropic normal distributions.
      state_gradients_are_stopped: Python `bool` indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      store_parameters_in_results: If `True`, then `step_size`,
        `momentum_distribution`, and `num_leapfrog_steps` are written to and
        read from eponymous fields in the kernel results objects returned from
        `one_step` and `bootstrap_results`. This allows wrapper kernels to
        adjust those parameters on the fly. In case this is `True`, the
        `momentum_distribution` must be a `CompositeTensor`. See
        `tfp.experimental.as_composite` and `tfp.experimental.auto_composite`.
        This is incompatible with `step_size_update_fn`, which must be set to
        `None`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._set_core_kernel(
        preconditioned_hmc.PreconditionedHamiltonianMonteCarlo,
        {'num_leapfrog_steps': num_leapfrog_steps,
         'momentum_distribution': momentum_distribution,
         'state_gradients_are_stopped': state_gradients_are_stopped,
         'store_parameters_in_results': store_parameters_in_results,
         'name': name})

  def rwm(self,
          new_state_fn=None,
          name=None):
    """Use the `RandomWalkMetropolis` core kernel.

    See the `RandomWalkMetropolis` docs for more details.

    Args:
      new_state_fn: Python callable which takes a list of state parts and a
        seed; returns a same-type `list` of `Tensor`s, each being a perturbation
        of the input state parts. The perturbation distribution is assumed to be
        a symmetric distribution centered at the input state part.
        Default value: `None` which is mapped to
          `tfp.mcmc.random_walk_normal_fn()`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'rwm_kernel').

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._set_core_kernel(
        random_walk_metropolis.RandomWalkMetropolis,
        {'new_state_fn': new_state_fn,
         'name': name})

  def _set_core_kernel(self, cls, params):
    if (self.step_adapter_class is not None
        and cls not in CORE_KERNELS_ADAPTABLE_STEPS):
      raise ValueError('Core kernel not compatible with step size adaptation.')
    return self._replace(core_class=cls, core_params=params)

  def _build_core_params(self, target_log_prob_fn):
    extra = {'target_log_prob_fn': target_log_prob_fn}
    if self.core_class in CORE_KERNELS_ADAPTABLE_STEPS:
      extra['step_size'] = self.get_step_size()
    return dict(self.core_params, **extra)

  ## Step size adaptation

  def get_target_accept_prob(self):
    """Return the set `target_accept_prob` or the default for the core kernel."""
    return (DEFAULT_TARGET_ACCEPT_PROB[self.core_class]
            if self.default_target_accept_prob_on
            else self.target_accept_prob)

  def set_target_accept_prob(self, target_accept_prob):
    """Set the target acceptance for step adaptation kernels.

    Args:
      target_accept_prob: A floating point `Tensor` representing desired
        acceptance probability. Must be a positive number less than 1. This can
        either be a scalar, or have shape [num_chains].  By default, this is
        0.25 for `RandomWalkMetropolis` and 0.75 for `HamiltonianMonteCarlo`,
        `MetropolisAdjustedLangevinAlgorithm` and `NoUTurnSampler`.

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._replace(default_target_accept_prob_on=False,
                         target_accept_prob=target_accept_prob)

  def use_default_target_accept_prob(self, on=True):
    """Use per-core class default target acceptance probability (or not.)"""
    return self._replace(default_target_accept_prob_on=on)

  def dual_averaging_adaptation(
      self,
      exploration_shrinkage=0.05,
      shrinkage_target=None,
      step_count_smoothing=10,
      decay_rate=0.75,
      step_size_setter_fn=
      simple_step_size_adaptation.hmc_like_step_size_setter_fn,
      step_size_getter_fn=
      simple_step_size_adaptation.hmc_like_step_size_getter_fn,
      log_accept_prob_getter_fn=
      simple_step_size_adaptation.hmc_like_log_accept_prob_getter_fn,
      validate_args=False,
      name=None):
    """Use `DualAveragingStepSizeAdaptation`.

    See the `DualAveragingStepSizeAdaptation` docs for more details.

    Args:
      exploration_shrinkage: Floating point scalar `Tensor`. How strongly the
        exploration rate is biased towards the shrinkage target.
      shrinkage_target: `Tensor` or list of tensors. Value the exploration
        step size(s) is/are biased towards.
        As `num_adaptation_steps --> infinity`, this bias goes to zero.
        Defaults to 10 times the initial step size.
      step_count_smoothing: Int32 scalar `Tensor`. Number of "pseudo-steps"
        added to the number of steps taken to prevents noisy exploration during
        the early samples.
      decay_rate: Floating point scalar `Tensor`. How much to favor recent
        iterations over earlier ones. A value of 1 gives equal weight to all
        history. A value of 0 gives weight only to the most recent iteration.
      step_size_setter_fn: A callable with the signature `(kernel_results,
        new_step_size) -> new_kernel_results` where `kernel_results` are the
        results of the `inner_kernel`, `new_step_size` is a `Tensor` or a nested
        collection of `Tensor`s with the same structure as returned by the
        `step_size_getter_fn`, and `new_kernel_results` are a copy of
        `kernel_results` with the step size(s) set.
      step_size_getter_fn: A callable with the signature `(kernel_results) ->
        step_size` where `kernel_results` are the results of the `inner_kernel`,
        and `step_size` is a floating point `Tensor` or a nested collection of
        such `Tensor`s.
      log_accept_prob_getter_fn: A callable with the signature `(kernel_results)
        -> log_accept_prob` where `kernel_results` are the results of the
        `inner_kernel`, and `log_accept_prob` is a floating point `Tensor`.
        `log_accept_prob` can either be a scalar, or have shape [num_chains]. If
        it's the latter, `step_size` should also have the same leading
        dimension.
      validate_args: Python `bool`. When `True` kernel parameters are checked
        for validity. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'dual_averaging_step_size_adaptation').

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._set_step_size_adapter(
        dual_averaging_step_size_adaptation.DualAveragingStepSizeAdaptation,
        {'exploration_shrinkage': exploration_shrinkage,
         'shrinkage_target': shrinkage_target,
         'step_count_smoothing': step_count_smoothing,
         'decay_rate': decay_rate,
         'step_size_setter_fn': step_size_setter_fn,
         'step_size_getter_fn': step_size_getter_fn,
         'log_accept_prob_getter_fn': log_accept_prob_getter_fn,
         'validate_args': validate_args,
         'name': name})

  def simple_adaptation(
      self,
      adaptation_rate=0.01,
      step_size_setter_fn=
      simple_step_size_adaptation.hmc_like_step_size_setter_fn,
      step_size_getter_fn=
      simple_step_size_adaptation.hmc_like_step_size_getter_fn,
      log_accept_prob_getter_fn=
      simple_step_size_adaptation.hmc_like_log_accept_prob_getter_fn,
      validate_args=False,
      name=None):
    """Use `SimpleStepSizeAdaptation`.

    See the `SimpleStepSizeAdaptation` docs for more details.

    Args:
      adaptation_rate: `Tensor` representing amount to scale the current
        `step_size`.
      step_size_setter_fn: A callable with the signature
        `(kernel_results, new_step_size) -> new_kernel_results` where
        `kernel_results` are the results of the `inner_kernel`, `new_step_size`
        is a `Tensor` or a nested collection of `Tensor`s with the same
        structure as returned by the `step_size_getter_fn`, and
        `new_kernel_results` are a copy of `kernel_results` with the step
        size(s) set.
      step_size_getter_fn: A callable with the signature
        `(kernel_results) -> step_size` where `kernel_results` are the results
        of the `inner_kernel`, and `step_size` is a floating point `Tensor` or a
        nested collection of such `Tensor`s.
      log_accept_prob_getter_fn: A callable with the signature
        `(kernel_results) -> log_accept_prob` where `kernel_results` are the
        results of the `inner_kernel`, and `log_accept_prob` is a floating point
        `Tensor`. `log_accept_prob` can either be a scalar, or have shape
        [num_chains]. If it's the latter, `step_size` should also have the same
        leading dimension.
      validate_args: Python `bool`. When `True` kernel parameters are checked
        for validity. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class. Default:
        'simple_step_size_adaptation'.

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._set_step_size_adapter(
        simple_step_size_adaptation.SimpleStepSizeAdaptation,
        {'adaptation_rate': adaptation_rate,
         'step_size_setter_fn': step_size_setter_fn,
         'step_size_getter_fn': step_size_getter_fn,
         'log_accept_prob_getter_fn': log_accept_prob_getter_fn,
         'validate_args': validate_args,
         'name': name})

  def clear_step_adapter(self):
    """Removes step adaptation."""
    return self._replace(
        step_adapter_class=None,
        step_adapter_params=None)

  def _set_step_size_adapter(self, cls, params):
    if (self.core_class is not None
        and self.core_class not in CORE_KERNELS_ADAPTABLE_STEPS):
      raise ValueError('Core kernel not compatible with step size adaptation.')
    return self._replace(
        step_adapter_class=cls,
        step_adapter_params=params)

  def _build_step_adapter_params(self, kernel, num_adaptation_steps):
    return dict(self.step_adapter_params,
                inner_kernel=kernel,
                num_adaptation_steps=num_adaptation_steps,
                target_accept_prob=self.get_target_accept_prob())

  # Replica exchange

  def replica_exchange(
      self,
      inverse_temperatures,
      swap_proposal_fn=replica_exchange_mc.default_swap_proposal_fn(1.),
      state_includes_replicas=False,
      validate_args=False,
      name=None):
    """Use `ReplicaExchangeMC`.

    See the `ReplicaExchangeMC` docs for more details.

    Args:
      inverse_temperatures: `Tensor` of inverse temperatures to temper each
        replica. The leftmost dimension is the `num_replica` and the
        second dimension through the rightmost can provide different temperature
        to different batch members, doing a left-justified broadcast.
      swap_proposal_fn: Python callable which take a number of replicas, and
        returns `swaps`, a shape `[num_replica] + batch_shape` `Tensor`, where
        axis 0 indexes a permutation of `{0,..., num_replica-1}`, designating
        replicas to swap.
      state_includes_replicas: Boolean indicating whether the leftmost dimension
        of each state sample should index replicas. If `True`, the leftmost
        dimension of the `current_state` kwarg to `tfp.mcmc.sample_chain` will
        be interpreted as indexing replicas.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "remc_kernel").

    Raises:
      ValueError: `inverse_temperatures` doesn't have statically known 1D shape.

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._replace(replica_exchange_params={
        'inverse_temperatures': inverse_temperatures,
        'swap_proposal_fn': swap_proposal_fn,
        'state_includes_replicas': state_includes_replicas,
        'validate_args': validate_args,
        'name': name,
    })

  def clear_replica_exchange(self):
    return self._replace(replica_exchange_params=None)

  ## Transformations

  def transform(self, bijector, name=None):
    """Use `TransformedTransitionKernel`.

    See `TransformedTransitionKernel` docs for more details.

    Args:
      bijector: `tfp.distributions.Bijector` or list of
        `tfp.distributions.Bijector`s. These bijectors use `forward` to map the
        `inner_kernel` state space to the state expected by
        `inner_kernel.target_log_prob_fn`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "transformed_kernel").

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._replace(transform_params={
        'bijector': bijector,
        'name': name,
    })

  def clear_transform(self):
    """Remove previously set `TransformedTransitionKernel`."""
    return self._replace(transform_params=None)

  ## Thinning

  def set_num_steps_between_results(self, num_steps_between_results):
    """Thin sampling by `num_steps_between_results`."""
    return self._replace(num_steps_between_results=num_steps_between_results)

  ## Progress bar

  def set_show_progress(self, on=True):
    return self._replace(show_progress=on)

  ## Tracing

  def set_tracing(self, trace_fn=_trace_all_results, size=None, name=None):
    """Trace sampling state and results.

    See the `TracingReducer` docs for more details.

    Args:
      trace_fn: A callable that takes in the current chain state and the
        previous kernel results and return a `Tensor` or a nested collection
        of `Tensor`s that is accumulated across samples.
      size: Integer or scalar `Tensor` denoting the size of the accumulated
        `TensorArray`. If this is `None` (which is the default), a
        dynamic-shaped `TensorArray` will be used.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'tracing_reducer').

    Returns:
      self: Returns the builder for more method chaining.
    """
    def real_trace_fn(state, results):
      return state, trace_fn(state, results)
    return self._replace(tracing_params={
        'trace_fn': real_trace_fn,
        'size': size,
        'name': name,
    })

  def set_auto_tracing(self, on=True):
    """Add smart tracing."""
    # Do something smart here.
    return self._replace(auto_tracing_on=on)

  def clear_tracing(self):
    """Remove `TracingReducer`."""
    return self._replace(tracing_params=None, auto_tracing_on=False)

  ## Reductions

  def set_reducer(self, reducer):
    """Use `tfp.experimental.mcmc.WithReductions`.

    See the `WithReductions` docs for more details.

    Args:
      reducer: A (possibly nested) structure of `Reducer`s to be evaluated
        on the `inner_kernel`'s samples.

    Returns:
      self: Returns the builder for more method chaining.
    """
    return self._replace(user_reducer=reducer)

  def clear_reducer(self):
    """Remove previously set reductions."""
    return self.set_reducer(reducer=None)

  def _build_reducers(self, size=None):
    reducers = {}
    if self.user_reducer is not None:
      reducers['user'] = self.user_reducer
    tracing_params = self.tracing_params
    if tracing_params is None and self.auto_tracing_on:
      tracing_params = self.set_tracing(size=size).tracing_params
    if tracing_params is not None:
      reducers['tracing'] = tracing_reducer.TracingReducer(**tracing_params)
    if self.show_progress:
      reducers['progress'] = progress_bar_reducer.ProgressBarReducer(size)
    return reducers

  ## Add common reducers here
