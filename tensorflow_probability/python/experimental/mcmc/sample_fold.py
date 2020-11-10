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
"""Drivers for streaming reductions framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import sample as exp_sample_lib
from tensorflow_probability.python.experimental.mcmc import sample_discarding_kernel
from tensorflow_probability.python.experimental.mcmc import tracing_reducer
from tensorflow_probability.python.experimental.mcmc import with_reductions
from tensorflow_probability.python.mcmc import sample
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'sample_chain',
    'sample_fold',
]


def sample_fold(
    num_steps,
    current_state,
    previous_kernel_results=None,
    kernel=None,
    reducer=None,
    previous_reducer_state=None,
    return_final_reducer_states=False,
    num_burnin_steps=0,
    num_steps_between_results=0,
    parallel_iterations=10,
    seed=None,
    name=None,
):
  """Computes the requested reductions over the `kernel`'s samples.

  To wit, runs the given `kernel` for `num_steps` steps, and consumes
  the stream of samples with the given `Reducer`s' `one_step` method(s).
  This runs in constant memory (unless a given `Reducer` builds a
  large structure).

  The driver internally composes the correct onion of `WithReductions`
  and `SampleDiscardingKernel` to implement the requested optionally
  thinned reduction; however, the kernel results of those applied
  Transition Kernels will not be returned. Hence, if warm-restarting
  reductions is desired, one should manually build the Transition Kernel
  onion and use `tfp.experimental.mcmc.step_kernel`.

  An arbitrary collection of `reducer` can be provided, and the resulting
  finalized statistic(s) will be returned in an identical structure.

  This function can sample from and reduce over multiple chains, in parallel.
  Whether or not there are multiple chains is dictated by how the `kernel`
  treats its inputs.  Typically, the shape of the independent chains is shape of
  the result of the `target_log_prob_fn` used by the `kernel` when applied to
  the given `current_state`.

  Args:
    num_steps: Integer or scalar `Tensor` representing the number of `Reducer`
      steps.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s).
    previous_kernel_results: A `Tensor` or a nested collection of `Tensor`s.
      Warm-start for the auxiliary state needed by the given `kernel`.
      If not supplied, `sample_fold` will cold-start with
      `kernel.bootstrap_results`.
    kernel: An instance of `tfp.mcmc.TransitionKernel` which implements one step
      of the Markov chain.
    reducer: A (possibly nested) structure of `Reducer`s to be evaluated
      on the `kernel`'s samples. If no reducers are given (`reducer=None`),
      then `None` will be returned in place of streaming calculations.
    previous_reducer_state: A (possibly nested) structure of running states
      corresponding to the structure in `reducer`.  For resuming streaming
      reduction computations begun in a previous run.
    return_final_reducer_states: A Python `bool` giving whether to return
      resumable final reducer states.
    num_burnin_steps: Integer or scalar `Tensor` representing the number
        of chain steps to take before starting to collect results.
        Defaults to 0 (i.e., no burn-in).
    num_steps_between_results: Integer or scalar `Tensor` representing
      the number of chain steps between collecting a result. Only one out
      of every `num_steps_between_samples + 1` steps is included in the
      returned results. Defaults to 0 (i.e., no thinning).
    parallel_iterations: The number of iterations allowed to run in parallel. It
      must be a positive integer. See `tf.while_loop` for more details.
    seed: Optional seed for reproducible sampling.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'mcmc_sample_fold').

  Returns:
    reduction_results: A (possibly nested) structure of finalized reducer
      statistics. The structure identically mimics that of `reducer`.
    end_state: The final state of the Markov chain(s).
    final_kernel_results: `collections.namedtuple` of internal calculations
      used to advance the supplied `kernel`. These results do not include
      the kernel results of `WithReductions` or `SampleDiscardingKernel`.
    final_reducer_states: A (possibly nested) structure of final running reducer
      states, if `return_final_reducer_states` was `True`.  Can be used to
      resume streaming reductions when continuing sampling.
  """
  with tf.name_scope(name or 'mcmc_sample_fold'):
    num_steps = tf.convert_to_tensor(
        num_steps, dtype=tf.int32, name='num_steps')
    current_state = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(x, name='current_state'),
        current_state)
    reducer_was_none = False
    if reducer is None:
      reducer = []
      reducer_was_none = True
    thinning_kernel = sample_discarding_kernel.SampleDiscardingKernel(
        inner_kernel=kernel,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results)
    reduction_kernel = with_reductions.WithReductions(
        inner_kernel=thinning_kernel,
        reducer=reducer,
        # Strip thinning kernel results layer
        adjust_kr_fn=lambda kr: kr.inner_results,
    )
    if previous_kernel_results is None:
      previous_kernel_results = kernel.bootstrap_results(current_state)
    thinning_pkr = thinning_kernel.bootstrap_results(
        current_state, previous_kernel_results)
    reduction_pkr = reduction_kernel.bootstrap_results(
        current_state, thinning_pkr, previous_reducer_state)

    end_state, final_kernel_results = exp_sample_lib.step_kernel(
        num_steps=num_steps,
        current_state=current_state,
        previous_kernel_results=reduction_pkr,
        kernel=reduction_kernel,
        return_final_kernel_results=True,
        parallel_iterations=parallel_iterations,
        seed=seed,
        name=name,
    )
    reduction_results = nest.map_structure_up_to(
        reducer,
        lambda r, s: r.finalize(s),
        reducer,
        final_kernel_results.reduction_results,
        check_types=False)
    if reducer_was_none:
      reduction_results = None
    # TODO(axch): Choose a friendly return value convention that
    # - Doesn't burden the user with needless stuff when they don't want it
    # - Supports warm restart when the user does want it
    # - Doesn't trigger Pylint's unbalanced-tuple-unpacking warning.
    if return_final_reducer_states:
      return (reduction_results,
              end_state,
              final_kernel_results.inner_results.inner_results,
              final_kernel_results.reduction_results)
    else:
      return (reduction_results,
              end_state,
              final_kernel_results.inner_results.inner_results)


def _trace_kernel_results(current_state, kernel_results):
  del current_state
  return kernel_results


def sample_chain(
    num_results,
    current_state,
    previous_kernel_results=None,
    kernel=None,
    num_burnin_steps=0,
    num_steps_between_results=0,
    trace_fn=_trace_kernel_results,
    return_final_kernel_results=False,
    parallel_iterations=10,
    seed=None,
    name=None,
):
  """Implements Markov chain Monte Carlo via repeated `TransitionKernel` steps.

  This function samples from a Markov chain at `current_state` whose
  stationary distribution is governed by the supplied `TransitionKernel`
  instance (`kernel`).

  This function can sample from multiple chains, in parallel. (Whether or not
  there are multiple chains is dictated by the `kernel`.)

  The `current_state` can be represented as a single `Tensor` or a `list` of
  `Tensors` which collectively represent the current state.

  Since MCMC states are correlated, it is sometimes desirable to produce
  additional intermediate states, and then discard them, ending up with a set of
  states with decreased autocorrelation.  See [Owen (2017)][1]. Such 'thinning'
  is made possible by setting `num_steps_between_results > 0`. The chain then
  takes `num_steps_between_results` extra steps between the steps that make it
  into the results. The extra steps are never materialized, and thus do not
  increase memory requirements.

  In addition to returning the chain state, this function supports tracing of
  auxiliary variables used by the kernel. The traced values are selected by
  specifying `trace_fn`. By default, all kernel results are traced but in the
  future the default will be changed to no results being traced, so plan
  accordingly. See below for some examples of this feature.

  Args:
    num_results: Integer number of Markov chain draws.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s).
    previous_kernel_results: A `Tensor` or a nested collection of `Tensor`s
      representing internal calculations made within the previous call to this
      function (or as returned by `bootstrap_results`).
    kernel: An instance of `tfp.mcmc.TransitionKernel` which implements one step
      of the Markov chain.
    num_burnin_steps: Integer number of chain steps to take before starting to
      collect results.
      Default value: 0 (i.e., no burn-in).
    num_steps_between_results: Integer number of chain steps between collecting
      a result. Only one out of every `num_steps_between_samples + 1` steps is
      included in the returned results.  The number of returned chain states is
      still equal to `num_results`.  Default value: 0 (i.e., no thinning).
    trace_fn: A callable that takes in the current chain state and the previous
      kernel results and return a `Tensor` or a nested collection of `Tensor`s
      that is then traced along with the chain state.
    return_final_kernel_results: If `True`, then the final kernel results are
      returned alongside the chain state and the trace specified by the
      `trace_fn`.
    parallel_iterations: The number of iterations allowed to run in parallel. It
      must be a positive integer. See `tf.while_loop` for more details.
    seed: Optional, a seed for reproducible sampling.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'experimental_mcmc_sample_chain').

  Returns:
    checkpointable_states_and_trace: if `return_final_kernel_results` is
      `True`. The return value is an instance of
      `CheckpointableStatesAndTrace`.
    all_states: if `return_final_kernel_results` is `False` and `trace_fn` is
      `None`. The return value is a `Tensor` or Python list of `Tensor`s
      representing the state(s) of the Markov chain(s) at each result step. Has
      same shape as input `current_state` but with a prepended
      `num_results`-size dimension.
    states_and_trace: if `return_final_kernel_results` is `False` and
      `trace_fn` is not `None`. The return value is an instance of
      `StatesAndTrace`.

  #### References

  [1]: Art B. Owen. Statistically efficient thinning of a Markov chain sampler.
       _Technical Report_, 2017.
       http://statweb.stanford.edu/~owen/reports/bestthinning.pdf
  """
  with tf.name_scope(name or 'experimental_mcmc_sample_chain'):
    if not kernel.is_calibrated:
      warnings.warn('supplied `TransitionKernel` is not calibrated. Markov '
                    'chain may not converge to intended target distribution.')

    if trace_fn is None:
      trace_fn = lambda *args: ()
      no_trace = True
    else:
      no_trace = False

    if trace_fn is sample_chain.__defaults__[4]:
      warnings.warn('Tracing all kernel results by default is deprecated. Set '
                    'the `trace_fn` argument to None (the future default '
                    'value) or an explicit callback that traces the values '
                    'you are interested in.')

    def real_trace_fn(curr_state, kr):
      return curr_state, trace_fn(curr_state, kr)
    trace_reducer = tracing_reducer.TracingReducer(
        trace_fn=real_trace_fn,
        size=num_results
    )
    # pylint: disable=unbalanced-tuple-unpacking
    trace_results, _, final_kernel_results = sample_fold(
        num_steps=num_results,
        current_state=current_state,
        previous_kernel_results=previous_kernel_results,
        kernel=kernel,
        reducer=trace_reducer,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results,
        parallel_iterations=parallel_iterations,
        seed=seed,
        name=name,
    )

    all_states, trace = trace_results
    if return_final_kernel_results:
      return sample.CheckpointableStatesAndTrace(
          all_states=all_states,
          trace=trace,
          final_kernel_results=final_kernel_results)
    else:
      if no_trace:
        return all_states
      else:
        return sample.StatesAndTrace(all_states=all_states, trace=trace)
