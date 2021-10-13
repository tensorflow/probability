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
"""High(er) level driver for streaming MCMC."""

import collections
import warnings
# Dependency imports

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import step
from tensorflow_probability.python.experimental.mcmc import tracing_reducer
from tensorflow_probability.python.experimental.mcmc import with_reductions
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'sample_chain',
]


def _trace_everything(chain_state, kernel_results, *reduction_results):
  del kernel_results
  return chain_state, reduction_results


class SampleChainResults(collections.namedtuple(
    'SampleChainResults', ['trace', 'reduction_results', 'final_state',
                           'final_kernel_results', 'resume_kwargs'])):
  """Result from a sampling run.

  Attributes:
    trace: A `Tensor` or a nested collection of `Tensor`s representing the
      values during the run, if any.

    reduction_results: A `Tensor` or a nested collection of `Tensor`s giving the
      results of any requested reductions.

    final_state: A `Tensor` or a nested collection of `Tensor`s giving the final
      state of the Markov chain.

    final_kernel_results: The last auxiliary state of the `kernel` that was run.

    resume_kwargs: A dict of `Tensor` or nested collections of `Tensor`s giving
      keyword arguments that can be used to continue the Markov chain (and
      auxiliaries) where it left off.
  """
  # This list of fields is meant to grow as we decide what metrics, diagnostics,
  # or auxiliary information MCMC entry points should return.  Part of the idea,
  # like scipy.optimize, is to admit multiple entry points; insofar as they all
  # need to return similar information, we should use consistent field names to
  # store them, so users can change entry points without having to write (as
  # much) glue code.

  # Specific possible fields to add:
  # - Performance diagnostics such as number of log_prob and gradient
  #   evaluations
  # - Statistical diagnostics such as ESS or R-hat
  # - Internal diagnostics about adaptation convergence, etc
  # - Once our methods become sophisticated enough to evaluate their own
  #   efficacy, we can also adopt a "success" boolean, failure reason message,
  #   and things like that.
  __slots__ = ()


def sample_chain(
    kernel,
    num_results,
    current_state,
    previous_kernel_results=None,
    reducer=(),
    previous_reducer_state=None,
    trace_fn=_trace_everything,
    parallel_iterations=10,
    seed=None,
    name=None,
):
  """Runs a Markov chain defined by the given `TransitionKernel`.

  This is meant as a (more) helpful frontend to the low-level
  `TransitionKernel`-based MCMC API, supporting several main features:

  - Running a batch of multiple independent chains using SIMD parallelism
  - Tracing the history of the chains, or not tracing it to save memory
  - Computing reductions over chain history, whether it is also traced or not
  - Warm (re-)start, including auxiliary state

  This function samples from a Markov chain at `current_state` whose
  stationary distribution is governed by the supplied `TransitionKernel`
  instance (`kernel`).

  The `current_state` can be represented as a single `Tensor` or a `list` of
  `Tensors` which collectively represent the current state.

  This function can sample from multiple chains, in parallel.  Whether or not
  there are multiple chains is dictated by how the `kernel` treats its inputs.
  Typically, the shape of the independent chains is shape of the result of the
  `target_log_prob_fn` used by the `kernel` when applied to the given
  `current_state`.

  This function can compute reductions over the samples in tandem with sampling,
  for example to return summary statistics without materializing all the
  samples.  To request reductions, pass a `Reducer` object, or a nested
  structure of `Reducer` objects, as the `reducer=` argument.

  In addition to the chain state, this function supports tracing of auxiliary
  variables used by the kernel, as well as intermediate values of any supplied
  reductions. The traced values are selected by specifying `trace_fn`.  The
  `trace_fn` must be a callable accepting three arguments: the chain state, the
  kernel_results of the `kernel`, and the current results of the reductions, if
  any are supplied.  The return value of `trace_fn` (which may be a `Tensor` or
  a nested structure of `Tensor`s) is accumulated, such that each `Tensor` gains
  a new outmost dimension representing time in the chain history.

  Since MCMC states are correlated, it is sometimes desirable to produce
  additional intermediate states, and then discard them, ending up with a set of
  states with decreased autocorrelation.  See [Owen (2017)][1]. Such 'thinning'
  is made possible by setting `num_steps_between_results > 0`. The chain then
  takes `num_steps_between_results` extra steps between the steps that make it
  into the results, or are shown to any supplied reductions. The extra steps
  are never materialized, and thus do not increase memory requirements.

  Args:
    kernel: An instance of `tfp.mcmc.TransitionKernel` which implements one step
      of the Markov chain.
    num_results: Integer number of (non-discarded) Markov chain draws to
      compute.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      initial state(s) of the Markov chain(s).
    previous_kernel_results: A `Tensor` or a nested collection of `Tensor`s
      representing internal calculations made within the previous call to this
      function (or as returned by `bootstrap_results`).
    reducer: A (possibly nested) structure of `Reducer`s to be evaluated
      on the `kernel`'s samples. If no reducers are given (`reducer=None`),
      their states will not be passed to any supplied `trace_fn`.
    previous_reducer_state: A (possibly nested) structure of running states
      corresponding to the structure in `reducer`.  For resuming streaming
      reduction computations begun in a previous run.
    trace_fn: A callable that takes in the current chain state, the current
      auxiliary kernel state, and the current result of any reducers, and
      returns a `Tensor` or a nested collection of `Tensor`s that is then
      traced.  If `None`, nothing is traced.
    parallel_iterations: The number of iterations allowed to run in parallel. It
      must be a positive integer. See `tf.while_loop` for more details.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'mcmc_sample_chain').

  Returns:
    result: A `SampleChainResults` instance containing information about the
      sampling run.  Main fields are `trace`, the history of outputs of
      `trace_fn`, and `reduction_results`, the final outputs of all supplied
      `Reducer`s.  See `SampleChainResults` for contents of other fields.
  """
  # Features omitted for simplicity:
  # - Can only warm start either all the reducers or none of them, not
  #   piecemeal.
  #
  # Defects admitted for simplicity:
  # - All reducers are finalized internally at every step, whether the user
  #   wished to trace them or not.  We expect graph mode TF to avoid that unused
  #   computation, but eager mode will not.
  # - The user is not given the opportunity to trace the running state of
  #   reducers.  For example, the user cannot trace the sum and count of a
  #   running mean, only the running mean itself.  Arguably this is a feature,
  #   because the sum and count can be considered implementation details, the
  #   hiding of which is the purpose of the `finalize` method.
  with tf.name_scope(name or 'mcmc_sample_chain'):
    if not kernel.is_calibrated:
      warnings.warn('supplied `TransitionKernel` is not calibrated. Markov '
                    'chain may not converge to intended target distribution.')

    if trace_fn is None:
      trace_fn = lambda *args: ()

    # Form kernel onion
    reduction_kernel = with_reductions.WithReductions(
        inner_kernel=kernel,
        reducer=reducer)

    # User trace function should be called with
    # - current chain state
    # - kernel results structure of the passed-in kernel
    # - if there were any reducers, their intermediate results
    #
    # `WithReductions` will show the TracingReducer the intermediate state as
    # the kernel results of the onion named `reduction_kernel` above.  This
    # wrapper converts from that to what the user-supplied trace function needs
    # to see.
    def internal_trace_fn(curr_state, kr):
      if reducer:
        def fin(reducer, red_state):
          return reducer.finalize(red_state)
        # Extra level of list will be unwrapped by *reduction_args, below.
        reduction_args = [nest.map_structure_up_to(
            reducer, fin, reducer, kr.reduction_results)]
      else:
        reduction_args = []
      return trace_fn(curr_state, kr.inner_results, *reduction_args)

    trace_reducer = tracing_reducer.TracingReducer(
        trace_fn=internal_trace_fn,
        size=num_results
    )
    tracing_kernel = with_reductions.WithReductions(
        inner_kernel=reduction_kernel,
        reducer=trace_reducer,
    )

    # Bootstrap corresponding warm start
    if previous_kernel_results is None:
      previous_kernel_results = kernel.bootstrap_results(current_state)
    reduction_pkr = reduction_kernel.bootstrap_results(
        current_state, previous_kernel_results, previous_reducer_state)
    tracing_pkr = tracing_kernel.bootstrap_results(
        current_state, reduction_pkr)

    # pylint: disable=unbalanced-tuple-unpacking
    final_state, tracing_kernel_results = step.step_kernel(
        num_steps=num_results,
        current_state=current_state,
        previous_kernel_results=tracing_pkr,
        kernel=tracing_kernel,
        return_final_kernel_results=True,
        parallel_iterations=parallel_iterations,
        seed=seed,
        name=name,
    )

    trace = trace_reducer.finalize(
        tracing_kernel_results.reduction_results)

    reduction_kernel_results = tracing_kernel_results.inner_results
    reduction_results = nest.map_structure_up_to(
        reducer,
        lambda r, s: r.finalize(s),
        reducer,
        reduction_kernel_results.reduction_results,
        check_types=False)

    user_kernel_results = reduction_kernel_results.inner_results

    resume_kwargs = {
        'current_state': final_state,
        'previous_kernel_results': user_kernel_results,
        'kernel': kernel,
        'reducer': reducer,
        'previous_reducer_state': reduction_kernel_results.reduction_results,
    }

    return SampleChainResults(
        trace=trace,
        reduction_results=reduction_results,
        final_state=final_state,
        final_kernel_results=user_kernel_results,
        resume_kwargs=resume_kwargs)
