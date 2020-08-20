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
"""TracingReducer for pre-packaged accumulation of samples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.mcmc import reducer as reducer_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'TracingReducer',
]


class TracingState(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'TracingState', ['num_samples', 'trace_state'])):
  __slots__ = ()


def _trace_state_and_kernel_results(current_state, kernel_results):
  return current_state, kernel_results


class TracingReducer(reducer_base.Reducer):
  """`Reducer` that accumulates trace results at each sample.

  Trace results are defined by an appropriate `trace_fn`, which accepts the
  current chain state and kernel results, and returns the desired result.
  At each sample, the traced values are added to a `TensorArray` which
  accumulates all results. By default, all kernel results are traced but in the
  future the default will be changed to no results being traced, so plan
  accordingly.

  If wrapped in a `tfp.experimental.mcmc.WithReductions` Transition Kernel,
  TracingReducer will not accumulate the kernel results of `WithReductions`.
  Rather, the top level kernel results will be that of `WithReductions`' inner
  kernel.

  As with all reducers, `TracingReducer` does not hold state information;
  rather, it stores supplied metadata. Intermediate calculations are held in
  a `TracingState` named tuple, which is returned via `initialize` and
  `one_step` method calls.
  """

  def __init__(
      self,
      trace_fn=_trace_state_and_kernel_results,
      size=None,
      name=None):
    """Instantiates this object.

    TracingReducer can accumulate samples in a dynamic or static shape
    `TensorArray`. Specifying the `size` at instantiation will cause all
    subsequent state objects to hold a `TensorArray` of the same static size.
    If `size` is `None`, all `TensorArray`s will have dynamic size.

    Args:
      trace_fn: A callable that takes in the current chain state and the
        previous kernel results and return a `Tensor` or a nested collection
        of `Tensor`s that is accumulated across samples.
      size: Integer or scalar `Tensor` denoting the size of the accumulated
        `TensorArray`. If this is `None` (which is the default), a
        dynamic-shaped `TensorArray` will be used.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'tracing_reducer').
    """
    self._parameters = dict(
        trace_fn=trace_fn,
        size=size,
        name=name or 'tracing_reducer',
    )

  def initialize(self, initial_chain_state, initial_kernel_results=None):
    """Initializes a `TracingState` using previously defined metadata.

    Both the `initial_chain_state` and `initial_kernel_results` do not count
    as a sample, and hence, will not be traced. This is a deliberate decision
    that ensures consistency across sampling procedures.

    Args:
      initial_chain_state: A (possibly nested) structure of `Tensor`s or Python
        `list`s of `Tensor`s representing the current state(s) of the Markov
        chain(s). It is used to infer the structure of future trace results.
      initial_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related `TransitionKernel`.
        It is used to infer the structure of future trace results.

    Returns:
      state: `TracingState` with an empty `TensorArray` in its `trace_state`
        field.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'tracing_reducer', 'initialize')):
      initial_chain_state = tf.nest.map_structure(
          tf.convert_to_tensor,
          initial_chain_state)
      trace_result = self.trace_fn(initial_chain_state, initial_kernel_results)
      if self.size is None:
        size = 0
        dynamic_size = True
      else:
        size = self.size
        dynamic_size = False

      def _map_body(trace_state):
        if not tf.is_tensor(trace_state):
          trace_state = tf.convert_to_tensor(trace_state)
        return tf.TensorArray(
            dtype=trace_state.dtype,
            size=size,
            dynamic_size=dynamic_size,
            element_shape=trace_state.shape,
            clear_after_read=False)

      trace_state = tf.nest.map_structure(
          _map_body,
          trace_result)
      return TracingState(0, trace_state)

  def one_step(
      self, new_chain_state, current_reducer_state, previous_kernel_results):
    """Update the `current_reducer_state` with a new trace result.

    The trace result will be computed by evaluating the `trace_fn` provided
    at instantiation with the `new_chain_state` and `previous_kernel_results`.

    Args:
      new_chain_state: A (possibly nested) structure of incoming chain state(s)
        with shape and dtype compatible with those used to initialize the
        `TracingState`.
      current_reducer_state: `TracingState`s representing all previously traced
        results.
      previous_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related
        `TransitionKernel`.

    Returns:
      new_reducer_state: `TracingState` with updated trace. Its `trace_state`
        field holds a `TensorArray` that includes the newly computed trace
        result.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'tracing_reducer', 'one_step')):
      num_samples, trace_state = current_reducer_state
      new_trace_results = self.trace_fn(
          new_chain_state, previous_kernel_results)
      new_trace_state = tf.nest.map_structure(
          lambda state, new_trace: state.write(num_samples, new_trace),
          trace_state, new_trace_results)
      return TracingState(
          num_samples + 1,
          new_trace_state)

  def finalize(self, final_reducer_state):
    """Finalizes tracing by stacking the accumulated `TensorArray`.

    Args:
      final_reducer_state: `TracingState` that holds all desired traced results.

    Returns:
      trace: `Tensor` that represents stacked tracing results.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'tracing_reducer', 'finalize')):
      return tf.nest.map_structure(
          lambda state: state.stack(),
          final_reducer_state.trace_state)

  @property
  def name(self):
    return self._parameters['name']

  @property
  def trace_fn(self):
    return self._parameters['trace_fn']

  @property
  def size(self):
    return self._parameters['size']

  @property
  def parameters(self):
    return self._parameters
