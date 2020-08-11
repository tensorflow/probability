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
"""ExpectationsReducer for pre-packaged streaming expectations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.mcmc import reducer as reducer_base
from tensorflow_probability.python.experimental.stats import sample_stats
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'ExpectationsReducer',
]


def _get_sample(sample_and_kr):
  return sample_and_kr[0]


ExpectationsReducerState = collections.namedtuple(
    'ExpectationsReducerState', 'init_state, expectation_state')


class ExpectationsReducer(reducer_base.Reducer):
  """`Reducer` that computes a running expectation.

  `ExpectationsReducer` calculates expectation over some arbitrary structure
  of `callables`. A callable is a function that accepts a tuple of a Markov
  chain sample and kernel results, and outputs the relevant value for
  expectation calculation. In other words, if we denote a callable as f(x,y),
  `ExpectationsReducer` computes E[f(x,y)] for all provided functions. The
  finalized expectation will also identically mimic the structure of
  `callables`.

  As with all reducers, ExpectationsReducer does not hold state information;
  rather, it stores supplied metadata. Intermediate calculations are held in
  a state object, which is returned via `initialize` and `one_step` method
  calls.
  """

  def __init__(self, callables=_get_sample, name=None):
    """Instantiates this object.

    Args:
      callables: A (possibly nested) structure of functions that accept a
        chain state and kernel results. Defaults to simply returning the
        incoming chain state.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'expectations_reducer').
    """
    self._parameters = dict(
        callables=callables,
        name=name or 'expectations_reducer'
    )

  def initialize(self, initial_chain_state, initial_kernel_results=None):
    """Initializes an empty `ExpectationsReducerState`.

    Args:
      initial_chain_state: A (possibly nested) structure of `Tensor`s or Python
        `list`s of `Tensor`s representing the current state(s) of the Markov
        chain(s).
      initial_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related `TransitionKernel`.

    Returns:
      state: `ExpectationsReducerState` representing a stream of no inputs.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'expectations_reducer', 'initialize')):
      initial_chain_state = tf.nest.map_structure(
          tf.convert_to_tensor,
          initial_chain_state)
      stream = sample_stats.RunningExpectations(
          shape=initial_chain_state.shape,
          callables=self.callables,
          dtype=initial_chain_state.dtype
      )
      return ExpectationsReducerState(initial_chain_state, stream.initialize())

  def one_step(
      self,
      new_chain_state,
      current_reducer_state,
      previous_kernel_results=None,
      axis=None):
    """Update the `current_reducer_state` with a new chain state.

    Chunking semantics are specified by the `axis` parameter. If chunking is
    enabled (axis is not `None`), all elements along the specified `axis` will
    be treated as separate samples. If a single scalar value is provided for a
    non-scalar sample structure, that value will be used for all elements in the
    structure. If not, an identical structure must be provided.

    Args:
      new_chain_state: A (possibly nested) structure of incoming chain state(s)
        with shape and dtype compatible with those used to initialize the
        `current_reducer_state`.
      current_reducer_state: `ExpectationsReducerState` representing the current
        reducer state.
      previous_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related
        `TransitionKernel`.
      axis: If chunking is desired, this is a (possibly nested) structure of
        integers that specifies the axis with chunked samples. For individual
        samples, set this to `None`. By default, samples are not chunked
        (`axis` is None).

    Returns:
      new_reducer_state: `ExpectationsReducerState` with updated running
        statistics. It tracks a running total and the number of processed
        samples.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'expectations_reducer', 'one_step')):
      new_chain_state = tf.nest.map_structure(
          tf.convert_to_tensor,
          new_chain_state)
      stream = sample_stats.RunningExpectations(
          shape=new_chain_state.shape,
          callables=self.callables,
          dtype=new_chain_state.dtype
      )
      if previous_kernel_results is None:
        # having a `None` pkr means it's not convertible to a `Tensor`, which
        # breaks `RunningExpectations`
        previous_kernel_results = ()
      updated_expectation = stream.update(
          current_reducer_state.expectation_state,
          (new_chain_state, previous_kernel_results),
          axis=axis)
      return ExpectationsReducerState(
          current_reducer_state.init_state,
          updated_expectation)

  def finalize(self, final_reducer_state):
    """Finalizes expectation calculation from the `final_reducer_state`.

    If the finalized method is invoked on a stream of no inputs, a corresponding
    structure of `tf.ones` will be returned.

    Args:
      final_reducer_state: `ExpectationsReducerState` that represents the
        final reducer state.

    Returns:
      expectation: an estimate of the expectation with identical structure to
        `final_reducer_state.running_total`.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'expectations_reducer', 'finalize')):
      stream = sample_stats.RunningExpectations(
          shape=final_reducer_state.init_state.shape,
          callables=self.callables,
          dtype=final_reducer_state.init_state.dtype,
      )
      return stream.finalize(final_reducer_state.expectation_state)

  @property
  def callables(self):
    return self._parameters['callables']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    return self._parameters
