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
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'ExpectationsReducer',
]


ExpectationsReducerState = collections.namedtuple(
    'ExpectationsReducerState', 'num_samples, running_total')


class ExpectationsReducer(reducer_base.Reducer):
  """`Reducer` that computes a running expectation.

  `ExpectationsReducer` calculates expectation over some arbitrary structure
  of `callables`. A callable is a function that accepts a Markov chain sample
  and kernel results, and outputs the relevant value for expectation
  calculation. In other words, if we denote a callable as f(x,y),
  `ExpectationsReducer` computes E[f(x,y)] for all provided functions. The
  finalized expectation will also identically mimic the structure of
  `callables`.

  As with all reducers, ExpectationsReducer does not hold state information;
  rather, it stores supplied metadata. Intermediate calculations are held in
  a state object, which is returned via `initialize` and `one_step` method
  calls.
  """

  def __init__(self, callables=lambda sample, kr: sample, name=None):
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
      return ExpectationsReducerState(
          num_samples=tf.zeros((), dtype=tf.int32),
          running_total=tf.nest.map_structure(
              lambda _: tf.zeros(
                  initial_chain_state.shape, initial_chain_state.dtype),
              self.callables
          )
      )

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
      axis, new_chain_state, previous_kernel_results = _prepare_args(
          self.callables, axis, new_chain_state, previous_kernel_results)
      num_new_samples = 1

      def _update_running_total(
          running_total, new_chain_state, pkr, fn, axis):
        """Updates the running total and facilitates chunking (if enabled)."""
        nonlocal num_new_samples
        if axis is None:
          chunked_new_chain_state = new_chain_state
          chunked_result = fn(chunked_new_chain_state, pkr)
        else:
          rank = ps.rank(new_chain_state)
          num_new_samples = ps.shape(new_chain_state)[axis]
          chunked_new_chain_state = tf.transpose(
              new_chain_state,
              [axis] + list(range(0, axis)) + list(range(axis + 1, rank)))
          chunked_result = tf.reduce_sum(
              tf.map_fn(lambda chain_state: fn(chain_state, pkr),
                        chunked_new_chain_state),
              axis=0)
        return running_total + chunked_result

      new_running_total = nest.map_structure_up_to(
          current_reducer_state.running_total,
          _update_running_total,
          current_reducer_state.running_total,
          new_chain_state,
          previous_kernel_results,
          self.callables,
          axis,
          check_types=False,
      )
      return ExpectationsReducerState(
          num_samples=current_reducer_state.num_samples + num_new_samples,
          running_total=new_running_total)

  def finalize(self, final_reducer_state):
    """Finalizes expectation calculation from the `final_reducer_state`.

    Args:
      final_reducer_state: `ExpectationsReducerState` that represents the
        final reducer state.

    Returns:
      expectation: an estimate of the expectation with identical structure to
        `final_reducer_state.running_total`.
    """
    if final_reducer_state.num_samples == 0:
      finalize_fn = lambda x, y: None
    else:
      finalize_fn = lambda num_samples, total: total / tf.cast(
          num_samples, total.dtype)
    return tf.nest.map_structure(
        finalize_fn,
        nest_util.broadcast_structure(
            final_reducer_state.running_total,
            final_reducer_state.num_samples),
        final_reducer_state.running_total,
        check_types=False,
    )

  @property
  def callables(self):
    return self._parameters['callables']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    return self._parameters


def _prepare_args(target, axis, chain_state, pkr):
  """Broadcasts arguments to match the structure of `target`."""
  axis = nest_util.broadcast_structure(target, axis)
  # using `nest_util.broadcast_structure` for the following arguments
  # isn't robust as they may already be in some nested structure.
  chain_state = tf.nest.map_structure(
      lambda _: chain_state,
      target
  )
  pkr = tf.nest.map_structure(
      lambda _: pkr,
      target
  )
  return axis, chain_state, pkr
