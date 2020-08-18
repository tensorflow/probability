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
"""RhatReducer for calculating streaming potential scale reduction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.mcmc import reducer as reducer_base
from tensorflow_probability.python.experimental.stats import sample_stats
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'RhatReducer',
]


RhatReducerState = collections.namedtuple(
    'RhatReducerState', 'sample_shape, rhat_state')


class RhatReducer(reducer_base.Reducer):
  """`Reducer` that computes a running R-hat diagnostic statistic.

  `RhatReducer` assumes that all provided chain states include samples from
  multiple independent Markov chains, and that all of these chains are to
  be included in the same calculation. `RhatReducer` also assumes that
  incoming samples have shape `[Ni, Ci1, Ci2,...,CiD]`. Dimension `0` indexes
  the `Ni > 1` result steps of the Markov Chain. Dimensions `1` through `D`
  index the `Ci1 x ... x CiD` independent chains to be tested for convergence
  to the same target. A possible chunking dimension can also be specified via
  the `axis` parameter of the `one_step` method. This dimension indexes multiple
  dependent samples from the same Markov chain, but disregards the first sample
  dimension.

  To illustrate, with no chunking enabled, a chain state of shape `[5, 2, 3, 4]`
  can be inferred as shape `[2, 3, 4]` samples drawn from 5 independent Markov
  chains. If chunking is enabled and `axis=1` then the same chain state can be
  inferred as 3 samples of shape `[2, 4]` from 5 independent Markov chains. Note
  how the chunking dimension indexes into only the event dimensions, and not the
  batch dimension.

  As with all reducers, RhatReducer does not hold state information;
  rather, it stores supplied metadata. Intermediate calculations are held in
  a state object, which is returned via `initialize` and `one_step` method
  calls.

  `RhatReducer` is meant to fit into the larger Streaming MCMC framework.
  `RunningPotentialScaleReduction` in `tfp.experimental.stats` is
  better suited for more generic streaming R-hat needs. More precise
  algorithmic details can also be found by referencing
  `RunningPotentialScaleReduction`.
  """

  def __init__(self, num_parallel_chains, name=None):
    """Instantiates this object.

    Args:
      num_parallel_chains: Integer number of independent Markov chains
        ran in parallel.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'rhat_reducer').

    Raises:
      ValueError: if `num_parallel_chains < 2`. This results in undefined
        intermediate variance calculations.
    """
    if num_parallel_chains < 2:
      raise ValueError('At least 2 parallel chains are required for '
                       'well-defined between_chain variances')
    self._parameters = dict(
        num_parallel_chains=num_parallel_chains,
        name=name or 'rhat_reducer'
    )

  def initialize(self, initial_chain_state, initial_kernel_results=None):
    """Initializes a `RhatReducerState` using previously defined metadata.

    For calculation purposes, the `initial_chain_state` does not count as a
    sample. This is a deliberate decision that ensures consistency across
    sampling procedures (i.e. `tfp.mcmc.sample_chain` follows similar
    semantics).

    Args:
      initial_chain_state: A (possibly nested) structure of `Tensor`s or Python
        `list`s of `Tensor`s representing the current state(s) of the Markov
        chain(s). It is used to infer the shape and dtype of future samples.
      initial_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related `TransitionKernel`.
        For streaming R-hat, this argument has no influence on the
        computation; hence, it is `None` by default. However, it's
        still accepted to fit the `Reducer` base class.

    Returns:
      state: `RhatReducerState` with `rhat_state` field representing
        a stream of no inputs.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'rhat_reducer', 'initialize')):
      initial_chain_state = tf.convert_to_tensor(initial_chain_state)
      if ps.rank(initial_chain_state) == 1:
        sample_shape = ()
      else:
        sample_shape = ps.shape(initial_chain_state)[1:]
      running_rhat = sample_stats.RunningPotentialScaleReduction(
          shape=sample_shape,
          num_chains=self.num_parallel_chains,
          dtype=initial_chain_state.dtype
      )
      return RhatReducerState(sample_shape, running_rhat.initialize())

  def one_step(
      self,
      new_chain_state,
      current_reducer_state,
      previous_kernel_results=None,
      axis=None):
    """Update the `current_reducer_state` with a new chain state.

    Args:
      new_chain_state: A (possibly nested) structure of incoming chain state(s)
        with shape and dtype compatible with those used to initialize the
        `current_reducer_state`.
      current_reducer_state: `RhatReducerState` representing the current
        state of the running R-hat statistic.
      previous_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related
        `TransitionKernel`. For streaming R-hat, this argument has no
        influence on computation; hence, it is `None` by default. However, it's
        still accepted to fit the `Reducer` base class.
      axis: If chunking is desired, this is a (possibly nested) structure of
        integers that specifies the axis with chunked samples. For individual
        samples, set this to `None`. By default, samples are not chunked
        (`axis` is None).

    Returns:
      new_reducer_state: `RhatReducerState` with updated running
        statistics.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'rhat_reducer', 'one_step')):
      new_chain_state = tf.convert_to_tensor(new_chain_state)
      running_rhat = sample_stats.RunningPotentialScaleReduction(
          shape=current_reducer_state.sample_shape,
          num_chains=self.num_parallel_chains,
          dtype=new_chain_state.dtype
      )
      new_rhat_state = running_rhat.update(
          current_reducer_state.rhat_state,
          new_chain_state,
          chunk_axis=axis)
      return RhatReducerState(
          current_reducer_state.sample_shape, new_rhat_state)

  def finalize(self, final_reducer_state):
    """Finalizes covariance calculation from the `final_reducer_state`.

    Args:
      final_reducer_state: `RhatReducerState` that represents the
        final state of the running R-hat statistic.

    Returns:
      R-hat: an estimate of the R-hat.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'rhat_reducer', 'finalize')):
      running_rhat = sample_stats.RunningPotentialScaleReduction(
          shape=final_reducer_state.sample_shape,
          num_chains=self.num_parallel_chains,
          dtype=final_reducer_state.rhat_state.chain_var[0].mean.dtype
      )
      return running_rhat.finalize(final_reducer_state.rhat_state)

  @property
  def parameters(self):
    return self._parameters

  @property
  def num_parallel_chains(self):
    return self._parameters['num_parallel_chains']

  @property
  def name(self):
    return self._parameters['name']
