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
"""PotentialScaleReductionReducer for calculating streaming R-hat."""

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
    'PotentialScaleReductionReducer',
]


PotentialScaleReductionReducerState = collections.namedtuple(
    'PotentialScaleReductionReducerState', 'init_state, rhat_state')


class PotentialScaleReductionReducer(reducer_base.Reducer):
  """`Reducer` that computes a running R-hat diagnostic statistic.

  `PotentialScaleReductionReducer` assumes that all provided chain states
  include samples from multiple independent Markov chains, and that all of
  these chains are to be included in the same calculation.
  `PotentialScaleReductionReducer` also assumes that incoming samples have
  shape `[Ci1, Ci2,...,CiD] + A`. Dimensions `0` through `D - 1` index the
  `Ci1 x ... x CiD` independent chains to be tested for convergence to the
  same target. The remaining dimensions, `A`, represent the event shape and
  hence, can have any shape (even empty, which implies scalar samples). The
  number of independent chain dimensions is defined by the
  `independent_chain_ndims` parameter at initialization.

  As with all reducers, PotentialScaleReductionReducer does not hold state
  information; rather, it stores supplied metadata. Intermediate calculations
  are held in a state object, which is returned via `initialize` and `one_step`
  method calls.

  `PotentialScaleReductionReducer` is meant to fit into the larger Streaming
  MCMC framework. `RunningPotentialScaleReduction` in `tfp.experimental.stats`
  is better suited for more generic streaming R-hat needs. More precise
  algorithmic details can also be found by referencing
  `RunningPotentialScaleReduction`.
  """

  def __init__(self, independent_chain_ndims=1, name=None):
    """Instantiates this object.

    Args:
      independent_chain_ndims: Integer type `Tensor` with value `>= 1` giving
        the number of dimensions, from `dim = 1` to `dim = D`, holding
        independent chain results to be tested for convergence.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'potential_scale_reduction_reducer').

    Raises:
      ValueError: if `independent_chain_ndims < 1`. This results in undefined
        intermediate variance calculations.
    """
    if independent_chain_ndims < 1:
      raise ValueError('Must specify at least 1 dimension to represent'
                       'independent Markov chains. Note, at least 2 parallel'
                       'chains are required for well-defined between_chain'
                       'variances')
    self._parameters = dict(
        independent_chain_ndims=independent_chain_ndims,
        name=name or 'potential_scale_reduction_reducer'
    )

  def initialize(self, initial_chain_state, initial_kernel_results=None):
    """Initializes an empty `PotentialScaleReductionReducerState`.

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
      state: `PotentialScaleReductionReducerState` with `rhat_state` field
        representing a stream of no inputs.
    """
    with tf.name_scope(
        mcmc_util.make_name(
            self.name, 'potential_scale_reduction_reducer', 'initialize')):
      initial_chain_state = tf.nest.map_structure(
          tf.convert_to_tensor,
          initial_chain_state)
      sample_shape, chain_ndims, dtype = _prepare_args(
          initial_chain_state, self.independent_chain_ndims
      )
      running_rhat = sample_stats.RunningPotentialScaleReduction(
          shape=sample_shape,
          independent_chain_ndims=chain_ndims,
          dtype=dtype
      )
      return PotentialScaleReductionReducerState(
          initial_chain_state, running_rhat.initialize())

  def one_step(
      self,
      new_chain_state,
      current_reducer_state,
      previous_kernel_results=None):
    """Update the `current_reducer_state` with a new chain state.

    Args:
      new_chain_state: A (possibly nested) structure of incoming chain state(s)
        with shape and dtype compatible with those used to initialize the
        `current_reducer_state`.
      current_reducer_state: `PotentialScaleReductionReducerState` representing
        the current state of the running R-hat statistic.
      previous_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related
        `TransitionKernel`. For streaming R-hat, this argument has no
        influence on computation; hence, it is `None` by default. However, it's
        still accepted to fit the `Reducer` base class.

    Returns:
      new_reducer_state: `PotentialScaleReductionReducerState` with updated
        running statistics.
    """
    with tf.name_scope(
        mcmc_util.make_name(
            self.name, 'potential_scale_reduction_reducer', 'one_step')):
      new_chain_state = tf.nest.map_structure(
          tf.convert_to_tensor,
          new_chain_state)
      sample_shape, chain_ndims, dtype = _prepare_args(
          new_chain_state, self.independent_chain_ndims
      )
      running_rhat = sample_stats.RunningPotentialScaleReduction(
          shape=sample_shape,
          independent_chain_ndims=chain_ndims,
          dtype=dtype
      )
      new_rhat_state = running_rhat.update(
          current_reducer_state.rhat_state,
          new_chain_state)
      return PotentialScaleReductionReducerState(
          current_reducer_state.init_state,
          new_rhat_state)

  def finalize(self, final_reducer_state):
    """Finalizes R-hat calculation from the `final_reducer_state`.

    Args:
      final_reducer_state: `PotentialScaleReductionReducerState` that
        represents the final state of the running R-hat statistic.

    Returns:
      rhat: an estimate of the R-hat.
    """
    sample_shape, chain_ndims, dtype = _prepare_args(
        final_reducer_state.init_state, self.independent_chain_ndims
    )
    with tf.name_scope(
        mcmc_util.make_name(
            self.name, 'potential_scale_reduction_reducer', 'finalize')):
      running_rhat = sample_stats.RunningPotentialScaleReduction(
          shape=sample_shape,
          independent_chain_ndims=chain_ndims,
          dtype=dtype,
      )
      return running_rhat.finalize(final_reducer_state.rhat_state)

  @property
  def parameters(self):
    return self._parameters

  @property
  def independent_chain_ndims(self):
    return self._parameters['independent_chain_ndims']

  @property
  def name(self):
    return self._parameters['name']


def _prepare_args(target, chain_ndims):
  """Infers metadata to instantiate a streaming rhat object from `target`."""
  sample_shape = tf.nest.map_structure(
      lambda chain_state: tuple(ps.shape(chain_state)),
      target
  )
  nested_chain_ndims = tf.nest.map_structure(
      lambda _: chain_ndims,
      target
  )
  dtype = tf.nest.map_structure(
      lambda chain_state: chain_state.dtype,
      target
  )
  return sample_shape, nested_chain_ndims, dtype
