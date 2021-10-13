# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
"""Markov chain Monte Carlo Reducer base class."""

import abc

# dependency imports
import six


__all__ = [
    'Reducer',
]


@six.add_metaclass(abc.ABCMeta)
class Reducer(object):
  """Base class for all MCMC `Reducer`s.

  This class defines the minimal requirements to implement a Markov chain Monte
  Carlo (MCMC) reducer. A reducer updates a streaming computation by reducing
  new samples to a summary statistic. `Reducer`s can be defined to return "side
  information" if desired, but they do not remember state. Hence, reducers
  should be seen as objects that hold metadata (i.e. shape and dtype of
  incoming samples) and all reducer method calls must be coupled with a state
  object, as first returned by the `initialize` method.
  """

  @abc.abstractmethod
  def initialize(self, initial_chain_state, initial_inner_kernel_results):
    """Initializes a reducer state corresponding to the stream of no samples.

    This is an abstract method and must be overridden by subclasses.

    Args:
      initial_chain_state: `Tensor` or Python `list` of `Tensor`s representing
        the current state(s) of the Markov chain(s). This is to be used for
        any needed (shape, dtype, etc.) information, but should not be
        considered part of the stream being reduced.
      initial_inner_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made in a related
        `TransitionKernel`. This allows for introspection of deeper layers of
        `TransitionKernel`s that have bearing to the nature of the initial
        reducer state.

    Returns:
      init_reducer_state: `tuple`, `namedtuple` or `list` of `Tensor`s
        representing the stream of no samples.
    """

  @abc.abstractmethod
  def one_step(
      self, new_chain_state, current_reducer_state, previous_kernel_results):
    """Takes one step of the `Reducer`.

    This is an abstract method and must be overridden by subclasses.

    Args:
      new_chain_state: Incoming chain state(s) with shape and dtype compatible
        with the `initial_chain_state` with which the `current_reducer_state`
        was produced by `initialize`.
      current_reducer_state: A `tuple`, `namedtuple` or `list` of `Tensor`s
        representing the current state of reduced statistics.
      previous_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made in a related
        `TransitionKernel`. This allows for introspection of deeper layers of
        `TransitionKernel`s that have bearing to the nature of the updated
        reducer state (i.e. updating based on a value in the kernel results of
        some `TransitionKernel`).

    Returns:
      new_state: The new reducer state after updates. This has the same type and
        structure as `current_reducer_state`.
    """

  def finalize(self, final_reducer_state):
    """Finalizes target statistic calculation from the `final_state`.

    This is an identity function of the `final_state` by default. Subclasses
    can override it for streaming calculations whose running state is not the
    same as the desired result.

    Args:
      final_reducer_state: A `tuple`, `namedtuple` or `list` of `Tensor`s
        representing the final state of the reduced statistic.

    Returns:
      statistic: An estimate of the target statistic
    """
    return final_reducer_state  # Default finalize is the identity
