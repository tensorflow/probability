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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


__all__ = [
    'Reducer',
]


@six.add_metaclass(abc.ABCMeta)
class Reducer(object):
  """Base class for all MCMC `Reducer`s.

  This class defines the minimal requirements to implement a Markov chain Monte
  Carlo (MCMC) reducer. A reducer updates streaming computation by reducing
  new samples to a summary statistic. Unlike `TransitionKernel`s, `Reducer`s do
  not return any "side information". Moreover, they do not remember state.
  Hence, reducers should be seen as objects that hold metadata (i.e. shape and
  dtype of incoming samples) and all reducer method calls must be coupled with
  a state object, as first returned by the `initialize` method.
  """

  @abc.abstractmethod
  def initialize(self, initial_chain_state, initial_inner_kernel_results):
    """Initializes a reducer state corresponding to the stream of no samples.

    This is an abstract method and must be overridden by subclasses.

    Args:
      initial_chain_state: `Tensor` or Python `list` of `Tensor`s representing
        the current state(s) of the Markov chain(s).
      initial_inner_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made in a related
        `TransitionKernel`. This allows for introspection if the kernel results
        of deeper layers of `TransitionKernel`s has has bearing to the nature of
        the initial reducer state.

    Returns:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the
        stream of no samples.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def one_step(self, sample, current_state, previous_kernel_results, axis=None):
    """Takes one step of the `Reducer`.

    This is an abstract method and must be overridden by subclasses.

    Args:
      sample: Incoming sample with shape and dtype compatible with those
        specified when initializing the `Reducer`.
      current_state: A `tuple`, `namedtuple` or `list` of `Tensor`s representing
        the current state of the reduced statistic.
      previous_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made in a related
        `TransitionKernel`. This allows for introspection if the kernel results
        of deeper layers of `TransitionKernel`s has has bearing to the nature of
        the updated reducer state (i.e. updating based on a value in the kernel
        results of a nested `TransitionKernel`).
      axis: If chunking is desired, this is an integer that specifies the axis
        with chunked samples. For individual samples, set this to `None`. By
        default, samples are not chunked (`axis` is None).

    Returns:
      new_state: The new reducer state after updates. This has the same type and
        structure as `current_state`.
    """
    raise NotImplementedError()

  def finalize(self, final_state):
    """Finalizes target statistic calculation from the `final_state`.

    This is not an abstract method, but the base implementation is an
    identity function of the `final_state`. Hence, it should be overriden
    for more complex streaming calculations.

    Args:
      final_state: A `tuple`, `namedtuple` or `list` of `Tensor`s
        representing the final state of the reduced statistic.

    Returns:
      statistic: An estimate of the target statistic
    """
    return final_state  # Default finalize is the identity
