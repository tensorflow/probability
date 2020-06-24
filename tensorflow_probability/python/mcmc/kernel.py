# Copyright 2018 The TensorFlow Probability Authors.
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
"""Transition Kernel base class and utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


__all__ = [
    "TransitionKernel",
]


# TODO(b/74235190): Add additional documentation/examples to the
# `TransitionKernel` docstring.


@six.add_metaclass(abc.ABCMeta)
class TransitionKernel(object):
  """Base class for all MCMC `TransitionKernel`s.

  This class defines the minimal requirements to efficiently implement a Markov
  chain Monte Carlo (MCMC) transition kernel. A transition kernel returns a new
  state given some old state. It also takes (and returns) "side information"
  which may be used for debugging or optimization purposes (i.e, to "recycle"
  previously computed results).
  """

  @abc.abstractmethod
  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Takes one step of the TransitionKernel.

    Must be overridden by subclasses.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s).
      previous_kernel_results: A (possibly nested) `tuple`, `namedtuple` or
        `list` of `Tensor`s representing internal calculations made within the
        previous call to this function (or as returned by `bootstrap_results`).
      seed: Optional, a seed for reproducible sampling.

    Returns:
      next_state: `Tensor` or Python `list` of `Tensor`s representing the
        next state(s) of the Markov chain(s).
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
    """
    raise NotImplementedError()

  @abc.abstractproperty
  def is_calibrated(self):
    """Returns `True` if Markov chain converges to specified distribution.

    `TransitionKernel`s which are "uncalibrated" are often calibrated by
    composing them with the `tfp.mcmc.MetropolisHastings` `TransitionKernel`.
    """
    raise NotImplementedError()

  def bootstrap_results(self, init_state):  # pylint: disable=unused-argument
    """Returns an object with the same type as returned by `one_step(...)[1]`.

    Args:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the
        initial state(s) of the Markov chain(s).

    Returns:
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
    """
    return []

  def copy(self, **override_parameter_kwargs):
    """Non-destructively creates a deep copy of the kernel.

    Args:
      **override_parameter_kwargs: Python String/value `dictionary` of
        initialization arguments to override with new values.

    Returns:
      new_kernel: `TransitionKernel` object of same type as `self`,
        initialized with the union of self.parameters and
        override_parameter_kwargs, with any shared keys overridden by the
        value of override_parameter_kwargs, i.e.,
        `dict(self.parameters, **override_parameters_kwargs)`.
    """
    parameters = dict(self.parameters, **override_parameter_kwargs)
    new_kernel = type(self)(**parameters)
    return new_kernel
