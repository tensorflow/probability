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

import abc
import six


__all__ = [
    "TransitionKernel",
]


@six.add_metaclass(abc.ABCMeta)
class TransitionKernel(object):
  """Base class for all MCMC `TransitionKernel`s.

  This class defines the minimal requirements to efficiently implement a Markov
  chain Monte Carlo (MCMC) transition kernel. A transition kernel returns a new
  state given some old state. It also takes (and returns) "side information"
  which may be used for debugging or optimization purposes (i.e, to "recycle"
  previously computed results).

  #### Example (random walk transition kernel):

  In this example we make isotropic Gaussian proposals of a given step size.

  ```python
  from tensorflow_probability.python.mcmc import kernel as kernel_base
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp

  tfd = tfp.distributions

  RWResult = collections.namedtuple("RWResult", 'target_log_prob')

  class RandomWalkProposalKernel(kernel_base.TransitionKernel):
    def __init__(self, target_log_prob_fn, step_size):
      self._parameters = dict(
        target_log_prob_fn = target_log_prob_fn,
        step_size = step_size)

    @property
    def target_log_prob_fn(self):
      return self._parameters['target_log_prob_fn']

    @property
    def step_size(self):
      return self._parameters['step_size']

    @property
    def is_calibrated(self):
      return False

    def one_step(self, current_state, previous_kernel_results, seed=None):
      scale = tf.broadcast_to(self.step_size, tf.shape(current_state))
      isotropic_normal = tfd.Normal(loc=current_state, scale=scale)

      next_state = isotropic_normal.sample(seed=seed)
      next_target_log_prob = self.target_log_prob_fn(next_state)
      new_kernel_results = previous_kernel_results._replace(
        target_log_prob = next_target_log_prob)

      return next_state, new_kernel_results

    def bootstrap_results(self, init_state):
      kernel_results = RWResult(
        target_log_prob = self.target_log_prob_fn(init_state))
      return kernel_results
  ```

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
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      next_state: `Tensor` or Python `list` of `Tensor`s representing the
        next state(s) of the Markov chain(s).
      kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
    """
    raise NotImplementedError()

  @property
  @abc.abstractmethod
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

  @property
  def experimental_shard_axis_names(self):
    """The shard axis names for members of the state."""
    return []

  def experimental_with_shard_axes(self, shard_axis_names):
    """Returns a copy of the kernel with the provided shard axis names.

    Args:
      shard_axis_names: a structure of strings indicating the shard axis names
        for each component of this kernel's state.
    Returns:
      A copy of the current kernel with the shard axis information.
    """
    del shard_axis_names
    return self

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
