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
"""Sample Discarding Kernel for Thinning and Burn-in."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import step
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'SampleDiscardingKernel',
]


class SampleDiscardingKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('SampleDiscardingKernelResults',
                           ['call_counter',
                            'inner_results'])):
  __slots__ = ()


class SampleDiscardingKernel(kernel_base.TransitionKernel):
  """Appropriately discards samples to conduct thinning and burn-in.

  `SampleDiscardingKernel` is a composable `TransitionKernel` that
  applies thinning and burn-in to samples returned by its
  `inner_kernel`. All Transition Kernels wrapping it will only
  see non-discarded samples.

  The burn-in step conducts both burn-in and one step of thinning.
  In other words, the first call to `one_step` will skip
  `num_burnin_steps + num_steps_between_results` samples. All
  subsequent calls skip only `num_steps_between_results` samples.
  """

  def __init__(
      self,
      inner_kernel,
      num_burnin_steps=0,
      num_steps_between_results=0,
      name=None):
    """Instantiates this object.

    Args:
      inner_kernel: `TransitionKernel` whose `one_step` will generate
        MCMC results.
      num_burnin_steps: Integer or scalar `Tensor` representing the number
        of chain steps to take before starting to collect results.
        Defaults to 0 (i.e., no burn-in).
      num_steps_between_results: Integer or scalar `Tensor` representing
        the number of chain steps between collecting a result. Only one out
        of every `num_steps_between_samples + 1` steps is included in the
        returned results. Defaults to 0 (i.e., no thinning).
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "sample_discarding_kernel").
    """
    self._parameters = dict(
        inner_kernel=inner_kernel,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results,
        name=name or 'sample_discarding_kernel'
    )

  def _num_samples_to_skip(self, call_counter):
    """Calculates how many samples to skip based on the call number."""
    # If `self.num_burnin_steps` is statically known to be 0,
    # `self.num_steps_between_results` will be returned outright.
    num_burnin_steps_ = tf.get_static_value(self.num_burnin_steps)
    if num_burnin_steps_ == 0:
      return self.num_steps_between_results
    else:
      return (tf.where(tf.equal(call_counter, 0), self.num_burnin_steps, 0) +
              self.num_steps_between_results)

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Collects one non-discarded chain state.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s
        representing the current state(s) of the Markov chain(s),
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function).
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      new_chain_state: Newest non-discarded MCMC chain state drawn from
        the `inner_kernel`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'sample_discarding_kernel', 'one_step')):
      new_chain_state, inner_kernel_results = step.step_kernel(
          num_steps=self._num_samples_to_skip(
              previous_kernel_results.call_counter
          ) + 1,
          current_state=current_state,
          previous_kernel_results=previous_kernel_results.inner_results,
          kernel=self.inner_kernel,
          return_final_kernel_results=True,
          seed=seed,
          name=self.name)
      new_kernel_results = SampleDiscardingKernelResults(
          previous_kernel_results.call_counter + 1, inner_kernel_results
      )
      return new_chain_state, new_kernel_results

  def bootstrap_results(self, init_state, inner_results=None):
    """Instantiates a new kernel state with no calls.

    Args:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the
        state(s) of the Markov chain(s).
      inner_results: Optional results tuple for the inner kernel.  Will be
        re-bootstrapped if omitted.

    Returns:
      kernel_results: `collections.namedtuple` of `Tensor`s representing
        internal calculations made within this function.
    """
    with tf.name_scope(
        mcmc_util.make_name(
            self.name, 'sample_discarding_kernel', 'bootstrap_results')):
      if inner_results is None:
        inner_results = self.inner_kernel.bootstrap_results(init_state)
      return SampleDiscardingKernelResults(
          tf.zeros((), dtype=tf.int32), inner_results)

  @property
  def is_calibrated(self):
    return self.inner_kernel.is_calibrated

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernel']

  @property
  def num_burnin_steps(self):
    return self._parameters['num_burnin_steps']

  @property
  def num_steps_between_results(self):
    return self._parameters['num_steps_between_results']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    return self._parameters
