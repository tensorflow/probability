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
"""Kernel for Thinning."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import step
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'ThinningKernel',
]


class ThinningKernel(kernel_base.TransitionKernel):
  """Discards samples to perform thinning.

  `ThinningKernel` is a composable `TransitionKernel` that thins samples
  returned by its `inner_kernel`. All Transition Kernels wrapping it will only
  see non-discarded samples.
  """

  def __init__(
      self,
      inner_kernel,
      num_steps_to_skip,
      name=None):
    """Instantiates this object.

    Args:
      inner_kernel: `TransitionKernel` whose `one_step` will generate
        MCMC results.
      num_steps_to_skip: Integer or scalar `Tensor` representing
        the number of chain steps skipped before collecting a result.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "thinning_kernel").
    """
    self._parameters = dict(
        inner_kernel=inner_kernel,
        num_steps_to_skip=num_steps_to_skip,
        name=name or 'thinning_kernel'
    )

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Collects one non-thinned chain state.

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
        mcmc_util.make_name(self.name, 'thinned_kernel', 'one_step')):
      return step.step_kernel(
          num_steps=self.num_steps_to_skip + 1,
          current_state=current_state,
          previous_kernel_results=previous_kernel_results,
          kernel=self.inner_kernel,
          return_final_kernel_results=True,
          seed=seed,
          name=self.name)

  def bootstrap_results(self, init_state):
    """Instantiates a new kernel state with no calls.

    Args:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the
        state(s) of the Markov chain(s).

    Returns:
      kernel_results: `collections.namedtuple` of `Tensor`s representing
        internal calculations made within this function.
    """
    return self.inner_kernel.bootstrap_results(init_state)

  @property
  def is_calibrated(self):
    return self.inner_kernel.is_calibrated

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernel']

  @property
  def num_steps_to_skip(self):
    return self._parameters['num_steps_to_skip']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    return self._parameters
