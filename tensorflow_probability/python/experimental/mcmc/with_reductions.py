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
"""WithReductions Transition Kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.mcmc import step_kernel
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow.python.util import nest


__all__ = [
    'WithReductions',
]


class WithReductionsKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('WithReductionsKernelResults',
                           ['streaming_calculations',
                            'inner_results'])):
  """Reducer state and diagnositics for `WithReductions`"""
  __slots__ = ()


class WithReductions(kernel_base.TransitionKernel):
  """Applies `Reducer`s to stream over MCMC samples.

  `WithReductions` is intended to be a composable piece in the
  Streaming MCMC framework that seemlessly faciliates the use of
  `Reducer`s. Its purpose is twofold: it generates an appropriate
  sample from its `inner_kernel`, then invokes each reducer's
  `one_step` method on that sample. The updated reducer states are
  stored in the `streaming_calculations` field of `WithReductions`'
  kernel results.

  This `TransitionKernel` can also accept an arbitrary structure of
  reducers. It then invokes all reducers in that structure with
  one `WithReductions.one_step` call. The resulting reducer state
  will identically mimic the structure of the provided reducers.
  """

  def __init__(self, inner_kernel, reducers, name=None):
    """Instantiates this object.

    Args:
      inner_kernel: `TransitionKernel`-like object whose `one_step` will
        generate MCMC sample(s).
      reducers: A (possibly nested) structure of `Reducer`s to be evaluated
        on the `inner_kernel`'s samples.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., "reduced_kernel").
    """
    self._parameters = dict(
        inner_kernel=inner_kernel,
        reducers=reducers,
        name=name or 'reduced_kernel'
    )

  def one_step(
      self, current_state, previous_kernel_results=None, seed=None
  ):
    """Updates all `Reducer`s with a new sample from the `inner_kernel`.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s
        representing the current state(s) of the Markov chain(s),
      previous_kernel_results: `WithReductionsKernelResults` named tuple,
        or `None`. `WithReductionsKernelResults` contain the state of
        `streaming_calculations` and a reference to kernel results of
        nested `TransitionKernel`s. If `None`, the kernel will cold
        start with a call to `bootstrap_results`.
      seed: Optional seed for reproducible sampling.

    Returns:
      sample: Newest MCMC sample drawn from the `inner_kernel`.
      kernel_results: `WithReductionsKernelResults` representing updated
        kernel results. Reducer states are stored in the
        `streaming_calculations` field. The state structure is identical
        to `self.reducers`.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'with_reductions', 'one_step')):
      if previous_kernel_results is None:
        previous_kernel_results = self.bootstrap_results(current_state)
      sample, inner_kernel_results = step_kernel(
          num_steps=1,
          current_state=current_state,
          previous_kernel_results=previous_kernel_results.inner_results,
          kernel=self.inner_kernel,
          return_final_kernel_results=True,
          seed=seed,
          name=self.name,
      )
      new_reducers_state = nest.map_structure_up_to(
          self.reducers,
          lambda r, state: r.one_step(
              sample,
              state,
              previous_kernel_results=inner_kernel_results),
          self.reducers, previous_kernel_results.streaming_calculations,
          check_types=False)
      kernel_results = WithReductionsKernelResults(
          new_reducers_state, inner_kernel_results)
      return sample, kernel_results

  def bootstrap_results(self, init_state):
    """Instantiates reducer states with identical structure to the `init_state`.

    Args:
      init_state: `Tensor` or Python `list` of `Tensor`s representing the a
        state(s) of the Markov chain(s). For consistency, the initial state does
        not count as a "sample". Hence, all reducer states will reflect empty
        streams.

    Returns:
      kernel_results: `WithReductionsKernelResults` representing updated
        kernel results. Reducer states are stored in the
        `streaming_calculations` field. The state structure is identical
        to `self.reducers`.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'with_reductions', 'bootstrap_results')):
      inner_kernel_results = self.inner_kernel.bootstrap_results(init_state)
      return WithReductionsKernelResults(
          tf.nest.map_structure(
              lambda r: r.initialize(init_state, inner_kernel_results),
              self.reducers),
          inner_kernel_results)

  @property
  def is_calibrated(self):
    return self.inner_kernel.is_calibrated

  @property
  def parameters(self):
    return self._parameters

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernel']

  @property
  def reducers(self):
    return self._parameters['reducers']

  @property
  def name(self):
    return self._parameters['name']
