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
"""Sample Discarding Kernel for Thinning and Burn-in"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import step_kernel
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

  def __init__(
      self,
      inner_kernel,
      num_burnin_steps=0,
      num_steps_between_results=0,
      name=None):
    if tf.get_static_value(num_burnin_steps):
      num_burnin_steps = tf.get_static_value(num_burnin_steps)
    if tf.get_static_value(num_steps_between_results):
      num_steps_between_results = tf.get_static_value(
          num_steps_between_results)
    self._parameters = dict(
        inner_kernel=inner_kernel,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results,
        name=name or 'sample_discarding_kernel'
    )

  def _num_samples_to_skip(self, call_counter):
    # not using `tf.equal(self.num_burnin_steps, 0)` here is intentional.
    # We are checking to see if `self.num_burnin_steps` is statically known.
    # In the case where it's a `Tensor` holding 0, a `Tensor` will be
    # returned in the else clause.
    if self.num_burnin_steps == 0:
      return self.num_steps_between_results
    else:
      return (tf.where(tf.equal(call_counter, 0), self.num_burnin_steps, 0) +
              self.num_steps_between_results)

  def one_step(self, current_state, previous_kernel_results=None, seed=None):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'sample_discarding_kernel', 'one_step')):
      if previous_kernel_results is None:
        previous_kernel_results = self.bootstrap_results(current_state)
      new_sample, inner_kernel_results = step_kernel(
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
      return new_sample, new_kernel_results

  def bootstrap_results(self, init_state):
    with tf.name_scope(
        mcmc_util.make_name(
            self.name, 'sample_discarding_kernel', 'bootstrap_results')):
      return SampleDiscardingKernelResults(
          tf.zeros((), dtype=tf.int32),
          self.inner_kernel.bootstrap_results(init_state))

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
