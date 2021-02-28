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
"""Test Fixtures for the Streaming MCMC Framework."""

import collections

# Dependency imports
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


TestTransitionKernelResults = collections.namedtuple(
    'TestTransitionKernelResults', 'counter_1, counter_2')


class TestTransitionKernel(tfp.mcmc.TransitionKernel):
  """Fake deterministic Transition Kernel."""

  def __init__(
      self,
      shape=(),
      target_log_prob_fn=None,
      is_calibrated=True,
      accepts_seed=True):
    self._is_calibrated = is_calibrated
    self._shape = shape
    # for composition purposes
    self.parameters = dict(
        target_log_prob_fn=target_log_prob_fn)
    self._accepts_seed = accepts_seed

  def one_step(self, current_state, previous_kernel_results, seed=None):
    if seed is not None and not self._accepts_seed:
      raise TypeError('seed arg not accepted')
    current_state = tf.convert_to_tensor(current_state)
    return (current_state + tf.ones(self._shape, dtype=current_state.dtype),
            TestTransitionKernelResults(
                counter_1=previous_kernel_results.counter_1 + 1,
                counter_2=previous_kernel_results.counter_2 + 2))

  def bootstrap_results(self, current_state):
    return TestTransitionKernelResults(
        counter_1=tf.zeros(()),
        counter_2=tf.zeros(()))

  @property
  def is_calibrated(self):
    return self._is_calibrated


class RandomTransitionKernel(tfp.mcmc.TransitionKernel):
  """Outputs a random next state following a Rayleigh distribution."""

  def __init__(self, shape=(), is_calibrated=True, accepts_seed=True):
    self._shape = shape
    self._is_calibrated = is_calibrated
    self._accepts_seed = accepts_seed

  def one_step(self, current_state, previous_kernel_results, seed=None):
    if seed is not None and not self._accepts_seed:
      raise TypeError('seed arg not accepted')
    new_state = tfp.random.rayleigh(self._shape, seed=seed)
    return new_state, TestTransitionKernelResults(
        counter_1=previous_kernel_results.counter_1 + 1,
        counter_2=previous_kernel_results.counter_2 + 2)

  def bootstrap_results(self, current_state):
    return TestTransitionKernelResults(
        counter_1=tf.zeros(()), counter_2=tf.zeros(()))

  @property
  def is_calibrated(self):
    return self._is_calibrated


class NaiveMeanReducer(tfp.experimental.mcmc.Reducer):
  """Simple Reducer that (naively) computes the mean."""

  def initialize(self, initial_chain_state=None, initial_kernel_results=None):
    return tf.zeros((2,))

  def one_step(self, sample, current_state, previous_kernel_results, axis=None):
    return current_state + tf.convert_to_tensor([1, sample])

  def finalize(self, final_state):
    return final_state[1] / final_state[0]


class TestReducer(tfp.experimental.mcmc.Reducer):
  """Simple Reducer that just keeps track of the last sample."""

  def initialize(self, initial_chain_state, initial_kernel_results=None):
    return tf.zeros_like(initial_chain_state)

  def one_step(
      self, new_chain_state, current_reducer_state, previous_kernel_results):
    return new_chain_state


def reduce(reducer, elems):
  """Reduces `elems` along the first dimension with `reducer`."""
  elems = tf.convert_to_tensor(elems)
  state = reducer.initialize(elems[0])
  def body(i, state):
    return i + 1, reducer.one_step(elems[i], state)
  _, state = tf.while_loop(lambda i, _: i < elems.shape[0], body, (0, state))
  return reducer.finalize(state)
