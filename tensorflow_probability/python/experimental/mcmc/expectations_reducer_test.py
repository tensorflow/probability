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
"""Tests for ExpectationsReducer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


FakeKernelResults = collections.namedtuple(
        'FakeKernelResults', 'value, inner_results')


FakeInnerResults = collections.namedtuple(
    'FakeInnerResults', 'value')


TestTransitionKernelResults = collections.namedtuple(
    'TestTransitionKernelResults', 'counter_1, counter_2')


class TestTransitionKernel(tfp.mcmc.TransitionKernel):
  """Fake deterministic `TransitionKernel` for testing purposes."""

  def __init__(self, is_calibrated=True, accepts_seed=True):
    self._is_calibrated = is_calibrated
    self._accepts_seed = accepts_seed

  def one_step(self, current_state, previous_kernel_results, seed=None):
    if seed is not None and not self._accepts_seed:
      raise TypeError('seed arg not accepted')
    return current_state + 1, TestTransitionKernelResults(
        counter_1=previous_kernel_results.counter_1 + 1,
        counter_2=previous_kernel_results.counter_2 + 2)

  def bootstrap_results(self, current_state):
    return TestTransitionKernelResults(
        counter_1=tf.constant(0, dtype=tf.int32),
        counter_2=tf.constant(0, dtype=tf.int32))

  @property
  def is_calibrated(self):
    return self._is_calibrated


@test_util.test_all_tf_execution_regimes
class ExpectationsReducerTest(test_util.TestCase):

  def test_simple_operation(self):
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer()
    state = mean_reducer.initialize(0)
    for sample in range(6):
      state = mean_reducer.one_step(sample, state)
    mean = self.evaluate(mean_reducer.finalize(state))
    self.assertEqual(2.5, mean)

  def test_with_callables(self):
    callables = [lambda x, y: x + 1, lambda x, y: x + 2]
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer(
        callables=callables
    )
    state = mean_reducer.initialize(0)
    for sample in range(6):
      state = mean_reducer.one_step(sample, state)
    mean = self.evaluate(mean_reducer.finalize(state))
    self.assertEqual([3.5, 4.5], mean)

  def test_with_nested_callables(self):
    callables = [
        {'add_one': lambda x, y: x + 1},
        {'add_two': lambda x, y: x + 2, 'zero': lambda x, y: 0}
    ]
    expectations_reducer = tfp.experimental.mcmc.ExpectationsReducer(
        callables=callables
    )
    state = expectations_reducer.initialize(0)
    for sample in range(6):
      state = expectations_reducer.one_step(sample, state)
    mean = self.evaluate(expectations_reducer.finalize(state))
    self.assertEqual([
        {'add_one': 3.5},
        {'add_two': 4.5, 'zero': 0}
    ], mean)

  def test_with_kernel_results(self):
    def kernel_average(sample, kr):
      return kr.value
    def inner_average(sample, kr):
      return kr.inner_results.value

    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer(
        callables=[kernel_average, inner_average]
    )
    kernel_results = FakeKernelResults(
        0, FakeInnerResults(1))
    state = mean_reducer.initialize(0, kernel_results)
    for sample in range(6):
      kernel_results = FakeKernelResults(
          sample, FakeInnerResults(sample + 1))
      state = mean_reducer.one_step(sample, state, kernel_results)
    mean = self.evaluate(mean_reducer.finalize(state))
    self.assertEqual([2.5, 3.5], mean)

  def test_chunking(self):
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer()
    state = mean_reducer.initialize(tf.ones((3,)))
    kernel_results = FakeKernelResults(
        0, FakeInnerResults(1))
    for sample in range(6):
      state = mean_reducer.one_step(
          tf.ones((3, 9)) * sample, state, kernel_results, axis=1)
    mean = self.evaluate(mean_reducer.finalize(state))
    self.assertEqual((3,), mean.shape)
    self.assertAllEqual([2.5, 2.5, 2.5], mean)

  def test_no_steps(self):
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer()
    state = mean_reducer.initialize(0)
    mean = self.evaluate(mean_reducer.finalize(state))
    if tf.executing_eagerly():
      self.assertEqual(None, mean)
    else:
      self.assertTrue(np.isnan(mean))

  def test_in_with_reductions(self):
    fake_kernel = TestTransitionKernel()
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer()
    reduced_kernel = tfp.experimental.mcmc.WithReductions(
        fake_kernel, mean_reducer,
    )
    pkr = reduced_kernel.bootstrap_results(8)
    _, kernel_results = reduced_kernel.one_step(8, pkr)
    streaming_calculations = self.evaluate(
        mean_reducer.finalize(kernel_results.streaming_calculations))
    self.assertEqual(9, streaming_calculations)

  def test_in_step_kernel(self):
    fake_kernel = TestTransitionKernel()
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer()
    reduced_kernel = tfp.experimental.mcmc.WithReductions(
        fake_kernel, mean_reducer,
    )
    _, kernel_results = tfp.experimental.mcmc.step_kernel(
        num_steps=5,
        current_state=8,
        kernel=reduced_kernel,
        return_final_kernel_results=True,
    )
    streaming_calculations = self.evaluate(
        mean_reducer.finalize(kernel_results.streaming_calculations))
    self.assertEqual(11, streaming_calculations)


if __name__ == '__main__':
  tf.test.main()
