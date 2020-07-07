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
"""Tests for drivers in a Streaming Reductions Framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import collections
import warnings

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.experimental.mcmc.streaming_reductions import step_kernel


TestTransitionKernelResults = collections.namedtuple(
    'TestTransitionKernelResults', 'counter_1, counter_2')


class TestTransitionKernel(tfp.mcmc.TransitionKernel):

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
    return TestTransitionKernelResults(counter_1=0, counter_2=0)

  @property
  def is_calibrated(self):
    return self._is_calibrated


@test_util.test_all_tf_execution_regimes
class StepKernelTest(test_util.TestCase):

  def test_simple_operation(self):
    fake_kernel = TestTransitionKernel()
    final_state, kernel_results = step_kernel(
        num_steps=2,
        current_state=0,
        kernel=fake_kernel,
        return_final_kernel_results=True
    )
    final_state = self.evaluate(final_state)
    self.assertEqual(final_state, 2)
    self.assertEqual(kernel_results.counter_1, 2)
    self.assertEqual(kernel_results.counter_2, 4)

  def test_defined_pkr(self):
    fake_kernel = TestTransitionKernel()
    final_state, kernel_results = step_kernel(
        num_steps=2,
        current_state=0,
        previous_kernel_results=TestTransitionKernelResults(2, 3),
        kernel=fake_kernel,
        return_final_kernel_results=True
    )
    final_state = self.evaluate(final_state)
    self.assertEqual(final_state, 2)
    self.assertEqual(kernel_results.counter_1, 4)
    self.assertEqual(kernel_results.counter_2, 7)

  @parameterized.parameters(1, 2)
  def test_initial_states(self, init_state):
    fake_kernel = TestTransitionKernel()
    final_state = step_kernel(
        num_steps=2,
        current_state=init_state,
        kernel=fake_kernel,
    )
    final_state = self.evaluate(final_state)
    self.assertEqual(final_state, init_state + 2)

  def test_seed(self):
    fake_kernel = TestTransitionKernel()
    final_state, kernel_results = step_kernel(
        num_steps=2,
        current_state=0,
        kernel=fake_kernel,
        return_final_kernel_results=True,
        seed=test_util.test_seed()
    )
    final_state = self.evaluate(final_state)
    self.assertEqual(final_state, 2)
    self.assertEqual(kernel_results.counter_1, 2)
    self.assertEqual(kernel_results.counter_2, 4)

  def test_calibration_warning(self):
    with warnings.catch_warnings(record=True) as triggered:
      kernel = TestTransitionKernel(is_calibrated=False)
      tfp.mcmc.sample_chain(
          num_results=2,
          current_state=0,
          kernel=kernel,
          trace_fn=lambda current_state, kernel_results: kernel_results,
          seed=test_util.test_seed())
    self.assertTrue(
        any('supplied `TransitionKernel` is not calibrated.' in str(
            warning.message) for warning in triggered))


if __name__ == '__main__':
  tf.test.main()
