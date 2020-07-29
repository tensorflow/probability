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
"""Tests for SampleDiscardingKernel TransitionKernel (thinning and burn-in)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


TestTransitionKernelResults = collections.namedtuple(
    'TestTransitionKernelResults', 'counter_1, counter_2')


class TestTransitionKernel(tfp.mcmc.TransitionKernel):
  """Fake deterministic Transition Kernel"""

  def __init__(self, shape=(), target_log_prob_fn=None, is_calibrated=True):
    self._is_calibrated = is_calibrated
    self._shape = shape
    # for composition purposes
    self.parameters = dict(
        target_log_prob_fn=target_log_prob_fn)

  def one_step(self, current_state, previous_kernel_results, seed=None):
    return (current_state + tf.ones(self._shape),
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


@test_util.test_all_tf_execution_regimes
class SampleDiscardingTest(test_util.TestCase):

  def test_thinning(self):
    fake_inner_kernel = TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_steps_between_results=1,)
    first_state, kernel_results = discarder.one_step(0.)
    second_state, kernel_results = discarder.one_step(
        first_state, kernel_results)
    first_state, second_state, inner_results = self.evaluate([
        first_state, second_state, kernel_results.inner_results])
    self.assertEqual(first_state, 2)
    self.assertEqual(second_state, 4)
    self.assertEqual(kernel_results.call_counter, 2)
    self.assertEqual(inner_results.counter_1, 4)
    self.assertEqual(inner_results.counter_2, 8)

  def test_burnin(self):
    fake_inner_kernel = TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=5,)
    sample, kernel_results = discarder.one_step(0.)
    sample, inner_results = self.evaluate([
        sample, kernel_results.inner_results])
    self.assertEqual(sample, 6)
    self.assertEqual(kernel_results.call_counter, 1)
    self.assertEqual(inner_results.counter_1, 6)
    self.assertEqual(inner_results.counter_2, 12)

  def test_no_thinning_or_burnin(self):
    fake_inner_kernel = TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,)
    first_state, kernel_results = discarder.one_step(0.)
    second_state, kernel_results = discarder.one_step(
        first_state, kernel_results)
    first_state, second_state, inner_results = self.evaluate([
        first_state, second_state, kernel_results.inner_results])
    self.assertEqual(first_state, 1)
    self.assertEqual(second_state, 2)
    self.assertEqual(kernel_results.call_counter, 2)
    self.assertEqual(inner_results.counter_1, 2)
    self.assertEqual(inner_results.counter_2, 4)

  def test_both_thinning_and_burnin(self):
    fake_inner_kernel = TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=10,
        num_steps_between_results=1,)
    first_state, kernel_results = discarder.one_step(0.)
    second_state, kernel_results = discarder.one_step(
        first_state, kernel_results)
    first_state, second_state, inner_results = self.evaluate([
        first_state, second_state, kernel_results.inner_results])
    self.assertEqual(first_state, 12)
    self.assertEqual(second_state, 14)
    self.assertEqual(kernel_results.call_counter, 2)
    self.assertEqual(inner_results.counter_1, 14)
    self.assertEqual(inner_results.counter_2, 28)

  def test_cold_start(self):
    fake_inner_kernel = TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=10,
        num_steps_between_results=1,)
    first_state, _ = discarder.one_step(0.)
    second_state, kernel_results = discarder.one_step(first_state)
    first_state, second_state, inner_results = self.evaluate([
        first_state, second_state, kernel_results.inner_results])
    self.assertEqual(first_state, 12)
    self.assertEqual(second_state, 24)
    self.assertEqual(kernel_results.call_counter, 1)
    self.assertEqual(inner_results.counter_1, 12)
    self.assertEqual(inner_results.counter_2, 24)

  def test_is_calibrated(self):
    calibrated_kernel = TestTransitionKernel()
    uncalibrated_kernel = TestTransitionKernel(is_calibrated=False)
    calibrated_discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        calibrated_kernel)
    uncalibrated_discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        uncalibrated_kernel)
    self.assertTrue(calibrated_discarder.is_calibrated)
    self.assertFalse(uncalibrated_discarder.is_calibrated)

  def test_with_composed_kernel(self):
    fake_inner_kernel = TestTransitionKernel()
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=tfp.experimental.mcmc.SampleDiscardingKernel(
            inner_kernel=fake_inner_kernel,
            num_burnin_steps=10,
            num_steps_between_results=2,),
        reducers=cov_reducer
    )
    current_state, kernel_results = 0., None
    for _ in range(2):
      current_state, kernel_results = reducer_kernel.one_step(
          current_state, kernel_results)
    current_state, most_inner_results, cov = self.evaluate([
        current_state,
        kernel_results.inner_results.inner_results,
        cov_reducer.finalize(kernel_results.streaming_calculations),
    ])
    self.assertEqual(current_state, 16)
    self.assertEqual(kernel_results.inner_results.call_counter, 2)
    self.assertEqual(most_inner_results.counter_1, 16)
    self.assertEqual(most_inner_results.counter_2, 32)
    self.assertEqual(cov, np.var([13, 16]))

  def test_tf_while(self):
    fake_inner_kernel = TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=10,
        num_steps_between_results=1,)

    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = discarder.one_step(
          curr_state, pkr,
      )
      return (i + 1, new_state, kernel_results)

    pkr = discarder.bootstrap_results(0.)
    _, final_sample, kernel_results = tf.while_loop(
        lambda i, _, __: i < 2,
        _loop_body,
        (0., 0., pkr),
    )
    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(final_sample, 14)
    self.assertEqual(kernel_results.call_counter, 2)
    self.assertEqual(kernel_results.inner_results.counter_1, 14)
    self.assertEqual(kernel_results.inner_results.counter_2, 28)

  def test_tensor_thinning_and_burnin(self):
    fake_inner_kernel = TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=tf.convert_to_tensor(10),
        num_steps_between_results=tf.convert_to_tensor(1),)

    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = discarder.one_step(
          curr_state, pkr,
      )
      return (i + 1, new_state, kernel_results)

    pkr = discarder.bootstrap_results(0.)
    _, final_sample, kernel_results = tf.while_loop(
        lambda i, _, __: i < 2,
        _loop_body,
        (0., 0., pkr),
    )
    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(final_sample, 14)
    self.assertEqual(kernel_results.call_counter, 2)
    self.assertEqual(kernel_results.inner_results.counter_1, 14)
    self.assertEqual(kernel_results.inner_results.counter_2, 28)

  def test_tensor_no_burnin(self):
    fake_inner_kernel = TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=tf.convert_to_tensor(0),
        num_steps_between_results=tf.convert_to_tensor(1),)

    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = discarder.one_step(
          curr_state, pkr,
      )
      return (i + 1, new_state, kernel_results)

    pkr = discarder.bootstrap_results(0.)
    _, final_sample, kernel_results = tf.while_loop(
        lambda i, _, __: i < 2,
        _loop_body,
        (0., 0., pkr),
    )
    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(final_sample, 4)
    self.assertEqual(kernel_results.call_counter, 2)
    self.assertEqual(kernel_results.inner_results.counter_1, 4)
    self.assertEqual(kernel_results.inner_results.counter_2, 8)


if __name__ == '__main__':
  tf.test.main()
