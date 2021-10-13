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
"""Tests for drivers in a Streaming Reductions Framework."""

import collections
import warnings

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.experimental.mcmc.step import step_kernel
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


class StepKernelTest(test_util.TestCase):

  @test_util.test_all_tf_execution_regimes
  def test_simple_operation(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    final_state, kernel_results = step_kernel(
        num_steps=2,
        current_state=0,
        kernel=fake_kernel,
        return_final_kernel_results=True
    )
    final_state, kernel_results = self.evaluate([final_state, kernel_results])
    self.assertEqual(final_state, 2)
    self.assertEqual(kernel_results.counter_1, 2)
    self.assertEqual(kernel_results.counter_2, 4)

  @test_util.test_all_tf_execution_regimes
  def test_defined_pkr(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    init_pkr = test_fixtures.TestTransitionKernelResults(
        tf.constant(2, dtype=tf.int32), tf.constant(3, dtype=tf.int32))
    final_state, kernel_results = step_kernel(
        num_steps=2,
        current_state=0,
        previous_kernel_results=init_pkr,
        kernel=fake_kernel,
        return_final_kernel_results=True
    )
    final_state, kernel_results = self.evaluate([final_state, kernel_results])
    self.assertEqual(final_state, 2)
    self.assertEqual(kernel_results.counter_1, 4)
    self.assertEqual(kernel_results.counter_2, 7)

  @test_util.test_all_tf_execution_regimes
  def test_initial_state(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    final_state = step_kernel(
        num_steps=2,
        current_state=1,
        kernel=fake_kernel,
    )
    final_state = self.evaluate(final_state)
    self.assertEqual(final_state, 3)

  @test_util.test_all_tf_execution_regimes
  def test_calibration_warning(self):
    with warnings.catch_warnings(record=True) as triggered:
      kernel = test_fixtures.TestTransitionKernel(is_calibrated=False)
      tfp.mcmc.sample_chain(
          num_results=2,
          current_state=0,
          kernel=kernel,
          trace_fn=lambda current_state, kernel_results: kernel_results,
          seed=test_util.test_seed())
    self.assertTrue(
        any('supplied `TransitionKernel` is not calibrated.' in str(
            warning.message) for warning in triggered))

  @test_util.test_all_tf_execution_regimes
  def test_seed_reproducibility(self):
    first_fake_kernel = test_fixtures.RandomTransitionKernel()
    second_fake_kernel = test_fixtures.RandomTransitionKernel()
    seed = samplers.sanitize_seed(test_util.test_seed())
    last_state_t = step_kernel(
        num_steps=1,
        current_state=0,
        kernel=test_fixtures.RandomTransitionKernel(),
        seed=seed,
    )
    for num_steps in range(2, 5):
      first_final_state_t = step_kernel(
          num_steps=num_steps,
          current_state=0.,
          kernel=first_fake_kernel,
          seed=seed,
      )
      second_final_state_t = step_kernel(
          num_steps=num_steps,
          current_state=1.,  # difference should be irrelevant
          kernel=second_fake_kernel,
          seed=seed,
      )
      last_state, first_final_state, second_final_state = self.evaluate([
          last_state_t, first_final_state_t, second_final_state_t
      ])
      self.assertEqual(first_final_state, second_final_state)
      self.assertNotEqual(first_final_state, last_state)
      last_state_t = first_final_state_t

  @test_util.test_graph_mode_only
  def test_smart_for_loop_uses_tf_while(self):
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5],
                      [0.5, 1]])
    true_cov_chol = np.linalg.cholesky(true_cov)
    num_results = 100
    counter = collections.Counter()

    def target_log_prob(x, y):
      counter['target_calls'] += 1
      z = tf.stack([x, y], axis=-1) - true_mean
      z = tf.linalg.triangular_solve(true_cov_chol, z[..., tf.newaxis])[..., 0]
      return -0.5 * tf.reduce_sum(z**2., axis=-1)

    _ = tfp.experimental.mcmc.step_kernel(
        num_steps=num_results,
        current_state=[dtype(-2), dtype(2)],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            step_size=[0.5, 0.5],
            num_leapfrog_steps=2),
        return_final_kernel_results=False,
        seed=test_util.test_seed())

    target_calls = 4 if tf.executing_eagerly() else 2
    self.assertAllEqual(dict(target_calls=target_calls), counter)


if __name__ == '__main__':
  test_util.main()
