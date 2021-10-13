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
"""Tests for ProgressBarReducer."""

# Dependency imports

import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.internal import test_util


def test_progress_bar_fn(num_steps):
  return iter(range(num_steps))


@test_util.test_all_tf_execution_regimes
class ProgressBarReducerTest(test_util.TestCase):

  def test_noop(self):
    num_results = 10
    pbar = tfp.experimental.mcmc.ProgressBarReducer(
        num_results, test_progress_bar_fn)
    self.assertEqual(pbar.num_results, num_results)
    pbar.initialize(None)
    for _ in range(num_results):
      pbar.one_step(None, None, None)

  def test_too_many_steps_is_ok(self):
    num_results = 10
    pbar = tfp.experimental.mcmc.ProgressBarReducer(
        num_results, test_progress_bar_fn)
    pbar.initialize(None)
    for _ in range(num_results):
      pbar.one_step(None, None, None)
    pbar.one_step(None, None, None)

  def test_sample_fold(self):
    num_results = 3
    pbar = tfp.experimental.mcmc.ProgressBarReducer(
        num_results, test_progress_bar_fn)
    fake_kernel = test_fixtures.TestTransitionKernel()
    reductions, final_state, kernel_results = tfp.experimental.mcmc.sample_fold(
        num_steps=num_results,
        current_state=0.,
        kernel=fake_kernel,
        reducer=pbar,
    )
    reductions, final_state, kernel_results = self.evaluate([
        reductions,
        final_state,
        kernel_results
    ])
    self.assertEqual([], reductions)
    self.assertEqual(num_results, final_state)
    self.assertEqual(num_results, kernel_results.counter_1)
    self.assertEqual(num_results * 2, kernel_results.counter_2)


if __name__ == '__main__':
  test_util.main()
