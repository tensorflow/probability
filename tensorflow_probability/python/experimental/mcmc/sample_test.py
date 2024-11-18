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
"""Tests for high(er) level drivers for streaming MCMC."""

from tensorflow_probability.python.experimental.mcmc import sample
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class RunTest(test_util.TestCase):

  def test_simple_reduction(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.NaiveMeanReducer()
    seed1, seed2 = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'))
    result = sample.sample_chain(
        num_results=5,
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducer,
        seed=seed1,
    )
    last_sample, reduction_result, kernel_results = self.evaluate([
        result.final_state, result.reduction_results,
        result.final_kernel_results
    ])
    self.assertEqual(5, last_sample)
    self.assertEqual(3, reduction_result)
    self.assertEqual(5, kernel_results.counter_1)
    self.assertEqual(10, kernel_results.counter_2)

    # Warm-restart the underlying kernel but not the reduction
    result_2 = sample.sample_chain(
        num_results=5,
        current_state=last_sample,
        kernel=fake_kernel,
        reducer=fake_reducer,
        previous_kernel_results=kernel_results,
        seed=seed2,
    )
    last_sample_2, reduction_result_2, kernel_results_2 = self.evaluate([
        result_2.final_state, result_2.reduction_results,
        result_2.final_kernel_results
    ])
    self.assertEqual(10, last_sample_2)
    self.assertEqual(8, reduction_result_2)
    self.assertEqual(10, kernel_results_2.counter_1)
    self.assertEqual(20, kernel_results_2.counter_2)

  def test_reducer_warm_restart(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.NaiveMeanReducer()
    seed1, seed2 = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'))
    result = sample.sample_chain(
        num_results=5,
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducer,
        seed=seed1,
    )
    last_sample, red_res, kernel_results = self.evaluate([
        result.final_state, result.reduction_results,
        result.final_kernel_results
    ])
    self.assertEqual(3, red_res)
    self.assertEqual(5, last_sample)
    self.assertEqual(5, kernel_results.counter_1)
    self.assertEqual(10, kernel_results.counter_2)

    # Warm-restart the underlying kernel and the reduction using the provided
    # restart package
    result_2 = sample.sample_chain(
        num_results=5, seed=seed2, **result.resume_kwargs)
    last_sample_2, reduction_result_2, kernel_results_2 = self.evaluate([
        result_2.final_state, result_2.reduction_results,
        result_2.final_kernel_results
    ])
    self.assertEqual(5.5, reduction_result_2)
    self.assertEqual(10, last_sample_2)
    self.assertEqual(10, kernel_results_2.counter_1)
    self.assertEqual(20, kernel_results_2.counter_2)

  def test_tracing_a_reduction(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.NaiveMeanReducer()
    result = sample.sample_chain(
        num_results=5,
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducer,
        trace_fn=lambda _state, _kr, reductions: reductions,
        seed=test_util.test_seed(),
    )
    trace = self.evaluate(result.trace)
    self.assertAllEqual(trace, [1.0, 1.5, 2.0, 2.5, 3.0])

  def test_tracing_no_reduction(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    result = sample.sample_chain(
        num_results=5,
        current_state=0.,
        kernel=fake_kernel,
        trace_fn=lambda state, _kr: state + 10,
        seed=test_util.test_seed(),
    )
    trace = self.evaluate(result.trace)
    self.assertAllEqual(trace, [11.0, 12.0, 13.0, 14.0, 15.0])


if __name__ == '__main__':
  test_util.main()
