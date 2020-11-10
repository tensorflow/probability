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
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.internal import test_util


FakeKernelResults = collections.namedtuple(
        'FakeKernelResults', 'value, inner_results')


FakeInnerResults = collections.namedtuple(
    'FakeInnerResults', 'value')


@test_util.test_all_tf_execution_regimes
class ExpectationsReducerTest(test_util.TestCase):

  def test_simple_operation(self):
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer()
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = mean_reducer.initialize(0, fake_kr)
    for sample in range(6):
      state = mean_reducer.one_step(sample, state, fake_kr)
    mean = self.evaluate(mean_reducer.finalize(state))
    self.assertEqual(2.5, mean)

  def test_with_transform_fn(self):
    transform_fn = [lambda x, y: x + 1, lambda x, y: x + 2]
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer(
        transform_fn=transform_fn
    )
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = mean_reducer.initialize(0, fake_kr)
    for sample in range(6):
      state = mean_reducer.one_step(sample, state, fake_kr)
    mean = self.evaluate(mean_reducer.finalize(state))
    self.assertEqual([3.5, 4.5], mean)

  def test_with_nested_transform_fn(self):
    transform_fn = [
        {'add_one': lambda x, y: x + 1},
        {'add_two': lambda x, y: x + 2, 'zero': lambda x, y: tf.zeros(())}
    ]
    expectations_reducer = tfp.experimental.mcmc.ExpectationsReducer(
        transform_fn=transform_fn
    )
    fake_kr = FakeKernelResults(0, FakeInnerResults(0))
    state = expectations_reducer.initialize(0, fake_kr)
    for sample in range(6):
      state = expectations_reducer.one_step(sample, state, fake_kr)
    mean = self.evaluate(expectations_reducer.finalize(state))
    self.assertEqual([
        {'add_one': 3.5},
        {'add_two': 4.5, 'zero': 0}
    ], mean)

  def test_with_kernel_results(self):
    def kernel_average(sample, kr):
      del sample
      return kr.value
    def inner_average(sample, kr):
      del sample
      return kr.inner_results.value

    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer(
        transform_fn=[kernel_average, inner_average]
    )
    kernel_results = FakeKernelResults(
        0, FakeInnerResults(0))
    state = mean_reducer.initialize(0, kernel_results)
    for sample in range(6):
      kernel_results = FakeKernelResults(
          sample, FakeInnerResults(sample + 1))
      state = mean_reducer.one_step(sample, state, kernel_results)
    mean = self.evaluate(mean_reducer.finalize(state))
    self.assertEqual([2.5, 3.5], mean)

  def test_chunking(self):
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer()
    kernel_results = FakeKernelResults(
        tf.zeros((3, 9)), FakeInnerResults(tf.ones((3, 9))))
    state = mean_reducer.initialize(tf.ones((3,)), kernel_results)
    for sample in range(6):
      state = mean_reducer.one_step(
          tf.ones((3, 9)) * sample, state, kernel_results, axis=1)
    mean = self.evaluate(mean_reducer.finalize(state))
    self.assertEqual((3,), mean.shape)
    self.assertAllEqual([2.5, 2.5, 2.5], mean)

  def test_no_steps(self):
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer()
    kernel_results = FakeKernelResults(
        0, FakeInnerResults(0))
    state = mean_reducer.initialize(0, kernel_results)
    mean = self.evaluate(mean_reducer.finalize(state))
    self.assertEqual(0, mean)

  def test_in_with_reductions(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    mean_reducer = tfp.experimental.mcmc.ExpectationsReducer()
    reduced_kernel = tfp.experimental.mcmc.WithReductions(
        fake_kernel, mean_reducer,
    )
    pkr = reduced_kernel.bootstrap_results(8)
    _, kernel_results = reduced_kernel.one_step(8, pkr)
    reduction_results = self.evaluate(
        mean_reducer.finalize(kernel_results.reduction_results))
    self.assertEqual(9, reduction_results)

  def test_in_step_kernel(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
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
    reduction_results = self.evaluate(
        mean_reducer.finalize(kernel_results.reduction_results))
    self.assertEqual(11, reduction_results)


if __name__ == '__main__':
  tf.test.main()
