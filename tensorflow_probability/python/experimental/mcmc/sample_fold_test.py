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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import collections
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


TestTransitionKernelResults = collections.namedtuple(
    'TestTransitionKernelResults', 'counter_1, counter_2')


class TestTransitionKernel(tfp.mcmc.TransitionKernel):
  "Deterministic Transition Kernel for testing purposes"

  def __init__(self, shape=(), is_calibrated=True, accepts_seed=True):
    self._shape = shape
    self._is_calibrated = is_calibrated
    self._accepts_seed = accepts_seed

  def one_step(self, current_state, previous_kernel_results, seed=None):
    if seed is not None and not self._accepts_seed:
      raise TypeError('seed arg not accepted')
    return current_state + tf.ones((self._shape)), TestTransitionKernelResults(
        counter_1=previous_kernel_results.counter_1 + tf.ones(self._shape),
        counter_2=previous_kernel_results.counter_2 + tf.ones(self._shape) * 2)

  def bootstrap_results(self, current_state):
    return TestTransitionKernelResults(
        counter_1=tf.zeros(self._shape),
        counter_2=tf.zeros(self._shape))

  @property
  def is_calibrated(self):
    return self._is_calibrated


class RandomTransitionKernel(tfp.mcmc.TransitionKernel):
  "Outputs a random next state following a Rayleigh distribution."

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


class TestReducer(tfp.experimental.mcmc.Reducer):
  """Simple Reducer that (naively) computes the mean"""

  def initialize(self, initial_chain_state=None, initial_kernel_results=None):
    return tf.zeros((2,))

  def one_step(self, sample, current_state, previous_kernel_results, axis=None):
    return current_state + tf.convert_to_tensor([1, sample])

  def finalize(self, final_state):
    return final_state[1] / final_state[0]


@test_util.test_all_tf_execution_regimes
class SampleFoldTest(test_util.TestCase):

  def test_simple_operation(self):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    reduction_rslt, last_sample, kr = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,
    )
    reduction_rslt, last_sample, innermost_results = self.evaluate([
        reduction_rslt,
        last_sample,
        kr.inner_results.inner_results
    ])
    self.assertEqual(3, reduction_rslt)
    self.assertEqual(5, last_sample)
    self.assertEqual(5, innermost_results.counter_1)
    self.assertEqual(10, innermost_results.counter_2)

  @parameterized.parameters(1., 2.)
  def test_current_state(self, curr_state):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    reduction_rslt, last_sample, kr = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=curr_state,
        kernel=fake_kernel,
        reducers=fake_reducer,
    )
    reduction_rslt, last_sample, innermost_results = self.evaluate([
        reduction_rslt,
        last_sample,
        kr.inner_results.inner_results
    ])
    self.assertEqual(
        np.mean(np.arange(curr_state + 1, curr_state + 6)), reduction_rslt)
    self.assertEqual(curr_state + 5, last_sample)
    self.assertEqual(5, innermost_results.counter_1)
    self.assertEqual(10, innermost_results.counter_2)

  def test_warm_restart(self):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    _, last_sample, kr = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,)
    # verify `warm_restart_pkg` works as intended
    reduction_rslt, last_sample, kr = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=last_sample,
        kernel=fake_kernel,
        reducers=fake_reducer,
        previous_kernel_results=kr
    )
    reduction_rslt, last_sample, innermost_results = self.evaluate([
        reduction_rslt,
        last_sample,
        kr.inner_results.inner_results
    ])
    self.assertEqual(5.5, reduction_rslt)
    self.assertEqual(10, last_sample)
    self.assertEqual(10, innermost_results.counter_1)
    self.assertEqual(20, innermost_results.counter_2)

  def test_nested_reducers(self):
    fake_kernel = TestTransitionKernel()
    fake_reducers = [
        [TestReducer(), tfp.experimental.mcmc.CovarianceReducer()],
        [TestReducer()]
    ]
    reduction_rslt, last_sample, kr = tfp.experimental.mcmc.sample_fold(
        num_steps=3,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducers,
    )
    reduction_rslt, last_sample, innermost_results = self.evaluate([
        reduction_rslt,
        last_sample,
        kr.inner_results.inner_results
    ])
    self.assertEqual(2, len(reduction_rslt))
    self.assertEqual(2, len(reduction_rslt[0]))
    self.assertEqual(1, len(reduction_rslt[1]))

    self.assertEqual(2, reduction_rslt[0][0])
    self.assertNear(2/3, reduction_rslt[0][1], err=1e-6)
    self.assertEqual(3, last_sample)
    self.assertEqual(3, innermost_results.counter_1)
    self.assertEqual(6, innermost_results.counter_2)

  def test_true_streaming_covariance(self):
    seed = test_util.test_seed()
    fake_kernel = TestTransitionKernel(())
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reduction_rslt, _, _ = tfp.experimental.mcmc.sample_fold(
        num_steps=20,
        current_state=tf.convert_to_tensor([0., 0.]),
        kernel=fake_kernel,
        reducers=cov_reducer,
        seed=seed,)
    reduction_rslt = self.evaluate(reduction_rslt)
    self.assertAllClose(
        np.cov(np.column_stack((np.arange(20), np.arange(20))).T, ddof=0),
        reduction_rslt,
        rtol=1e-5)

  def test_batched_streaming_covariance(self):
    fake_kernel = TestTransitionKernel((2, 3))
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(event_ndims=1)
    reduction_rslt, last_sample, kr = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=tf.convert_to_tensor(
            [[0., 0., 0.], [0., 0., 0.]]),
        kernel=fake_kernel,
        reducers=cov_reducer,
    )
    reduction_rslt, streaming_calc = self.evaluate([
        reduction_rslt, kr.streaming_calculations
    ])
    mean = streaming_calc.cov_state.mean
    self.assertEqual((2, 3, 3), reduction_rslt.shape)
    self.assertAllEqual(np.ones(reduction_rslt.shape) * 2, reduction_rslt)
    self.assertEqual((2, 3), mean.shape)
    self.assertAllEqual(np.ones(mean.shape) * 3, mean)
    self.assertAllEqualNested(last_sample, [[5., 5., 5.], [5., 5., 5.]])

  def test_seed_reproducibility(self):
    seed = samplers.sanitize_seed(test_util.test_seed())
    fake_kernel = RandomTransitionKernel()
    fake_reducer = TestReducer()
    first_reduction_rslt, _, _ = tfp.experimental.mcmc.sample_fold(
        num_steps=3,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,
        seed=seed
    )
    second_reduction_rslt, _, _ = tfp.experimental.mcmc.sample_fold(
        num_steps=3,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,
        seed=seed
    )
    first_reduction_rslt, second_reduction_rslt = self.evaluate([
        first_reduction_rslt, second_reduction_rslt
    ])
    self.assertEqual(first_reduction_rslt, second_reduction_rslt)

  def test_thinning_and_burnin(self):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    reduction_rslt, last_sample, kr = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,
        num_burnin_steps=10,
        num_steps_between_results=1,
    )
    reduction_rslt, last_sample, kernel_results = self.evaluate([
        reduction_rslt,
        last_sample,
        kr
    ])
    self.assertEqual(16, reduction_rslt)
    self.assertEqual(20, last_sample)
    self.assertEqual(
        20, kernel_results.inner_results.inner_results.counter_1)
    self.assertEqual(
        40, kernel_results.inner_results.inner_results.counter_2)
    self.assertEqual(
        5, kernel_results.inner_results.call_counter)

  def test_tensor_thinning_and_burnin(self):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    reduction_rslt, last_sample, kr = tfp.experimental.mcmc.sample_fold(
        num_steps=tf.convert_to_tensor(5),
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,
        num_burnin_steps=tf.convert_to_tensor(10),
        num_steps_between_results=tf.convert_to_tensor(1),
    )
    reduction_rslt, last_sample, kernel_results = self.evaluate([
        reduction_rslt,
        last_sample,
        kr
    ])
    self.assertEqual(16, reduction_rslt)
    self.assertEqual(20, last_sample)
    self.assertEqual(
        20, kernel_results.inner_results.inner_results.counter_1)
    self.assertEqual(
        40, kernel_results.inner_results.inner_results.counter_2)
    self.assertEqual(
        5, kernel_results.inner_results.call_counter)

  def test_no_reducer(self):
    fake_kernel = TestTransitionKernel()
    reduction_rslt, last_sample, kr = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        num_burnin_steps=10,
        num_steps_between_results=1,
    )
    last_sample, innermost_results = self.evaluate([
        last_sample,
        kr.inner_results.inner_results
    ])
    self.assertEqual(None, reduction_rslt)
    self.assertEqual(20, last_sample)
    self.assertEqual(None, kr.streaming_calculations)
    self.assertEqual(20, innermost_results.counter_1)
    self.assertEqual(40, innermost_results.counter_2)


if __name__ == '__main__':
  tf.test.main()
