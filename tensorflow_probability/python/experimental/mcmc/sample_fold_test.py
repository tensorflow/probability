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
    reduction_rslt, warm_restart_pkg = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,
    )
    reduction_rslt, last_sample, innermost_results = self.evaluate([
        reduction_rslt,
        warm_restart_pkg[0],
        warm_restart_pkg[1].inner_results.inner_results
    ])
    self.assertEqual(reduction_rslt, 3)
    self.assertEqual(last_sample, 5)
    self.assertEqual(innermost_results.counter_1, 5)
    self.assertEqual(innermost_results.counter_2, 10)

  @parameterized.parameters(1., 2.)
  def test_current_state(self, curr_state):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    reduction_rslt, warm_restart_pkg = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=curr_state,
        kernel=fake_kernel,
        reducers=fake_reducer,
    )
    reduction_rslt, last_sample, innermost_results = self.evaluate([
        reduction_rslt,
        warm_restart_pkg[0],
        warm_restart_pkg[1].inner_results.inner_results
    ])
    self.assertEqual(reduction_rslt, np.mean(
        np.arange(curr_state + 1, curr_state + 6)))
    self.assertEqual(last_sample, curr_state + 5)
    self.assertEqual(innermost_results.counter_1, 5)
    self.assertEqual(innermost_results.counter_2, 10)

  def test_warm_restart(self):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    _, warm_restart_pkg = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,)
    # verify `warm_restart_pkg` works as intended
    reduction_rslt, warm_restart_pkg = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=warm_restart_pkg[0],
        kernel=fake_kernel,
        reducers=fake_reducer,
        previous_kernel_results=warm_restart_pkg[1]
    )
    reduction_rslt, last_sample, innermost_results = self.evaluate([
        reduction_rslt,
        warm_restart_pkg[0],
        warm_restart_pkg[1].inner_results.inner_results
    ])
    self.assertEqual(reduction_rslt, 5.5)
    self.assertEqual(last_sample, 10)
    self.assertEqual(innermost_results.counter_1, 10)
    self.assertEqual(innermost_results.counter_2, 20)

  def test_nested_reducers(self):
    fake_kernel = TestTransitionKernel()
    fake_reducers = [
        [TestReducer(), tfp.experimental.mcmc.CovarianceReducer()],
        [TestReducer()]
    ]
    reduction_rslt, warm_restart_pkg = tfp.experimental.mcmc.sample_fold(
        num_steps=3,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducers,
    )
    reduction_rslt, last_sample, innermost_results = self.evaluate([
        reduction_rslt,
        warm_restart_pkg[0],
        warm_restart_pkg[1].inner_results.inner_results
    ])
    self.assertEqual(len(reduction_rslt), 2)
    self.assertEqual(len(reduction_rslt[0]), 2)
    self.assertEqual(len(reduction_rslt[1]), 1)

    self.assertEqual(reduction_rslt[0][0], 2)
    self.assertNear(reduction_rslt[0][1], 2/3, err=1e-6)
    self.assertEqual(last_sample, 3)
    self.assertEqual(innermost_results.counter_1, 3)
    self.assertEqual(innermost_results.counter_2, 6)

  def test_true_streaming_covariance(self):
    seed = test_util.test_seed()
    fake_kernel = TestTransitionKernel(())
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reduction_rslt, _ = tfp.experimental.mcmc.sample_fold(
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
    reduction_rslt, warm_restart_pkg = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=tf.convert_to_tensor(
            [[0., 0., 0.], [0., 0., 0.]]),
        kernel=fake_kernel,
        reducers=cov_reducer,
    )
    reduction_rslt, streaming_calc = self.evaluate([
        reduction_rslt, warm_restart_pkg[1].streaming_calculations
    ])
    mean = streaming_calc.cov_state.mean
    self.assertEqual(reduction_rslt.shape, (2, 3, 3))
    self.assertAllEqual(reduction_rslt, np.ones(reduction_rslt.shape) * 2)
    self.assertEqual(mean.shape, (2, 3))
    self.assertAllEqual(mean, np.ones(mean.shape) * 3)
    self.assertAllEqualNested(warm_restart_pkg[0], [[5., 5., 5.], [5., 5., 5.]])

  def test_seed_reproducibility(self):
    seed = samplers.sanitize_seed(test_util.test_seed())
    fake_kernel = RandomTransitionKernel()
    fake_reducer = TestReducer()
    first_reduction_rslt, _ = tfp.experimental.mcmc.sample_fold(
        num_steps=3,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,
        seed=seed
    )
    second_reduction_rslt, _ = tfp.experimental.mcmc.sample_fold(
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
    reduction_rslt, warm_restart_pkg = tfp.experimental.mcmc.sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducers=fake_reducer,
        num_burnin_steps=10,
        num_steps_between_results=1,
    )
    reduction_rslt, last_sample, innermost_results = self.evaluate([
        reduction_rslt,
        warm_restart_pkg[0],
        warm_restart_pkg[1].inner_results.inner_results
    ])
    self.assertEqual(reduction_rslt, 16)
    self.assertEqual(last_sample, 20)
    self.assertEqual(innermost_results.counter_1, 20)
    self.assertEqual(innermost_results.counter_2, 40)
    self.assertEqual(
        warm_restart_pkg[1].inner_results.call_counter, 5)


if __name__ == '__main__':
  tf.test.main()
