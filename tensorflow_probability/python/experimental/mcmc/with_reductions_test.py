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
"""Tests for WithReductions TransitionKernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc import CovarianceReducer
from tensorflow_probability.python.experimental.mcmc import VarianceReducer
from tensorflow_probability.python.experimental.mcmc import WithReductions
from tensorflow_probability.python.internal import test_util


class TestReducer(tfp.experimental.mcmc.Reducer):
  """Simple Reducer that just keeps track of the last sample"""

  def initialize(self, initial_chain_state, initial_kernel_results=None):
    return tf.zeros_like(initial_chain_state)

  def one_step(
      self, new_chain_state, current_reducer_state, previous_kernel_results):
    return new_chain_state


class MeanReducer(tfp.experimental.mcmc.Reducer):
  """Simple Reducer that (naively) computes the mean"""

  def initialize(self, initial_chain_state=None, initial_kernel_results=None):
    return tf.zeros((2,))

  def one_step(
      self, new_chain_state, current_reducer_state, previous_kernel_results):
    return current_reducer_state + tf.convert_to_tensor([1, new_chain_state])

  def finalize(self, final_state):
    return final_state[1] / final_state[0]


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
class WithReductionsTest(test_util.TestCase):

  def test_simple_operation(self):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=fake_reducer,
    )
    pkr = reducer_kernel.bootstrap_results(0.,)
    new_sample, kernel_results = reducer_kernel.one_step(0., pkr)
    new_sample, kernel_results = self.evaluate([
        new_sample, kernel_results])
    self.assertEqual(kernel_results.streaming_calculations, 1)
    self.assertEqual(new_sample, 1)
    self.assertEqual(kernel_results.inner_results.counter_1, 1)
    self.assertEqual(kernel_results.inner_results.counter_2, 2)

  def test_boostrap_results(self):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=fake_reducer,
    )
    pkr = self.evaluate(reducer_kernel.bootstrap_results(9.))
    self.assertEqual(pkr.streaming_calculations, 0)
    self.assertEqual(pkr.inner_results.counter_1, 0)
    self.assertEqual(pkr.inner_results.counter_2, 0)

  def test_is_calibrated(self):
    fake_calibrated_kernel = TestTransitionKernel()
    fake_uncalibrated_kernel = TestTransitionKernel(is_calibrated=False)
    fake_reducer = TestReducer()
    calibrated_reducer_kernel = WithReductions(
        inner_kernel=fake_calibrated_kernel,
        reducers=fake_reducer,
    )
    uncalibrated_reducer_kernel = WithReductions(
        inner_kernel=fake_uncalibrated_kernel,
        reducers=fake_reducer,
    )
    self.assertTrue(calibrated_reducer_kernel.is_calibrated)
    self.assertFalse(uncalibrated_reducer_kernel.is_calibrated)

  def test_tf_while(self):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=fake_reducer,
    )

    initial_state = 0.
    initial_kernel_results = reducer_kernel.bootstrap_results(
        initial_state
    )
    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = reducer_kernel.one_step(curr_state, pkr)
      return (i + 1, new_state, kernel_results)

    _, new_sample, kernel_results = tf.while_loop(
        lambda i, _, __: i < 6,
        _loop_body,
        (0., initial_state, initial_kernel_results)
    )

    new_sample, kernel_results = self.evaluate([
        new_sample, kernel_results])
    self.assertEqual(kernel_results.streaming_calculations, 6)
    self.assertEqual(new_sample, 6)
    self.assertEqual(kernel_results.inner_results.counter_1, 6)
    self.assertEqual(kernel_results.inner_results.counter_2, 12)

  def test_nested_reducers(self):
    fake_kernel = TestTransitionKernel()
    nested_reducers = [[TestReducer(), TestReducer()], [TestReducer()]]
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=nested_reducers,
    )
    pkr = reducer_kernel.bootstrap_results(0.)
    new_sample, kernel_results = reducer_kernel.one_step(0., pkr)
    new_sample, kernel_results = self.evaluate([
        new_sample, kernel_results])

    self.assertEqual(
        len(kernel_results.streaming_calculations[0]), 2)
    self.assertEqual(
        len(kernel_results.streaming_calculations[1]), 1)
    self.assertEqual(
        np.array(kernel_results.streaming_calculations).shape,
        (2,))

    self.assertAllEqual(
        kernel_results.streaming_calculations,
        [[1, 1], [1]])
    self.assertEqual(new_sample, 1)
    self.assertEqual(kernel_results.inner_results.counter_1, 1)
    self.assertEqual(kernel_results.inner_results.counter_2, 2)

  def test_nested_state_dependent_reducers(self):
    fake_kernel = TestTransitionKernel()
    nested_reducers = [[MeanReducer(), MeanReducer()], [MeanReducer()]]
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=nested_reducers,
    )
    pkr = reducer_kernel.bootstrap_results(0.)
    new_sample, kernel_results = reducer_kernel.one_step(0., pkr)
    new_sample, kernel_results = self.evaluate([
        new_sample, kernel_results])

    self.assertEqual(
        len(kernel_results.streaming_calculations[0]), 2)
    self.assertEqual(
        len(kernel_results.streaming_calculations[1]), 1)
    self.assertEqual(
        np.array(kernel_results.streaming_calculations).shape,
        (2,))

    self.assertAllEqualNested(
        kernel_results.streaming_calculations,
        [[[1, 1], [1, 1]], [[1, 1]]])
    self.assertEqual(new_sample, 1)
    self.assertEqual(kernel_results.inner_results.counter_1, 1)
    self.assertEqual(kernel_results.inner_results.counter_2, 2)


@test_util.test_all_tf_execution_regimes
class CovarianceWithReductionsTest(test_util.TestCase):

  @parameterized.parameters(0, 1)
  def test_covariance_reducer(self, ddof):
    fake_kernel = TestTransitionKernel()
    cov_reducer = CovarianceReducer(ddof=ddof)
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=cov_reducer,
    )

    chain_state, kernel_results = 0., reducer_kernel.bootstrap_results(0.)
    for _ in range(6):
      chain_state, kernel_results = reducer_kernel.one_step(
          chain_state, kernel_results)

    kernel_results, final_cov = self.evaluate([
        kernel_results,
        cov_reducer.finalize(
            kernel_results.streaming_calculations)])
    self.assertEqual(kernel_results.streaming_calculations.cov_state.mean, 3.5)
    self.assertNear(
        final_cov,
        np.cov(np.arange(1, 7), ddof=ddof).tolist(),
        err=1e-6)
    self.assertEqual(kernel_results.inner_results.counter_1, 6)
    self.assertEqual(kernel_results.inner_results.counter_2, 12)

  def test_covariance_with_batching(self):
    fake_kernel = TestTransitionKernel((9, 3))
    cov_reducer = CovarianceReducer(event_ndims=1)
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=cov_reducer,
    )
    state = tf.zeros((9, 3))
    kernel_results = reducer_kernel.bootstrap_results(state)
    for _ in range(6):
      state, kernel_results = reducer_kernel.one_step(
          state, kernel_results)
    kernel_results, final_cov = self.evaluate([
        kernel_results,
        cov_reducer.finalize(kernel_results.streaming_calculations)
    ])
    self.assertEqual(
        kernel_results.streaming_calculations.cov_state.mean.shape, (9, 3))
    self.assertEqual(final_cov.shape, (9, 3, 3))

  @parameterized.parameters(0, 1)
  def test_variance_reducer(self, ddof):
    fake_kernel = TestTransitionKernel()
    reducer = VarianceReducer(ddof=ddof)
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=reducer,
    )

    chain_state, kernel_results = 0., reducer_kernel.bootstrap_results(0.)
    for _ in range(6):
      chain_state, kernel_results = reducer_kernel.one_step(
          chain_state, kernel_results)

    kernel_results, final_var = self.evaluate([
        kernel_results,
        reducer.finalize(
            kernel_results.streaming_calculations)])
    self.assertEqual(kernel_results.streaming_calculations.cov_state.mean, 3.5)
    self.assertNear(
        final_var,
        np.var(np.arange(1, 7), ddof=ddof).tolist(),
        err=1e-6)
    self.assertEqual(kernel_results.inner_results.counter_1, 6)
    self.assertEqual(kernel_results.inner_results.counter_2, 12)

  def test_multivariate_normal_covariance_with_sample_chain(self):
    mu = [1, 2, 3]
    cov = [[0.36, 0.12, 0.06],
           [0.12, 0.29, -0.13],
           [0.06, -0.13, 0.26]]
    target = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mu, covariance_matrix=cov
    )
    fake_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target.log_prob,
        step_size=1/3,
        num_leapfrog_steps=27
    )
    cov_reducer = CovarianceReducer()
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=cov_reducer,
    )
    samples, _, kernel_results = tfp.mcmc.sample_chain(
        num_results=20,
        current_state=tf.convert_to_tensor([1., 2., 3.]),
        kernel=reducer_kernel,
        trace_fn=None,
        return_final_kernel_results=True,
        seed=test_util.test_seed()
    )
    samples, kernel_results, final_cov = self.evaluate([
        samples,
        kernel_results,
        cov_reducer.finalize(kernel_results.streaming_calculations)
    ])
    self.assertAllClose(
        kernel_results.streaming_calculations.cov_state.mean,
        np.mean(samples, axis=0),
        rtol=1e-6)
    self.assertAllClose(
        final_cov, np.cov(samples.T, ddof=0), rtol=1e-6)

  def test_covariance_with_step_kernel(self):
    fake_kernel = TestTransitionKernel()
    cov_reducer = CovarianceReducer()
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=cov_reducer,
    )
    chain_state, kernel_results = tfp.experimental.mcmc.step_kernel(
        num_steps=6,
        current_state=0.,
        kernel=reducer_kernel,
        return_final_kernel_results=True,
    )
    chain_state, kernel_results, final_cov = self.evaluate([
        chain_state,
        kernel_results,
        cov_reducer.finalize(kernel_results.streaming_calculations)
    ])
    self.assertEqual(chain_state, 6)
    self.assertEqual(kernel_results.streaming_calculations.cov_state.mean, 3.5)
    self.assertNear(
        final_cov,
        np.cov(np.arange(1, 7), ddof=0).tolist(),
        err=1e-6)
    self.assertEqual(kernel_results.inner_results.counter_1, 6)
    self.assertEqual(kernel_results.inner_results.counter_2, 12)

  def test_covariance_before_transformation(self):
    fake_kernel = TestTransitionKernel(lambda x: -x**2 / 2)
    cov_reducer = CovarianceReducer()
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=cov_reducer,
    )
    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=reducer_kernel,
        bijector=tfp.bijectors.Exp(),
    )
    samples, _, kernel_results = tfp.mcmc.sample_chain(
        num_results=10,
        current_state=1.,
        kernel=transformed_kernel,
        trace_fn=None,
        return_final_kernel_results=True,
        seed=test_util.test_seed())
    samples, kernel_results, final_cov = self.evaluate([
        samples,
        kernel_results,
        cov_reducer.finalize(
            kernel_results.inner_results.streaming_calculations)
    ])
    self.assertAllClose(
        kernel_results.inner_results.streaming_calculations.cov_state.mean,
        np.mean(np.log(samples), axis=0),
        rtol=1e-6)
    self.assertAllClose(
        final_cov, np.cov(np.log(samples).T, ddof=0), rtol=1e-6)

  def test_covariance_after_transformation(self):
    fake_kernel = TestTransitionKernel(lambda x: -x**2 / 2)
    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=fake_kernel,
        bijector=tfp.bijectors.Exp(),
    )
    cov_reducer = CovarianceReducer()
    reducer_kernel = WithReductions(
        inner_kernel=transformed_kernel,
        reducers=cov_reducer,
    )
    samples, _, kernel_results = tfp.mcmc.sample_chain(
        num_results=10,
        current_state=1.,
        kernel=reducer_kernel,
        trace_fn=None,
        return_final_kernel_results=True,
        seed=test_util.test_seed())
    samples, kernel_results, final_cov = self.evaluate([
        samples,
        kernel_results,
        cov_reducer.finalize(
            kernel_results.streaming_calculations)
    ])
    self.assertAllClose(
        kernel_results.streaming_calculations.cov_state.mean,
        np.mean(samples, axis=0),
        rtol=1e-6)
    self.assertAllClose(
        final_cov, np.cov(samples.T, ddof=0), rtol=1e-6)

  def test_nested_in_step_size_adaptation(self):
    target_dist = tfp.distributions.MultivariateNormalDiag(
        loc=[0., 0.], scale_diag=[1., 10.])
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_dist.log_prob,
        num_leapfrog_steps=27,
        step_size=10)
    cov_reducer = CovarianceReducer()
    reducer_kernel = WithReductions(
        inner_kernel=hmc_kernel,
        reducers=cov_reducer
    )
    step_adapted_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=reducer_kernel,
        adaptation_rate=0.8,
        num_adaptation_steps=9)
    samples, _, kernel_results = tfp.mcmc.sample_chain(
        num_results=10,
        current_state=tf.convert_to_tensor([0., 0.]),
        kernel=step_adapted_kernel,
        trace_fn=None,
        return_final_kernel_results=True,
        seed=test_util.test_seed())
    samples, kernel_results, final_cov = self.evaluate([
        samples,
        kernel_results,
        cov_reducer.finalize(
            kernel_results.inner_results.streaming_calculations)
    ])

    mean = kernel_results.inner_results.streaming_calculations.cov_state.mean
    self.assertEqual(mean.shape, (2,))
    self.assertAllClose(mean, np.mean(samples, axis=0), rtol=1e-6)
    self.assertEqual(final_cov.shape, (2, 2))
    self.assertAllClose(
        final_cov, np.cov(samples.T, ddof=0), rtol=1e-6)

  def test_nested_reducers(self):
    fake_kernel = TestTransitionKernel()
    fake_reducer = TestReducer()
    mean_reducer = MeanReducer()
    cov_reducer = CovarianceReducer()
    reducer_kernel = WithReductions(
        inner_kernel=fake_kernel,
        reducers=[[mean_reducer, cov_reducer], [fake_reducer]],
    )

    chain_state, kernel_results = 0., reducer_kernel.bootstrap_results(0.)
    for _ in range(6):
      chain_state, kernel_results = reducer_kernel.one_step(
          chain_state, kernel_results)

    kernel_results, final_cov, final_mean = self.evaluate([
        kernel_results,
        cov_reducer.finalize(
            kernel_results.streaming_calculations[0][1]),
        mean_reducer.finalize(
            kernel_results.streaming_calculations[0][0])
        ])
    self.assertEqual(len(kernel_results.streaming_calculations), 2)
    self.assertEqual(len(kernel_results.streaming_calculations[0]), 2)
    self.assertEqual(len(kernel_results.streaming_calculations[1]), 1)

    self.assertEqual(final_mean, 3.5)
    self.assertEqual(
        kernel_results.streaming_calculations[0][1].cov_state.mean, 3.5)
    self.assertEqual(kernel_results.streaming_calculations[1][0], 6)
    self.assertNear(
        final_cov,
        np.cov(np.arange(1, 7), ddof=0).tolist(),
        err=1e-6)
    self.assertEqual(kernel_results.inner_results.counter_1, 6)
    self.assertEqual(kernel_results.inner_results.counter_2, 12)


if __name__ == '__main__':
  tf.test.main()
