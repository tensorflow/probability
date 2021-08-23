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
"""Tests for tfp.experimental.mcmc.WithReductions TransitionKernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class WithReductionsTest(test_util.TestCase):

  def test_simple_operation(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.TestReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=fake_reducer,
    )
    pkr = reducer_kernel.bootstrap_results(0.,)
    new_sample, kernel_results = reducer_kernel.one_step(0., pkr)
    new_sample, kernel_results = self.evaluate([
        new_sample, kernel_results])
    self.assertEqual(1, kernel_results.reduction_results)
    self.assertEqual(1, new_sample)
    self.assertEqual(1, kernel_results.inner_results.counter_1)
    self.assertEqual(2, kernel_results.inner_results.counter_2)

  def test_boostrap_results(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.TestReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=fake_reducer,
    )
    pkr = self.evaluate(reducer_kernel.bootstrap_results(9.))
    self.assertEqual(0, pkr.reduction_results, 0)
    self.assertEqual(0, pkr.inner_results.counter_1, 0)
    self.assertEqual(0, pkr.inner_results.counter_2, 0)

  def test_is_calibrated(self):
    fake_calibrated_kernel = test_fixtures.TestTransitionKernel()
    fake_uncalibrated_kernel = test_fixtures.TestTransitionKernel(
        is_calibrated=False)
    fake_reducer = test_fixtures.TestReducer()
    calibrated_reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_calibrated_kernel,
        reducer=fake_reducer,
    )
    uncalibrated_reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_uncalibrated_kernel,
        reducer=fake_reducer,
    )
    self.assertTrue(calibrated_reducer_kernel.is_calibrated)
    self.assertFalse(uncalibrated_reducer_kernel.is_calibrated)

  def test_tf_while(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.TestReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=fake_reducer,
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
    self.assertEqual(6, kernel_results.reduction_results)
    self.assertEqual(6, new_sample)
    self.assertEqual(6, kernel_results.inner_results.counter_1)
    self.assertEqual(12, kernel_results.inner_results.counter_2)

  def test_nested_reducers(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    nested_reducer = [
        [test_fixtures.TestReducer(), test_fixtures.TestReducer()],
        [test_fixtures.TestReducer()]
    ]
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=nested_reducer,
    )
    pkr = reducer_kernel.bootstrap_results(0.)
    new_sample, kernel_results = reducer_kernel.one_step(0., pkr)
    new_sample, kernel_results = self.evaluate([
        new_sample, kernel_results])

    self.assertEqual(
        2, len(kernel_results.reduction_results[0]))
    self.assertEqual(
        1, len(kernel_results.reduction_results[1]))
    self.assertEqual(
        (2,),
        np.array(kernel_results.reduction_results).shape)

    self.assertAllEqual(
        [[1, 1], [1]],
        kernel_results.reduction_results)
    self.assertEqual(1, new_sample)
    self.assertEqual(1, kernel_results.inner_results.counter_1)
    self.assertEqual(2, kernel_results.inner_results.counter_2)

  def test_nested_state_dependent_reducers(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    nested_reducer = [
        [test_fixtures.NaiveMeanReducer(), test_fixtures.NaiveMeanReducer()],
        [test_fixtures.NaiveMeanReducer()]
    ]
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=nested_reducer,
    )
    pkr = reducer_kernel.bootstrap_results(0.)
    new_sample, kernel_results = reducer_kernel.one_step(0., pkr)
    new_sample, kernel_results = self.evaluate([
        new_sample, kernel_results])

    self.assertEqual(
        2,
        len(kernel_results.reduction_results[0]))
    self.assertEqual(
        1,
        len(kernel_results.reduction_results[1]))
    self.assertEqual(
        (2,),
        np.array(kernel_results.reduction_results).shape)

    self.assertAllEqualNested(
        kernel_results.reduction_results,
        [[[1, 1], [1, 1]], [[1, 1]]],
    )
    self.assertEqual(1, new_sample)
    self.assertEqual(1, kernel_results.inner_results.counter_1)
    self.assertEqual(2, kernel_results.inner_results.counter_2)


@test_util.test_all_tf_execution_regimes
class CovarianceWithReductionsTest(test_util.TestCase):

  @parameterized.parameters(0, 1)
  def test_covariance_reducer(self, ddof):
    fake_kernel = test_fixtures.TestTransitionKernel()
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(ddof=ddof)
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=cov_reducer,
    )

    chain_state, kernel_results = 0., reducer_kernel.bootstrap_results(0.)
    for _ in range(6):
      chain_state, kernel_results = reducer_kernel.one_step(
          chain_state, kernel_results)

    final_cov = self.evaluate(
        cov_reducer.finalize(kernel_results.reduction_results))
    self.assertAllEqual(
        3.5, kernel_results.reduction_results.cov_state.mean)
    self.assertNear(
        np.cov(np.arange(1, 7), ddof=ddof).tolist(),
        final_cov,
        err=1e-6)
    self.assertAllEqual(6, kernel_results.inner_results.counter_1)
    self.assertAllEqual(12, kernel_results.inner_results.counter_2)

  def test_covariance_with_batching(self):
    fake_kernel = test_fixtures.TestTransitionKernel((9, 3))
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer(event_ndims=1)
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=cov_reducer,
    )
    state = tf.zeros((9, 3))
    kernel_results = reducer_kernel.bootstrap_results(state)
    for _ in range(6):
      state, kernel_results = reducer_kernel.one_step(
          state, kernel_results)
    final_cov = cov_reducer.finalize(kernel_results.reduction_results)
    self.assertEqual(
        (9, 3), kernel_results.reduction_results.cov_state.mean.shape)
    self.assertEqual((9, 3, 3), final_cov.shape)

  @parameterized.parameters(0, 1)
  def test_variance_reducer(self, ddof):
    fake_kernel = test_fixtures.TestTransitionKernel()
    reducer = tfp.experimental.mcmc.VarianceReducer(ddof=ddof)
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=reducer,
    )

    chain_state, kernel_results = 0., reducer_kernel.bootstrap_results(0.)
    for _ in range(6):
      chain_state, kernel_results = reducer_kernel.one_step(
          chain_state, kernel_results)

    final_var = self.evaluate(
        reducer.finalize(kernel_results.reduction_results))
    self.assertAllEqual(
        3.5, kernel_results.reduction_results.cov_state.mean)
    self.assertNear(
        np.var(np.arange(1, 7), ddof=ddof).tolist(),
        final_var,
        err=1e-6)
    self.assertAllEqual(6, kernel_results.inner_results.counter_1)
    self.assertAllEqual(12, kernel_results.inner_results.counter_2)

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
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=cov_reducer,
    )
    samples, _, kernel_results = tfp.mcmc.sample_chain(
        num_results=20,
        current_state=tf.convert_to_tensor([1., 2., 3.]),
        kernel=reducer_kernel,
        trace_fn=None,
        return_final_kernel_results=True,
        seed=test_util.test_seed(sampler_type='stateless')
    )
    samples, mean, final_cov = self.evaluate([
        samples,
        kernel_results.reduction_results.cov_state.mean,
        cov_reducer.finalize(kernel_results.reduction_results)
    ])
    self.assertAllClose(np.mean(samples, axis=0), mean, rtol=1e-6)
    self.assertAllClose(np.cov(samples.T, ddof=0), final_cov, rtol=1e-6)

  def test_covariance_with_step_kernel(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=cov_reducer,
    )
    chain_state, kernel_results = tfp.experimental.mcmc.step_kernel(
        num_steps=6,
        current_state=0.,
        kernel=reducer_kernel,
        return_final_kernel_results=True,
    )
    final_cov = self.evaluate(
        cov_reducer.finalize(kernel_results.reduction_results))
    self.assertAllEqual(6, chain_state)
    self.assertAllEqual(
        3.5, kernel_results.reduction_results.cov_state.mean)
    self.assertNear(
        np.cov(np.arange(1, 7), ddof=0).tolist(),
        final_cov,
        err=1e-6)
    self.assertAllEqual(6, kernel_results.inner_results.counter_1)
    self.assertAllEqual(12, kernel_results.inner_results.counter_2)

  def test_covariance_before_transformation(self):
    fake_kernel = test_fixtures.TestTransitionKernel(
        target_log_prob_fn=lambda x: -x**2 / 2)
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=cov_reducer,
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
    samples, final_cov = self.evaluate([
        samples,
        cov_reducer.finalize(
            kernel_results.inner_results.reduction_results)
    ])
    self.assertAllClose(
        np.mean(np.log(samples), axis=0),
        kernel_results.inner_results.reduction_results.cov_state.mean,
        rtol=1e-6)
    self.assertAllClose(
        np.cov(np.log(samples).T, ddof=0), final_cov, rtol=1e-6)

  def test_covariance_after_transformation(self):
    fake_kernel = test_fixtures.TestTransitionKernel(
        target_log_prob_fn=lambda x: -x**2 / 2)
    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=fake_kernel,
        bijector=tfp.bijectors.Exp(),
    )
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=transformed_kernel,
        reducer=cov_reducer,
    )
    samples, _, kernel_results = tfp.mcmc.sample_chain(
        num_results=10,
        current_state=1.,
        kernel=reducer_kernel,
        trace_fn=None,
        return_final_kernel_results=True,
        seed=test_util.test_seed())
    samples, final_cov = self.evaluate([
        samples,
        cov_reducer.finalize(
            kernel_results.reduction_results)
    ])
    self.assertAllClose(
        np.mean(samples, axis=0),
        kernel_results.reduction_results.cov_state.mean,
        rtol=1e-6)
    self.assertAllClose(
        np.cov(samples.T, ddof=0), final_cov, rtol=1e-6)

  def test_nested_in_step_size_adaptation(self):
    target_dist = tfp.distributions.MultivariateNormalDiag(
        loc=[0., 0.], scale_diag=[1., 10.])
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_dist.log_prob,
        num_leapfrog_steps=27,
        step_size=10)
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=hmc_kernel,
        reducer=cov_reducer
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
    mean = kernel_results.inner_results.reduction_results.cov_state.mean
    samples, mean, final_cov = self.evaluate([
        samples,
        mean,
        cov_reducer.finalize(
            kernel_results.inner_results.reduction_results)
    ])

    self.assertEqual((2,), mean.shape)
    self.assertAllClose(np.mean(samples, axis=0), mean, rtol=1e-6)
    self.assertEqual((2, 2), final_cov.shape)
    self.assertAllClose(
        np.cov(samples.T, ddof=0), final_cov, rtol=1e-6)

  def test_nested_reducers(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.TestReducer()
    mean_reducer = test_fixtures.NaiveMeanReducer()
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=[[mean_reducer, cov_reducer], [fake_reducer]],
    )

    chain_state, kernel_results = 0., reducer_kernel.bootstrap_results(0.)
    for _ in range(6):
      chain_state, kernel_results = reducer_kernel.one_step(
          chain_state, kernel_results)

    final_cov, final_mean = self.evaluate([
        cov_reducer.finalize(
            kernel_results.reduction_results[0][1]),
        mean_reducer.finalize(
            kernel_results.reduction_results[0][0])
        ])
    self.assertEqual(2, len(kernel_results.reduction_results))
    self.assertEqual(2, len(kernel_results.reduction_results[0]))
    self.assertEqual(1, len(kernel_results.reduction_results[1]))

    self.assertEqual(3.5, final_mean)
    self.assertAllEqual(
        3.5, kernel_results.reduction_results[0][1].cov_state.mean)
    self.assertAllEqual(6, kernel_results.reduction_results[1][0])
    self.assertNear(
        np.cov(np.arange(1, 7), ddof=0).tolist(),
        final_cov,
        err=1e-6)
    self.assertAllEqual(6, kernel_results.inner_results.counter_1)
    self.assertAllEqual(12, kernel_results.inner_results.counter_2)


if __name__ == '__main__':
  test_util.main()
