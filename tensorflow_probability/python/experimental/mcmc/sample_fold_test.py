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
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.mcmc import covariance_reducer
from tensorflow_probability.python.experimental.mcmc import expectations_reducer
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.experimental.mcmc.sample_fold import sample_chain_with_burnin
from tensorflow_probability.python.experimental.mcmc.sample_fold import sample_fold
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc import hmc


JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class SampleFoldTest(test_util.TestCase):

  def test_simple_operation(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.NaiveMeanReducer()
    seed1, seed2 = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'))
    reduction_rslt, last_sample, kr = sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducer,
        seed=seed1)
    reduction_rslt, last_sample, kernel_results = self.evaluate([
        reduction_rslt, last_sample, kr])
    self.assertEqual(3, reduction_rslt)
    self.assertEqual(5, last_sample)
    self.assertEqual(5, kernel_results.counter_1)
    self.assertEqual(10, kernel_results.counter_2)

    # Warm-restart the underlying kernel but not the reduction
    reduction_rslt_2, last_sample_2, kr_2 = sample_fold(
        num_steps=5,
        current_state=last_sample,
        kernel=fake_kernel,
        reducer=fake_reducer,
        previous_kernel_results=kernel_results,
        seed=seed2)
    reduction_rslt_2, last_sample_2, kernel_results_2 = self.evaluate([
        reduction_rslt_2, last_sample_2, kr_2])
    self.assertEqual(8, reduction_rslt_2)
    self.assertEqual(10, last_sample_2)
    self.assertEqual(10, kernel_results_2.counter_1)
    self.assertEqual(20, kernel_results_2.counter_2)

  def test_reducer_warm_restart(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.NaiveMeanReducer()
    seed1, seed2 = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'))
    red_res, last_sample, kr, red_states = sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducer,
        return_final_reducer_states=True,
        seed=seed1)
    red_res, last_sample, kernel_results, red_states = self.evaluate([
        red_res, last_sample, kr, red_states])
    self.assertEqual(3, red_res)
    self.assertEqual(5, last_sample)
    self.assertEqual(5, kernel_results.counter_1)
    self.assertEqual(10, kernel_results.counter_2)

    # Warm-restart the underlying kernel and the reduction
    reduction_rslt_2, last_sample_2, kr_2 = sample_fold(
        num_steps=5,
        current_state=last_sample,
        previous_kernel_results=kernel_results,
        kernel=fake_kernel,
        reducer=fake_reducer,
        previous_reducer_state=red_states,
        seed=seed2)
    reduction_rslt_2, last_sample_2, kernel_results_2 = self.evaluate([
        reduction_rslt_2, last_sample_2, kr_2])
    self.assertEqual(5.5, reduction_rslt_2)
    self.assertEqual(10, last_sample_2)
    self.assertEqual(10, kernel_results_2.counter_1)
    self.assertEqual(20, kernel_results_2.counter_2)

  @parameterized.parameters(1., 2.)
  def test_current_state(self, curr_state):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.NaiveMeanReducer()
    reduction_rslt, last_sample, kr = sample_fold(
        num_steps=5,
        current_state=curr_state,
        kernel=fake_kernel,
        reducer=fake_reducer,
        seed=test_util.test_seed())
    reduction_rslt, last_sample, kernel_results = self.evaluate([
        reduction_rslt, last_sample, kr])
    self.assertEqual(
        np.mean(np.arange(curr_state + 1, curr_state + 6)), reduction_rslt)
    self.assertEqual(curr_state + 5, last_sample)
    self.assertEqual(5, kernel_results.counter_1)
    self.assertEqual(10, kernel_results.counter_2)

  def test_reducing_kernel_results(self):
    kernel = test_fixtures.TestTransitionKernel()
    def reduction_target(current_state, kernel_results):
      del current_state
      assert isinstance(
          kernel_results, test_fixtures.TestTransitionKernelResults)
      return kernel_results.counter_2

    reduction = expectations_reducer.ExpectationsReducer(reduction_target)
    reduction_rslt, last_sample, kr = sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=kernel,
        reducer=reduction,
        seed=test_util.test_seed())
    reduction_rslt, last_sample, kernel_results = self.evaluate([
        reduction_rslt, last_sample, kr])
    self.assertEqual(np.mean(np.arange(2, 12, 2)), reduction_rslt)
    self.assertEqual(5, last_sample)
    self.assertEqual(5, kernel_results.counter_1)
    self.assertEqual(10, kernel_results.counter_2)

  def test_nested_reducers(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducers = [[
        test_fixtures.NaiveMeanReducer(),
        covariance_reducer.CovarianceReducer()
    ], [test_fixtures.NaiveMeanReducer()]]
    reduction_rslt, last_sample, kr = sample_fold(
        num_steps=3,
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducers,
        seed=test_util.test_seed())
    reduction_rslt, last_sample, kernel_results = self.evaluate([
        reduction_rslt, last_sample, kr])
    self.assertEqual(2, len(reduction_rslt))
    self.assertEqual(2, len(reduction_rslt[0]))
    self.assertEqual(1, len(reduction_rslt[1]))

    self.assertEqual(2, reduction_rslt[0][0])
    self.assertNear(2/3, reduction_rslt[0][1], err=1e-6)
    self.assertEqual(3, last_sample)
    self.assertEqual(3, kernel_results.counter_1)
    self.assertEqual(6, kernel_results.counter_2)

  def test_true_streaming_covariance(self):
    fake_kernel = test_fixtures.TestTransitionKernel(())
    cov_reducer = covariance_reducer.CovarianceReducer()
    reduction_rslt, _, _ = sample_fold(
        num_steps=20,
        current_state=tf.convert_to_tensor([0., 0.]),
        kernel=fake_kernel,
        reducer=cov_reducer,
        seed=test_util.test_seed())
    reduction_rslt = self.evaluate(reduction_rslt)
    self.assertAllClose(
        np.cov(np.column_stack((np.arange(20), np.arange(20))).T, ddof=0),
        reduction_rslt,
        rtol=1e-5)

  def test_batched_streaming_covariance(self):
    fake_kernel = test_fixtures.TestTransitionKernel((2, 3))
    cov_reducer = covariance_reducer.CovarianceReducer(event_ndims=1)
    reduction_rslt, last_sample, _ = sample_fold(
        num_steps=5,
        current_state=tf.convert_to_tensor([[0., 0., 0.], [0., 0., 0.]]),
        kernel=fake_kernel,
        reducer=cov_reducer,
        seed=test_util.test_seed())
    reduction_rslt = self.evaluate(reduction_rslt)
    self.assertEqual((2, 3, 3), reduction_rslt.shape)
    self.assertAllEqual(np.ones(reduction_rslt.shape) * 2, reduction_rslt)
    self.assertAllEqualNested(last_sample, [[5., 5., 5.], [5., 5., 5.]])

  def test_seed_reproducibility(self):
    fake_kernel = test_fixtures.RandomTransitionKernel()
    fake_reducer = test_fixtures.NaiveMeanReducer()
    seed = test_util.test_seed(sampler_type='stateless')
    first_reduction_rslt, _, _ = sample_fold(
        num_steps=3,
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducer,
        seed=seed)
    seed = test_util.clone_seed(seed)
    second_reduction_rslt, _, _ = sample_fold(
        num_steps=3,
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducer,
        seed=seed)
    first_reduction_rslt, second_reduction_rslt = self.evaluate([
        first_reduction_rslt, second_reduction_rslt])
    self.assertEqual(first_reduction_rslt, second_reduction_rslt)

  def test_thinning_and_burnin(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.NaiveMeanReducer()
    reduction_rslt, last_sample, kr = sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducer,
        num_burnin_steps=10,
        num_steps_between_results=1,
        seed=test_util.test_seed())
    reduction_rslt, last_sample, kernel_results = self.evaluate([
        reduction_rslt,
        last_sample,
        kr])
    self.assertEqual(16, reduction_rslt)
    self.assertEqual(20, last_sample)
    self.assertEqual(
        20, kernel_results.counter_1)
    self.assertEqual(
        40, kernel_results.counter_2)

  def test_tensor_thinning_and_burnin(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    fake_reducer = test_fixtures.NaiveMeanReducer()
    reduction_rslt, last_sample, kr = sample_fold(
        num_steps=tf.convert_to_tensor(5),
        current_state=0.,
        kernel=fake_kernel,
        reducer=fake_reducer,
        num_burnin_steps=tf.convert_to_tensor(10),
        num_steps_between_results=tf.convert_to_tensor(1),
        seed=test_util.test_seed())
    reduction_rslt, last_sample, kernel_results = self.evaluate([
        reduction_rslt,
        last_sample,
        kr])
    self.assertEqual(16, reduction_rslt)
    self.assertEqual(20, last_sample)
    self.assertEqual(
        20, kernel_results.counter_1)
    self.assertEqual(
        40, kernel_results.counter_2)

  def test_none_reducer(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    reduction_rslt, last_sample, kr = sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducer=None,
        num_burnin_steps=10,
        num_steps_between_results=1,
        seed=test_util.test_seed())
    last_sample, kernel_results = self.evaluate([
        last_sample, kr])
    self.assertIsNone(reduction_rslt)
    self.assertEqual(20, last_sample)
    self.assertEqual(20, kernel_results.counter_1)
    self.assertEqual(40, kernel_results.counter_2)

  def test_empty_reducer(self):
    fake_kernel = test_fixtures.TestTransitionKernel()
    reduction_rslt, last_sample, kr = sample_fold(
        num_steps=5,
        current_state=0.,
        kernel=fake_kernel,
        reducer=[],
        num_burnin_steps=10,
        num_steps_between_results=1,
        seed=test_util.test_seed())
    last_sample, kernel_results = self.evaluate([
        last_sample, kr])
    self.assertEqual([], reduction_rslt)
    self.assertEqual(20, last_sample)
    self.assertEqual(20, kernel_results.counter_1)
    self.assertEqual(40, kernel_results.counter_2)


@test_util.test_all_tf_execution_regimes
class SampleChainTest(test_util.TestCase):

  def setUp(self):
    self._shape_param = 5.
    self._rate_param = 10.
    super(SampleChainTest, self).setUp()

  def test_basic_operation(self):
    kernel = test_fixtures.TestTransitionKernel()
    seed1, seed2 = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'))
    result = sample_chain_with_burnin(
        num_results=2, current_state=0., kernel=kernel, seed=seed1)
    samples = result.trace
    kernel_results = result.final_kernel_results
    self.assertAllClose(
        [2], tensorshape_util.as_list(samples.shape))

    samples, kernel_results = self.evaluate([samples, kernel_results])
    self.assertAllClose([1, 2], samples)
    self.assertAllClose(2, kernel_results.counter_1)
    self.assertAllClose(4, kernel_results.counter_2)

    # Warm-restart the underlying kernel.  The Trace does not support warm
    # restart.
    result_2 = sample_chain_with_burnin(
        num_results=2, seed=seed2, **result.resume_kwargs)
    samples_2, kernel_results_2 = self.evaluate(
        [result_2.trace, result_2.final_kernel_results])
    self.assertAllClose([3, 4], samples_2)
    self.assertAllClose(4, kernel_results_2.counter_1)
    self.assertAllClose(8, kernel_results_2.counter_2)

  def test_burn_in(self):
    kernel = test_fixtures.TestTransitionKernel()
    result = sample_chain_with_burnin(
        num_results=2,
        current_state=0.,
        kernel=kernel,
        num_burnin_steps=1,
        seed=test_util.test_seed())
    samples = result.trace
    kernel_results = result.final_kernel_results

    self.assertAllClose([2], tensorshape_util.as_list(samples.shape))

    samples, kernel_results = self.evaluate([samples, kernel_results])
    self.assertAllClose([2, 3], samples)
    self.assertAllClose(3, kernel_results.counter_1)
    self.assertAllClose(6, kernel_results.counter_2)

  def test_thinning(self):
    kernel = test_fixtures.TestTransitionKernel()
    result = sample_chain_with_burnin(
        num_results=2,
        current_state=0.,
        kernel=kernel,
        num_steps_between_results=2,
        seed=test_util.test_seed())
    samples = result.trace
    kernel_results = result.final_kernel_results

    self.assertAllClose([2], tensorshape_util.as_list(samples.shape))

    samples, kernel_results = self.evaluate([samples, kernel_results])
    self.assertAllClose([3, 6], samples)
    self.assertAllClose(6, kernel_results.counter_1)
    self.assertAllClose(12, kernel_results.counter_2)

  def test_custom_trace(self):
    kernel = test_fixtures.TestTransitionKernel()
    res = sample_chain_with_burnin(
        num_results=2,
        current_state=0.,
        kernel=kernel,
        trace_fn=lambda *args: args,
        seed=test_util.test_seed())
    trace = res.trace

    self.assertAllClose([2], tensorshape_util.as_list(trace[0].shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(trace[1].counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(trace[1].counter_2.shape))

    trace = self.evaluate(trace)
    self.assertAllClose([1, 2], trace[0])
    self.assertAllClose([1, 2], trace[1].counter_1)
    self.assertAllClose([2, 4], trace[1].counter_2)

  def test_is_calibrated(self):
    with warnings.catch_warnings(record=True) as triggered:
      kernel = test_fixtures.TestTransitionKernel(is_calibrated=False)
      sample_chain_with_burnin(
          num_results=2,
          current_state=0.,
          kernel=kernel,
          trace_fn=lambda current_state, kernel_results: kernel_results,
          seed=test_util.test_seed())
    self.assertTrue(
        any('supplied `TransitionKernel` is not calibrated.' in str(
            warning.message) for warning in triggered))

  @test_util.jax_disable_test_missing_functionality('TF-only bug')
  @test_util.numpy_disable_test_missing_functionality('TF-only bug')
  def test_reproduce_bug159550941(self):
    # Reproduction for b/159550941.
    input_signature = [tf.TensorSpec([], tf.int32)]

    @tf.function(input_signature=input_signature)
    def sample(chains):
      initial_state = tf.zeros([chains, 1])
      def log_prob(x):
        return tf.reduce_sum(normal.Normal(0, 1).log_prob(x), -1)

      kernel = hmc.HamiltonianMonteCarlo(
          target_log_prob_fn=log_prob, num_leapfrog_steps=3, step_size=1e-3)
      results = sample_chain_with_burnin(
          num_results=5,
          num_burnin_steps=4,
          current_state=initial_state,
          kernel=kernel)
      return results.trace

    # Checking that shape inference doesn't fail.
    sample(2)

  def test_seed_reproducibility(self):
    first_fake_kernel = test_fixtures.RandomTransitionKernel()
    second_fake_kernel = test_fixtures.RandomTransitionKernel()
    seed = test_util.test_seed(sampler_type='stateless')
    first_trace = sample_chain_with_burnin(
        num_results=5, current_state=0., kernel=first_fake_kernel,
        seed=seed).trace
    seed = test_util.clone_seed(seed)
    second_trace = sample_chain_with_burnin(
        num_results=5,
        current_state=1.,  # difference should be irrelevant
        kernel=second_fake_kernel,
        seed=seed).trace
    first_trace, second_trace = self.evaluate([
        first_trace, second_trace])
    self.assertAllCloseNested(
        first_trace, second_trace, rtol=1e-6)


@test_util.test_graph_mode_only
class SampleChainGraphTest(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def test_chain_works_correlated_multivariate(self):
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5],
                      [0.5, 1]])
    true_cov_chol = np.linalg.cholesky(true_cov)
    num_results = 3000
    counter = collections.Counter()

    @tf.function
    def target_log_prob(x, y):
      counter['target_calls'] += 1
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      z = tf.stack([x, y], axis=-1) - true_mean
      z = tf.linalg.triangular_solve(true_cov_chol, z[..., tf.newaxis])[..., 0]
      return -0.5 * tf.reduce_sum(z**2., axis=-1)

    states = sample_chain_with_burnin(
        num_results=num_results,
        current_state=[dtype(-2), dtype(2)],
        kernel=hmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            step_size=[0.5, 0.5],
            num_leapfrog_steps=2),
        num_burnin_steps=20,
        num_steps_between_results=1,
        seed=test_util.test_seed()).trace

    # TODO(https://github.com/google/jax/issues/7155)
    self.assertAllEqual(dict(target_calls=3 if JAX_MODE else 1), counter)
    states = tf.stack(states, axis=-1)
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    x = states - sample_mean
    sample_cov = tf.matmul(x, x, transpose_a=True) / dtype(num_results)
    sample_mean_, sample_cov_ = self.evaluate([sample_mean, sample_cov])
    self.assertAllClose(true_mean, sample_mean_,
                        atol=0.1, rtol=0.)
    self.assertAllClose(true_cov, sample_cov_,
                        atol=0., rtol=0.175)


if __name__ == '__main__':
  test_util.main()
