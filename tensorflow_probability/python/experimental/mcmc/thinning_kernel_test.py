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
"""Tests for ThinningKernel TransitionKernel."""

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import covariance_reducer
from tensorflow_probability.python.experimental.mcmc import thinning_kernel
from tensorflow_probability.python.experimental.mcmc import with_reductions
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ThinningTest(test_util.TestCase):

  def test_thinning(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    thinner = thinning_kernel.ThinningKernel(
        fake_inner_kernel,
        num_steps_to_skip=1,
    )
    seed1, seed2 = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'))
    first_state, kernel_results = thinner.one_step(
        0., thinner.bootstrap_results(0.), seed=seed1)
    second_state, kernel_results = thinner.one_step(
        first_state, kernel_results, seed=seed2)
    first_state, second_state, kernel_results = self.evaluate([
        first_state, second_state, kernel_results])
    self.assertEqual(2, first_state)
    self.assertEqual(4, second_state)
    self.assertEqual(4, kernel_results.counter_1)
    self.assertEqual(8, kernel_results.counter_2)

  def test_no_thinning(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    thinner = thinning_kernel.ThinningKernel(
        fake_inner_kernel,
        num_steps_to_skip=0,
    )
    seed1, seed2 = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'))
    first_state, kernel_results = thinner.one_step(
        0., thinner.bootstrap_results(0.), seed=seed1)
    second_state, kernel_results = thinner.one_step(
        first_state, kernel_results, seed=seed2)
    first_state, second_state, kernel_results = self.evaluate([
        first_state, second_state, kernel_results])
    self.assertEqual(1, first_state)
    self.assertEqual(2, second_state)
    self.assertEqual(2, kernel_results.counter_1)
    self.assertEqual(4, kernel_results.counter_2)

  def test_cold_start(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    thinner = thinning_kernel.ThinningKernel(
        fake_inner_kernel,
        num_steps_to_skip=1,
    )
    seed1, seed2 = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'))
    first_state, _ = thinner.one_step(
        0., thinner.bootstrap_results(0.), seed=seed1)
    second_state, kernel_results = thinner.one_step(
        first_state, thinner.bootstrap_results(first_state), seed=seed2)
    first_state, second_state, kernel_results = self.evaluate([
        first_state, second_state, kernel_results])
    self.assertEqual(2, first_state)
    self.assertEqual(4, second_state)
    self.assertEqual(2, kernel_results.counter_1)
    self.assertEqual(4, kernel_results.counter_2)

  def test_is_calibrated(self):
    calibrated_kernel = test_fixtures.TestTransitionKernel()
    uncalibrated_kernel = test_fixtures.TestTransitionKernel(
        is_calibrated=False)
    calibrated_thinner = thinning_kernel.ThinningKernel(calibrated_kernel, 0)
    uncalibrated_thinner = thinning_kernel.ThinningKernel(
        uncalibrated_kernel, 0)
    self.assertTrue(calibrated_thinner.is_calibrated)
    self.assertFalse(uncalibrated_thinner.is_calibrated)

  def test_with_composed_kernel(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    cov_reducer = covariance_reducer.CovarianceReducer()
    reducer_kernel = with_reductions.WithReductions(
        inner_kernel=thinning_kernel.ThinningKernel(
            inner_kernel=fake_inner_kernel,
            num_steps_to_skip=2,
        ),
        reducer=cov_reducer)
    current_state, kernel_results = 0., reducer_kernel.bootstrap_results(0.)
    seed = test_util.test_seed(sampler_type='stateless')
    for _ in range(2):
      mcmc_seed, seed = samplers.split_seed(seed)
      current_state, kernel_results = reducer_kernel.one_step(
          current_state, kernel_results, seed=mcmc_seed)
    cov = self.evaluate(cov_reducer.finalize(kernel_results.reduction_results))
    self.assertAllEqual(6, current_state)
    self.assertAllEqual(6, kernel_results.inner_results.counter_1)
    self.assertAllEqual(12, kernel_results.inner_results.counter_2)
    self.assertNear(np.var([3, 6]), cov, err=1e-6)

  def test_tf_while(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    thinner = thinning_kernel.ThinningKernel(
        fake_inner_kernel,
        num_steps_to_skip=1,
    )

    def _loop_body(i, seed, curr_state, pkr):
      mcmc_seed, seed = samplers.split_seed(seed)
      new_state, kernel_results = thinner.one_step(
          curr_state, pkr, seed=mcmc_seed,
      )
      return (i + 1, seed, new_state, kernel_results)

    pkr = thinner.bootstrap_results(0.)
    _, _, final_sample, kernel_results = tf.while_loop(
        lambda i, *_: i < 2,
        _loop_body,
        (0., test_util.test_seed(sampler_type='stateless'), 0., pkr),
    )
    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(4, final_sample)
    self.assertEqual(4, kernel_results.counter_1)
    self.assertEqual(8, kernel_results.counter_2)

  def test_tensor_thinning(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    thinner = thinning_kernel.ThinningKernel(
        fake_inner_kernel,
        num_steps_to_skip=tf.convert_to_tensor(1),
    )

    def _loop_body(i, seed, curr_state, pkr):
      mcmc_seed, seed = samplers.split_seed(seed)
      new_state, kernel_results = thinner.one_step(
          curr_state, pkr, seed=mcmc_seed,
      )
      return (i + 1, seed, new_state, kernel_results)

    pkr = thinner.bootstrap_results(0.)
    _, _, final_sample, kernel_results = tf.while_loop(
        lambda i, *_: i < 2,
        _loop_body,
        (0., test_util.test_seed(sampler_type='stateless'), 0., pkr),
    )

    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(4, final_sample)
    self.assertEqual(4, kernel_results.counter_1)
    self.assertEqual(8, kernel_results.counter_2)

  @test_util.jax_disable_variable_test
  def test_non_static_thinning(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    num_steps_to_skip = tf.Variable(1, dtype=tf.int32)
    thinner = thinning_kernel.ThinningKernel(
        fake_inner_kernel, num_steps_to_skip=num_steps_to_skip)

    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = thinner.one_step(
          curr_state, pkr,
      )
      return (i + 1, new_state, kernel_results)

    pkr = thinner.bootstrap_results(0.)
    _, final_sample, kernel_results = tf.while_loop(
        lambda i, *_: i < 2,
        _loop_body,
        (0., 0., pkr),
    )
    self.evaluate([num_steps_to_skip.initializer])
    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(4, final_sample)
    self.assertEqual(4, kernel_results.counter_1)
    self.assertEqual(8, kernel_results.counter_2)


if __name__ == '__main__':
  test_util.main()
