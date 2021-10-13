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
"""Tests for SampleDiscardingKernel TransitionKernel (thinning and burn-in)."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class SampleDiscardingTest(test_util.TestCase):

  def test_thinning(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_steps_between_results=1,)
    first_state, kernel_results = discarder.one_step(
        0., discarder.bootstrap_results(0.))
    second_state, kernel_results = discarder.one_step(
        first_state, kernel_results)
    first_state, second_state, kernel_results = self.evaluate([
        first_state, second_state, kernel_results])
    self.assertEqual(2, first_state)
    self.assertEqual(4, second_state)
    self.assertEqual(2, kernel_results.call_counter)
    self.assertEqual(4, kernel_results.inner_results.counter_1)
    self.assertEqual(8, kernel_results.inner_results.counter_2)

  def test_burnin(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=5,)
    sample, kernel_results = discarder.one_step(
        0., discarder.bootstrap_results(0.))
    sample, kernel_results = self.evaluate([
        sample, kernel_results])
    self.assertEqual(6, sample)
    self.assertEqual(1, kernel_results.call_counter)
    self.assertEqual(6, kernel_results.inner_results.counter_1)
    self.assertEqual(12, kernel_results.inner_results.counter_2)

  def test_no_thinning_or_burnin(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,)
    first_state, kernel_results = discarder.one_step(
        0., discarder.bootstrap_results(0.))
    second_state, kernel_results = discarder.one_step(
        first_state, kernel_results)
    first_state, second_state, kernel_results = self.evaluate([
        first_state, second_state, kernel_results])
    self.assertEqual(1, first_state)
    self.assertEqual(2, second_state)
    self.assertEqual(2, kernel_results.call_counter)
    self.assertEqual(2, kernel_results.inner_results.counter_1)
    self.assertEqual(4, kernel_results.inner_results.counter_2)

  def test_both_thinning_and_burnin(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=10,
        num_steps_between_results=1,)
    first_state, kernel_results = discarder.one_step(
        0., discarder.bootstrap_results(0.))
    second_state, kernel_results = discarder.one_step(
        first_state, kernel_results)
    first_state, second_state, kernel_results = self.evaluate([
        first_state, second_state, kernel_results])
    # the first `one_step` performs burn-in and thinning to skip
    # `num_burnin_steps` + `num_steps_between_results` samples
    self.assertEqual(12, first_state)
    # subsequent `one_step` calls only perform thinning
    self.assertEqual(14, second_state)
    self.assertEqual(2, kernel_results.call_counter)
    self.assertEqual(14, kernel_results.inner_results.counter_1)
    self.assertEqual(28, kernel_results.inner_results.counter_2)

  def test_cold_start(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=10,
        num_steps_between_results=1,)
    first_state, _ = discarder.one_step(
        0., discarder.bootstrap_results(0.))
    second_state, kernel_results = discarder.one_step(
        first_state, discarder.bootstrap_results(first_state))
    first_state, second_state, kernel_results = self.evaluate([
        first_state, second_state, kernel_results])
    # the first `one_step` performs burn-in and thinning to skip
    # `num_burnin_steps` + `num_steps_between_results` samples
    self.assertEqual(12, first_state)
    # the step count is stored in the kernel results. Cold
    # restarting resets the count back to 0 and hence, both
    # thinning and burn-in are once again performed.
    self.assertEqual(24, second_state)
    self.assertEqual(1, kernel_results.call_counter)
    self.assertEqual(12, kernel_results.inner_results.counter_1)
    self.assertEqual(24, kernel_results.inner_results.counter_2)

  def test_is_calibrated(self):
    calibrated_kernel = test_fixtures.TestTransitionKernel()
    uncalibrated_kernel = test_fixtures.TestTransitionKernel(
        is_calibrated=False)
    calibrated_discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        calibrated_kernel)
    uncalibrated_discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        uncalibrated_kernel)
    self.assertTrue(calibrated_discarder.is_calibrated)
    self.assertFalse(uncalibrated_discarder.is_calibrated)

  def test_with_composed_kernel(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    cov_reducer = tfp.experimental.mcmc.CovarianceReducer()
    reducer_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=tfp.experimental.mcmc.SampleDiscardingKernel(
            inner_kernel=fake_inner_kernel,
            num_burnin_steps=10,
            num_steps_between_results=2,),
        reducer=cov_reducer
    )
    current_state, kernel_results = 0., reducer_kernel.bootstrap_results(0.)
    for _ in range(2):
      current_state, kernel_results = reducer_kernel.one_step(
          current_state, kernel_results)
    cov = cov_reducer.finalize(kernel_results.reduction_results)
    self.assertAllEqual(16, current_state)
    self.assertAllEqual(2, kernel_results.inner_results.call_counter)
    self.assertAllEqual(
        16, kernel_results.inner_results.inner_results.counter_1)
    self.assertAllEqual(
        32, kernel_results.inner_results.inner_results.counter_2)
    self.assertAllEqual(np.var([13, 16]), cov)

  def test_tf_while(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=10,
        num_steps_between_results=1,)

    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = discarder.one_step(
          curr_state, pkr,
      )
      return (i + 1, new_state, kernel_results)

    pkr = discarder.bootstrap_results(0.)
    _, final_sample, kernel_results = tf.while_loop(
        lambda i, *_: i < 2,
        _loop_body,
        (0., 0., pkr),
    )
    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(14, final_sample)
    self.assertEqual(2, kernel_results.call_counter)
    self.assertEqual(14, kernel_results.inner_results.counter_1)
    self.assertEqual(28, kernel_results.inner_results.counter_2)

  def test_tensor_thinning_and_burnin(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=tf.convert_to_tensor(10),
        num_steps_between_results=tf.convert_to_tensor(1),)

    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = discarder.one_step(
          curr_state, pkr,
      )
      return (i + 1, new_state, kernel_results)

    pkr = discarder.bootstrap_results(0.)
    _, final_sample, kernel_results = tf.while_loop(
        lambda i, _, __: i < 2,
        _loop_body,
        (0., 0., pkr),
    )

    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(14, final_sample)
    self.assertEqual(2, kernel_results.call_counter)
    self.assertEqual(14, kernel_results.inner_results.counter_1)
    self.assertEqual(28, kernel_results.inner_results.counter_2)

  def test_non_static_thinning_and_burnin(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    num_burnin_steps = tf.Variable(10, dtype=tf.int32)
    num_steps_between_results = tf.Variable(1, dtype=tf.int32)
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results)

    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = discarder.one_step(
          curr_state, pkr,
      )
      return (i + 1, new_state, kernel_results)

    pkr = discarder.bootstrap_results(0.)
    _, final_sample, kernel_results = tf.while_loop(
        lambda i, _, __: i < 2,
        _loop_body,
        (0., 0., pkr),
    )
    self.evaluate([
        num_burnin_steps.initializer, num_steps_between_results.initializer])
    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(14, final_sample)
    self.assertEqual(2, kernel_results.call_counter)
    self.assertEqual(14, kernel_results.inner_results.counter_1)
    self.assertEqual(28, kernel_results.inner_results.counter_2)

  def test_tensor_no_burnin(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=tf.convert_to_tensor(0),
        num_steps_between_results=tf.convert_to_tensor(1),)

    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = discarder.one_step(
          curr_state, pkr,
      )
      return (i + 1, new_state, kernel_results)

    pkr = discarder.bootstrap_results(0.)
    _, final_sample, kernel_results = tf.while_loop(
        lambda i, _, __: i < 2,
        _loop_body,
        (0., 0., pkr),
    )
    final_sample, kernel_results = self.evaluate([
        final_sample, kernel_results])
    self.assertEqual(4, final_sample)
    self.assertEqual(2, kernel_results.call_counter)
    self.assertEqual(4, kernel_results.inner_results.counter_1)
    self.assertEqual(8, kernel_results.inner_results.counter_2)

  def test_call_count_is_int32(self):
    fake_inner_kernel = test_fixtures.TestTransitionKernel()
    discarder = tfp.experimental.mcmc.SampleDiscardingKernel(
        fake_inner_kernel,
        num_burnin_steps=10,
        num_steps_between_results=1,)

    def _loop_body(i, curr_state, pkr):
      new_state, kernel_results = discarder.one_step(
          curr_state, pkr,
      )
      return (i + 1, new_state, kernel_results)

    pkr = discarder.bootstrap_results(0.)
    _, _, kernel_results = tf.while_loop(
        lambda i, *_: i < 2,
        _loop_body,
        (0., 0., pkr),
    )
    self.assertTrue(tf.int32, kernel_results.call_counter.dtype)


if __name__ == '__main__':
  test_util.main()
