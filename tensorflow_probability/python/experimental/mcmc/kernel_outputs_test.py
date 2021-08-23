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
"""Tests for `KernelOutputs`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import kernel_builder
from tensorflow_probability.python.internal import test_util


def fake_target_log_prob(x):
  return - 0.5 * tf.reduce_sum(x**2)


@test_util.test_all_tf_execution_regimes
class TestKernelOutputs(test_util.TestCase):

  def test_basic(self):
    builder = kernel_builder.KernelBuilder.make(fake_target_log_prob)
    builder = (
        builder
        .hmc(num_leapfrog_steps=3))
    outputs1 = builder.sample(10, tf.cast(0.5, tf.float32))
    self.assertIsNotNone(outputs1.kernel)
    self.assertIsNotNone(outputs1.current_state)
    self.assertIsNotNone(outputs1.results)
    self.assertIsNotNone(outputs1.reductions)
    self.assertIsNotNone(outputs1.all_states)
    self.assertIsNotNone(outputs1.trace)

  def test_simple_adaptation(self):
    builder = kernel_builder.KernelBuilder.make(fake_target_log_prob)
    builder = (
        builder
        .hmc(num_leapfrog_steps=3)
        .simple_adaptation(adaptation_rate=.03))
    outputs1 = builder.sample(10, tf.cast(0.5, tf.float32))
    self.assertIsNotNone(outputs1.new_step_size)
    builder = (
        builder
        .set_step_size(outputs1.new_step_size)
        .clear_step_adapter())
    outputs2 = builder.sample(10, outputs1.current_state)
    diagnostics = outputs2.get_diagnostics()
    self.assertIn('realized_acceptance_rate', diagnostics)


if __name__ == '__main__':
  test_util.main()
