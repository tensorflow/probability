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
"""Tests for TracingReducer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class TracingReducerTest(test_util.TestCase):

  def test_tf_while(self):
    def trace_fn(sample, pkr):
      return sample, (sample, pkr), {'one': sample, 'two': pkr}
    tracer = tfp.experimental.mcmc.TracingReducer(trace_fn=trace_fn)
    state = tracer.initialize(tf.zeros(()), tf.zeros(()))
    def _body(sample, pkr, state):
      new_state = tracer.one_step(sample, state, pkr)
      return (sample + 1, pkr + 2, new_state)
    _, _, state = tf.while_loop(
        cond=lambda i, _, __: i < 3,
        body=_body,
        loop_vars=(1., 2., state))
    final_trace = self.evaluate(tracer.finalize(state))
    self.assertEqual(3, len(final_trace))
    self.assertAllEqual([1, 2], final_trace[0])
    self.assertAllEqual(([1, 2], [2, 4]), final_trace[1])
    self.assertAllEqualNested(final_trace[2], ({'one': [1, 2], 'two': [2, 4]}))

  def test_in_sample_fold(self):
    tracer = tfp.experimental.mcmc.TracingReducer()
    fake_kernel = test_fixtures.TestTransitionKernel()
    trace, final_state, kernel_results = tfp.experimental.mcmc.sample_fold(
        num_steps=3,
        current_state=0.,
        kernel=fake_kernel,
        reducer=tracer)
    trace, final_state, kernel_results = self.evaluate([
        trace,
        final_state,
        kernel_results])
    self.assertAllEqual([1, 2, 3], trace[0])
    self.assertAllEqual([1, 2, 3], trace[1].counter_1)
    self.assertAllEqual([2, 4, 6], trace[1].counter_2)
    self.assertEqual(3, final_state)
    self.assertEqual(3, kernel_results.counter_1)
    self.assertEqual(6, kernel_results.counter_2)

  def test_known_size(self):
    tracer = tfp.experimental.mcmc.TracingReducer(size=3)
    self.assertEqual(tracer.size, 3)
    state = tracer.initialize(tf.zeros(()), tf.zeros(()))
    for sample in range(3):
      state = tracer.one_step(sample, state, sample)
    all_states, final_trace = tracer.finalize(state)
    self.assertAllEqual([3], tensorshape_util.as_list(all_states.shape))
    self.assertAllEqual([3], tensorshape_util.as_list(final_trace.shape))
    all_states, final_trace = self.evaluate([all_states, final_trace])
    self.assertAllEqual([0, 1, 2], all_states)
    self.assertAllEqual([0, 1, 2], final_trace)


if __name__ == '__main__':
  test_util.main()
