# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for loop utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import loop_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class SmartForLoopTest(test_util.TestCase):

  @parameterized.parameters(0, 1, 10)
  def test_static_num_iters(self, iters):
    counter = None
    # following loop variables not @parameterized because the tf.constants
    # would be executed outside the Eager mode that
    # @test_util.test_all_tf_execution_regimes creates
    for n in [iters, tf.constant(iters, dtype=tf.int64),
              tf.constant(iters, dtype=tf.int32)]:
      counter = collections.Counter()
      def body(x):
        counter['body_calls'] += 1
        return [x + 1]

      result = loop_util.smart_for_loop(
          loop_num_iter=n, body_fn=body, initial_loop_vars=[tf.constant(1)])
      if JAX_MODE:  # JAX always traces loop bodies exactly once
        self.assertEqual(1, counter['body_calls'])
      elif tf.executing_eagerly():
        self.assertEqual(iters, counter['body_calls'])
      else:
        expected_num_calls = 1 if iters > 0 else 0
        self.assertEqual(expected_num_calls, counter['body_calls'])
      self.assertAllClose([iters + 1], self.evaluate(result))

  def test_placeholder_num_iters(self):
    iters = 10
    n = tf1.placeholder_with_default(np.int64(iters), shape=())
    counter = collections.Counter()
    def body(x):
      counter['body_calls'] += 1
      return [x + 1]

    result = loop_util.smart_for_loop(
        loop_num_iter=n, body_fn=body, initial_loop_vars=[tf.constant(1)])
    if tf.executing_eagerly() and not JAX_MODE:  # JAX always traces loops
      self.assertEqual(iters, counter['body_calls'])
    else:
      self.assertEqual(1, counter['body_calls'])
    self.assertAllClose([11], self.evaluate(result))

  def test_unroll_threshold(self):
    iters = 50
    counter = collections.Counter()
    def body(x):
      counter['body_calls'] += 1
      return [x + 1]

    result = loop_util.smart_for_loop(
        loop_num_iter=iters,
        body_fn=body,
        initial_loop_vars=[tf.constant(1)],
        unroll_threshold=iters)
    if JAX_MODE:  # JAX always traces loop bodies exactly once
      self.assertEqual(1, counter['body_calls'])
    else:
      self.assertEqual(iters, counter['body_calls'])
    self.assertAllClose([iters + 1], self.evaluate(result))


@test_util.test_all_tf_execution_regimes
class TraceScanTest(test_util.TestCase):

  def testBasic(self):

    def _loop_fn(state, element):
      return state + element

    def _trace_fn(state):
      return [state, state * 2]

    final_state, trace = loop_util.trace_scan(
        loop_fn=_loop_fn, initial_state=0, elems=[1, 2], trace_fn=_trace_fn)

    self.assertAllClose([], tensorshape_util.as_list(final_state.shape))
    self.assertAllClose([2], tensorshape_util.as_list(trace[0].shape))
    self.assertAllClose([2], tensorshape_util.as_list(trace[1].shape))

    final_state, trace = self.evaluate([final_state, trace])

    self.assertAllClose(3, final_state)
    self.assertAllClose([1, 3], trace[0])
    self.assertAllClose([2, 6], trace[1])

  @test_util.jax_disable_test_missing_functionality('b/157611426')
  @parameterized.named_parameters(
      ('static_length', True),
      ('dynamic_length', False))
  def testTraceCriterion(self, static_length):
    final_state, trace = self.evaluate(
        loop_util.trace_scan(
            loop_fn=lambda state, element: state + element,
            initial_state=0,
            elems=[1, 2, 3, 4, 5, 6, 7],
            trace_fn=lambda state: state / 2,
            trace_criterion_fn=lambda state: tf.equal(state % 2, 0),
            static_trace_allocation_size=3 if static_length else None))
    self.assertAllClose(7 + 6 + 5 + 4 + 3 + 2 + 1, final_state)
    self.assertAllClose([3, 5, 14], trace)

  @test_util.jax_disable_test_missing_functionality('b/157611426')
  @parameterized.named_parameters(
      ('static_length', True),
      ('dynamic_length', False))
  def testConditionFn(self, static_length):
    final_state, trace = self.evaluate(
        loop_util.trace_scan(
            loop_fn=lambda state, element: state + element,
            initial_state=0,
            elems=[1, 2, 3, 4, 5, 6, 7],
            trace_fn=lambda state: state / 2,
            condition_fn=lambda step, state, num_traced, trace: state < 9,
            static_trace_allocation_size=4 if static_length else None))
    self.assertAllClose(10, final_state)
    self.assertAllClose([.5, 1.5, 3, 5], trace)

  @test_util.jax_disable_test_missing_functionality('b/171298381')
  @test_util.numpy_disable_test_missing_functionality('No expanding composites')
  def testComposite(self):
    auto_normal = auto_composite_tensor.auto_composite_tensor(
        normal.Normal, omit_kwargs=('name',))

    def _loop_fn(state, element):
      return state + element

    def _trace_fn(state):
      return [state, 2 * state, auto_normal(state, 0.1)]

    final_state, trace = loop_util.trace_scan(
        loop_fn=_loop_fn, initial_state=0., elems=[1., 2.], trace_fn=_trace_fn)

    self.assertAllClose([], tensorshape_util.as_list(final_state.shape))
    self.assertAllClose([2], tensorshape_util.as_list(trace[0].shape))
    self.assertAllClose([2], tensorshape_util.as_list(trace[1].shape))

    self.assertAllClose(3, final_state)
    self.assertAllClose([1, 3], trace[0])
    self.assertAllClose([2, 6], trace[1])

    self.assertIsInstance(trace[2], normal.Normal)
    self.assertAllClose([1., 3.], trace[2].loc)
    self.assertAllClose([0.1, 0.1], trace[2].scale)

  @test_util.numpy_disable_gradient_test
  def test_can_take_loop_gradient_inside_xla(self):
    def loss_fn(v):
      return loop_util.trace_scan(lambda x, t: x + v,
                                  0.,
                                  tf.range(10),
                                  trace_fn=lambda x: x)[0]

    xla_grad = tf.function(
        lambda v: gradient.value_and_gradient(loss_fn, v)[1],
        jit_compile=True)(0.)
    self.assertAllClose(xla_grad, 10.)

if __name__ == '__main__':
  test_util.main()
