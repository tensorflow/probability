# Copyright 2018 The TensorFlow Probability Authors.
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
"""Tests for minimization utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class MinimizeTests(test_util.TestCase):

  def test_custom_trace_fn(self):

    init_x = np.array([0., 0.]).astype(np.float32)
    target_x = np.array([3., 4.]).astype(np.float32)

    x = tf.Variable(init_x)
    loss_fn = lambda: tf.reduce_sum((x - target_x)**2)

    # The trace_fn should determine the structure and values of the results.
    def trace_fn(traceable_quantities):
      return {'loss': traceable_quantities.loss, 'x': x,
              'sqdiff': (x - target_x)**2}

    results = tfp.math.minimize(loss_fn, num_steps=100,
                                optimizer=tf.optimizers.Adam(0.1),
                                trace_fn=trace_fn)
    self.evaluate(tf1.global_variables_initializer())
    results_ = self.evaluate(results)
    self.assertAllClose(results_['x'][0], init_x, atol=0.5)
    self.assertAllClose(results_['x'][-1], target_x, atol=0.2)
    self.assertAllClose(results_['sqdiff'][-1], [0., 0.], atol=0.1)

  def test_can_trace_all_traceable_quantities(self):
    x = tf.Variable(5.0)
    trace_fn = lambda traceable_quantities: traceable_quantities
    results = tfp.math.minimize(loss_fn=lambda: tf.reduce_sum((x - 1.0)**2),
                                num_steps=10,
                                optimizer=tf.optimizers.Adam(0.1),
                                trace_fn=trace_fn)
    self.evaluate(tf1.global_variables_initializer())
    self.evaluate(results)

  def test_respects_trainable_variables(self):
    # Variables not included in `trainable_variables` should stay fixed.
    x = tf.Variable(5.)
    y = tf.Variable(2.)
    loss_fn = lambda: tf.reduce_sum((x - y)**2)

    loss = tfp.math.minimize(loss_fn, num_steps=100,
                             optimizer=tf.optimizers.Adam(0.1),
                             trainable_variables=[x])
    with tf.control_dependencies([loss]):
      final_x = tf.identity(x)
      final_y = tf.identity(y)

    self.evaluate(tf1.global_variables_initializer())
    final_x_, final_y_ = self.evaluate((final_x, final_y))
    self.assertAllClose(final_x_, 2, atol=0.1)
    self.assertEqual(final_y_, 2.)  # `y` was untrained, so should be unchanged.

  def test_works_when_results_have_dynamic_shape(self):

    # Create a variable (and thus loss) with dynamically-shaped result.
    x = tf.Variable(initial_value=tf1.placeholder_with_default(
        [5., 3.], shape=None))

    num_steps = 10
    losses, grads = tfp.math.minimize(
        loss_fn=lambda: (x - 2.)**2,
        num_steps=num_steps,
        # TODO(b/137299119) Replace with TF2 optimizer.
        optimizer=tf1.train.AdamOptimizer(0.1),
        trace_fn=lambda t: (t.loss, t.gradients),
        trainable_variables=[x])
    with tf.control_dependencies([losses]):
      final_x = tf.identity(x)

    self.evaluate(tf1.global_variables_initializer())
    final_x_, losses_, grads_ = self.evaluate((final_x, losses, grads))
    self.assertAllEqual(final_x_.shape, [2])
    self.assertAllEqual(losses_.shape, [num_steps, 2])
    self.assertAllEqual(grads_[0].shape, [num_steps, 2])

  def test_preserves_static_num_steps(self):
    x = tf.Variable([5., 3.])
    num_steps = 23

    # Check that we preserve static shapes with static `num_steps`.
    losses = tfp.math.minimize(
        loss_fn=lambda: (x - 2.)**2,
        num_steps=num_steps,
        optimizer=tf.optimizers.Adam(0.1))
    self.assertAllEqual(losses.shape, [num_steps, 2])

  def test_works_with_dynamic_num_steps(self):
    x = tf.Variable([5., 3.])
    num_steps_ = 23
    num_steps = tf1.placeholder_with_default(num_steps_, shape=[])

    losses = tfp.math.minimize(
        loss_fn=lambda: (x - 2.)**2,
        num_steps=num_steps,
        optimizer=tf.optimizers.Adam(0.1))
    with tf.control_dependencies([losses]):
      final_x = tf.identity(x)
    self.evaluate(tf1.global_variables_initializer())
    final_x_, losses_ = self.evaluate((final_x, losses))
    self.assertAllEqual(final_x_.shape, [2])
    self.assertAllEqual(losses_.shape, [num_steps_, 2])

  def test_obeys_convergence_criterion(self):
    init_x = np.array([0., 0.]).astype(np.float32)
    target_x = np.array([3., 4.]).astype(np.float32)

    x = tf.Variable(init_x)
    loss_fn = lambda: tf.reduce_sum((x - target_x)**2)

    # Check that we can trace the convergence criterion's moving average of
    # decrease in loss.
    trace_fn = (
        lambda tq:  # pylint: disable=g-long-lambda
        (tq.loss, tq.convergence_criterion_state.average_decrease_in_loss))
    atol = 0.1
    results = tfp.math.minimize(
        loss_fn, num_steps=100,
        optimizer=tf.optimizers.SGD(0.1),
        convergence_criterion=(
            tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=atol)),
        trace_fn=trace_fn,
        return_full_length_trace=False)
    self.evaluate(tf1.global_variables_initializer())
    losses_, moving_average_decreases_ = self.evaluate(results)
    self.assertLess(moving_average_decreases_[-1], atol)
    self.assertGreater(moving_average_decreases_[-3], atol)
    self.assertAllEqual(losses_.shape, [35])

    # Check that the second-step loss decreases from the first step. This could
    # fail in graph mode if we were sloppy with `control_dependencies`, so that
    # the steps ran simultaneously or in the wrong order.
    self.assertGreater(losses_[0] - losses_[1], 1e-4)

  def test_convergence_criterion_follows_batch_reduction(self):
    init_x = np.zeros([100]).astype(np.float32)
    target_x = np.arange(100).astype(np.float32)

    x = tf.Variable(init_x)
    loss_fn = lambda: (x - target_x)**2

    # Stop the optimization when 70% of the threads have converged.
    target_portion_converged = 0.7
    batch_convergence_reduce_fn = (
        lambda has_converged: tf.reduce_mean(  # pylint: disable=g-long-lambda
            tf.cast(has_converged, tf.float32)) > target_portion_converged)
    results = tfp.math.minimize(
        loss_fn, num_steps=200,
        optimizer=tf.optimizers.Adam(1.0),
        convergence_criterion=(
            tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=0.1)),
        batch_convergence_reduce_fn=batch_convergence_reduce_fn,
        trace_fn=lambda traceable: traceable.has_converged,
        return_full_length_trace=False)
    self.evaluate(tf1.global_variables_initializer())
    has_converged_by_step = self.evaluate(results)

    self.assertLessEqual(
        np.mean(has_converged_by_step[-2]), target_portion_converged)
    self.assertGreater(
        np.mean(has_converged_by_step[-1]), target_portion_converged)

  def test_criteria_can_run_under_xla_with_static_shape(self):
    if not tf.config.experimental_functions_run_eagerly():
      self.skipTest('XLA test does not make sense without tf.function')

    init_x = np.array([0., 0.]).astype(np.float32)
    target_x = np.array([3., 4.]).astype(np.float32)

    x = tf.Variable(init_x)
    loss_fn = lambda: tf.reduce_sum((x - target_x)**2)
    optimizer = tf.optimizers.Adam(0.1)
    num_steps = 100

    # This test verifies that it works to compile the entire optimization loop,
    # as opposed to the `jit_compile` argument to `minimize`, which only
    # compiles an optimization step.
    @tf.function(jit_compile=True)
    def do_minimization(return_full_length_trace):
      return tfp.math.minimize(
          loss_fn=loss_fn,
          num_steps=num_steps,
          optimizer=optimizer,
          trace_fn=lambda ms: (ms.loss, ms.has_converged),
          convergence_criterion=(
              tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=0.1)),
          return_full_length_trace=return_full_length_trace)

    trace = do_minimization(return_full_length_trace=True)
    self.evaluate(tf1.global_variables_initializer())
    losses, has_converged = self.evaluate(trace)
    self.assertEqual(num_steps, losses.shape[0])
    self.assertEqual(num_steps, has_converged.shape[0])

    # Verify that the test is interesting, i.e., that we actually converged
    # before the end.
    self.assertTrue(has_converged[-2])

    # Verify that the final loss is tiled up to the end of the array.
    converged_at_step = np.argmax(has_converged)
    self.assertTrue(np.all(
        losses[converged_at_step + 1:] == losses[converged_at_step]))

  def test_jit_compile_applies_xla_context(self):
    if tf.config.functions_run_eagerly():
      self.skipTest('XLA test does not make sense without tf.function')

    x = tf.Variable(0.)
    optimizer = tf.optimizers.SGD(0.1)

    # Define a 'loss' that returns a constant value indicating
    # whether it is executing in an XLA context.
    using_xla, not_using_xla = 42., -9999.
    def xla_detecting_loss_fn():
      control_flow_context = tf1.get_default_graph()._control_flow_context
      if (control_flow_context is not None and
          control_flow_context.IsXLAContext()):
        return using_xla + (x - x)  # Refer to `x` to ensure loss is trainable.
      return not_using_xla + (x - x)

    xla_losses = tfp.math.minimize(
        loss_fn=xla_detecting_loss_fn,
        num_steps=1,
        optimizer=optimizer,
        jit_compile=True)
    self.evaluate(tf1.global_variables_initializer())
    self.assertAllClose(xla_losses, [using_xla])

    non_xla_losses = tfp.math.minimize(
        loss_fn=xla_detecting_loss_fn,
        num_steps=1,
        optimizer=optimizer,
        jit_compile=False)
    self.assertAllClose(non_xla_losses, [not_using_xla])

  def test_jit_compiled_optimization_makes_progress(self):
    x = tf.Variable([5., 3.])
    losses = tfp.math.minimize(
        loss_fn=lambda: tf.reduce_sum((x - 2.)**2),
        num_steps=10,
        optimizer=tf.optimizers.Adam(0.1),
        jit_compile=True)
    self.evaluate(tf1.global_variables_initializer())
    losses_ = self.evaluate(losses)
    # Final loss should be lower than initial loss.
    self.assertAllGreater(losses_[0], losses_[-1])

if __name__ == '__main__':
  tf.test.main()
