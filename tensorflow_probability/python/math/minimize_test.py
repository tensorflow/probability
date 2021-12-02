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

# Dependency imports
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class MinimizeTests(test_util.TestCase):

  @test_util.jax_disable_variable_test
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

  @test_util.jax_disable_variable_test
  def test_can_trace_all_traceable_quantities(self):
    x = tf.Variable(5.0)
    trace_fn = lambda traceable_quantities: traceable_quantities
    results = tfp.math.minimize(loss_fn=lambda: tf.reduce_sum((x - 1.0)**2),
                                num_steps=10,
                                optimizer=tf.optimizers.Adam(0.1),
                                trace_fn=trace_fn)
    self.evaluate(tf1.global_variables_initializer())
    self.evaluate(results)

  @test_util.jax_disable_variable_test
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

  @test_util.jax_disable_variable_test
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

  @test_util.jax_disable_variable_test
  def test_preserves_static_num_steps(self):
    x = tf.Variable([5., 3.])
    num_steps = 23

    # Check that we preserve static shapes with static `num_steps`.
    losses = tfp.math.minimize(
        loss_fn=lambda: (x - 2.)**2,
        num_steps=num_steps,
        optimizer=tf.optimizers.Adam(0.1))
    self.assertAllEqual(losses.shape, [num_steps, 2])

  @test_util.jax_disable_variable_test
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

  @test_util.jax_disable_variable_test
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

  @test_util.jax_disable_variable_test
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

  @test_util.jax_disable_variable_test
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

  @test_util.jax_disable_variable_test
  def test_jit_compile_applies_xla_context(self):
    if tf.config.functions_run_eagerly():
      self.skipTest('XLA test does not make sense without tf.function')

    x = tf.Variable(0.)
    optimizer = tf.optimizers.SGD(0.1)

    # Define a 'loss' that returns a constant value indicating
    # whether it is executing in an XLA context.
    using_xla, not_using_xla = 42., -9999.
    def xla_detecting_loss_fn():
      # Search the graph hierarchy for an XLA context.
      graph = tf1.get_default_graph()
      while True:
        if (graph._control_flow_context is not None and
            graph._control_flow_context.IsXLAContext()):
          return using_xla + (x - x)  # Refer to `x` to ensure gradient.
        try:
          graph = graph.outer_graph
        except AttributeError:
          break
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

  @test_util.jax_disable_variable_test
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

  @test_util.jax_disable_variable_test
  def test_deterministic_results_with_seed(self):
    stochastic_loss_fn = lambda seed: tf.random.stateless_normal([], seed=seed)
    optimizer = tf.optimizers.SGD(1e-3)
    seed = test_util.test_seed(sampler_type='stateless')
    losses1 = self.evaluate(
        tfp.math.minimize(loss_fn=stochastic_loss_fn,
                          num_steps=10,
                          optimizer=optimizer,
                          seed=seed))
    losses2 = self.evaluate(
        tfp.math.minimize(loss_fn=stochastic_loss_fn,
                          num_steps=10,
                          optimizer=optimizer,
                          seed=seed))
    self.assertAllEqual(losses1, losses2)
    # Make sure we got different samples at each step.
    self.assertAllGreater(tf.abs(losses1[1:] - losses1[:-1]), 1e-4)


class StatelessSGD(object):
  """Optax-like optimizer for testing pure functional optimization in TF."""

  def __init__(self, learning_rate, decay_learning_rate=False):
    self._learning_rate = learning_rate
    self._decay_learning_rate = decay_learning_rate

  def init(self, params):
    del params  # Unused.
    return 0.  # Step counter.

  def update(self, grads, step, params=None):
    del params  # Unused.
    learning_rate = self._learning_rate
    if self._decay_learning_rate:
      learning_rate /= (step + 1.)
    return (tf.nest.map_structure(lambda x: -learning_rate * x, grads),
            step + 1.)


class MinimizeStatelessTests(test_util.TestCase):

  def test_basic_minimization(self):
    init_x = np.array([0., 0.]).astype(np.float32)
    target_x = np.array([3., 4.]).astype(np.float32)
    loss_fn = lambda x: tf.reduce_sum((x - target_x)**2)

    # The trace_fn should determine the structure and values of the results.
    def trace_fn(traceable_quantities):
      return {'loss': traceable_quantities.loss,
              'x': traceable_quantities.parameters,
              'sqdiff': (traceable_quantities.parameters - target_x)**2}

    final_x, results = self.evaluate(
        tfp.math.minimize_stateless(loss_fn,
                                    init=init_x,
                                    num_steps=100,
                                    optimizer=StatelessSGD(0.05),
                                    trace_fn=trace_fn))
    self.assertAllClose(results['x'][0], init_x, atol=0.5)
    self.assertAllClose(results['x'][-1], final_x)
    self.assertAllClose(results['x'][-1], target_x, atol=0.2)
    self.assertAllClose(results['sqdiff'][-1], [0., 0.], atol=0.1)

  def test_updates_optimizer_state(self):
    final_x, losses = self.evaluate(
        tfp.math.minimize_stateless(lambda x: x**2,
                                    init=10.,
                                    num_steps=3,
                                    optimizer=StatelessSGD(
                                        learning_rate=1.0,
                                        decay_learning_rate=True)))
    # The loss has gradient `2 * x`, so with decaying learning rate the
    # optimization path should be:
    # step   x    loss   decayed_learning_rate   grad    new_x
    # 0     10.   100.   1.                       20.     -10.
    # 1    -10.   100.   1. / 2.                 -20.       0.
    # 2      0.   0.     1. / 3.                   0.       0.
    # ...    0.   0.     ...
    # (with no decay, `x` will just bounce between -10. and 10. forever).
    self.assertAllClose(final_x, 0.)
    self.assertAllClose(losses, [100., 100., 0.])

  def test_works_with_optax_optimizer(self):
    if not JAX_MODE:
      return
    import optax  # pylint: disable=g-import-not-at-top
    target_x = np.array([3., 4.]).astype(np.float32)
    final_x, losses = self.evaluate(
        tfp.math.minimize_stateless(
            lambda x: tf.reduce_sum((x - target_x)**2),
            init=np.array([0., 0.]).astype(np.float32),
            num_steps=100,
            optimizer=optax.adam(0.1)))
    self.assertAllClose(final_x, target_x, atol=0.2)
    self.assertAllClose(losses[-1], 0., atol=0.2)

  def test_batch_of_optimizations(self):
    init_x = np.array([0., 0.]).astype(np.float32)
    target_x = np.array([3., 4.]).astype(np.float32)
    loss_fn = lambda x: (x - target_x)**2
    final_x, losses = self.evaluate(
        tfp.math.minimize_stateless(loss_fn,
                                    init=init_x,
                                    num_steps=100,
                                    optimizer=StatelessSGD(0.05)))
    self.assertAllClose(final_x, target_x, atol=0.2)
    self.assertAllClose(losses[0], [9., 16.], atol=0.5)
    self.assertAllClose(losses[-1], [0., 0.], atol=0.1)

  def test_custom_trace_fn(self):
    init_x = np.array([0., 0.]).astype(np.float32)
    target_x = np.array([3., 4.]).astype(np.float32)
    loss_fn = lambda x: tf.reduce_sum((x - target_x)**2)

    # The trace_fn should determine the structure and values of the results.
    def trace_fn(traceable_quantities):
      return {'loss': traceable_quantities.loss,
              'x': traceable_quantities.parameters,
              'sqdiff': (traceable_quantities.parameters - target_x)**2}

    _, results = tfp.math.minimize_stateless(
        loss_fn,
        init=init_x,
        num_steps=100,
        optimizer=StatelessSGD(0.05),
        trace_fn=trace_fn)
    results_ = self.evaluate(results)
    self.assertAllClose(results_['x'][0], init_x, atol=0.5)
    self.assertAllClose(results_['x'][-1], target_x, atol=0.2)
    self.assertAllClose(results_['sqdiff'][-1], [0., 0.], atol=0.1)

  @test_util.jax_disable_test_missing_functionality('dynamic shape')
  def test_works_when_results_have_dynamic_shape(self):

    num_steps = 10
    final_x, (losses, grads) = self.evaluate(
        tfp.math.minimize_stateless(
            loss_fn=lambda x: (x - 2.)**2,
            # Create a parameter (and thus loss) with dynamic shape.
            init=tf1.placeholder_with_default([5., 3.], shape=None),
            num_steps=num_steps,
            optimizer=StatelessSGD(0.01),
            trace_fn=lambda t: (t.loss, t.gradients)))
    self.assertAllEqual(final_x.shape, [2])
    self.assertAllEqual(losses.shape, [num_steps, 2])
    self.assertAllEqual(grads.shape, [num_steps, 2])

  @test_util.jax_disable_test_missing_functionality('dynamic shape')
  def test_preserves_static_num_steps(self):
    num_steps = 23
    # Check that we preserve static shapes with static `num_steps`.
    _, losses = tfp.math.minimize_stateless(
        loss_fn=lambda x: (x - 2.)**2,
        init=tf.constant([5., 3.]),
        num_steps=num_steps,
        optimizer=StatelessSGD(0.1))
    self.assertAllEqual(losses.shape, [num_steps, 2])

  @test_util.jax_disable_test_missing_functionality('dynamic shape')
  def test_works_with_dynamic_num_steps(self):
    num_steps_ = 23
    num_steps = tf1.placeholder_with_default(num_steps_, shape=[])

    final_x, losses = self.evaluate(
        tfp.math.minimize_stateless(
            loss_fn=lambda x: (x - 2.)**2,
            init=tf.constant([5., 3.]),
            num_steps=num_steps,
            optimizer=StatelessSGD(0.1)))
    self.assertAllEqual(final_x.shape, [2])
    self.assertAllEqual(losses.shape, [num_steps_, 2])

  def test_obeys_convergence_criterion(self):
    init_x = np.array([0., 0.]).astype(np.float32)
    target_x = np.array([3., 4.]).astype(np.float32)

    loss_fn = lambda x: tf.reduce_sum((x - target_x)**2)

    # Check that we can trace the convergence criterion's moving average of
    # decrease in loss.
    trace_fn = (
        lambda tq:  # pylint: disable=g-long-lambda
        (tq.loss, tq.convergence_criterion_state.average_decrease_in_loss))
    atol = 0.1
    _, (losses, moving_average_decreases) = self.evaluate(
        tfp.math.minimize_stateless(
            loss_fn,
            init=init_x,
            num_steps=100,
            optimizer=StatelessSGD(0.1),
            convergence_criterion=(
                tfp.optimizer.convergence_criteria.LossNotDecreasing(
                    atol=atol)),
            trace_fn=trace_fn,
            return_full_length_trace=False))
    self.assertLess(moving_average_decreases[-1], atol)
    self.assertGreater(moving_average_decreases[-3], atol)
    self.assertAllEqual(losses.shape, [35])

    # Check that the second-step loss decreases from the first step. This could
    # fail in graph mode if we were sloppy with `control_dependencies`, so that
    # the steps ran simultaneously or in the wrong order.
    self.assertGreater(losses[0] - losses[1], 1e-4)

  def test_jit_compiled_optimization_makes_progress(self):
    _, losses = self.evaluate(tfp.math.minimize_stateless(
        loss_fn=lambda x: tf.reduce_sum((x - 2.)**2),
        init=tf.constant([5., 3.]),
        num_steps=10,
        optimizer=StatelessSGD(0.1),
        jit_compile=True))
    # Final loss should be lower than initial loss.
    self.assertAllGreater(losses[0], losses[-1])

  def test_deterministic_results_with_seed(self):
    stochastic_loss_fn = (
        lambda x, seed: (x - x) + tf.random.stateless_normal([], seed=seed))
    optimizer = StatelessSGD(1e-3)
    seed = test_util.test_seed(sampler_type='stateless')
    _, losses1 = self.evaluate(
        tfp.math.minimize_stateless(
            loss_fn=stochastic_loss_fn,
            init=0.,
            num_steps=10,
            optimizer=optimizer,
            seed=seed))
    _, losses2 = self.evaluate(
        tfp.math.minimize_stateless(
            loss_fn=stochastic_loss_fn,
            init=0.,
            num_steps=10,
            optimizer=optimizer,
            seed=seed))
    self.assertAllEqual(losses1, losses2)
    # Make sure we got different samples at each step.
    self.assertAllGreater(tf.abs(losses1[1:] - losses1[:-1]), 1e-4)

if __name__ == '__main__':
  test_util.main()
