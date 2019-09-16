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


from tensorflow_probability.python.internal import test_case
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class MinimizeTests(test_case.TestCase):

  def test_custom_trace_fn(self):

    init_x = np.array([0., 0.]).astype(np.float32)
    target_x = np.array([3., 4.]).astype(np.float32)

    x = tf.Variable(init_x)
    loss_fn = lambda: tf.reduce_sum(input_tensor=(x - target_x)**2)

    # The trace_fn should determine the structure and values of the results.
    def trace_fn(loss, grads, variables):
      del grads
      del variables
      return {'loss': loss, 'x': x, 'sqdiff': (x - target_x)**2}

    results = tfp.math.minimize(loss_fn, num_steps=100,
                                optimizer=tf.optimizers.Adam(0.1),
                                trace_fn=trace_fn)
    self.evaluate(tf1.global_variables_initializer())
    results_ = self.evaluate(results)
    self.assertAllClose(results_['x'][0], init_x, atol=0.5)
    self.assertAllClose(results_['x'][-1], target_x, atol=0.2)
    self.assertAllClose(results_['sqdiff'][-1], [0., 0.], atol=0.1)

  def test_respects_trainable_variables(self):
    # Variables not included in `trainable_variables` should stay fixed.
    x = tf.Variable(5.)
    y = tf.Variable(2.)
    loss_fn = lambda: tf.reduce_sum(input_tensor=(x - y)**2)

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
        trace_fn=lambda loss, grads, vars: (loss, grads),
        trainable_variables=[x])
    with tf.control_dependencies([losses]):
      final_x = tf.identity(x)

    self.evaluate(tf1.global_variables_initializer())
    final_x_, losses_, grads_ = self.evaluate((final_x, losses, grads))
    self.assertAllEqual(final_x_.shape, [2])
    self.assertAllEqual(losses_.shape, [num_steps, 2])
    self.assertAllEqual(grads_[0].shape, [num_steps, 2])

if __name__ == '__main__':
  tf.test.main()
