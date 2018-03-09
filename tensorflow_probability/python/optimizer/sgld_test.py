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
"""Functional test for GradientDescent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


class StochasticGradientLangevinDynamicsOptimizerTest(tf.test.TestCase):

  def testBasic(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.1, 2.1], dtype=dtype)
        var1 = tf.Variable([3., 4.], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        decay_rate = 0.53
        sgd_optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(
            3., preconditioner_decay_rate=decay_rate)
        sgd_op = sgd_optimizer.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.1, 2.1], var0.eval())
        self.assertAllCloseAccordingToType([3., 4.], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled], var0.eval())
        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1. - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled], var1.eval())
        self.assertAllCloseAccordingToType(1, sgd_optimizer._counter.eval())

  def testBasicMultiInstance(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.1, 2.1], dtype=dtype)
        var1 = tf.Variable([3., 4.], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        vara = tf.Variable([1.1, 2.1], dtype=dtype)
        varb = tf.Variable([3., 4.], dtype=dtype)
        gradsa = tf.constant([0.1, 0.1], dtype=dtype)
        gradsb = tf.constant([0.01, 0.01], dtype=dtype)
        decay_rate = 0.5
        sgd_optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(
            3., preconditioner_decay_rate=decay_rate)
        sgd_op = sgd_optimizer.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        sgd_optimizer2 = tfp.optimizer.StochasticGradientLangevinDynamics(
            3., preconditioner_decay_rate=decay_rate)
        sgd_op2 = sgd_optimizer2.apply_gradients(
            zip([gradsa, gradsb], [vara, varb]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.1, 2.1], var0.eval())
        self.assertAllCloseAccordingToType([3., 4.], var1.eval())
        self.assertAllCloseAccordingToType([1.1, 2.1], vara.eval())
        self.assertAllCloseAccordingToType([3., 4.], varb.eval())

        # Run 1 step of sgd
        sgd_op.run()
        sgd_op2.run()
        # Validate updated params
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled], var0.eval())
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled], vara.eval())

        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1 - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled], var1.eval())
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled], varb.eval())
        self.assertNotEqual(sgd_optimizer.variable_scope,
                            sgd_optimizer2.variable_scope)
        self.assertNotEqual(sgd_optimizer.variable_scope.name,
                            sgd_optimizer2.variable_scope.name)
        self.assertAllCloseAccordingToType(1, sgd_optimizer._counter.eval())
        self.assertAllCloseAccordingToType(1, sgd_optimizer2._counter.eval())

  def testTensorLearningRate(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.1, 2.1], dtype=dtype)
        var1 = tf.Variable([3., 4.], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        lrate = tf.constant(3.0)
        decay_rate = 0.5
        sgd_op = tfp.optimizer.StochasticGradientLangevinDynamics(
            lrate, preconditioner_decay_rate=tf.constant(
                decay_rate)).apply_gradients(
                    zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.1, 2.1], var0.eval())
        self.assertAllCloseAccordingToType([3., 4.], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled], var0.eval())
        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1. - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled], var1.eval())

  def testGradWrtRef(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        opt = tfp.optimizer.StochasticGradientLangevinDynamics(3.0)
        values = [1., 3.]
        vars_ = [tf.Variable([v], dtype=dtype) for v in values]
        grads_and_vars = opt.compute_gradients(vars_[0] + vars_[1], vars_)
        tf.global_variables_initializer().run()
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.], grad.eval())

  def testWithGlobalStep(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        global_step = tf.Variable(0, trainable=False)
        var0 = tf.Variable([1.1, 2.1], dtype=dtype)
        var1 = tf.Variable([3., 4.], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        decay_rate = 0.1
        sgd_op = tfp.optimizer.StochasticGradientLangevinDynamics(
            3., preconditioner_decay_rate=decay_rate).apply_gradients(
                zip([grads0, grads1], [var0, var1]),
                global_step=global_step)
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.1, 2.1], var0.eval())
        self.assertAllCloseAccordingToType([3., 4.], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()

        # Validate updated params and global_step
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled], var0.eval())
        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1. - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled], var1.eval())
        self.assertAllCloseAccordingToType(1, global_step.eval())

  def testSparseBasic(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([[1.1], [2.1]], dtype=dtype)
        var1 = tf.Variable([[3.], [4.]], dtype=dtype)
        grads0 = tf.IndexedSlices(
            tf.constant([0.1], shape=[1, 1], dtype=dtype),
            tf.constant([0]), tf.constant([2, 1]))
        grads1 = tf.IndexedSlices(
            tf.constant([0.01], shape=[1, 1], dtype=dtype),
            tf.constant([1]), tf.constant([2, 1]))
        decay_rate = 0.9
        sgd_op = tfp.optimizer.StochasticGradientLangevinDynamics(
            3., preconditioner_decay_rate=decay_rate).apply_gradients(
                zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.1], [2.1]], var0.eval())
        self.assertAllCloseAccordingToType([[3.], [4.]], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        self.assertAllCloseAccordingToType([[1.1 - 3. * grads_scaled], [2.1]],
                                           var0.eval())
        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1. - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [[3. - 3. * 0], [4. - 3. * grads_scaled]], var1.eval())


if __name__ == '__main__':
  tf.test.main()
