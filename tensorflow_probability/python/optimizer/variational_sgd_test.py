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
"""Functional test for VariationalSGD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


class VariationalSGDTest(tf.test.TestCase):

  def testBasic(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.1, 2.1], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        decay_rate = 0.53
        sgd_op = tfp.optimizer.VariationalSGD(
            1,
            1,
            preconditioner_decay_rate=decay_rate,
            max_learning_rate=3.0,
            burnin_max_learning_rate=3.0,
            use_single_learning_rate=True).apply_gradients(
                zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        self.assertAllCloseAccordingToType([1.1 - 3. * 0.1, 2.1 - 3. * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3. - 3. * 0.01, 4. - 3. * 0.01],
                                           self.evaluate(var1))

  def testBasicMultiInstance(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.1, 2.1], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        vara = tf.Variable([1.1, 2.1], dtype=dtype)
        varb = tf.Variable([3.0, 4.0], dtype=dtype)
        gradsa = tf.constant([0.1, 0.1], dtype=dtype)
        gradsb = tf.constant([0.01, 0.01], dtype=dtype)
        decay_rate = 0.5
        batch_size = 2
        total_num_examples = 10
        optimizer = tfp.optimizer.VariationalSGD(
            batch_size,
            total_num_examples,
            max_learning_rate=1.0,
            burnin_max_learning_rate=3.0,
            preconditioner_decay_rate=decay_rate)
        sgd_op = optimizer.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        optimizer2 = tfp.optimizer.VariationalSGD(
            batch_size,
            total_num_examples,
            max_learning_rate=1.0,
            burnin_max_learning_rate=10.0,
            burnin=0,
            preconditioner_decay_rate=decay_rate)
        sgd_op2 = optimizer2.apply_gradients(
            zip([gradsa, gradsb], [vara, varb]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(vara))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(varb))

        # Run 1 step of sgd
        sgd_op.run()
        sgd_op2.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.1 - 3. * 0.1, 2.1 - 3. * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            [1.1 - 0.1, 2.1 - 0.1], self.evaluate(vara))

        self.assertAllCloseAccordingToType([3. - 3. * 0.01, 4. - 3. * 0.01],
                                           self.evaluate(var1))
        self.assertAllCloseAccordingToType([3. - 0.01, 4. - 0.01],
                                           self.evaluate(varb))
        self.assertNotEqual(optimizer.variable_scope,
                            optimizer2.variable_scope)
        self.assertNotEqual(optimizer.variable_scope.name,
                            optimizer2.variable_scope.name)
        self.assertAllCloseAccordingToType(1, self.evaluate(optimizer._counter))
        self.assertAllCloseAccordingToType(
            1, self.evaluate(optimizer2._counter))

  def testTensorLearningRate(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.1, 2.1], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        lrate = tf.constant(3.0)
        decay_rate = 0.5
        batch_size = 2
        total_num_examples = 10
        sgd_op = tfp.optimizer.VariationalSGD(
            batch_size,
            total_num_examples,
            max_learning_rate=lrate,
            burnin=0,
            preconditioner_decay_rate=decay_rate).apply_gradients(
                zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.1 - 3. * 0.1, 2.1 - 3. * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3. - 3. * 0.01, 4. - 3. * 0.01],
                                           self.evaluate(var1))

  def testTensorDecayLearningRate(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.1, 2.1], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        lrate = tf.Variable(3.0)
        lrate_decay_op = lrate.assign_add(-3.)
        decay_rate = 0.5
        batch_size = 2
        total_num_examples = 10
        optimizer = tfp.optimizer.VariationalSGD(
            batch_size,
            total_num_examples,
            max_learning_rate=lrate,
            burnin=0,
            preconditioner_decay_rate=decay_rate)
        sgd_op = optimizer.apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([1.1 - 3. * 0.1, 2.1 - 3. * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3. - 3. * 0.01, 4. - 3. * 0.01],
                                           self.evaluate(var1))
        # Update learning rate to 0
        self.evaluate(lrate_decay_op)
        sgd_op.run()
        # Validate params haven't changed
        self.assertAllCloseAccordingToType([1.1 - 3. * 0.1, 2.1 - 3. * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3. - 3. * 0.01, 4. - 3. * 0.01],
                                           self.evaluate(var1))
        self.evaluate(lrate_decay_op)

        with self.assertRaises(tf.errors.InvalidArgumentError):
          sgd_op.run()

  def testGradWrtRef(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        opt = tfp.optimizer.VariationalSGD(1, 1, max_learning_rate=1.0)
        values = [1.0, 3.0]
        vars_ = [tf.Variable([v], dtype=dtype) for v in values]
        grads_and_vars = opt.compute_gradients(vars_[0] + vars_[1], vars_)
        tf.global_variables_initializer().run()
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.0], self.evaluate(grad))

  def testWithGlobalStep(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        global_step = tf.Variable(0, trainable=False)
        var0 = tf.Variable([1.1, 2.1], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        decay_rate = 0.1
        batch_size = 2
        total_num_examples = 10
        sgd_optimizer = tfp.optimizer.VariationalSGD(
            batch_size,
            total_num_examples,
            max_learning_rate=3.0,
            burnin=0,
            preconditioner_decay_rate=decay_rate)
        sgd_op = sgd_optimizer.apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()

        # Validate updated params and global_step
        self.assertAllCloseAccordingToType([1.1 - 3. * 0.1, 2.1 - 3. * 0.1],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType([3. - 3. * 0.01, 4. - 3. * 0.01],
                                           self.evaluate(var1))
        self.assertAllCloseAccordingToType(1, self.evaluate(global_step))
        self.assertAllCloseAccordingToType(
            1, self.evaluate(sgd_optimizer._counter))

  def testSparseBasic(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([[1.1], [2.1]], dtype=dtype)
        var1 = tf.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = tf.IndexedSlices(
            tf.constant([0.1], shape=[1, 1], dtype=dtype),
            tf.constant([0]), tf.constant([2, 1]))
        grads1 = tf.IndexedSlices(
            tf.constant([0.01], shape=[1, 1], dtype=dtype),
            tf.constant([1]), tf.constant([2, 1]))
        decay_rate = 0.1
        batch_size = 2
        total_num_examples = 10
        sgd_op = tfp.optimizer.VariationalSGD(
            batch_size,
            total_num_examples,
            max_learning_rate=3.0,
            burnin=0,
            preconditioner_decay_rate=decay_rate).apply_gradients(
                zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.1], [2.1]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0]], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([[1.1 - 3. * 0.1], [2.1]],
                                           self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            [[3. - 3. * 0], [4. - 3. * 0.01]], self.evaluate(var1))


if __name__ == '__main__':
  tf.test.main()
