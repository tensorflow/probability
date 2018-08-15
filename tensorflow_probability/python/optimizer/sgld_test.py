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
from tensorflow_probability.python.math import diag_jacobian

tfd = tfp.distributions


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
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3., 4.], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        # Note that `tfp.math.diag_jacobian(xs=var, ys=grad)` returns zero
        # tensor
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled],
            self.evaluate(var0))
        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1. - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled],
            self.evaluate(var1))
        self.assertAllCloseAccordingToType(
            1, self.evaluate(sgd_optimizer._counter))

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
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3., 4.], self.evaluate(var1))
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(vara))
        self.assertAllCloseAccordingToType([3., 4.], self.evaluate(varb))

        # Run 1 step of sgd
        sgd_op.run()
        sgd_op2.run()
        # Validate updated params
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled],
            self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled],
            self.evaluate(vara))

        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1 - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled],
            self.evaluate(var1))
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled],
            self.evaluate(varb))
        self.assertNotEqual(sgd_optimizer.variable_scope,
                            sgd_optimizer2.variable_scope)
        self.assertNotEqual(sgd_optimizer.variable_scope.name,
                            sgd_optimizer2.variable_scope.name)
        self.assertAllCloseAccordingToType(
            1, self.evaluate(sgd_optimizer._counter))
        self.assertAllCloseAccordingToType(
            1, self.evaluate(sgd_optimizer2._counter))

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
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3., 4.], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        # Note that `tfp.math.diag_jacobian(xs=var, ys=grad)` returns zero
        # tensor
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled],
            self.evaluate(var0))
        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1. - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled],
            self.evaluate(var1))

  def testGradWrtRef(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        opt = tfp.optimizer.StochasticGradientLangevinDynamics(3.0)
        values = [1., 3.]
        vars_ = [tf.Variable([v], dtype=dtype) for v in values]
        grads_and_vars = opt.compute_gradients(vars_[0] + vars_[1], vars_)
        tf.global_variables_initializer().run()
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.], self.evaluate(grad))

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
        self.assertAllCloseAccordingToType([1.1, 2.1], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3., 4.], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()

        # Validate updated params and global_step
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        # Note that `tfp.math.diag_jacobian(xs=var, ys=grad)` returns zero
        # tensor
        self.assertAllCloseAccordingToType(
            [1.1 - 3. * grads_scaled, 2.1 - 3. * grads_scaled],
            self.evaluate(var0))
        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1. - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [3. - 3. * grads_scaled, 4. - 3. * grads_scaled],
            self.evaluate(var1))
        self.assertAllCloseAccordingToType(1, self.evaluate(global_step))

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
        self.assertAllCloseAccordingToType([[1.1], [2.1]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.], [4.]], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        grads_scaled = (0.5 * 0.1 /
                        np.sqrt(decay_rate + (1. - decay_rate) * 0.1**2 + 1e-8))
        # Note that `tfp.math.diag_jacobian(xs=var, ys=grad)` returns zero
        # tensor
        self.assertAllCloseAccordingToType([[1.1 - 3. * grads_scaled], [2.1]],
                                           self.evaluate(var0))
        grads_scaled = (0.5 * 0.01 / np.sqrt(
            decay_rate + (1. - decay_rate) * 0.01**2 + 1e-8))
        self.assertAllCloseAccordingToType(
            [[3. - 3. * 0], [4. - 3. * grads_scaled]], self.evaluate(var1))

  def testPreconditionerComputedCorrectly(self):
    """Test that SGLD step is computed correctly for a 3D Gaussian energy."""
    with self.test_session(graph=tf.Graph()) as sess:
      tf.set_random_seed(42)
      dtype = np.float32
      # Target function is the energy function of normal distribution
      true_mean = dtype([0, 0, 0])
      true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
      # Target distribution is defined through the Cholesky decomposition
      chol = tf.linalg.cholesky(true_cov)
      target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)
      var_1 = tf.get_variable(
          'var_1', initializer=[1., 1.])
      var_2 = tf.get_variable(
          'var_2', initializer=[1.])

      var = [var_1, var_2]

      # Set up the learning rate and the optimizer
      learning_rate = .5
      optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(
          learning_rate=learning_rate, burnin=1)

      # Target function
      def target_fn(x, y):
        # Stack the input tensors together
        z = tf.concat([x, y], axis=-1) - true_mean
        return -target.log_prob(z)

      grads = tf.gradients(target_fn(*var), var)

      # Update value of `var` with one iteration of the SGLD (without the
      # normal perturbation, since `burnin > 0`)
      step = optimizer_kernel.apply_gradients(zip(grads, var))

      # True theoretical value of `var` after one iteration
      decay_tensor = tf.cast(optimizer_kernel._decay_tensor, var[0].dtype)
      diagonal_bias = tf.cast(optimizer_kernel._diagonal_bias, var[0].dtype)
      learning_rate = tf.cast(optimizer_kernel._learning_rate, var[0].dtype)
      velocity = [(decay_tensor * tf.ones_like(v)
                   + (1 - decay_tensor) * tf.square(g))
                  for v, g in zip(var, grads)]
      preconditioner = [tf.rsqrt(vel + diagonal_bias) for vel in velocity]
      # Compute second order gradients
      _, grad_grads = diag_jacobian(
          xs=var,
          ys=grads)
      # Compute gradient of the preconsitioner (compute the gradient manually)
      preconditioner_grads = [-(g * g_g * (1. - decay_tensor) * p**3.)
                              for g, g_g, p in zip(grads, grad_grads,
                                                   preconditioner)]

      # True theoretical value of `var` after one iteration
      var_true = [v - learning_rate * 0.5 * (p * g - p_g)
                  for v, p, g, p_g in zip(var, preconditioner, grads,
                                          preconditioner_grads)]
      sess.run(tf.initialize_all_variables())
      var_true_ = sess.run(var_true)
      sess.run(step)
      var_ = sess.run(var)  # new `var` after one SGLD step
      self.assertAllClose(var_true_,
                          var_, atol=0.001, rtol=0.001)

  def testDiffusionBehavesCorrectly(self):
    """Test that for the SGLD finds minimum of the 3D Gaussian energy."""
    with self.test_session(graph=tf.Graph()) as sess:
      # Set up random seed for the optimizer
      tf.set_random_seed(42)
      dtype = np.float32
      true_mean = dtype([0, 0, 0])
      true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
      # Loss is defined through the Cholesky decomposition
      chol = tf.linalg.cholesky(true_cov)
      var_1 = tf.get_variable(
          'var_1', initializer=[1., 1.])
      var_2 = tf.get_variable(
          'var_2', initializer=[1.])

      var = tf.concat([var_1, var_2], axis=-1)
      # Partially defined loss function
      loss_part = tf.cholesky_solve(chol, tf.expand_dims(var, -1))
      # Loss function
      loss = 0.5 * tf.squeeze(tf.matmul(loss_part, tf.expand_dims(var, -1),
                                        transpose_a=True))

      # Set up the learning rate with a polynomial decay
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = .3
      end_learning_rate = 1e-4
      decay_steps = 1e4
      learning_rate = tf.train.polynomial_decay(starter_learning_rate,
                                                global_step, decay_steps,
                                                end_learning_rate, power=1.)

      # Set up the optimizer
      optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(
          learning_rate=learning_rate, preconditioner_decay_rate=0.99)

      optimizer = optimizer_kernel.minimize(loss)

      init = tf.global_variables_initializer()
      # Number of training steps
      training_steps = 5000
      # Record the steps as and treat them as samples
      samples = [np.zeros([training_steps, 2]), np.zeros([training_steps, 1])]
      sess.run(init)
      for step in range(training_steps):
        sess.run([optimizer, loss])
        sample = [sess.run(var_1), sess.run(var_2)]
        samples[0][step, :] = sample[0]
        samples[1][step, :] = sample[1]

    samples_ = np.concatenate(samples, axis=-1)
    sample_mean = np.mean(samples_, 0)
    self.assertAllClose(sample_mean, true_mean, atol=0.1, rtol=0.1)

if __name__ == '__main__':
  tf.test.main()
