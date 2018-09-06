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
"""Tests for the unconstrained BFGS optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import special_ortho_group


import tensorflow as tf
import tensorflow_probability as tfp


class BfgsTest(tf.test.TestCase):
  """Tests for BFGS optimization algorithm."""

  def test_quadratic_bowl_2d(self):
    """Can minimize a two dimensional quadratic function."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])
    def quadratic(x):
      value = tf.reduce_sum(scales * (x - minimum) ** 2)
      return value, tf.gradients(value, x)[0]
    with self.test_session() as session:
      start = tf.constant([0.6, 0.8])
      results = session.run(tfp.optimizer.bfgs_minimize(
          quadratic, initial_position=start, tolerance=1e-8))
      self.assertTrue(results.converged)
      final_gradient = results.objective_gradient
      final_gradient_norm = np.sqrt(np.sum(final_gradient * final_gradient))
      self.assertTrue(final_gradient_norm <= 1e-8)
      self.assertArrayNear(results.position, minimum, 1e-5)

  def test_inverse_hessian_spec(self):
    """Checks that specifying the 'initial_inverse_hessian_estimate' works."""
    minimum = np.array([1.0, 1.0], dtype=np.float32)
    scales = np.array([2.0, 3.0], dtype=np.float32)
    def quadratic(x):
      value = tf.reduce_sum(scales * (x - minimum) ** 2)
      return value, tf.gradients(value, x)[0]

    start = tf.constant([0.6, 0.8])
    test_inv_hessian = tf.constant([[2.0, 1.0], [1.0, 2.0]],
                                   dtype=np.float32)
    results = self.evaluate(tfp.optimizer.bfgs_minimize(
        quadratic, initial_position=start, tolerance=1e-8,
        initial_inverse_hessian_estimate=test_inv_hessian))
    self.assertTrue(results.converged)
    final_gradient = results.objective_gradient
    final_gradient_norm = np.sqrt(np.sum(final_gradient * final_gradient))
    self.assertTrue(final_gradient_norm <= 1e-8)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_quadratic_bowl_10d(self):
    """Can minimize a ten dimensional quadratic function."""
    dim = 10
    np.random.seed(14159)
    minimum = np.random.randn(dim)
    scales = np.exp(np.random.randn(dim))
    def quadratic(x):
      value = tf.reduce_sum(scales * (x - minimum) ** 2)
      return value, tf.gradients(value, x)[0]
    with self.test_session() as session:
      start = tf.ones_like(minimum)
      results = session.run(tfp.optimizer.bfgs_minimize(
          quadratic, initial_position=start, tolerance=1e-8))
      self.assertTrue(results.converged)
      final_gradient = results.objective_gradient
      final_gradient_norm = np.sqrt(np.sum(final_gradient * final_gradient))
      self.assertTrue(final_gradient_norm <= 1e-8)
      self.assertArrayNear(results.position, minimum, 1e-5)

  def test_quadratic_with_skew(self):
    """Can minimize a general quadratic function."""
    dim = 3
    np.random.seed(26535)
    minimum = np.random.randn(dim)
    principal_values = np.diag(np.exp(np.random.randn(dim)))
    rotation = special_ortho_group.rvs(dim)
    hessian = np.dot(np.transpose(rotation), np.dot(principal_values, rotation))
    def quadratic(x):
      y = x - minimum
      yp = tf.tensordot(hessian, y, axes=[1, 0])
      value = tf.reduce_sum(y * yp) / 2
      return value, tf.gradients(value, x)[0]

    with self.test_session() as session:
      start = tf.ones_like(minimum)
      results = session.run(tfp.optimizer.bfgs_minimize(
          quadratic, initial_position=start, tolerance=1e-8))
      self.assertTrue(results.converged)
      final_gradient = results.objective_gradient
      final_gradient_norm = np.sqrt(np.sum(final_gradient * final_gradient))
      print (final_gradient_norm)
      self.assertTrue(final_gradient_norm <= 1e-8)
      self.assertArrayNear(results.position, minimum, 1e-5)

  def test_quadratic_with_strong_skew(self):
    """Can minimize a strongly skewed quadratic function."""
    np.random.seed(89793)
    minimum = np.random.randn(3)
    principal_values = np.diag(np.array([0.1, 2.0, 50.0]))
    rotation = special_ortho_group.rvs(3)
    hessian = np.dot(np.transpose(rotation), np.dot(principal_values, rotation))
    def quadratic(x):
      y = x - minimum
      yp = tf.tensordot(hessian, y, axes=[1, 0])
      value = tf.reduce_sum(y * yp) / 2
      return value, tf.gradients(value, x)[0]

    with self.test_session() as session:
      start = tf.ones_like(minimum)
      results = session.run(tfp.optimizer.bfgs_minimize(
          quadratic, initial_position=start, tolerance=1e-8))
      self.assertTrue(results.converged)
      final_gradient = results.objective_gradient
      final_gradient_norm = np.sqrt(np.sum(final_gradient * final_gradient))
      print (final_gradient_norm)
      self.assertTrue(final_gradient_norm <= 1e-8)
      self.assertArrayNear(results.position, minimum, 1e-5)

  def test_rosenbrock_2d(self):
    """Tests BFGS on the Rosenbrock function.

    The Rosenbrock function is a standard optimization test case. In two
    dimensions, the function is (a, b > 0):
      f(x, y) = (a - x)^2 + b (y - x^2)^2
    The function has a global minimum at (a, a^2). This minimum lies inside
    a parabolic valley (y = x^2).
    """
    def rosenbrock(coord):
      """The Rosenbrock function in two dimensions with a=1, b=100.

      Args:
        coord: A Tensor of shape [2]. The coordinate of the point to evaluate
          the function at.

      Returns:
        fv: A scalar tensor containing the value of the Rosenbrock function at
          the supplied point.
        dfx: Scalar tensor. The derivative of the function with respect to x.
        dfy: Scalar tensor. The derivative of the function with respect to y.
      """
      x, y = coord[0], coord[1]
      fv = (1 - x)**2 + 100 * (y - x**2)**2
      dfx = 2 * (x - 1) + 400 * x * (x**2 - y)
      dfy = 200 * (y - x**2)
      return fv, tf.stack([dfx, dfy])

    with self.test_session() as session:
      start = tf.constant([-1.2, 1.0])
      results = session.run(tfp.optimizer.bfgs_minimize(
          rosenbrock, initial_position=start, tolerance=1e-5))
      self.assertTrue(results.converged)
      final_gradient = results.objective_gradient
      final_gradient_norm = np.sqrt(np.sum(final_gradient * final_gradient))
      self.assertTrue(final_gradient_norm <= 1e-5)
      self.assertArrayNear(results.position, np.array([1.0, 1.0]), 1e-5)

  def test_determinism(self):
    """Tests that the results are determinsitic."""
    dim = 5
    def rastrigin(x):
      """The value and gradient of the Rastrigin function.

      The Rastrigin function is a standard optimization test case. It is a
      multimodal non-convex function. While it has a large number of local
      minima, the global minimum is located at the origin and where the function
      value is zero. The standard search domain for optimization problems is the
      hypercube [-5.12, 5.12]**d in d-dimensions.

      Args:
        x: Real `Tensor` of shape [2]. The position at which to evaluate the
          function.

      Returns:
        value_and_gradient: A tuple of two `Tensor`s containing
          value: A scalar `Tensor` of the function value at the supplied point.
          gradient: A `Tensor` of shape [2] containing the gradient of the
            function along the two axes.
      """
      value = tf.reduce_sum(x**2 - 10.0 * tf.cos(2 * np.pi * x)) + 10.0 * dim
      gradient = tf.gradients(value, x)[0]
      return value, gradient

    start_position = np.random.rand(dim) * 2.0 * 5.12 - 5.12

    def get_results():
      with self.test_session() as session:
        start = tf.constant(start_position)
        results = session.run(tfp.optimizer.bfgs_minimize(
            rastrigin, initial_position=start, tolerance=1e-5))
        return results

    res1, res2 = get_results(), get_results()

    self.assertTrue(res1.converged)
    self.assertEqual(res1.converged, res2.converged)
    self.assertEqual(res1.failed, res2.failed)
    self.assertEqual(res1.num_objective_evaluations,
                     res2.num_objective_evaluations)
    self.assertArrayNear(res1.position, res2.position, 1e-5)
    self.assertAlmostEqual(res1.objective_value, res2.objective_value)
    self.assertArrayNear(res1.objective_gradient, res2.objective_gradient, 1e-5)
    self.assertArrayNear(res1.inverse_hessian_estimate.reshape([-1]),
                         res2.inverse_hessian_estimate.reshape([-1]), 1e-5)


if __name__ == "__main__":
  tf.test.main()
