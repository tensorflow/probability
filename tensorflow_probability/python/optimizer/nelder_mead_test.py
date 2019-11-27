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
"""Tests for the unconstrained Nelder Mead optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import special_ortho_group


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class NelderMeadTest(test_util.TestCase):
  """Tests for Nelder-Mead optimization algorithm."""

  def test_quadratic_bowl_2d(self):
    """Can minimize a two dimensional quadratic function."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])
    def quadratic(x):
      return tf.reduce_sum(
          scales * tf.math.squared_difference(x, minimum), axis=-1)

    start = tf.constant([0.6, 0.8])
    results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        quadratic,
        initial_vertex=start,
        func_tolerance=1e-12))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-6)

  def test_quadratic_bowl_with_initial_simplex(self):
    """Can minimize a quadratic function with initial simplex."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])
    def quadratic(x):
      return tf.reduce_sum(
          scales * tf.math.squared_difference(x, minimum), axis=-1)

    initial_simplex = tf.constant([[0.6, 0.8], [5.0, 4.1], [-1.4, -3.2]])
    results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        quadratic,
        initial_simplex=initial_simplex,
        func_tolerance=1e-12,
        batch_evaluate_objective=True))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-6)

  def test_quadratic_bowl_with_step_sizes(self):
    """Can minimize a quadratic function with step size specification."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])
    def quadratic(x):
      return tf.reduce_sum(
          scales * tf.math.squared_difference(x, minimum), axis=-1)

    initial_vertex = tf.constant([1.29, -0.88])
    step_sizes = tf.constant([0.2, 1.3])
    results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        quadratic,
        initial_vertex=initial_vertex,
        step_sizes=step_sizes,
        func_tolerance=1e-12,
        batch_evaluate_objective=True))

    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-6)

  def test_quadratic_bowl_10d(self):
    """Can minimize a ten dimensional quadratic function."""
    dim = 10
    np.random.seed(14159)
    minimum = np.random.randn(dim)
    scales = np.exp(np.random.randn(dim))

    def quadratic(x):
      return tf.reduce_sum(
          scales * tf.math.squared_difference(x, minimum), axis=-1)

    start = tf.ones_like(minimum)
    results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        quadratic,
        initial_vertex=start,
        func_tolerance=1e-12,
        batch_evaluate_objective=True))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-6)

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
      return value

    start = tf.ones_like(minimum)
    results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        quadratic,
        initial_vertex=start,
        func_tolerance=1e-12,
        batch_evaluate_objective=False))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-6)

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
      return tf.reduce_sum(y * yp) / 2

    start = tf.ones_like(minimum)
    results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        quadratic,
        initial_vertex=start,
        func_tolerance=1e-12,
        batch_evaluate_objective=False))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_sqrt_quadratic_function(self):
    """Can minimize the square root function."""
    minimum = np.array([0.0, 0.0, 0.0, 0.0])
    def sqrt_quad(x):
      return tf.sqrt(tf.reduce_sum(x**2, axis=-1))

    start = tf.constant([1.2, 0.4, -1.8, 2.9])
    results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        sqrt_quad,
        initial_vertex=start,
        func_tolerance=1e-12,
        batch_evaluate_objective=True))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-6)

  def test_abs_function(self):
    """Can minimize the absolute value function."""
    minimum = np.array([0.0, 0.0, 0.0])
    def abs_func(x):
      return tf.reduce_sum(tf.abs(x), axis=-1)

    start = tf.constant([0.6, 1.8, -4.3], dtype=tf.float64)
    results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        abs_func,
        initial_vertex=start,
        func_tolerance=1e-12))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_rosenbrock_2d(self):
    """Tests Nelder Mead on the Rosenbrock function.

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
      """
      x, y = coord[0], coord[1]
      fv = (1 - x)**2 + 100 * (y - x**2)**2
      return fv

    start = tf.constant([-1.0, 1.0])
    results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        rosenbrock,
        initial_vertex=start,
        func_tolerance=1e-12,
        batch_evaluate_objective=False))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, [1.0, 1.0], 1e-5)

  def test_batch_consistent_with_nonbatch(self):
    """Tests that results with batch evaluate same as non-batch evaluate."""
    def easom(z):
      """The value of the two dimensional Easom function.

      The Easom function is a standard optimization test function. It has
      a single global minimum at (pi, pi) which is located inside a deep
      funnel. The expression for the function is:

      ```None
      f(x, y) = -cos(x) cos(y) exp(-(x-pi)**2 - (y-pi)**2)
      ```

      Args:
        z: `Tensor` of shape [2] and real dtype. The argument at which to
          evaluate the function.

      Returns:
        value: Scalar real `Tensor`. The value of the Easom function at the
          supplied argument.
      """
      f1 = tf.reduce_prod(tf.cos(z), axis=-1)
      f2 = tf.exp(-tf.reduce_sum((z - np.pi)**2, axis=-1))
      return -f1 * f2

    start = tf.constant([1.3, 2.2], dtype=tf.float64)
    results_non_batch = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        easom,
        initial_vertex=start,
        func_tolerance=1e-12,
        batch_evaluate_objective=False))
    results_batch = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        easom,
        initial_vertex=start,
        func_tolerance=1e-12,
        batch_evaluate_objective=True))
    self.assertTrue(results_batch.converged)
    self.assertEqual(results_batch.converged, results_non_batch.converged)
    self.assertArrayNear(results_batch.position,
                         results_non_batch.position,
                         1e-5)

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
        x: Real `Tensor` of shape [d]. The position at which to evaluate the
          function.

      Returns:
        value: A scalar `Tensor` of the function value at the supplied point.
      """
      value = tf.reduce_sum(
          x**2 - 10.0 * tf.cos(2 * np.pi * x), axis=-1) + 10.0 * dim
      return value

    start_position = np.random.rand(dim) * 2.0 * 5.12 - 5.12

    def get_results():
      start = tf.constant(start_position)
      results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
          rastrigin,
          initial_vertex=start,
          func_tolerance=1e-12))
      return results

    res1, res2 = get_results(), get_results()

    self.assertTrue(res1.converged)
    self.assertEqual(res1.converged, res2.converged)
    self.assertEqual(res1.num_objective_evaluations,
                     res2.num_objective_evaluations)
    self.assertArrayNear(res1.position, res2.position, 1e-5)
    self.assertAlmostEqual(res1.objective_value, res2.objective_value)
    self.assertArrayNear(res1.final_simplex.reshape([-1]),
                         res2.final_simplex.reshape([-1]), 1e-5)
    self.assertArrayNear(res1.final_objective_values.reshape([-1]),
                         res2.final_objective_values.reshape([-1]), 1e-5)
    self.assertEqual(res1.num_iterations, res2.num_iterations)

  def test_max_iteration_bounds_work(self):
    """Tests that max iteration bound specification works as expected."""
    def beale(coord):
      """The Beale function in two dimensions.

      Beale function is another standard test function for optimization. It is
      characterized by having a single minimum at (3, 0.5) which lies is
      a very flat region surrounded by steep walls.

      Args:
        coord: A Tensor of shape [2]. The coordinate of the point to evaluate
          the function at.

      Returns:
        fv: A scalar tensor containing the value of the Beale function at
          the supplied point.
      """
      x, y = coord[0], coord[1]
      fv = ((1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 +
            (2.625 - x + x * y**3)**2)
      return fv

    start = tf.constant([0.1, 1.0])
    # First evaluate without any iteration bounds to find the number of
    # iterations it takes to converge.
    unbounded_results = self.evaluate(tfp.optimizer.nelder_mead_minimize(
        beale,
        initial_vertex=start,
        func_tolerance=1e-12,
        batch_evaluate_objective=False))
    # Check that this converged.
    self.assertTrue(unbounded_results.converged)
    minimum = [3, 0.5]
    self.assertArrayNear(unbounded_results.position, minimum, 1e-5)

    # Next we evaluate this with exactly the number of iterations it should
    # take and assert it converges.
    bounded_converged_results = self.evaluate(
        tfp.optimizer.nelder_mead_minimize(
            beale,
            initial_vertex=start,
            func_tolerance=1e-12,
            batch_evaluate_objective=False,
            max_iterations=unbounded_results.num_iterations))
    self.assertTrue(bounded_converged_results.converged)
    self.assertEqual(bounded_converged_results.num_iterations,
                     unbounded_results.num_iterations)
    # Next, we reduce the number of allowed iterations to be one less than
    # needed and check that it doesn't converge.
    bounded_unconverged_results = self.evaluate(
        tfp.optimizer.nelder_mead_minimize(
            beale,
            initial_vertex=start,
            func_tolerance=1e-12,
            batch_evaluate_objective=False,
            max_iterations=unbounded_results.num_iterations-1))
    self.assertFalse(bounded_unconverged_results.converged)
    self.assertEqual(bounded_unconverged_results.num_iterations,
                     unbounded_results.num_iterations-1)


if __name__ == "__main__":
  tf.test.main()
