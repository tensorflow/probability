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
"""Tests for the differential evolution optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import special_ortho_group


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class DifferentialEvolutionTest(test_util.TestCase):
  """Tests for Differential Evolution optimization algorithm."""

  def test_quadratic_bowl_2d(self):
    """Can minimize a two dimensional quadratic function."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])
    def quadratic(x):
      return tf.reduce_sum(
          scales * tf.math.squared_difference(x, minimum), axis=-1)

    start = tf.constant([0.6, 0.8])
    results = self.evaluate(tfp.optimizer.differential_evolution_minimize(
        quadratic,
        initial_position=start,
        func_tolerance=1e-12,
        seed=1234))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-6)

  def test_quadratic_bowl_with_initial_simplex(self):
    """Can minimize a quadratic function with initial simplex."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])
    def quadratic(x):
      return tf.reduce_sum(
          scales * tf.math.squared_difference(x, minimum), axis=-1)

    initial_population = tf.random.uniform([40, 2], seed=1243)
    results = self.evaluate(tfp.optimizer.differential_evolution_minimize(
        quadratic,
        initial_population=initial_population,
        func_tolerance=1e-12,
        seed=2484))
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
    results = self.evaluate(tfp.optimizer.differential_evolution_minimize(
        quadratic,
        initial_position=start,
        func_tolerance=1e-12,
        max_iterations=400,
        seed=9844))
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
    def quadratic_single(x):
      y = x - minimum
      yp = tf.tensordot(hessian, y, axes=[1, 0])
      value = tf.reduce_sum(y * yp) / 2
      return value

    def objective_func(population):
      return tf.map_fn(quadratic_single, population)

    start = tf.ones_like(minimum)
    results = self.evaluate(tfp.optimizer.differential_evolution_minimize(
        objective_func,
        initial_position=start,
        func_tolerance=1e-12,
        max_iterations=150,
        seed=7393))
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

    def objective_func(population):
      return tf.map_fn(quadratic, population)

    start = tf.ones_like(minimum)
    results = self.evaluate(tfp.optimizer.differential_evolution_minimize(
        objective_func,
        initial_position=start,
        func_tolerance=1e-12,
        max_iterations=150,
        seed=3321))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_sqrt_quadratic_function(self):
    """Can minimize the square root function."""
    minimum = np.array([0.0, 0.0, 0.0, 0.0])
    def sqrt_quad(x):
      return tf.sqrt(tf.reduce_sum(x**2, axis=-1))

    start = tf.constant([1.2, 0.4, -1.8, 2.9])
    results = self.evaluate(tfp.optimizer.differential_evolution_minimize(
        sqrt_quad,
        initial_position=start,
        func_tolerance=1e-12,
        max_iterations=200,
        seed=1230))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-6)

  def test_abs_function(self):
    """Can minimize the absolute value function."""
    minimum = np.array([0.0, 0.0, 0.0])
    def abs_func(x):
      return tf.reduce_sum(tf.abs(x), axis=-1)

    start = tf.constant([0.6, 1.8, -4.3], dtype=tf.float64)
    results = self.evaluate(tfp.optimizer.differential_evolution_minimize(
        abs_func,
        initial_position=start,
        func_tolerance=1e-12,
        max_iterations=200,
        seed=1212))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_rosenbrock_2d(self):
    """Tests Differential Evolution on the Rosenbrock function.

    The Rosenbrock function is a standard optimization test case. In two
    dimensions, the function is (a, b > 0):
      f(x, y) = (a - x)^2 + b (y - x^2)^2
    The function has a global minimum at (a, a^2). This minimum lies inside
    a parabolic valley (y = x^2).
    """
    def rosenbrock(x, y):
      """The Rosenbrock function in two dimensions with a=1, b=100.

      Args:
        x: A `Tensor` of shape [K]. The first coordinate of the point to
          evaluate the function at.
        y: A `Tensor` of the same shape and dtype as `x`. The second coordinate
          of the point to evaluate the function at.

      Returns:
        fv: A `Tensor` of shape [K] containing the value of the Rosenbrock
          function at the supplied points.
      """
      fv = (1 - x)**2 + 100 * (y - x**2)**2
      return fv
    start = (tf.constant(-1.0), tf.constant(1.0))
    results = self.evaluate(tfp.optimizer.differential_evolution_minimize(
        rosenbrock,
        initial_position=start,
        func_tolerance=1e-12,
        max_iterations=200,
        seed=test_util.test_seed_stream()))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, [1.0, 1.0], 1e-5)

  def test_docstring_example(self):
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
    results = self.evaluate(tfp.optimizer.differential_evolution_minimize(
        easom,
        initial_position=start,
        func_tolerance=1e-12,
        seed=47564))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, [np.pi, np.pi], 1e-5)


if __name__ == "__main__":
  tf.test.main()
