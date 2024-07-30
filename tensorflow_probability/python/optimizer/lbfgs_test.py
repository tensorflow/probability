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
"""Tests for the unconstrained L-BFGS optimizer."""

from absl.testing import parameterized
import numpy as np
from scipy.stats import special_ortho_group

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.optimizer import bfgs_utils
from tensorflow_probability.python.optimizer import lbfgs


def _make_val_and_grad_fn(value_fn):
  def val_and_grad(x):
    return gradient.value_and_gradient(value_fn, x)
  return val_and_grad


def _norm(x):
  return np.linalg.norm(x, np.inf)


@test_util.test_all_tf_execution_regimes
class LBfgsTest(test_util.TestCase):
  """Tests for LBFGS optimization algorithm."""

  def test_quadratic_bowl_2d(self):
    """Can minimize a two dimensional quadratic function."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])

    @_make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(scales * tf.math.squared_difference(x, minimum))

    start = tf.constant([0.6, 0.8])
    results = self.evaluate(
        lbfgs.minimize(quadratic, initial_position=start, tolerance=1e-8))
    self.assertTrue(results.converged)
    self.assertLessEqual(_norm(results.objective_gradient), 1e-8)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_quadratic_bowl_3d_absolute_tolerance(self):
    """Can minimize a two dimensional quadratic function."""
    minimum = np.array([1.0, 1.0, 1.0])
    scales = np.array([2.0, 3.0, 4.0])

    @_make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(scales * tf.math.squared_difference(x, minimum))

    start = tf.constant([0.1, 0.8, 0.9])
    results_with_tolerance = self.evaluate(
        lbfgs.minimize(
            quadratic,
            initial_position=start,
            max_iterations=50,
            f_absolute_tolerance=1.))
    self.assertTrue(results_with_tolerance.converged)

    results_without_tolerance = self.evaluate(
        lbfgs.minimize(quadratic, initial_position=start, max_iterations=50))
    self.assertTrue(results_without_tolerance.converged)
    self.assertLess(results_with_tolerance.num_iterations,
                    results_without_tolerance.num_iterations)

  def test_high_dims_quadratic_bowl_trivial(self):
    """Can minimize a high-dimensional trivial bowl (sphere)."""
    ndims = 100
    minimum = np.ones([ndims], dtype='float64')
    scales = np.ones([ndims], dtype='float64')

    @_make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(scales * tf.math.squared_difference(x, minimum))

    start = np.zeros([ndims], dtype='float64')
    results = self.evaluate(
        lbfgs.minimize(quadratic, initial_position=start, tolerance=1e-8))
    self.assertTrue(results.converged)
    self.assertEqual(results.num_iterations, 1)  # Solved by first line search.
    self.assertLessEqual(_norm(results.objective_gradient), 1e-8)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_quadratic_bowl_40d(self):
    """Can minimize a high-dimensional quadratic function."""
    dim = 40
    np.random.seed(14159)
    minimum = np.random.randn(dim)
    scales = np.exp(np.random.randn(dim))

    @_make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(scales * tf.math.squared_difference(x, minimum))

    start = tf.ones_like(minimum)
    results = self.evaluate(
        lbfgs.minimize(quadratic, initial_position=start, tolerance=1e-8))
    self.assertTrue(results.converged)
    self.assertLessEqual(_norm(results.objective_gradient), 1e-8)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_quadratic_with_skew(self):
    """Can minimize a general quadratic function."""
    dim = 50
    np.random.seed(26535)
    minimum = np.random.randn(dim)
    principal_values = np.diag(np.exp(np.random.randn(dim)))
    rotation = special_ortho_group.rvs(dim)
    hessian = np.dot(np.transpose(rotation), np.dot(principal_values, rotation))

    @_make_val_and_grad_fn
    def quadratic(x):
      y = x - minimum
      yp = tf.tensordot(hessian, y, axes=[1, 0])
      return tf.reduce_sum(y * yp) / 2

    start = tf.ones_like(minimum)
    results = self.evaluate(
        lbfgs.minimize(quadratic, initial_position=start, tolerance=1e-8))
    self.assertTrue(results.converged)
    self.assertLessEqual(_norm(results.objective_gradient), 1e-8)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_quadratic_with_strong_skew(self):
    """Can minimize a strongly skewed quadratic function."""
    np.random.seed(89793)
    minimum = np.random.randn(3)
    principal_values = np.diag(np.array([0.1, 2.0, 50.0]))
    rotation = special_ortho_group.rvs(3)
    hessian = np.dot(np.transpose(rotation), np.dot(principal_values, rotation))

    @_make_val_and_grad_fn
    def quadratic(x):
      y = x - minimum
      yp = tf.tensordot(hessian, y, axes=[1, 0])
      return tf.reduce_sum(y * yp) / 2

    start = tf.ones_like(minimum)
    results = self.evaluate(
        lbfgs.minimize(quadratic, initial_position=start, tolerance=1e-8))
    self.assertTrue(results.converged)
    self.assertLessEqual(_norm(results.objective_gradient), 1e-8)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_rosenbrock_2d(self):
    """Tests L-BFGS on the Rosenbrock function.

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

    start = tf.constant([-1.2, 1.0])
    results = self.evaluate(
        lbfgs.minimize(rosenbrock, initial_position=start, tolerance=1e-5))
    self.assertTrue(results.converged)
    self.assertLessEqual(_norm(results.objective_gradient), 1e-5)
    self.assertArrayNear(results.position, np.array([1.0, 1.0]), 1e-5)

  def test_himmelblau(self):
    """Tests minimization on the Himmelblau's function.

    Himmelblau's function is a standard optimization test case. The function is
    given by:

      f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    The function has four minima located at (3, 2), (-2.805118, 3.131312),
    (-3.779310, -3.283186), (3.584428, -1.848126).

    All these minima may be reached from appropriate starting points. To keep
    the runtime of this test small, here we only find the first two minima.
    However, all four can be easily found in `test_himmelblau_batch_all` below
    with the help of batching.
    """
    @_make_val_and_grad_fn
    def himmelblau(coord):
      x, y = coord[0], coord[1]
      return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2

    starts_and_targets = [
        # Start Point, Target Minimum, Num evaluations expected.
        [(1, 1), (3, 2), 31],
        [(-2, 2), (-2.805118, 3.131312), 17],
    ]
    dtype = 'float64'
    for start, expected_minima, expected_evals in starts_and_targets:
      start = tf.constant(start, dtype=dtype)
      results = self.evaluate(
          lbfgs.minimize(himmelblau, initial_position=start, tolerance=1e-8))
      self.assertTrue(results.converged)
      self.assertArrayNear(results.position,
                           np.array(expected_minima, dtype=dtype),
                           1e-5)
      self.assertEqual(results.num_objective_evaluations, expected_evals)

  def test_himmelblau_batch_all(self):
    @_make_val_and_grad_fn
    def himmelblau(coord):
      x, y = coord[..., 0], coord[..., 1]
      return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2

    dtype = 'float64'
    starts = tf.constant([[[1, 1], [-2, 2]],
                          [[-1, -1], [1, -2]]], dtype=dtype)
    expected_minima = np.array([
        [[3, 2], [-2.805118, 3.131312]],
        [[-3.779310, -3.283186], [3.584428, -1.848126]]], dtype=dtype)
    batch_results = self.evaluate(
        lbfgs.minimize(
            himmelblau,
            initial_position=starts,
            stopping_condition=bfgs_utils.converged_all,
            tolerance=1e-8))

    self.assertFalse(np.any(batch_results.failed))  # None have failed.
    self.assertTrue(np.all(batch_results.converged))  # All converged.

    # All converged points are near expected minima.
    for actual, expected in zip(batch_results.position, expected_minima):
      self.assertAllClose(actual, expected)
    self.assertEqual(batch_results.num_objective_evaluations, 36)

  def test_himmelblau_batch_any(self):
    @_make_val_and_grad_fn
    def himmelblau(coord):
      x, y = coord[..., 0], coord[..., 1]
      return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2

    dtype = 'float64'
    starts = tf.constant([[1, 1],
                          [-2, 2],
                          [-1, -1],
                          [1, -2]], dtype=dtype)
    expected_minima = np.array([[3, 2],
                                [-2.805118, 3.131312],
                                [-3.779310, -3.283186],
                                [3.584428, -1.848126]], dtype=dtype)

    # Run with `converged_any` stopping condition, to stop as soon as any of
    # the batch members have converged.
    batch_results = self.evaluate(
        lbfgs.minimize(
            himmelblau,
            initial_position=starts,
            stopping_condition=bfgs_utils.converged_any,
            tolerance=1e-8))

    self.assertFalse(np.any(batch_results.failed))  # None have failed.
    self.assertTrue(np.any(batch_results.converged))  # At least one converged.
    self.assertFalse(np.all(batch_results.converged))  # But not all did.

    # Converged points are near expected minima.
    for actual, expected in zip(batch_results.position[batch_results.converged],
                                expected_minima[batch_results.converged]):
      self.assertArrayNear(actual, expected, 1e-5)
    self.assertEqual(batch_results.num_objective_evaluations, 28)

  def test_himmelblau_batch_any_resume_then_all(self):
    @_make_val_and_grad_fn
    def himmelblau(coord):
      x, y = coord[..., 0], coord[..., 1]
      return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2

    dtype = 'float64'
    starts = tf.constant([[1, 1],
                          [-2, 2],
                          [-1, -1],
                          [1, -2]], dtype=dtype)
    expected_minima = np.array([[3, 2],
                                [-2.805118, 3.131312],
                                [-3.779310, -3.283186],
                                [3.584428, -1.848126]], dtype=dtype)

    # Run with `converged_any` stopping condition, to stop as soon as any of
    # the batch members have converged.
    raw_batch_results = lbfgs.minimize(
        himmelblau,
        initial_position=starts,
        stopping_condition=bfgs_utils.converged_any,
        tolerance=1e-8)
    batch_results = self.evaluate(raw_batch_results)

    self.assertFalse(np.any(batch_results.failed))  # None have failed.
    self.assertTrue(np.any(batch_results.converged))  # At least one converged.
    self.assertFalse(np.all(batch_results.converged))  # But not all did.

    # Converged points are near expected minima.
    for actual, expected in zip(batch_results.position[batch_results.converged],
                                expected_minima[batch_results.converged]):
      self.assertArrayNear(actual, expected, 1e-5)
    self.assertEqual(batch_results.num_objective_evaluations, 28)

    # Run with `converged_all`, starting from previous state.
    batch_results = self.evaluate(
        lbfgs.minimize(
            himmelblau,
            initial_position=None,
            previous_optimizer_results=raw_batch_results,
            stopping_condition=bfgs_utils.converged_all,
            tolerance=1e-8))

    # All converged points are near expected minima and the nunmber of
    # evaluaitons is as if we never stopped.
    for actual, expected in zip(batch_results.position, expected_minima):
      self.assertArrayNear(actual, expected, 1e-5)
    self.assertEqual(batch_results.num_objective_evaluations, 36)

  def test_initial_position_and_previous_optimizer_results_are_exclusive(self):
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])

    @_make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(scales * tf.math.squared_difference(x, minimum))

    start = tf.constant([0.6, 0.8])

    def run(position, state):
      raw_results = lbfgs.minimize(
          quadratic,
          initial_position=position,
          previous_optimizer_results=state,
          tolerance=1e-8)
      self.evaluate(raw_results)
      return raw_results

    self.assertRaises(ValueError, run, None, None)
    results = run(start, None)
    self.assertRaises(ValueError, run, start, results)

  def test_data_fitting(self):
    """Tests MLE estimation for a simple geometric GLM."""
    n, dim = 100, 30
    dtype = tf.float64
    np.random.seed(234095)
    x = np.random.choice([0, 1], size=[dim, n])
    s = 0.01 * np.sum(x, 0)
    p = 1. / (1 + np.exp(-s))
    y = np.random.geometric(p)
    x_data = tf.convert_to_tensor(x, dtype=dtype)
    y_data = tf.convert_to_tensor(y, dtype=dtype)[..., tf.newaxis]

    @_make_val_and_grad_fn
    def neg_log_likelihood(state):
      state_ext = tf.expand_dims(state, 0)
      linear_part = tf.matmul(state_ext, x_data)
      linear_part_ex = tf.stack([tf.zeros_like(linear_part),
                                 linear_part], axis=0)
      term1 = tf.squeeze(
          tf.matmul(
              tf.reduce_logsumexp(linear_part_ex, axis=0), y_data),
          -1)
      term2 = (
          0.5 * tf.reduce_sum(state_ext * state_ext, axis=-1) -
          tf.reduce_sum(linear_part, axis=-1))
      return  tf.squeeze(term1 + term2)

    start = tf.ones(shape=[dim], dtype=dtype)

    results = self.evaluate(
        lbfgs.minimize(
            neg_log_likelihood, initial_position=start, tolerance=1e-6))
    self.assertTrue(results.converged)

  def test_determinism(self):
    """Tests that the results are determinsitic."""
    dim = 25

    @_make_val_and_grad_fn
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
      return tf.reduce_sum(x**2 - 10.0 * tf.cos(2 * np.pi * x)) + 10.0 * dim

    start_position = np.random.rand(dim) * 2.0 * 5.12 - 5.12

    def get_results():
      start = tf.constant(start_position)
      return self.evaluate(
          lbfgs.minimize(rastrigin, initial_position=start, tolerance=1e-5))

    res1, res2 = get_results(), get_results()

    self.assertTrue(res1.converged)
    self.assertEqual(res1.converged, res2.converged)
    self.assertEqual(res1.failed, res2.failed)
    self.assertEqual(res1.num_objective_evaluations,
                     res2.num_objective_evaluations)
    self.assertArrayNear(res1.position, res2.position, 1e-5)
    self.assertAlmostEqual(res1.objective_value, res2.objective_value)
    self.assertArrayNear(res1.objective_gradient, res2.objective_gradient, 1e-5)
    self.assertArrayNear(res1.position_deltas.reshape([-1]),
                         res2.position_deltas.reshape([-1]), 1e-5)
    self.assertArrayNear(res1.gradient_deltas.reshape([-1]),
                         res2.gradient_deltas.reshape([-1]), 1e-5)

  def test_compile(self):
    """Tests that the computation can be XLA-compiled."""

    self.skip_if_no_xla()

    dim = 25

    @_make_val_and_grad_fn
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
      return tf.reduce_sum(x**2 - 10.0 * tf.cos(2 * np.pi * x)) + 10.0 * dim

    start_position = np.random.rand(dim) * 2.0 * 5.12 - 5.12

    res = tf.function(
        lbfgs.minimize, jit_compile=True)(
            rastrigin,
            initial_position=tf.constant(start_position),
            tolerance=1e-5)

    # We simply verify execution & convergence.
    self.assertTrue(self.evaluate(res.converged))

  def test_dynamic_shapes(self):
    """Can build an lbfgs_op with dynamic shapes in graph mode."""
    if tf.executing_eagerly(): return
    ndims = 60
    minimum = np.ones([ndims], dtype='float64')
    scales = np.arange(ndims, dtype='float64') + minimum

    @_make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(scales * tf.math.squared_difference(x, minimum),
                           axis=-1)

    # Test with a vector of unknown dimension, and a fully unknown shape.
    for shape in ([None], None):
      start_value = np.arange(ndims, 0, -1, dtype='float64')
      start = tf1.placeholder_with_default(start_value, shape=shape)
      lbfgs_op = lbfgs.minimize(
          quadratic, initial_position=start, tolerance=1e-8)
      self.assertFalse(lbfgs_op.position.shape.is_fully_defined())

      results = self.evaluate(lbfgs_op)
      self.assertTrue(results.converged)
      self.assertLessEqual(_norm(results.objective_gradient), 1e-8)
      self.assertArrayNear(results.position, minimum, 1e-5)

  @parameterized.named_parameters(
      [{'testcase_name': '_from_start', 'start': np.array([0.8, 0.8])},
       {'testcase_name': '_during_opt', 'start': np.array([0.0, 0.0])},
       {'testcase_name': '_mixed', 'start': np.array([[0.8, 0.8], [0.0, 0.0]])},
      ])
  def test_stop_at_negative_infinity(self, start):
    """Stops gently when encountering a -inf objective."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])

    @_make_val_and_grad_fn
    def quadratic_with_hole(x):
      quadratic = tf.reduce_sum(
          scales * tf.math.squared_difference(x, minimum), axis=-1)
      square_hole = tf.reduce_all(tf.logical_and((x > 0.7), (x < 1.3)), axis=-1)
      minus_infty = tf.constant(float('-inf'), dtype=quadratic.dtype)
      answer = tf.where(square_hole, minus_infty, quadratic)
      return answer

    start = tf.constant(start)
    results = self.evaluate(
        lbfgs.minimize(
            quadratic_with_hole, initial_position=start, tolerance=1e-8))
    self.assertAllTrue(results.converged)
    self.assertAllFalse(results.failed)
    self.assertAllNegativeInf(results.objective_value)
    self.assertAllFinite(results.position)
    self.assertAllNegativeInf(quadratic_with_hole(results.position)[0])

  @parameterized.named_parameters(
      [{'testcase_name': '_from_start', 'start': np.array([0.8, 0.8])},
       {'testcase_name': '_during_opt', 'start': np.array([0.0, 0.0])},
       {'testcase_name': '_mixed', 'start': np.array([[0.8, 0.8], [0.0, 0.0]])},
      ])
  def test_fail_at_non_finite(self, start):
    """Fails promptly when encountering a non-finite but not -inf objective."""
    # Meaning, +inf (tested here) and nan (not tested separately due to nearly
    # identical code paths) objective values cause a "stop with failure".
    # Actually, there is a further nitpick: +inf is currently treated a little
    # inconsistently.  To wit, if the outer loop hits a +inf, it gives up and
    # reports failure, because it assumes the gradient from a +inf value is
    # garbage and no further progress is possible.  However, if the line search
    # encounters an intermediate +inf, it in some cases knows to just treat it
    # as a large finite value and avoid it.  So in principle, minimizing this
    # test function starting outside the +inf region could stop at the actual
    # minimum at the edge of said +inf region.  However, currently it happens to
    # fail.
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])

    @_make_val_and_grad_fn
    def quadratic_with_spike(x):
      quadratic = tf.reduce_sum(
          scales * tf.math.squared_difference(x, minimum), axis=-1)
      square_hole = tf.reduce_all(tf.logical_and((x > 0.7), (x < 1.3)), axis=-1)
      infty = tf.constant(float('+inf'), dtype=quadratic.dtype)
      answer = tf.where(square_hole, infty, quadratic)
      return answer

    start = tf.constant(start)
    results = self.evaluate(
        lbfgs.minimize(
            quadratic_with_spike, initial_position=start, tolerance=1e-8))
    self.assertAllFalse(results.converged)
    self.assertAllTrue(results.failed)
    self.assertAllFinite(results.position)


if __name__ == '__main__':
  test_util.main()
