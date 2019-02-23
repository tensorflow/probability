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
"""Tests for Hager Zhang line search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


def _is_exact_wolfe(x, f_x, df_x, f_0, df_0, delta, sigma):
  decrease_cond = (f_x <= f_0 + delta * x * df_0)
  curvature_cond = (df_x >= sigma * df_0)
  return decrease_cond & curvature_cond


def _is_approx_wolfe(_, f_x, df_x, f_0, df_0, delta, sigma, epsilon):
  flim = f_0 + epsilon * np.abs(f_0)
  decrease_cond = (2 * delta - 1) * df_0 >= df_x
  curvature_cond = (df_x >= sigma * df_0)
  return (f_x <= flim) & decrease_cond & curvature_cond

# Define value and gradient namedtuple
ValueAndGradient = collections.namedtuple('ValueAndGradient', ['f', 'df'])


@test_util.run_all_in_graph_and_eager_modes
class HagerZhangTest(tf.test.TestCase):
  """Tests for Hager Zhang line search algorithm."""

  def test_quadratic(self):
    fdf = lambda x: ValueAndGradient(f=(x-1.3)**2, df=2*(x-1.3))

    # Case 1: The starting value is close to 0 and doesn't bracket the min.
    close_start, far_start = tf.constant(0.1), tf.constant(7.0)
    results_close = self.evaluate(tfp.optimizer.linesearch.hager_zhang(
        fdf, initial_step_size=close_start))
    self.assertTrue(results_close.converged)
    self.assertAlmostEqual(results_close.left_pt, results_close.right_pt)
    f0, df0 = fdf(0.0)
    self.assertTrue(_is_exact_wolfe(results_close.left_pt,
                                    results_close.objective_at_left_pt,
                                    results_close.grad_objective_at_left_pt,
                                    f0,
                                    df0,
                                    0.1,
                                    0.9))

    results_far = self.evaluate(tfp.optimizer.linesearch.hager_zhang(
        fdf, initial_step_size=far_start))
    self.assertTrue(results_far.converged)
    self.assertAlmostEqual(results_far.left_pt, results_far.right_pt)
    self.assertTrue(_is_exact_wolfe(results_far.left_pt,
                                    results_far.objective_at_left_pt,
                                    results_far.grad_objective_at_left_pt,
                                    f0,
                                    df0,
                                    0.1,
                                    0.9))

  def test_multiple_minima(self):
    # This function has two minima in the direction of positive x.
    # The first is around x=0.46 and the second around 2.65.
    def fdf(x):
      val = (0.988 * x**5 - 4.96 * x**4 + 4.978 * x**3
             + 5.015 * x**2 - 6.043 * x - 1)
      dval = 4.94 * x**4 - 19.84 * x**3 + 14.934 * x**2 + 10.03 * x - 6.043
      return ValueAndGradient(val, dval)

    starts = (tf.constant(0.1), tf.constant(1.5), tf.constant(2.0),
              tf.constant(4.0))
    for start in starts:
      results = self.evaluate(tfp.optimizer.linesearch.hager_zhang(
          fdf, initial_step_size=start))
      self.assertTrue(results.converged)
      self.assertAlmostEqual(results.left_pt, results.right_pt)
      f0, df0 = fdf(0.0)
      self.assertTrue(_is_exact_wolfe(results.left_pt,
                                      results.objective_at_left_pt,
                                      results.grad_objective_at_left_pt,
                                      f0,
                                      df0,
                                      0.1,
                                      0.9))

  def test_rosenbrock(self):
    """Tests one pass of line search on the Rosenbrock function.

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

    x0 = tf.constant([-3.0, -4.0])
    dirn = tf.constant([0.5, 1.0])  # This is a descent direction at x0
    def fdf(t):
      """Value and derivative of Rosenbrock projected along a descent dirn."""
      coord = x0 + t * dirn
      ft, df = rosenbrock(coord)
      return ValueAndGradient(ft, tf.reduce_sum(input_tensor=df * dirn))

    results = self.evaluate(tfp.optimizer.linesearch.hager_zhang(
        fdf, initial_step_size=1.0))
    self.assertTrue(results.converged)

  def test_eval_count(self):
    """Tests that the evaluation count is reported correctly."""
    if tf.executing_eagerly():
      self._test_eval_count_eager()
    else:
      self._test_eval_count_graph()

  def _test_eval_count_eager(self):
    starts = [0.1, 4.0]

    def get_val_and_grad_fn():
      def _val_and_grad_fn(x):
        _val_and_grad_fn.num_calls += 1
        f = x * x - 2 * x + 1
        df = 2 * (x - 1)
        return ValueAndGradient(f, df)

      _val_and_grad_fn.num_calls = 0
      return _val_and_grad_fn

    for start in starts:
      fdf = get_val_and_grad_fn()
      results = self.evaluate(tfp.optimizer.linesearch.hager_zhang(
          fdf, initial_step_size=tf.constant(start)))
      self.assertEqual(fdf.num_calls, results.func_evals)

  def _test_eval_count_graph(self):
    starts = [0.1, 4.0]
    def get_fn():
      eval_count = tf.compat.v2.Variable(0)
      def _fdf(x):
        # Enabling locking is critical here. Otherwise, there are race
        # conditions between various call sites which causes some of the
        # invocations to be missed.
        inc = tf.compat.v1.assign_add(eval_count, 1, use_locking=True)
        with tf.control_dependencies([inc]):
          f = x * x - 2 * x + 1
          df = 2 * (x - 1)
          return ValueAndGradient(f, df)
      return _fdf, eval_count

    for start in starts:
      fdf, counter = get_fn()
      results = tfp.optimizer.linesearch.hager_zhang(
          fdf, initial_step_size=tf.constant(start))
      init = tf.compat.v1.global_variables_initializer()
      with self.cached_session() as session:
        session.run(init)
        results = session.run(results)
        actual_evals = session.run(counter)
        self.assertTrue(results.converged)
        self.assertEqual(actual_evals, results.func_evals)

  def test_approx_wolfe(self):
    """Tests appropriate usage of approximate Wolfe conditions."""
    # The approximate Wolfe conditions only kick in when we are very close to
    # the minimum. The following function is based on the discussion in Hager
    # Zhang, "A new conjugate gradient method with guaranteed descent and an
    # efficient line search" (2005). The argument of the function has been
    # shifted so that the minimum is extremely close to zero.
    dtype = np.float64
    def fdf(x):
      shift = dtype(0.99999999255000149)
      fv = dtype(1) - dtype(2) * (x + shift) + (x + shift) ** 2
      dfv = - dtype(2) + dtype(2) * (x + shift)
      return ValueAndGradient(fv, dfv)

    start = tf.constant(dtype(1e-8))
    results = self.evaluate(
        tfp.optimizer.linesearch.hager_zhang(
            fdf,
            initial_step_size=start,
            sufficient_decrease_param=0.1,
            curvature_param=0.9,
            threshold_use_approximate_wolfe_condition=1e-6))
    self.assertTrue(results.converged)
    f0, df0 = fdf(0.0)
    self.assertFalse(_is_exact_wolfe(results.left_pt,
                                     results.objective_at_left_pt,
                                     results.grad_objective_at_left_pt,
                                     f0,
                                     df0,
                                     0.1,
                                     0.9))
    self.assertTrue(_is_approx_wolfe(results.left_pt,
                                     results.objective_at_left_pt,
                                     results.grad_objective_at_left_pt,
                                     f0,
                                     df0,
                                     0.1,
                                     0.9,
                                     1e-6))

  def test_determinism(self):
    """Tests that the results are determinsitic."""
    fdf = lambda x: ValueAndGradient(f=(x - 1.8)**2, df=2 * (x - 1.8))

    def get_results():
      start = tf.constant(0.9)
      results = tfp.optimizer.linesearch.hager_zhang(
          fdf,
          initial_step_size=start,
          sufficient_decrease_param=0.1,
          curvature_param=0.9,
          threshold_use_approximate_wolfe_condition=1e-6)
      return self.evaluate(results)

    res1, res2 = get_results(), get_results()

    self.assertTrue(res1.converged)
    self.assertEqual(res1.converged, res2.converged)
    self.assertEqual(res1.func_evals, res1.func_evals)
    self.assertEqual(res1.left_pt, res2.left_pt)

  def test_consistency(self):
    """Tests that the results are consistent."""
    def rastrigin(x, use_np=False):
      z = x - 0.25
      sin, cos = (np.sin, np.cos) if use_np else (tf.sin, tf.cos)
      return ValueAndGradient(f=(10.0 + z*z - 10 * cos(2*np.pi*z)),
                              df=(2 * z + 10 * 2 * np.pi * sin(2*np.pi*z)))

    start = tf.constant(0.1, dtype=tf.float64)
    results = self.evaluate(
        tfp.optimizer.linesearch.hager_zhang(
            rastrigin,
            initial_step_size=start,
            sufficient_decrease_param=0.1,
            curvature_param=0.9,
            threshold_use_approximate_wolfe_condition=1e-6))
    self.assertTrue(results.converged)
    x = results.left_pt
    actual_f, actual_df = rastrigin(x, use_np=True)
    actual = ValueAndGradient(actual_f, actual_df)
    self.assertAlmostEqual(actual.f, results.objective_at_left_pt)
    self.assertAlmostEqual(actual.df, results.grad_objective_at_left_pt)


if __name__ == '__main__':
  tf.test.main()
