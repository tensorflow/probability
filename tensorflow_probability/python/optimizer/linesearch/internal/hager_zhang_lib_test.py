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
"""Tests for hager_zhang_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.gradient import value_and_gradient
from tensorflow_probability.python.optimizer.linesearch.internal import hager_zhang_lib as hzl


# Define value and gradient namedtuple
ValueAndGradient = collections.namedtuple('ValueAndGradient', ['x', 'f', 'df'])


LineSearchInterval = collections.namedtuple(
    'LineSearchInterval',
    ['converged', 'failed', 'func_evals', 'iterations', 'left', 'right'])


def _interval(val_left, val_right):
  false = tf.zeros_like(val_left.x, dtype=bool)
  return LineSearchInterval(
      converged=false,
      failed=false,
      func_evals=tf.constant(0),
      iterations=tf.constant(0),
      left=val_left,
      right=val_right)


def _test_function_x_y(x, y):
  """Builds a function that passes through the given points.

  Args:
    x: A tf.Tensor of shape [n].
    y: A tf.Tensor of shape [n] or [b, n] if batching is desired.

  Returns:
    A callable that takes a tf.Tensor `t` as input and returns as output the
    value and derivative of the interpolated function at `t`.
  """
  batches = y.shape[0] if len(y.shape) == 2 else None
  deg = len(x) - 1
  poly = np.polyfit(x, y.T, deg)
  poly = [tf.convert_to_tensor(c, dtype=tf.float32) for c in poly]

  def f(t):
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    if batches is not None and not tuple(t.shape):
      # Broadcast a scalar through all batches.
      t = tf.tile(t[..., tf.newaxis], [batches])
    f, df = value_and_gradient(lambda t_: tf.math.polyval(poly, t_), t)
    return ValueAndGradient(x=t, f=tf.squeeze(f), df=tf.squeeze(df))

  return f


def _test_function_x_y_dy(x, y, dy, eps=0.01):
  """Builds a polynomial with (approx) given values and derivatives."""
  x1 = x + eps
  y1 = y + eps * dy
  x2 = x - eps
  y2 = y - eps * dy
  return _test_function_x_y(
      np.concatenate([x1, x2], axis=-1), np.concatenate([y1, y2], axis=-1))


@test_util.test_all_tf_execution_regimes
class HagerZhangLibTest(test_util.TestCase):

  def test_secant2_batching_vs_mapping(self):
    # We build a simple example function with 2 batches, one where the wolfe
    # condition is satisfied immediately, and another which does not converge.
    wolfe_threshold = 1e-6
    x = np.array([0.0, 1.0, 1.5, 2.0])
    ys = np.array([[1.1, 1.0, 0.5, 1.0],
                   [1.1, 1.0, 1.0, 1.0]])
    dys = np.array([[-0.8, -0.8, 0.1, 0.8],
                    [-0.8, -0.8, -1.0, 0.8]])

    # Create each individual and batched functions.
    fun1 = _test_function_x_y_dy(x, ys[0], dys[0])
    fun2 = _test_function_x_y_dy(x, ys[1], dys[1])
    funs = _test_function_x_y_dy(x, ys, dys)

    def eval_secant2(fun):
      val_0 = fun(0.0)
      val_a = fun(1.0)
      val_b = fun(2.0)
      f_lim = val_0.f + (wolfe_threshold * tf.abs(val_0.f))
      return self.evaluate(
          hzl.secant2(fun, val_0, _interval(val_a, val_b), f_lim))

    result1 = eval_secant2(fun1)
    result2 = eval_secant2(fun2)
    results = eval_secant2(funs)

    # Assert secant2 converges on first function, but not the second one.
    self.assertTrue(result1.converged)
    self.assertTrue(results.converged[0])
    self.assertFalse(result2.converged)
    self.assertFalse(results.converged[1])

    # Batching is strictly better than not in number of evaluations.
    self.assertLess(results.num_evals, result1.num_evals + result2.num_evals)

    # Both batching/non-batching versions get the same results.
    self.assertEqual(result1.left.x, results.left.x[0])
    self.assertEqual(result1.right.x, results.right.x[0])
    self.assertEqual(result2.left.x, results.left.x[1])
    self.assertEqual(result2.right.x, results.right.x[1])

    # Left and right are the same on the one that converged.
    self.assertEqual(result1.left.x, result1.right.x)

  def test_update_simple(self):
    """Tests that update works on a single line function."""
    # Example where trial point works as new left end point.
    wolfe_threshold = 1e-6
    x = np.array([0.0, 0.6, 1.0])
    y = np.array([1.0, 0.9, 1.2])
    dy = np.array([-0.8, -0.7, 0.6])
    fun = _test_function_x_y_dy(x, y, dy)

    val_a = fun(0.0)
    val_b = fun(1.0)
    val_trial = fun(0.6)
    f_lim = val_a.f + (wolfe_threshold * tf.abs(val_a.f))

    result = self.evaluate(hzl.update(fun, val_a, val_b, val_trial, f_lim))
    self.assertEqual(result.num_evals, 0)  # No extra evaluations needed.
    self.assertTrue(result.stopped)
    self.assertTrue(~result.failed)
    self.assertAlmostEqual(result.left.x, 0.6)
    self.assertAlmostEqual(result.right.x, 1.0)
    self.assertLess(result.left.df, 0)  # Opposite slopes.
    self.assertGreaterEqual(result.right.df, 0)

  def test_update_batching(self):
    """Tests that update function works in batching mode."""
    wolfe_threshold = 1e-6
    # We build an example function with 4 batches, each for one of the
    # following cases:
    # - a) Trial point has positive slope, so works as right end point.
    # - b) Trial point has negative slope and value is not too large,
    #      so works as left end point.
    # - c) Trial point has negative slope but the value is too high,
    #      bisect is used to squeeze the interval.
    # - d) Trial point is outside of the (a, b) interval.
    x = np.array([0.0, 0.6, 1.0])
    y = np.array([[1.0, 1.2, 1.1],
                  [1.0, 0.9, 1.2],
                  [1.0, 1.1, 1.2],
                  [1.0, 1.1, 1.2]])
    dy = np.array([[-0.8, 0.6, 0.6],
                   [-0.8, -0.7, 0.6],
                   [-0.8, -0.7, 0.6],
                   [-0.8, -0.7, 0.6]])
    fun = _test_function_x_y_dy(x, y, dy)

    val_a = fun(0.0)  # Values at zero.
    val_b = fun(1.0)  # Values at initial step.
    val_trial = fun([0.6, 0.6, 0.6, 1.5])
    f_lim = val_a.f + (wolfe_threshold * tf.abs(val_a.f))

    expected_left = np.array([0.0, 0.6, 0.0, 0.0])
    expected_right = np.array([0.6, 1.0, 0.3, 1.0])

    result = self.evaluate(hzl.update(fun, val_a, val_b, val_trial, f_lim))
    self.assertEqual(result.num_evals, 1)  # Had to do bisect once.
    self.assertTrue(np.all(result.stopped))
    self.assertTrue(np.all(~result.failed))
    self.assertTrue(np.all(result.left.df < 0))  # Opposite slopes.
    self.assertTrue(np.all(result.right.df >= 0))
    self.assertArrayNear(result.left.x, expected_left, 1e-5)
    self.assertArrayNear(result.right.x, expected_right, 1e-5)

  def test_update_batching_vs_mapping(self):
    """Tests that update function works in batching mode."""
    wolfe_threshold = 1e-6
    x = np.array([0.0, 0.6, 1.0])
    ys = np.array([[1.0, 1.2, 1.1],
                   [1.0, 0.9, 1.2]])
    dys = np.array([[-0.8, 0.6, 0.6],
                    [-0.8, -0.7, 0.6]])

    # Create each individual and batched functions.
    fun1 = _test_function_x_y_dy(x, ys[0], dys[0])
    fun2 = _test_function_x_y_dy(x, ys[1], dys[1])
    funs = _test_function_x_y_dy(x, ys, dys)

    def eval_update(fun):
      val_a = fun(0.0)
      val_b = fun(1.0)
      val_trial = fun(0.6)
      f_lim = val_a.f + (wolfe_threshold * tf.abs(val_a.f))
      return self.evaluate(hzl.update(fun, val_a, val_b, val_trial, f_lim))

    result1 = eval_update(fun1)
    result2 = eval_update(fun2)
    results = eval_update(funs)

    # Both batching/non-batching versions get the same result.
    self.assertEqual(result1.left.x, results.left.x[0])
    self.assertEqual(result1.right.x, results.right.x[0])
    self.assertEqual(result2.left.x, results.left.x[1])
    self.assertEqual(result2.right.x, results.right.x[1])

  def test_bracket_simple(self):
    """Tests that bracketing works on a 1 variable scalar valued function."""
    # Example crafted to require one expansion during bracketing, and then
    # some bisection; same as case (d) in test_bracket_batching below.
    wolfe_threshold = 1e-6
    x = np.array([0.0, 1.0, 2.5, 5.0])
    y = np.array([1.0, 0.9, -2.0, 1.1])
    dy = np.array([-0.8, -0.7, 1.6, -0.8])
    fun = _test_function_x_y_dy(x, y, dy)

    val_a = fun(0.0)  # Value at zero.
    val_b = fun(1.0)  # Value at initial step.
    f_lim = val_a.f + (wolfe_threshold * tf.abs(val_a.f))

    result = self.evaluate(
        hzl.bracket(fun, _interval(val_a, val_b), f_lim, max_iterations=5))

    self.assertEqual(result.iteration, 1)  # One expansion.
    self.assertEqual(result.num_evals, 2)  # Once bracketing, once bisecting.
    self.assertEqual(result.left.x, 0.0)
    self.assertEqual(result.right.x, 2.5)
    self.assertLess(result.left.df, 0)  # Opposite slopes.
    self.assertGreaterEqual(result.right.df, 0)

  def test_bracket_batching(self):
    """Tests that bracketing works in batching mode."""
    wolfe_threshold = 1e-6
    # We build an example function with 4 batches, each for one of the
    # following cases:
    # - a) Minimum bracketed from the beginning.
    # - b) Minimum bracketed after one expansion.
    # - c) Needs bisect from the beginning.
    # - d) Needs one round of expansion and then bisect.
    x = np.array([0.0, 1.0, 2.5, 5.0])
    y = np.array([[1.0, 1.2, 1.4, 1.1],
                  [1.0, 0.9, -2.5, 1.2],
                  [1.0, 1.1, -3.0, 1.2],
                  [1.0, 0.9, -2.0, 1.1]])
    dy = np.array([[-0.8, 0.6, -0.5, -0.8],
                   [-0.8, -0.7, -0.5, 0.6],
                   [-0.8, -0.7, -0.3, -0.8],
                   [-0.8, -0.7, 1.6, -0.8]])
    fun = _test_function_x_y_dy(x, y, dy)

    val_a = fun(0.0)
    val_b = fun(1.0)
    f_lim = val_a.f + (wolfe_threshold * tf.abs(val_a.f))

    expected_left = np.array([0.0, 1.0, 0.0, 0.0])
    expected_right = np.array([1.0, 5.0, 0.5, 2.5])

    result = self.evaluate(hzl.bracket(fun, _interval(val_a, val_b), f_lim,
                                       max_iterations=5))
    self.assertEqual(result.num_evals, 2)  # Once bracketing, once bisecting.
    self.assertTrue(np.all(result.stopped))
    self.assertTrue(np.all(~result.failed))
    self.assertTrue(np.all(result.left.df < 0))  # Opposite slopes.
    self.assertTrue(np.all(result.right.df >= 0))
    self.assertArrayNear(result.left.x, expected_left, 1e-5)
    self.assertArrayNear(result.right.x, expected_right, 1e-5)

  def test_bisect_simple(self):
    """Tests that bisect works on a 1 variable scalar valued function."""
    wolfe_threshold = 1e-6
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([1.0, 0.6, 1.2])
    dy = np.array([-0.8, 0.6, -0.7])
    fun = _test_function_x_y_dy(x, y, dy)

    val_a = fun(0.0)  # Value at zero.
    val_b = fun(1.0)  # Value at initial step.
    f_lim = val_a.f + (wolfe_threshold * tf.abs(val_a.f))

    result = self.evaluate(hzl.bisect(fun, val_a, val_b, f_lim))
    self.assertEqual(result.right.x, 0.5)

  def test_bisect_batching(self):
    """Tests that bisect works in batching mode."""
    wolfe_threshold = 1e-6
    # Let's build our example function with 4 batches, each evaluating a
    # different poly. They all have negative slopes both on 0.0 and 1.0,
    # but different slopes (positive, negative) and values (low enough, too
    # high) on their midpoint.
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([[1.0, 0.6, 1.2],
                  [1.0, 0.6, 1.2],
                  [1.0, 1.6, 1.2],
                  [1.0, 1.6, 1.2]])
    dy = np.array([[-0.8, 0.6, -0.7],
                   [-0.8, -0.4, -0.7],
                   [-0.8, 0.8, -0.7],
                   [-0.8, -0.4, -0.7]])
    fun = _test_function_x_y_dy(x, y, dy)

    val_a = fun(0.0)  # Values at zero.
    val_b = fun(1.0)  # Values at initial step.
    f_lim = val_a.f + (wolfe_threshold * tf.abs(val_a.f))

    expected_left = np.array([0.0, 0.5, 0.0, 0.0])
    expected_right = np.array([0.5, 0.75, 0.5, 0.25])

    result = self.evaluate(hzl.bisect(fun, val_a, val_b, f_lim))
    self.assertTrue(np.all(result.stopped))
    self.assertTrue(np.all(~result.failed))
    self.assertTrue(np.all(result.left.df < 0))
    self.assertTrue(np.all(result.right.df >= 0))
    self.assertArrayNear(result.left.x, expected_left, 1e-5)
    self.assertArrayNear(result.right.x, expected_right, 1e-5)


if __name__ == '__main__':
  tf.test.main()
