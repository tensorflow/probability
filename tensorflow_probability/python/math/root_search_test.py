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
"""Tests for root finding functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import scipy.optimize as optimize

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class RootSearchTest(test_util.TestCase):

  def test_secant_finds_all_roots_from_one_initial_position(self):
    f = lambda x: (63 * x**5 - 70 * x**3 + 15 * x) / 8.

    x0, x1 = -1, 10
    guess = tf.constant([x0, x1], dtype=tf.float64)

    tolerance = 1e-8
    roots, value_at_roots, _ = self.evaluate(
        tfp.math.secant_root(f, guess, position_tolerance=tolerance))

    expected_roots = [optimize.newton(f, x0), optimize.newton(f, x1)]
    zeros = [0., 0.]

    self.assertAllClose(roots, expected_roots, atol=tolerance)
    self.assertAllClose(value_at_roots, zeros)

  def test_secant_finds_any_root_from_one_initial_position(self):
    f = lambda x: (63 * x**5 - 70 * x**3 + 15 * x) / 8.

    x0, x1 = -1, 10
    guess = tf.constant([x0, x1], dtype=tf.float64)

    tolerance = 1e-8
    # Only the root close to the first starting point will be found.
    roots, value_at_roots, _ = self.evaluate(
        tfp.math.secant_root(
            f,
            guess,
            position_tolerance=tolerance,
            stopping_policy_fn=tf.reduce_any))

    expected_roots = [optimize.newton(f, x0), optimize.newton(f, x1)]

    self.assertAllClose(roots[0], expected_roots[0], atol=tolerance)
    self.assertNotAllClose(roots[1], expected_roots[1], atol=tolerance)

    self.assertAllClose(value_at_roots[0], 0.)
    self.assertAllClose(value_at_roots[1], f(roots[1]))
    self.assertNotAllClose(value_at_roots[1], 0.)

  def test_secant_finds_all_roots_from_two_initial_positions(self):
    f = lambda x: (5 * x**3 - 3 * x) / 2.

    x0, x1 = -1, 10
    guess = tf.constant([x0, x1], dtype=tf.float64)
    guess_1 = tf.constant([x0 - 1, x1 - 1], dtype=tf.float64)

    tolerance = 1e-8
    roots, value_at_roots, _ = self.evaluate(
        tfp.math.secant_root(f, guess, guess_1, position_tolerance=tolerance))

    expected_roots = [optimize.newton(f, x0), optimize.newton(f, x1)]
    zeros = [0., 0.]

    self.assertAllClose(roots, expected_roots, atol=tolerance)
    self.assertAllClose(value_at_roots, zeros)

  def test_secant_finds_any_roots_from_two_initial_positions(self):
    f = lambda x: (5 * x**3 - 3 * x) / 2.

    x0, x1 = -1, 10
    guess = tf.constant([x0, x1], dtype=tf.float64)
    next_guess = tf.constant([x0 - 1, x1 - 1], dtype=tf.float64)

    tolerance = 1e-8
    roots, value_at_roots, _ = self.evaluate(
        tfp.math.secant_root(
            f,
            guess,
            next_guess,
            position_tolerance=tolerance,
            stopping_policy_fn=tf.reduce_any))

    expected_roots = [optimize.newton(f, x0), optimize.newton(f, x1)]

    self.assertAllClose(roots[0], expected_roots[0], atol=tolerance)
    self.assertNotAllClose(roots[1], expected_roots[1], atol=tolerance)

    self.assertAllClose(value_at_roots[0], 0.)
    self.assertAllClose(value_at_roots[1], f(roots[1]))
    self.assertNotAllClose(value_at_roots[1], 0.)

  def test_secant_finds_all_roots_using_float32(self):
    f = lambda x: (3 * x**2 - 1) / 2.

    x0, x1 = -5, 2
    guess = tf.constant([x0, x1], dtype=tf.float32)

    tolerance = 1e-8
    roots, value_at_roots, _ = self.evaluate(
        tfp.math.secant_root(f, guess, position_tolerance=tolerance))

    expected_roots = [optimize.newton(f, x0), optimize.newton(f, x1)]
    zeros = [0., 0.]

    self.assertAllClose(roots, expected_roots, atol=tolerance)
    self.assertAllClose(value_at_roots, zeros)

  def test_secant_skips_iteration(self):
    f = lambda x: (3 * x**2 - 1) / 2.

    x0, x1 = -5, 2
    guess = tf.constant([x0, x1], dtype=tf.float64)

    # Skip iteration entirely. This should be a no-op.
    guess, result = self.evaluate(
        [guess, tfp.math.secant_root(f, guess, max_iterations=0)])

    self.assertAllEqual(result.estimated_root, guess)

  def test_secant_invalid_position_tolerance(self):
    f = lambda x: (3 * x**2 - 1) / 2.

    guess = tf.constant(-2, dtype=tf.float64)
    with self.assertRaisesOpError(
        '`position_tolerance` must be greater than 0.'):
      self.evaluate(
          tfp.math.secant_root(
              f, guess, position_tolerance=-1e-8, validate_args=True))

  def test_secant_invalid_value_tolerance(self):
    f = lambda x: (3 * x**2 - 1) / 2.

    guess = tf.constant(-2, dtype=tf.float64)
    with self.assertRaisesOpError('`value_tolerance` must be greater than 0.'):
      self.evaluate(
          tfp.math.secant_root(
              f, guess, value_tolerance=-1e-8, validate_args=True))

  def test_secant_invalid_max_iterations(self):
    f = lambda x: (3 * x**2 - 1) / 2.

    guess = tf.constant(-2, dtype=tf.float64)
    with self.assertRaisesOpError('`max_iterations` must be nonnegative.'):
      self.evaluate(
          tfp.math.secant_root(f, guess, max_iterations=-1, validate_args=True))


if __name__ == '__main__':
  tf.test.main()
