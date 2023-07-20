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

# Dependency imports
import numpy as np
import scipy.optimize as optimize

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import root_search


@test_util.test_all_tf_execution_regimes
class SecantRootSearchTest(test_util.TestCase):

  def test_secant_finds_all_roots_from_one_initial_position(self):
    f = lambda x: (63 * x**5 - 70 * x**3 + 15 * x) / 8.

    x0, x1 = -1, 10
    guess = tf.constant([x0, x1], dtype=tf.float64)

    tolerance = 1e-8
    roots, value_at_roots, _ = self.evaluate(
        root_search.find_root_secant(f, guess, position_tolerance=tolerance))

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
        root_search.find_root_secant(
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
        root_search.find_root_secant(
            f, guess, guess_1, position_tolerance=tolerance))

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
        root_search.find_root_secant(
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
        root_search.find_root_secant(f, guess, position_tolerance=tolerance))

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
        [guess, root_search.find_root_secant(f, guess, max_iterations=0)])

    self.assertAllEqual(result.estimated_root, guess)

  def test_secant_invalid_position_tolerance(self):
    f = lambda x: (3 * x**2 - 1) / 2.

    guess = tf.constant(-2, dtype=tf.float64)
    with self.assertRaisesOpError(
        '`position_tolerance` must be greater than 0.'):
      self.evaluate(
          root_search.find_root_secant(
              f, guess, position_tolerance=-1e-8, validate_args=True))

  def test_secant_invalid_value_tolerance(self):
    f = lambda x: (3 * x**2 - 1) / 2.

    guess = tf.constant(-2, dtype=tf.float64)
    with self.assertRaisesOpError('`value_tolerance` must be greater than 0.'):
      self.evaluate(
          root_search.find_root_secant(
              f, guess, value_tolerance=-1e-8, validate_args=True))

  def test_secant_invalid_max_iterations(self):
    f = lambda x: (3 * x**2 - 1) / 2.

    guess = tf.constant(-2, dtype=tf.float64)
    with self.assertRaisesOpError('`max_iterations` must be nonnegative.'):
      self.evaluate(
          root_search.find_root_secant(
              f, guess, max_iterations=-1, validate_args=True))

  def test_secant_non_static_shape(self):
    if tf.executing_eagerly():
      self.skipTest('Test uses dynamic shapes.')

    f = lambda x: (x - 1.) * (x + 1)
    initial_position = tf1.placeholder_with_default([1., 1., 1.], shape=None)
    self.assertAllClose(
        root_search.find_root_secant(
            f, initial_position).objective_at_estimated_root, [0., 0., 0.])

  def test_secant_nan_objective(self):
    # With this setup, the find_root_secant should short circuit on the
    # first starting point immediately, and short circuit on the second
    # iteration for the other two values.
    f = lambda x: tf.where(tf.abs(x) < 1e-5, np.nan * tf.ones_like(x), x)
    initial_position = tf1.placeholder_with_default([0., -1., 10.], shape=None)

    search_result = root_search.find_root_secant(f, initial_position)

    self.assertAllNan(search_result.objective_at_estimated_root)
    self.assertAllNan(search_result.estimated_root)
    self.assertAllEqual(search_result.num_iterations, tf.constant([0, 2, 2]))


@test_util.test_all_tf_execution_regimes
class ChandrupatlaRootSearchTest(test_util.TestCase):

  def test_chandrupatla_scalar_inverse_gaussian_cdf(self):
    true_x = 3.14159
    u = special_math.ndtr(true_x)

    roots, value_at_roots, _ = root_search.find_root_chandrupatla(
        objective_fn=lambda x: special_math.ndtr(x) - u,
        low=-100.,
        high=100.,
        position_tolerance=1e-8)
    self.assertAllClose(value_at_roots, tf.zeros_like(value_at_roots))
    # The normal CDF function is not precise enough to be inverted to a
    # position tolerance of 1e-8 (the objective goes to zero relatively
    # far from the expected point), so check it at a lower tolerance.
    self.assertAllClose(roots, true_x, atol=1e-4)

  def test_chandrupatla_batch_high_degree_polynomial(self):
    seed = test_util.test_seed(sampler_type='stateless')
    expected_roots = self.evaluate(samplers.normal(
        [4, 3], seed=seed))
    roots, value_at_roots, _ = root_search.find_root_chandrupatla(
        objective_fn=lambda x: (x - expected_roots)**15,
        low=-20.,
        high=20.,
        position_tolerance=1e-8)
    self.assertAllClose(value_at_roots, tf.zeros_like(value_at_roots))
    # The function is not precise enough to be inverted to a
    # position tolerance of 1e-8, (the objective goes to zero relatively
    # far from the expected point), so check it at a lower tolerance.
    self.assertAllClose(roots, expected_roots, atol=1e-2)

  def test_chandrupatla_max_iterations(self):
    expected_roots = samplers.normal(
        [4, 3], seed=test_util.test_seed(sampler_type='stateless'))
    max_iterations = samplers.uniform(
        [4, 3], minval=1, maxval=6, dtype=tf.int32,
        seed=test_util.test_seed(sampler_type='stateless'))
    _, _, num_iterations = root_search.find_root_chandrupatla(
        objective_fn=lambda x: (x - expected_roots)**3,
        low=-1000000.,
        high=1000000.,
        position_tolerance=1e-8,
        max_iterations=max_iterations)
    self.assertAllClose(num_iterations,
                        max_iterations)

  def test_chandrupatla_halts_at_fixed_point(self):
    # This search would naively get stuck at the interval
    # {a=1.4717137813568115, b=1.471713662147522}, which does not quite
    # satisfy the tolerance, but will never be tightened further because it has
    # the property that `0.5 * a + 0.5 * b == a` in float32. The search should
    # detect the fixed point and halt early.
    max_iterations = 50
    _, _, num_iterations = root_search.find_root_chandrupatla(
        lambda ux: tf.math.igamma(2., tf.nn.softplus(ux)) - 0.5,
        low=-100.,
        high=100.,
        position_tolerance=1e-8,
        value_tolerance=1e-8,
        max_iterations=max_iterations)
    self.assertLess(self.evaluate(num_iterations), max_iterations)

  def test_chandrupatla_float64_high_precision(self):
    expected_roots = samplers.normal(
        [4, 3], seed=test_util.test_seed(sampler_type='stateless'),
        dtype=tf.float64)
    tolerance = 1e-12
    roots, value_at_roots, _ = root_search.find_root_chandrupatla(
        objective_fn=lambda x: (x - expected_roots)**3,
        low=tf.convert_to_tensor(-100., dtype=expected_roots.dtype),
        high=tf.convert_to_tensor(100., dtype=expected_roots.dtype),
        position_tolerance=tolerance)
    self.assertAllClose(roots, expected_roots, atol=tolerance)
    self.assertAllClose(value_at_roots, tf.zeros_like(value_at_roots))

  def test_chandrupatla_invalid_bounds(self):
    with self.assertRaisesOpError('must be on different sides of a root'):
      self.evaluate(
          root_search.find_root_chandrupatla(
              lambda x: x**2 - 2., 3., 4., validate_args=True))

  def test_chandrupatla_automatically_selects_bounds(self):
    expected_roots = 1e6 * samplers.normal(
        [4, 3], seed=test_util.test_seed(sampler_type='stateless'))
    _, value_at_roots, _ = root_search.find_root_chandrupatla(
        objective_fn=lambda x: (x - expected_roots)**5, position_tolerance=1e-8)
    self.assertAllClose(value_at_roots, tf.zeros_like(value_at_roots))

  def test_chandrupatla_non_static_shape(self):
    if tf.executing_eagerly():
      self.skipTest('Test uses dynamic shapes.')

    f = lambda x: (x - 1.) * (x + 1)
    low = tf1.placeholder_with_default([-100., -100., -100.], shape=None)
    high = tf1.placeholder_with_default([100., 100., 100.], shape=None)
    self.assertAllClose(
        root_search.find_root_chandrupatla(
            f, low=low, high=high).objective_at_estimated_root, [0., 0., 0.])


@test_util.test_all_tf_execution_regimes
class BracketRootTest(test_util.TestCase):

  def test_batch_with_nans(self):

    idxs = np.arange(20, dtype=np.float32)
    bounds = np.reshape(np.exp(idxs), [4, -1])
    roots = np.reshape(1. / (20. - idxs), [4, -1])
    def objective_fn(x):
      return tf.where(x < -bounds,
                      np.nan,
                      tf.where(x > bounds,
                               np.inf,
                               (x - roots)**3))

    low, high = self.evaluate(root_search.bracket_root(objective_fn))
    f_low, f_high = self.evaluate((objective_fn(low), objective_fn(high)))
    self.assertAllFinite(f_low)
    self.assertAllFinite(f_high)
    self.assertAllTrue(low < roots)
    self.assertAllTrue(high > roots)

  def test_negative_root(self):
    root = -17.314
    low, high = self.evaluate(root_search.bracket_root(lambda x: (x - root)))
    self.assertLess(low, root)
    self.assertGreater(high, root)

  def test_root_near_zero(self):
    root = tf.exp(-13.)
    low, high = self.evaluate(root_search.bracket_root(lambda x: (x - root)))
    self.assertLess(low, np.exp(-13.))
    self.assertGreater(high, np.exp(-13))
    self.assertAllClose(low, root, atol=1e-4)
    self.assertAllClose(high, root, atol=1e-4)

  def test_returns_zero_width_bracket_at_root(self):
    root = tf.exp(-10.)
    low, high = self.evaluate(root_search.bracket_root(lambda x: (x - root)))
    self.assertAllClose(low, root)
    self.assertAllClose(high, root)

  def test_backs_off_to_trivial_bracket(self):
    dtype_info = np.finfo(np.float32)
    low, high = self.evaluate(
        root_search.bracket_root(lambda x: np.nan * x, dtype=np.float32))
    self.assertEqual(low, dtype_info.min)
    self.assertEqual(high, dtype_info.max)

  def test_float64(self):
    low, high = self.evaluate(
        root_search.bracket_root(lambda x: (x - np.pi)**3, dtype=np.float64))
    self.assertEqual(low.dtype, np.float64)
    self.assertEqual(high.dtype, np.float64)
    self.assertLess(low, np.pi)
    self.assertGreater(high, np.pi)


if __name__ == '__main__':
  test_util.main()
