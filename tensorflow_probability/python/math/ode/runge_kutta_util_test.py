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
"""Tests for Runge-Kutta solver utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.ode import runge_kutta_util as rk_util


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters([
    ('float64', tf.float64),
    ('complex128', tf.complex128),
])
class RungeKuttaUtilTest(test_util.TestCase):

  def test_polynomial_fit(self, dtype):
    """Asserts that interpolation of 4th order polynomial is exact."""
    coefficients = [1 + 2j, 0.3 - 1j, 3.5 - 3.7j, 0.5 - 0.1j, 0.1 + 0.1j]
    coefficients = [tf.cast(c, dtype) for c in coefficients]

    def f(x):
      components = []
      for power, c in enumerate(reversed(coefficients)):
        components.append(c * x**power)
      return tf.add_n(components)

    def f_prime(x):
      components = []
      for power, c in enumerate(reversed(coefficients[:-1])):
        components.append(c * x**(power) * (power + 1))
      return tf.add_n(components)

    coeffs = rk_util._fourth_order_interpolation_coefficients(
        f(0.0), f(10.0), f(5.0), f_prime(0.0), f_prime(10.0), 10.0)
    times = np.linspace(0, 10, dtype=np.float32)
    y_fit = tf.stack(
        [rk_util.evaluate_interpolation(coeffs, 0.0, 10.0, t) for t in times])
    y_expected = f(times)
    self.assertAllClose(y_fit, y_expected)

  def test_weighted_sum_tensor(self, dtype):
    del dtype  # not used in this test case.
    weights = [0.5, -0.25, -0.25]
    states = [tf.eye(2) for _ in range(3)]
    weighted_tensor_sum = rk_util.weighted_sum(weights, states)
    self.assertAllClose(weighted_tensor_sum, tf.zeros((2, 2)))

    weights = [0.5, -0.25, -0.25, 1.0]
    states = [tf.ones(2) for _ in range(4)]
    weighted_tensor_sum = rk_util.weighted_sum(weights, states)
    self.assertAllClose(weighted_tensor_sum, tf.ones(2))

    weights = [0.5, -0.25, -0.25, 0.0]
    states = [tf.eye(2) for _ in range(4)]
    weighted_tensor_sum = rk_util.weighted_sum(weights, states)
    self.assertAllClose(weighted_tensor_sum, tf.zeros((2, 2)))

  def test_weighted_sum_nested_type(self, dtype):
    del dtype  # not used in this test case.
    weights = [0.5, -0.25, -0.25]
    states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(3)]
    weighted_state_sum = rk_util.weighted_sum(weights, states)
    self.assertIsInstance(weighted_state_sum, tuple)

  def test_weighted_sum_nested_values(self, dtype):
    del dtype  # not used in this test case.
    weights = [0.5, -0.25, -0.25]
    states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(3)]
    weighted_state_sum = rk_util.weighted_sum(weights, states)
    expected_result = (tf.zeros((2, 2)), tf.zeros((2, 2)))
    self.assertAllClose(weighted_state_sum, expected_result)

    weights = [0.5, -0.25, -0.25, 0]
    states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(4)]
    weighted_state_sum = rk_util.weighted_sum(weights, states)
    expected_result = (tf.zeros((2, 2)), tf.zeros((2, 2)))
    self.assertAllClose(weighted_state_sum, expected_result)

  def test_weighted_sum_value_errors(self, dtype):
    del dtype  # not used in this test case.
    empty_weights = []
    empty_states = []
    with self.assertRaises(ValueError):
      _ = rk_util.weighted_sum(empty_weights, empty_states)

    wrong_length_weights = [0.5, -0.25, -0.25, 0]
    wrong_length_states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(5)]
    with self.assertRaises(ValueError):
      _ = rk_util.weighted_sum(wrong_length_weights, wrong_length_states)

    weights = [0.5, -0.25, -0.25, 0]
    not_same_structure_states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(3)]
    not_same_structure_states.append(tf.eye(2))
    with self.assertRaises(ValueError):
      _ = rk_util.weighted_sum(weights, not_same_structure_states)

  def test_abs_square(self, dtype):
    test_values = np.array([1 + 2j, 0.3 - 1j, 3.5 - 3.7j])
    input_values = tf.cast(test_values, dtype)
    actual_abs_square = rk_util.abs_square(input_values)
    expected_abs_square = tf.math.square(tf.abs(input_values))
    self.assertAllClose(actual_abs_square, expected_abs_square)

  def test_nest_rms_norm_on_tensor(self, dtype):
    test_values = np.array([1.4 -1j, 2.7 + 0.23j, 7.3 + 9.4j])
    test_values = test_values.astype(dtype.as_numpy_dtype)
    input_values = tf.cast(test_values, dtype=dtype)
    actual_norm = rk_util.nest_rms_norm(input_values)
    expected_norm = np.linalg.norm(test_values) / np.sqrt(test_values.size)
    self.assertAllClose(actual_norm, expected_norm)

  def test_nest_rms_norm_on_nest(self, dtype):
    del dtype  # not used in this test case.
    a = np.array([1.4, 2.7, 7.3])
    b = 0.3 * np.eye(3, dtype=np.float32) + 0.64 * np.ones((3, 3))
    input_nest = (tf.convert_to_tensor(a), tf.convert_to_tensor(b))
    actual_norm_nest = rk_util.nest_rms_norm(input_nest)
    full_state = np.concatenate([np.expand_dims(a, 0), b])
    expected_norm_nest = np.linalg.norm(full_state) / np.sqrt(full_state.size)
    self.assertAllClose(expected_norm_nest, actual_norm_nest)

  def test_nest_constant(self, dtype):
    ndtype = dtype.as_numpy_dtype
    input_structure = (
        np.ones(4, dtype=ndtype),
        (np.eye(3, dtype=ndtype), np.zeros(4, dtype=ndtype))
    )
    ones_like_structure = rk_util.nest_constant(input_structure)
    tf.nest.assert_same_structure(input_structure, ones_like_structure)
    flat_ones_like_structure = tf.nest.flatten(ones_like_structure)
    for component in flat_ones_like_structure:
      self.assertAllClose(component, tf.ones(shape=component.shape))


if __name__ == '__main__':
  tf.test.main()
