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

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.math.ode import runge_kutta_util as rk_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
@parameterized.named_parameters([
    ('float64', tf.float64),
    ('complex128', tf.complex128),
])
class RungeKuttaUtilTest(parameterized.TestCase, test_case.TestCase):

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
    weighted_tensor_sum = rk_util._weighted_sum(weights, states)
    self.assertAllClose(weighted_tensor_sum, tf.zeros((2, 2)))

    weights = [0.5, -0.25, -0.25, 1.0]
    states = [tf.ones(2) for _ in range(4)]
    weighted_tensor_sum = rk_util._weighted_sum(weights, states)
    self.assertAllClose(weighted_tensor_sum, tf.ones(2))

    weights = [0.5, -0.25, -0.25, 0.0]
    states = [tf.eye(2) for _ in range(4)]
    weighted_tensor_sum = rk_util._weighted_sum(weights, states)
    self.assertAllClose(weighted_tensor_sum, tf.zeros((2, 2)))

  def test_weighted_sum_nested_type(self, dtype):
    del dtype  # not used in this test case.
    weights = [0.5, -0.25, -0.25]
    states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(3)]
    weighted_state_sum = rk_util._weighted_sum(weights, states)
    self.assertIsInstance(weighted_state_sum, tuple)

  def test_weighted_sum_nested_values(self, dtype):
    del dtype  # not used in this test case.
    weights = [0.5, -0.25, -0.25]
    states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(3)]
    weighted_state_sum = rk_util._weighted_sum(weights, states)
    expected_result = (tf.zeros((2, 2)), tf.zeros((2, 2)))
    self.assertAllClose(weighted_state_sum, expected_result)

    weights = [0.5, -0.25, -0.25, 0]
    states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(4)]
    weighted_state_sum = rk_util._weighted_sum(weights, states)
    expected_result = (tf.zeros((2, 2)), tf.zeros((2, 2)))
    self.assertAllClose(weighted_state_sum, expected_result)

  def test_weighted_sum_value_errors(self, dtype):
    del dtype  # not used in this test case.
    empty_weights = []
    empty_states = []
    with self.assertRaises(ValueError):
      _ = rk_util._weighted_sum(empty_weights, empty_states)

    wrong_length_weights = [0.5, -0.25, -0.25, 0]
    wrong_length_states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(5)]
    with self.assertRaises(ValueError):
      _ = rk_util._weighted_sum(wrong_length_weights, wrong_length_states)

    weights = [0.5, -0.25, -0.25, 0]
    not_same_structure_states = [(tf.eye(2), tf.ones((2, 2))) for _ in range(3)]
    not_same_structure_states.append(tf.eye(2))
    with self.assertRaises(ValueError):
      _ = rk_util._weighted_sum(weights, not_same_structure_states)


if __name__ == '__main__':
  tf.test.main()
