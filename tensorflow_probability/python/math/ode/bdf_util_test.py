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
"""Tests for Backward differentiation formula (BDF) solver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.ode import bdf_util


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters([
    ('float32', tf.float32),
    ('float64', tf.float64),
    ('complex128', tf.complex128),
])
class BDFUtilTest(test_util.TestCase):

  def test_first_step_size_is_large_when_ode_fn_is_constant(self, dtype):
    initial_state_vec = tf.constant([1.], dtype=dtype)
    real_dtype = abs(initial_state_vec).dtype
    atol = tf.constant(1e-12, dtype=real_dtype)
    first_order_bdf_coefficient = -0.1850
    first_order_error_coefficient = first_order_bdf_coefficient + 0.5
    initial_time = tf.constant(0., dtype=real_dtype)
    ode_fn_vec = lambda time, state: 1.
    rtol = tf.constant(1e-8, dtype=real_dtype)
    safety_factor = tf.constant(0.9, dtype=real_dtype)
    max_step_size = 1.
    step_size = bdf_util.first_step_size(
        atol,
        first_order_error_coefficient,
        initial_state_vec,
        initial_time,
        ode_fn_vec,
        rtol,
        safety_factor,
        max_step_size=max_step_size)
    # Step size should be maximal.
    self.assertAllClose(self.evaluate(step_size), max_step_size)

  def test_interpolation_matrix_unit_step_size_ratio(self, dtype):
    order = tf.constant(bdf_util.MAX_ORDER, dtype=tf.int32)
    step_size_ratio = tf.constant(1., dtype)
    interpolation_matrix = bdf_util.interpolation_matrix(
        dtype, order, step_size_ratio)
    self.assertAllClose(
        self.evaluate(interpolation_matrix), [
            [-1., -0., -0., -0., -0.],
            [-2., 1., 0., 0., 0.],
            [-3., 3., -1., -0., -0.],
            [-4., 6., -4., 1., 0.],
            [-5., 10., -10., 5., -1.],
        ])

  def test_interpolate_backward_differences_zeroth_order_is_unchanged(
      self, dtype):
    backward_differences = tf.constant(
        np.random.normal(size=((bdf_util.MAX_ORDER + 3, 3))), dtype=dtype)
    real_dtype = abs(backward_differences).dtype
    step_size_ratio = tf.constant(0.5, dtype=real_dtype)
    interpolated_backward_differences = (
        bdf_util.interpolate_backward_differences(backward_differences,
                                                  bdf_util.MAX_ORDER,
                                                  step_size_ratio))
    self.assertAllClose(
        self.evaluate(backward_differences[0]),
        self.evaluate(interpolated_backward_differences[0]))

  def test_newton_order_one(self, dtype):
    jacobian_mat = tf.constant([[-1.]], dtype=dtype)
    real_dtype = abs(jacobian_mat).dtype
    bdf_coefficient = tf.constant(-0.1850, dtype=dtype)
    first_order_newton_coefficient = 1. / (1. - bdf_coefficient)
    step_size = tf.constant(0.01, dtype=real_dtype)
    unitary, upper = bdf_util.newton_qr(jacobian_mat,
                                        first_order_newton_coefficient,
                                        step_size)

    backward_differences = tf.constant([[1.], [-1.], [0.], [0.], [0.], [0.]],
                                       dtype=dtype)
    ode_fn_vec = lambda time, state: -state
    order = tf.constant(1, dtype=tf.int32)
    time = tf.constant(0., dtype=real_dtype)
    tol = tf.constant(1e-6, dtype=real_dtype)

    # The equation we are trying to solve with Newton's method is linear.
    # Therefore, we should observe exact convergence after one iteration. An
    # additional iteration is required to obtain an accurate error estimate,
    # making the total number of iterations 2.
    max_num_newton_iters = 2

    converged, next_backward_difference, next_state, _ = bdf_util.newton(
        backward_differences, max_num_newton_iters,
        first_order_newton_coefficient, ode_fn_vec, order, step_size, time, tol,
        unitary, upper)
    self.assertEqual(self.evaluate(converged), True)

    state = backward_differences[0, :]
    step_size_cast = tf.cast(step_size, dtype=dtype)
    exact_next_state = ((1. - bdf_coefficient) * state + bdf_coefficient) / (
        1. + step_size_cast - bdf_coefficient)

    self.assertAllClose(
        self.evaluate(next_backward_difference),
        self.evaluate(exact_next_state))
    self.assertAllClose(
        self.evaluate(next_state), self.evaluate(exact_next_state))


if __name__ == '__main__':
  tf.test.main()
