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
"""Tests for utilities for TensorFlow Probability ODE solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.ode import util


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters([
    ('', False, False),
    ('use_pfor', False, True),
    ('use_automatic_differentiation', True, False),
    ('use_automatic_differentiation_and_pfor', True, True),
])
class JacobianTest(test_util.TestCase):

  def test_right_mult_by_jacobian_mat(self, use_automatic_differentiation,
                                      use_pfor):
    vec = np.float32([1., 2., 3.])
    jacobian = -np.float32([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    time = np.float32(0.)
    state_vec = np.float32([1., 1., 1.])

    def ode_fn(_, state):
      return tf.squeeze(tf.matmul(jacobian, state[:, tf.newaxis]))

    state_shape = tf.shape(state_vec)
    ode_fn_vec = util.get_ode_fn_vec(ode_fn, state_shape)
    jacobian_fn_mat = util.get_jacobian_fn_mat(
        None if use_automatic_differentiation else jacobian, ode_fn_vec,
        state_shape, use_pfor, dtype=tf.float32)
    result = util.right_mult_by_jacobian_mat(jacobian_fn_mat, ode_fn_vec, time,
                                             state_vec, vec)

    self.assertAllClose(self.evaluate(result), np.dot(vec, jacobian))


@test_util.test_all_tf_execution_regimes
class NestedJacobianTest(test_util.TestCase):

  def test_structured_jacobian_mat(self):
    """Test that structured jacobian gets flattened correctly."""
    state = [
        tf.constant([1., 2.]),
        tf.constant([3.]),
        tf.constant([[4., 5.], [6., 7.]])
    ]
    state_shape = [tf.shape(s) for s in state]
    mat = tf.convert_to_tensor(np.random.randn(7, 7), dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(state)
      state_vec = util.get_state_vec(state)
      tape.watch(state_vec)
      new_state_vec = tf.matmul(mat, state_vec[..., tf.newaxis])[..., 0]
      new_state = util.get_state_from_vec(new_state_vec, state_shape)

    jacobian_mat = self.evaluate(tape.jacobian(new_state_vec, state_vec))
    jacobian = [
        [self.evaluate(tape.jacobian(y, x)) for x in state] for y in new_state
    ]
    jacobian_mat2 = self.evaluate(
        util.get_jacobian_fn_mat(
            jacobian_fn=jacobian,
            ode_fn_vec=None,
            state_shape=state_shape,
            use_pfor=False,
            dtype=tf.float32)(None))
    self.assertAllEqual(jacobian_mat, jacobian_mat2)


if __name__ == '__main__':
  tf.test.main()
