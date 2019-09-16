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

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.math.ode import util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
@parameterized.named_parameters([
    ('', False, False),
    ('use_pfor', False, True),
    ('use_automatic_differentiation', True, False),
    ('use_automatic_differentiation_and_pfor', True, True),
])
class JacobianTest(test_case.TestCase):

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
        state_shape, use_pfor)
    result = util.right_mult_by_jacobian_mat(jacobian_fn_mat, ode_fn_vec, time,
                                             state_vec, vec)

    self.assertAllClose(self.evaluate(result), np.dot(vec, jacobian))


if __name__ == '__main__':
  tf.test.main()
