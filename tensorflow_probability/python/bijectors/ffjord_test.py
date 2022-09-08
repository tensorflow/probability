# Copyright 2019 The TensorFlow Probability Authors.
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
"""FFJORD tests."""

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import ffjord
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow_probability.python.math import gradient as tfp_gradient

from tensorflow.python import tf2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
@parameterized.named_parameters([
    ('float32', np.float32),
    ('float64', np.float64),
])
class FFJORDBijectorTest(tfp_test_util.TestCase):
  """Tests correctness of the Y = g(X) = FFJORD(X) transformation."""

  def testBijector(self, dtype):
    tf_dtype = tf.as_dtype(dtype)
    move_ode_fn = lambda t, z: tf.ones_like(z)
    trace_augmentation_fn = ffjord.trace_jacobian_exact
    bijector = ffjord.FFJORD(
        trace_augmentation_fn=trace_augmentation_fn,
        state_time_derivative_fn=move_ode_fn,
        dtype=tf_dtype)
    x = np.zeros((2, 5), dtype=dtype)
    y = np.ones((2, 5), dtype=dtype)
    expected_log_det_jacobian = np.zeros(2, dtype=dtype)

    self.assertStartsWith(bijector.name, 'ffjord')
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        expected_log_det_jacobian,
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1))
    )
    self.assertAllClose(
        expected_log_det_jacobian,
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1))
    )

  def testBijectorConditionKwargs(self, dtype):
    if not tf2.enabled():
      self.skipTest('b/152464477')

    tf_dtype = tf.as_dtype(dtype)
    def conditional_ode_fn(t, z, c):
      del t  # unused.
      return tf.ones_like(z) * c ** 2

    trace_augmentation_fn = ffjord.trace_jacobian_exact
    bijector = ffjord.FFJORD(
        trace_augmentation_fn=trace_augmentation_fn,
        state_time_derivative_fn=conditional_ode_fn,
        dtype=tf_dtype)
    x = tf.zeros((2, 5), dtype=tf_dtype)
    y = tf.ones((2, 5), dtype=tf_dtype) * 4
    c = tf.ones((2, 5), dtype=tf_dtype) * 2
    expected_log_det_jacobian = np.zeros(2, dtype=dtype)
    expected_dy_dc = np.ones((2, 5), dtype=dtype) * 4

    def grad_fn(c):
      y = bijector.forward(x, c=c)
      return y

    dy_dc = self.evaluate(
        tfp_gradient.value_and_gradient(grad_fn, c)[1])

    self.assertStartsWith(bijector.name, 'ffjord')
    self.assertAllClose(self.evaluate(y),
                        self.evaluate(bijector.forward(x, c=c)), atol=1e-5)
    self.assertAllClose(self.evaluate(x),
                        self.evaluate(bijector.inverse(y, c=c)), atol=1e-5)
    self.assertAllClose(
        expected_log_det_jacobian,
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1, c=c))
    )
    self.assertAllClose(
        expected_log_det_jacobian,
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1, c=c))
    )
    self.assertAllClose(expected_dy_dc, dy_dc)

  def testJacobianScaling(self, dtype):
    tf_dtype = tf.as_dtype(dtype)
    scaling_by_two_exp = np.log(2.0)
    scale_ode_fn = lambda t, z: scaling_by_two_exp * z
    trace_augmentation_fn = ffjord.trace_jacobian_exact
    bijector = ffjord.FFJORD(
        trace_augmentation_fn=trace_augmentation_fn,
        state_time_derivative_fn=scale_ode_fn,
        dtype=tf_dtype)
    x = np.array([[0.0], [1.0]], dtype=dtype)
    y = np.array([[0.0], [2.0]], dtype=dtype)
    expected_forward_log_det_jacobian = np.array([np.log(2.0)] * 2, dtype=dtype)
    expected_inverse_log_det_jacobian = np.array([np.log(0.5)] * 2, dtype=dtype)
    self.assertAllClose(
        y, self.evaluate(bijector.forward(x)), rtol=0.0, atol=1e-4)
    self.assertAllClose(
        x, self.evaluate(bijector.inverse(y)), rtol=0.0, atol=1e-4)
    self.assertAllClose(
        expected_inverse_log_det_jacobian,
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1)),
    )
    self.assertAllClose(
        expected_forward_log_det_jacobian,
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)),
    )

  def testJacobianDiagonalScaling(self, dtype):
    tf_dtype = tf.as_dtype(dtype)
    num_dims = 10
    matrix_diagonal = np.random.uniform(size=[num_dims]).astype(dtype)
    scaling_matrix = np.diag(matrix_diagonal)
    one_time_scale_matrix = np.diag(np.exp(matrix_diagonal))
    scale_ode_fn = lambda t, z: tf.linalg.matvec(scaling_matrix, z)
    trace_augmentation_fn = ffjord.trace_jacobian_exact
    bijector = ffjord.FFJORD(
        trace_augmentation_fn=trace_augmentation_fn,
        state_time_derivative_fn=scale_ode_fn,
        dtype=tf_dtype)
    x = np.random.uniform(size=[1, num_dims]).astype(dtype)
    y = np.matmul(x, one_time_scale_matrix)

    expected_forward_log_det_jacobian_value = np.log(
        np.prod(np.exp(matrix_diagonal)))
    expected_fldj = np.array([expected_forward_log_det_jacobian_value])
    expected_ildj = np.array([-expected_forward_log_det_jacobian_value])

    self.assertAllClose(
        y, self.evaluate(bijector.forward(x)), rtol=0.0, atol=1e-3)
    self.assertAllClose(
        x, self.evaluate(bijector.inverse(y)), rtol=0.0, atol=1e-3)
    self.assertAllClose(
        expected_ildj,
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1)),
    )
    self.assertAllClose(
        expected_fldj,
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)),
    )

  def testHutchinsonsNormalEstimator(self, dtype):
    seed = 42
    tf_dtype = tf.as_dtype(dtype)
    num_dims = 10
    np.random.seed(seed=seed)
    matrix_diagonal = np.random.uniform(size=[num_dims]).astype(dtype)
    scaling_matrix = np.diag(matrix_diagonal)
    one_time_scale_matrix = np.diag(np.exp(matrix_diagonal))
    scale_ode_fn = lambda t, z: tf.linalg.matvec(scaling_matrix, z)

    def trace_augmentation_fn(ode_fn, z_shape, dtype):
      return ffjord.trace_jacobian_hutchinson(
          ode_fn, z_shape, dtype, num_samples=128, seed=seed)

    bijector = ffjord.FFJORD(
        trace_augmentation_fn=trace_augmentation_fn,
        state_time_derivative_fn=scale_ode_fn,
        dtype=tf_dtype)
    x = np.random.uniform(size=[1, num_dims]).astype(dtype)
    y = np.matmul(x, one_time_scale_matrix)
    expected_forward_log_det_jacobian_value = np.log(
        np.prod(np.exp(matrix_diagonal)))
    expected_fldj = np.array([expected_forward_log_det_jacobian_value])
    expected_ildj = np.array([-expected_forward_log_det_jacobian_value])

    self.assertAllClose(
        y, self.evaluate(bijector.forward(x)), rtol=0.0, atol=1e-3)
    self.assertAllClose(
        x, self.evaluate(bijector.inverse(y)), rtol=0.0, atol=1e-3)
    self.assertAllClose(
        expected_ildj,
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1)),
        atol=7e-1
    )
    self.assertAllClose(
        expected_fldj,
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)),
        atol=7e-1
    )


if __name__ == '__main__':
  tfp_test_util.main()
