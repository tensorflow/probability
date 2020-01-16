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
"""Tests for CholeskyToInvCholesky bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class CholeskyToInvCholeskyTest(test_util.TestCase):

  def testBijector(self):
    bijector = tfb.CholeskyToInvCholesky()
    self.assertStartsWith(bijector.name, "cholesky_to_inv_cholesky")
    x = np.array([[3., 0.], [2., 7.]], dtype=np.float32)
    m = x.dot(x.T)
    m_inv = np.linalg.inv(m)
    y = np.linalg.cholesky(m_inv)
    x_fwd = bijector.forward(x)
    y_inv = bijector.inverse(x_fwd)
    x_fwd_, y_inv_ = self.evaluate([x_fwd, y_inv])
    self.assertAllClose(y, x_fwd_, atol=1.e-5, rtol=1.e-5)
    self.assertAllClose(x, y_inv_, atol=1.e-5, rtol=1.e-5)

  def testBijectorWithTensors(self):
    bijector = tfb.CholeskyToInvCholesky()
    x = np.array([
        [[3., 0.], [1., 4.]],
        [[2., 0.], [7., 1.]]], dtype=np.float32)
    y = bijector.forward(x)
    y0 = bijector.forward(x[0, :])
    y1 = bijector.forward(x[1, :])
    y_inv = bijector.inverse(y)
    y_inv0 = bijector.inverse(y[0, :])
    y_inv1 = bijector.inverse(y[1, :])
    y_, y0_, y1_, y_inv_, y_inv0_, y_inv1_ = self.evaluate(
        [y, y0, y1, y_inv, y_inv0, y_inv1])
    self.assertAllClose(y_[0, :], y0_, atol=1.e-5, rtol=1.e-5)
    self.assertAllClose(y_[1, :], y1_, atol=1.e-5, rtol=1.e-5)
    self.assertAllClose(y_inv_[0, :], y_inv0_, atol=1.e-5, rtol=1.e-5)
    self.assertAllClose(y_inv_[1, :], y_inv1_, atol=1.e-5, rtol=1.e-5)
    self.assertAllClose(y_inv_, x, atol=1.e-5, rtol=1.e-5)

  def _get_fldj_numerical(self, bijector, x, event_ndims,
                          eps=1.e-6,
                          input_to_vector=tfb.Identity,
                          output_to_vector=tfb.Identity):
    """Numerically approximate the forward log det Jacobian of a bijector.

    Args:
      bijector: the bijector whose Jacobian we wish to approximate
      x: the value for which we want to approximate the Jacobian
      event_ndims: number of dimensions in an event
      eps: epsilon to add when forming (f(x+eps)-f(x)) / eps
      input_to_vector: a bijector that maps the input value to a vector
      output_to_vector: a bijector that maps the output value to a vector

    Returns:
      A numerical approximation to the log det Jacobian of bijector.forward
      evaluated at x.
    """
    x_vector = input_to_vector.forward(x)                         # [B, n]
    n = tf.shape(x_vector)[-1]
    x_plus_eps_vector = (
        x_vector[..., tf.newaxis, :] +
        eps * tf.eye(n, dtype=x_vector.dtype))                    # [B, n, n]
    x_plus_eps = input_to_vector.inverse(x_plus_eps_vector)       # [B, n, d, d]
    f_x_plus_eps = bijector.forward(x_plus_eps)                   # [B, n, d, d]
    f_x_plus_eps_vector = output_to_vector.forward(f_x_plus_eps)  # [B, n, n]

    f_x = bijector.forward(x)                                     # [B, d, d]
    f_x_vector = output_to_vector.forward(f_x)                    # [B, n]

    jacobian_numerical = (f_x_plus_eps_vector -
                          f_x_vector[..., tf.newaxis, :]) / eps

    return (
        tf.math.log(tf.abs(tf.linalg.det(jacobian_numerical))) +
        input_to_vector.forward_log_det_jacobian(x, event_ndims=event_ndims) -
        output_to_vector.forward_log_det_jacobian(f_x, event_ndims=event_ndims)
    )

  def testJacobian(self):
    cholesky_to_vector = tfb.Invert(
        tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=None))
    bijector = tfb.CholeskyToInvCholesky()
    for x in [np.array([[2.]],
                       dtype=np.float64),
              np.array([[2., 0.],
                        [3., 4.]],
                       dtype=np.float64),
              np.array([[2., 0., 0.],
                        [3., 4., 0.],
                        [5., 6., 7.]],
                       dtype=np.float64)]:
      fldj = bijector.forward_log_det_jacobian(x, event_ndims=2)
      fldj_numerical = self._get_fldj_numerical(
          bijector, x, event_ndims=2,
          input_to_vector=cholesky_to_vector,
          output_to_vector=cholesky_to_vector)
      fldj_, fldj_numerical_ = self.evaluate([fldj, fldj_numerical])
      self.assertAllClose(fldj_, fldj_numerical_, rtol=1e-2)

  def testJacobianWithTensors(self):
    bijector = tfb.CholeskyToInvCholesky()
    x = np.array([
        [[3., 0.],
         [1., 4.]],
        [[2., 0.],
         [7., 1.]]], dtype=np.float32)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=2)
    fldj0 = bijector.forward_log_det_jacobian(x[0], event_ndims=2)
    fldj1 = bijector.forward_log_det_jacobian(x[1], event_ndims=2)
    fldj_, fldj0_, fldj1_ = self.evaluate([fldj, fldj0, fldj1])
    self.assertAllClose(fldj_[0], fldj0_, rtol=1e-5)
    self.assertAllClose(fldj_[1], fldj1_, rtol=1e-5)

if __name__ == "__main__":
  tf.test.main()
