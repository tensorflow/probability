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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class CholeskyOuterProductBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = X @ X.T transformation."""

  def testBijectorMatrix(self):
    bijector = tfb.CholeskyOuterProduct(validate_args=True)
    self.assertStartsWith(bijector.name, "cholesky_outer_product")
    x = [[[1., 0], [2, 1]], [[np.sqrt(2.), 0], [np.sqrt(8.), 1]]]
    y = np.matmul(x, np.transpose(x, axes=(0, 2, 1)))
    # Fairly easy to compute differentials since we have 2x2.
    dx_dy = [[[2. * 1, 0, 0],
              [2, 1, 0],
              [0, 2 * 2, 2 * 1]],
             [[2 * np.sqrt(2.), 0, 0],
              [np.sqrt(8.), np.sqrt(2.), 0],
              [0, 2 * np.sqrt(8.), 2 * 1]]]
    ildj = -np.sum(
        np.log(np.asarray(dx_dy).diagonal(
            offset=0, axis1=1, axis2=2)),
        axis=1)
    self.assertAllEqual((2, 2, 2), bijector.forward(x).shape)
    self.assertAllEqual((2, 2, 2), bijector.inverse(y).shape)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(
            bijector.inverse_log_det_jacobian(
                y, event_ndims=2)), atol=0., rtol=1e-7)
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(
            y, event_ndims=2)),
        self.evaluate(bijector.forward_log_det_jacobian(
            x, event_ndims=2)),
        atol=0.,
        rtol=1e-7)

  def testNoBatchStaticJacobian(self):
    x = np.eye(2)
    bijector = tfb.CholeskyOuterProduct()

    # The Jacobian matrix is 2 * tf.eye(2), which has jacobian determinant 4.
    self.assertAllClose(
        np.log(4),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=2)))

  def testNoBatchDynamicJacobian(self):
    bijector = tfb.CholeskyOuterProduct()
    x = tf1.placeholder_with_default(
        np.eye(2, dtype=np.float32), shape=None)

    log_det_jacobian = bijector.forward_log_det_jacobian(x, event_ndims=2)

    # The Jacobian matrix is 2 * tf.eye(2), which has jacobian determinant 4.
    self.assertAllClose(
        np.log(4), self.evaluate(log_det_jacobian))

  def testNoBatchStatic(self):
    x = np.array([[1., 0], [2, 1]])  # np.linalg.cholesky(y)
    y = np.array([[1., 2], [2, 5]])  # np.matmul(x, x.T)
    y_actual = tfb.CholeskyOuterProduct().forward(x=x)
    x_actual = tfb.CholeskyOuterProduct().inverse(y=y)
    [y_actual_, x_actual_] = self.evaluate([y_actual, x_actual])
    self.assertAllEqual([2, 2], y_actual.shape)
    self.assertAllEqual([2, 2], x_actual.shape)
    self.assertAllClose(y, y_actual_)
    self.assertAllClose(x, x_actual_)

  def testNoBatchDeferred(self):
    x_ = np.array([[1., 0], [2, 1]])  # np.linalg.cholesky(y)
    y_ = np.array([[1., 2], [2, 5]])  # np.matmul(x, x.T)
    x = tf1.placeholder_with_default(x_, shape=None)
    y = tf1.placeholder_with_default(y_, shape=None)
    y_actual = tfb.CholeskyOuterProduct().forward(x=x)
    x_actual = tfb.CholeskyOuterProduct().inverse(y=y)
    [y_actual_, x_actual_] = self.evaluate([y_actual, x_actual])
    # Shapes are always known in eager.
    if not tf.executing_eagerly():
      self.assertEqual(None, y_actual.shape)
      self.assertEqual(None, x_actual.shape)
    self.assertAllClose(y_, y_actual_)
    self.assertAllClose(x_, x_actual_)

  def testBatchStatic(self):
    x = np.array([[[1., 0],
                   [2, 1]],
                  [[3., 0],
                   [1, 2]]])  # np.linalg.cholesky(y)
    y = np.array([[[1., 2],
                   [2, 5]],
                  [[9., 3],
                   [3, 5]]])  # np.matmul(x, x.T)
    y_actual = tfb.CholeskyOuterProduct().forward(x=x)
    x_actual = tfb.CholeskyOuterProduct().inverse(y=y)
    [y_actual_, x_actual_] = self.evaluate([y_actual, x_actual])
    self.assertEqual([2, 2, 2], y_actual.shape)
    self.assertEqual([2, 2, 2], x_actual.shape)
    self.assertAllClose(y, y_actual_)
    self.assertAllClose(x, x_actual_)

  def testBatchDeferred(self):
    x_ = np.array([[[1., 0],
                    [2, 1]],
                   [[3., 0],
                    [1, 2]]])  # np.linalg.cholesky(y)
    y_ = np.array([[[1., 2],
                    [2, 5]],
                   [[9., 3],
                    [3, 5]]])  # np.matmul(x, x.T)
    x = tf1.placeholder_with_default(x_, shape=None)
    y = tf1.placeholder_with_default(y_, shape=None)
    y_actual = tfb.CholeskyOuterProduct().forward(x)
    x_actual = tfb.CholeskyOuterProduct().inverse(y)
    [y_actual_, x_actual_] = self.evaluate([y_actual, x_actual])

    # Shapes are always known in eager.
    if not tf.executing_eagerly():
      self.assertEqual(None, y_actual.shape)
      self.assertEqual(None, x_actual.shape)
    self.assertAllClose(y_, y_actual_)
    self.assertAllClose(x_, x_actual_)


if __name__ == "__main__":
  tf.test.main()
