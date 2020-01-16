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
"""AffineLinearOperator Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class AffineLinearOperatorTest(test_util.TestCase):

  def testIdentity(self):
    affine = tfb.AffineLinearOperator(validate_args=True)
    x = np.array([[1, 0, -1], [2, 3, 4]], dtype=np.float32)
    y = x
    ildj = 0.

    self.assertStartsWith(affine.name, "affine_linear_operator")
    self.assertAllClose(y, self.evaluate(affine.forward(x)))
    self.assertAllClose(x, self.evaluate(affine.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(
            affine.inverse_log_det_jacobian(
                y, event_ndims=2)))
    self.assertAllClose(
        self.evaluate(-affine.inverse_log_det_jacobian(y, event_ndims=2)),
        self.evaluate(affine.forward_log_det_jacobian(x, event_ndims=2)))

  def testDiag(self):
    shift = np.array([-1, 0, 1], dtype=np.float32)
    diag = np.array([[1, 2, 3],
                     [2, 5, 6]], dtype=np.float32)
    scale = tf.linalg.LinearOperatorDiag(diag, is_non_singular=True)
    affine = tfb.AffineLinearOperator(
        shift=shift, scale=scale, validate_args=True)

    x = np.array([[1, 0, -1], [2, 3, 4]], dtype=np.float32)
    y = diag * x + shift
    ildj = -np.sum(np.log(np.abs(diag)), axis=-1)

    self.assertStartsWith(affine.name, "affine_linear_operator")
    self.assertAllClose(y, self.evaluate(affine.forward(x)))
    self.assertAllClose(x, self.evaluate(affine.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(affine.inverse_log_det_jacobian(y, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(-affine.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(affine.forward_log_det_jacobian(x, event_ndims=1)))

  def testTriL(self):
    shift = np.array([-1, 0, 1], dtype=np.float32)
    tril = np.array([[[3, 0, 0],
                      [2, -1, 0],
                      [3, 2, 1]],
                     [[2, 0, 0],
                      [3, -2, 0],
                      [4, 3, 2]]],
                    dtype=np.float32)
    scale = tf.linalg.LinearOperatorLowerTriangular(
        tril, is_non_singular=True)
    affine = tfb.AffineLinearOperator(
        shift=shift, scale=scale, validate_args=True)

    x = np.array([[[1, 0, -1],
                   [2, 3, 4]],
                  [[4, 1, -7],
                   [6, 9, 8]]],
                 dtype=np.float32)
    # If we made the bijector do x*A+b then this would be simplified to:
    # y = np.matmul(x, tril) + shift.
    y = np.squeeze(np.matmul(tril, np.expand_dims(x, -1)), -1) + shift
    ildj = -np.sum(np.log(np.abs(np.diagonal(
        tril, axis1=-2, axis2=-1))))

    self.assertStartsWith(affine.name, "affine_linear_operator")
    self.assertAllClose(y, self.evaluate(affine.forward(x)))
    self.assertAllClose(x, self.evaluate(affine.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(
            affine.inverse_log_det_jacobian(
                y, event_ndims=2)))
    self.assertAllClose(
        self.evaluate(-affine.inverse_log_det_jacobian(y, event_ndims=2)),
        self.evaluate(affine.forward_log_det_jacobian(x, event_ndims=2)))

  def testTriLAdjoint(self):
    shift = np.array([-1, 0, 1], dtype=np.float32)
    tril = np.array([[[3, 0, 0],
                      [2, -1, 0],
                      [3, 2, 1]],
                     [[2, 0, 0],
                      [3, -2, 0],
                      [4, 3, 2]]],
                    dtype=np.float32)
    scale = tf.linalg.LinearOperatorLowerTriangular(
        tril, is_non_singular=True)
    affine = tfb.AffineLinearOperator(
        shift=shift, scale=scale, adjoint=True, validate_args=True)

    x = np.array([[[1, 0, -1],
                   [2, 3, 4]],
                  [[4, 1, -7],
                   [6, 9, 8]]],
                 dtype=np.float32)
    # If we made the bijector do x*A+b then this would be simplified to:
    # y = np.matmul(x, tril) + shift.
    triu = tril.transpose([0, 2, 1])
    y = np.matmul(triu, x[..., np.newaxis])[..., 0] + shift
    ildj = -np.sum(np.log(np.abs(np.diagonal(
        tril, axis1=-2, axis2=-1))))

    self.assertStartsWith(affine.name, "affine_linear_operator")
    self.assertAllClose(y, self.evaluate(affine.forward(x)))
    self.assertAllClose(x, self.evaluate(affine.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(
            affine.inverse_log_det_jacobian(
                y, event_ndims=2)))
    self.assertAllClose(
        self.evaluate(-affine.inverse_log_det_jacobian(y, event_ndims=2)),
        self.evaluate(affine.forward_log_det_jacobian(x, event_ndims=2)))

  @test_util.jax_disable_variable_test
  def testVariableGradient(self):
    b = tfb.AffineLinearOperator(shift=tf.Variable(-1.))

    with tf.GradientTape() as tape:
      y = b.forward(.1)
    self.assertIsNotNone(tape.gradient(y, b.shift))

if __name__ == "__main__":
  tf.test.main()
