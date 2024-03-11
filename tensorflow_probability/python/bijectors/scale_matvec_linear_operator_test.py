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
"""ScaleMatvecLinearOperator Tests."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import scale_matvec_linear_operator
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ScaleMatvecLinearOperatorTest(test_util.TestCase):

  def testDiag(self):
    diag = np.array([[1, 2, 3],
                     [2, 5, 6]], dtype=np.float32)
    scale = tf.linalg.LinearOperatorDiag(diag, is_non_singular=True)
    bijector = scale_matvec_linear_operator.ScaleMatvecLinearOperator(
        scale=scale, validate_args=True)

    x = np.array([[1, 0, -1], [2, 3, 4]], dtype=np.float32)
    y = diag * x
    ildj = -np.sum(np.log(np.abs(diag)), axis=-1)

    self.assertStartsWith(bijector.name, 'scale_matvec_linear_operator')
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=1)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))

  def testTriL(self):
    tril = np.array([[[3, 0, 0],
                      [2, -1, 0],
                      [3, 2, 1]],
                     [[2, 0, 0],
                      [3, -2, 0],
                      [4, 3, 2]]],
                    dtype=np.float32)
    scale = tf.linalg.LinearOperatorLowerTriangular(
        tril, is_non_singular=True)
    bijector = scale_matvec_linear_operator.ScaleMatvecLinearOperator(
        scale=scale, validate_args=True)

    x = np.array([[[1, 0, -1],
                   [2, 3, 4]],
                  [[4, 1, -7],
                   [6, 9, 8]]],
                 dtype=np.float32)
    # If we made the bijector do x*A+b then this would be simplified to:
    # y = np.matmul(x, tril).
    y = np.squeeze(np.matmul(tril, np.expand_dims(x, -1)), -1)
    ildj = -np.sum(np.log(np.abs(np.diagonal(
        tril, axis1=-2, axis2=-1))))

    self.assertStartsWith(bijector.name, 'scale_matvec_linear_operator')
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(
            bijector.inverse_log_det_jacobian(
                y, event_ndims=2)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=2)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=2)))

  def testTriLAdjoint(self):
    tril = np.array([[[3, 0, 0],
                      [2, -1, 0],
                      [3, 2, 1]],
                     [[2, 0, 0],
                      [3, -2, 0],
                      [4, 3, 2]]],
                    dtype=np.float32)
    scale = tf.linalg.LinearOperatorLowerTriangular(
        tril, is_non_singular=True)
    bijector = scale_matvec_linear_operator.ScaleMatvecLinearOperator(
        scale=scale, adjoint=True, validate_args=True)

    x = np.array([[[1, 0, -1],
                   [2, 3, 4]],
                  [[4, 1, -7],
                   [6, 9, 8]]],
                 dtype=np.float32)
    # If we made the bijector do x*A+b then this would be simplified to:
    # y = np.matmul(x, tril).
    triu = tril.transpose([0, 2, 1])
    y = np.matmul(triu, x[..., np.newaxis])[..., 0]
    ildj = -np.sum(np.log(np.abs(np.diagonal(
        tril, axis1=-2, axis2=-1))))

    self.assertStartsWith(bijector.name, 'scale_matvec_linear_operator')
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        ildj,
        self.evaluate(
            bijector.inverse_log_det_jacobian(
                y, event_ndims=2)))
    self.assertAllClose(
        self.evaluate(-bijector.inverse_log_det_jacobian(y, event_ndims=2)),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=2)))


class _ScaleMatvecLinearOperatorBlockTest(object):

  def testBijector(self):
    x = [np.array([4., 3., 3.]).astype(np.float32),
         np.array([0., -5.]).astype(np.float32)]
    op = self.build_operator()
    y = self.evaluate(op.matvec(x))
    ldj = self.evaluate(op.log_abs_determinant())

    bijector = scale_matvec_linear_operator.ScaleMatvecLinearOperatorBlock(
        scale=op, validate_args=True)
    self.assertStartsWith(bijector.name, 'scale_matvec_linear_operator_block')

    f_x = bijector.forward(x)
    self.assertAllClose(y, self.evaluate(f_x))

    inv_y = self.evaluate(bijector.inverse(y))
    self.assertAllClose(x, inv_y)

    # Calling `inverse` on an output of `bijector.forward` (that is equal to
    # `y`) is a cache hit and returns the original, non-broadcasted input `x`.
    for x_, z_ in zip(x, bijector.inverse(f_x)):
      self.assertIs(x_, z_)

    ldj_ = self.evaluate(
        bijector.forward_log_det_jacobian(x, event_ndims=[1, 1]))
    self.assertAllClose(ldj, ldj_)
    self.assertEmpty(ldj_.shape)

    self.assertAllClose(
        ldj_,
        self.evaluate(
            -bijector.inverse_log_det_jacobian(y, event_ndims=[1, 1])))

  def testOperatorBroadcast(self):
    x = [tf.ones((1, 1, 1, 4), dtype=tf.float32),
         tf.ones((1, 1, 1, 3), dtype=tf.float32)]
    op = self.build_batched_operator()
    bijector = scale_matvec_linear_operator.ScaleMatvecLinearOperatorBlock(
        op, validate_args=True)

    self.assertAllEqual(
        self.evaluate(tf.shape(bijector.forward_log_det_jacobian(x, [1, 1]))),
        self.evaluate(op.batch_shape_tensor()))

    # Broadcasting of event shape components with batched LinearOperators
    # raises.
    with self.assertRaisesRegex(ValueError, 'bijector parameters changes'):
      self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=[2, 2]))

    # Broadcasting of event shape components with batched LinearOperators
    # raises for `ldj_reduce_ndims > batch_ndims`.
    with self.assertRaisesRegex(ValueError, 'bijector parameters changes'):
      self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=[3, 3]))

  def testEventShapeBroadcast(self):
    op = self.build_operator()
    bijector = scale_matvec_linear_operator.ScaleMatvecLinearOperatorBlock(
        op, validate_args=True)
    x = [tf.broadcast_to(tf.constant(1., dtype=tf.float32), [2, 3, 3]),
         tf.broadcast_to(tf.constant(2., dtype=tf.float32), [2, 1, 2])]

    # Forward/inverse event shape methods return the correct value.
    self.assertAllEqual(
        self.evaluate(bijector.forward_event_shape_tensor(
            [tf.shape(x_) for x_ in x])),
        [self.evaluate(tf.shape(y_)) for y_ in bijector.forward(x)])
    self.assertAllEqual(
        bijector.inverse_event_shape([x_.shape for x_ in x]),
        [y_.shape for y_ in bijector.inverse(x)])

    # Broadcasting of inputs within `ldj_reduce_shape` raises.
    with self.assertRaisesRegex(ValueError, 'left of `min_event_ndims`'):
      self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=[2, 2]))

  def testAlignedEventDims(self):
    x = [tf.ones((3,), dtype=tf.float32), tf.ones((2, 2), tf.float32)]
    op = self.build_operator()
    bijector = scale_matvec_linear_operator.ScaleMatvecLinearOperatorBlock(
        op, validate_args=True)
    with self.assertRaisesRegex(ValueError, 'equal for all elements'):
      self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=[1, 2]))


@test_util.test_all_tf_execution_regimes
class ScaleMatvecLinearOperatorBlockDiagTest(
    test_util.TestCase, _ScaleMatvecLinearOperatorBlockTest):

  def build_operator(self):
    return tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorDiag(diag=[2., 3., 6.]),
         tf.linalg.LinearOperatorFullMatrix(matrix=[[12., 5.], [-1., 3.]])],
        is_non_singular=True)

  def build_batched_operator(self):
    seed = test_util.test_seed()
    return tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorDiag(
            tf.random.normal((2, 3, 4), dtype=tf.float32, seed=seed)),
         tf.linalg.LinearOperatorIdentity(3)], is_non_singular=True)


@test_util.test_all_tf_execution_regimes
class ScaleMatvecLinearOperatorBlockTrilTest(
    test_util.TestCase, _ScaleMatvecLinearOperatorBlockTest):

  def build_operator(self):
    return tf.linalg.LinearOperatorBlockLowerTriangular([
        [tf.linalg.LinearOperatorDiag(diag=[2., 3., 6.], is_non_singular=True)],
        [tf.linalg.LinearOperatorFullMatrix(
            matrix=[[12., 5., -1.], [3., 0., 1.]]),
         tf.linalg.LinearOperatorIdentity(2)]], is_non_singular=True)

  def build_batched_operator(self):
    seed1 = test_util.test_seed()
    seed2 = test_util.test_seed()
    return tf.linalg.LinearOperatorBlockLowerTriangular([
        [tf.linalg.LinearOperatorFullMatrix(
            tf.random.normal((3, 4, 4), dtype=tf.float32, seed=seed1),
            is_non_singular=True)],
        [tf.linalg.LinearOperatorZeros(
            3, 4, is_square=False, is_self_adjoint=False),
         tf.linalg.LinearOperatorFullMatrix(
             tf.random.normal((3, 3), dtype=tf.float32, seed=seed2),
             is_non_singular=True)]
    ], is_non_singular=True)

if __name__ == '__main__':
  test_util.main()
