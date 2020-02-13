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
"""Tests for MatrixInverseTriL bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class MatrixInverseTriLBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = inv(tril) transformation."""

  # The inverse of 0 is undefined, as the numbers above the main
  # diagonal must be zero, we zero out these numbers after running inverse.
  # See: https://github.com/numpy/numpy/issues/11445
  def _inv(self, x):
    y = np.linalg.inv(x)
    # Since triu_indices only works on 2d arrays we need to iterate over all the
    # 2d arrays in a x-dimensional array.
    for idx in np.ndindex(y.shape[0:-2]):
      y[idx][np.triu_indices(y[idx].shape[-1], 1)] = 0
    return y

  def testComputesCorrectValues(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    self.assertStartsWith(inv.name, 'matrix_inverse_tril')
    x_ = np.array([[0.7, 0., 0.],
                   [0.1, -1., 0.],
                   [0.3, 0.25, 0.5]], dtype=np.float32)
    x_inv_ = np.linalg.inv(x_)

    y = inv.forward(x_)
    x_back = inv.inverse(x_inv_)

    y_, x_back_ = self.evaluate([y, x_back])

    self.assertAllClose(x_inv_, y_, atol=0., rtol=1e-5)
    self.assertAllClose(x_, x_back_, atol=0., rtol=1e-5)

  def testOneByOneMatrix(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.array([[5.]], dtype=np.float32)
    x_inv_ = np.array([[0.2]], dtype=np.float32)

    y = inv.forward(x_)
    x_back = inv.inverse(x_inv_)

    y_, x_back_ = self.evaluate([y, x_back])

    self.assertAllClose(x_inv_, y_, atol=0., rtol=1e-5)
    self.assertAllClose(x_, x_back_, atol=0., rtol=1e-5)

  def testZeroByZeroMatrix(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.eye(0, dtype=np.float32)
    x_inv_ = np.eye(0, dtype=np.float32)

    y = inv.forward(x_)
    x_back = inv.inverse(x_inv_)

    y_, x_back_ = self.evaluate([y, x_back])

    self.assertAllClose(x_inv_, y_, atol=0., rtol=1e-5)
    self.assertAllClose(x_, x_back_, atol=0., rtol=1e-5)

  def testBatch(self):
    # Test batch computation with input shape (2, 1, 2, 2), i.e. batch shape
    # (2, 1).
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.array([[[[1., 0.],
                     [2., 3.]]],
                   [[[4., 0.],
                     [5., -6.]]]], dtype=np.float32)
    x_inv_ = self._inv(x_)

    y = inv.forward(x_)
    x_back = inv.inverse(x_inv_)

    y_, x_back_ = self.evaluate([y, x_back])

    self.assertAllClose(x_inv_, y_, atol=0., rtol=1e-5)
    self.assertAllClose(x_, x_back_, atol=0., rtol=1e-5)

  def testErrorOnInputRankTooLow(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.array([0.1], dtype=np.float32)
    rank_error_msg = 'must have rank at least 2'
    with self.assertRaisesWithPredicateMatch(ValueError, rank_error_msg):
      self.evaluate(inv.forward(x_))
    with self.assertRaisesWithPredicateMatch(ValueError, rank_error_msg):
      self.evaluate(inv.inverse(x_))
    with self.assertRaisesWithPredicateMatch(ValueError, rank_error_msg):
      self.evaluate(inv.forward_log_det_jacobian(x_, event_ndims=2))
    with self.assertRaisesWithPredicateMatch(ValueError, rank_error_msg):
      self.evaluate(inv.inverse_log_det_jacobian(x_, event_ndims=2))

  # TODO(b/80481923): Figure out why these assertions fail, and fix them.
  ## def testErrorOnInputNonSquare(self):
  ##   inv = tfb.MatrixInverseTriL(validate_args=True)
  ##   x_ = np.array([[1., 2., 3.],
  ##                  [4., 5., 6.]], dtype=np.float32)
  ##   square_error_msg = 'must be a square matrix'
  ##   with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError,
  ##                                            square_error_msg):
  ##     self.evaluate(inv.forward(x_))
  ##   with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError,
  ##                                            square_error_msg):
  ##     self.evaluate(inv.inverse(x_))
  ##   with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError,
  ##                                            square_error_msg):
  ##     self.evaluate(inv.forward_log_det_jacobian(x_, event_ndims=2))
  ##   with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError,
  ##                                            square_error_msg):
  ##     self.evaluate(inv.inverse_log_det_jacobian(x_, event_ndims=2))

  def testErrorOnInputNotLowerTriangular(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.array([[1., 2.],
                   [3., 4.]], dtype=np.float32)
    triangular_error_msg = 'must be lower triangular'
    with self.assertRaisesOpError(triangular_error_msg):
      self.evaluate(inv.forward(x_))
    with self.assertRaisesOpError(triangular_error_msg):
      self.evaluate(inv.inverse(x_))
    with self.assertRaisesOpError(triangular_error_msg):
      self.evaluate(inv.forward_log_det_jacobian(x_, event_ndims=2))
    with self.assertRaisesOpError(triangular_error_msg):
      self.evaluate(inv.inverse_log_det_jacobian(x_, event_ndims=2))

  def testErrorOnInputSingular(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.array([[1., 0.],
                   [0., 0.]], dtype=np.float32)
    nonsingular_error_msg = 'must have all diagonal entries nonzero'
    with self.assertRaisesOpError(nonsingular_error_msg):
      self.evaluate(inv.forward(x_))
    with self.assertRaisesOpError(nonsingular_error_msg):
      self.evaluate(inv.inverse(x_))
    with self.assertRaisesOpError(nonsingular_error_msg):
      self.evaluate(inv.forward_log_det_jacobian(x_, event_ndims=2))
    with self.assertRaisesOpError(nonsingular_error_msg):
      self.evaluate(inv.inverse_log_det_jacobian(x_, event_ndims=2))

  @test_util.numpy_disable_gradient_test
  def testJacobian(self):
    bijector = tfb.MatrixInverseTriL()
    batch_size = 5
    for ndims in range(2, 5):
      x_ = np.tril(
          np.random.uniform(
              -1., 1., size=[batch_size, ndims, ndims]).astype(np.float64))
      fldj = bijector.forward_log_det_jacobian(x_, event_ndims=2)
      fldj_theoretical = bijector_test_util.get_fldj_theoretical(
          bijector, x_, event_ndims=2,
          input_to_unconstrained=tfb.Invert(tfb.FillTriangular()),
          output_to_unconstrained=tfb.Invert(tfb.FillTriangular()))
      fldj_, fldj_theoretical_ = self.evaluate([fldj, fldj_theoretical])
      self.assertAllClose(fldj_, fldj_theoretical_)


if __name__ == '__main__':
  tf.test.main()
