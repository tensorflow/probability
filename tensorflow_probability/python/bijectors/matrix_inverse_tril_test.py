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
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow.python.framework import test_util


@test_util.run_all_in_graph_and_eager_modes
class MatrixInverseTriLBijectorTest(tf.test.TestCase):
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
    self.assertEqual("matrix_inverse_tril", inv.name)
    x_ = np.array([[0.7, 0., 0.],
                   [0.1, -1., 0.],
                   [0.3, 0.25, 0.5]], dtype=np.float32)
    x_inv_ = np.linalg.inv(x_)
    expected_fldj_ = -6. * np.sum(np.log(np.abs(np.diag(x_))))

    y = inv.forward(x_)
    x_back = inv.inverse(x_inv_)
    fldj = inv.forward_log_det_jacobian(x_, event_ndims=2)
    ildj = inv.inverse_log_det_jacobian(x_inv_, event_ndims=2)

    y_, x_back_, fldj_, ildj_ = self.evaluate([y, x_back, fldj, ildj])

    self.assertAllClose(x_inv_, y_, atol=0., rtol=1e-5)
    self.assertAllClose(x_, x_back_, atol=0., rtol=1e-5)
    self.assertNear(expected_fldj_, fldj_, err=1e-3)
    self.assertNear(-expected_fldj_, ildj_, err=1e-3)

  def testOneByOneMatrix(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.array([[5.]], dtype=np.float32)
    x_inv_ = np.array([[0.2]], dtype=np.float32)
    expected_fldj_ = np.log(0.04)

    y = inv.forward(x_)
    x_back = inv.inverse(x_inv_)
    fldj = inv.forward_log_det_jacobian(x_, event_ndims=2)
    ildj = inv.inverse_log_det_jacobian(x_inv_, event_ndims=2)

    y_, x_back_, fldj_, ildj_ = self.evaluate([y, x_back, fldj, ildj])

    self.assertAllClose(x_inv_, y_, atol=0., rtol=1e-5)
    self.assertAllClose(x_, x_back_, atol=0., rtol=1e-5)
    self.assertNear(expected_fldj_, fldj_, err=1e-3)
    self.assertNear(-expected_fldj_, ildj_, err=1e-3)

  def testZeroByZeroMatrix(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.eye(0, dtype=np.float32)
    x_inv_ = np.eye(0, dtype=np.float32)
    expected_fldj_ = 0.

    y = inv.forward(x_)
    x_back = inv.inverse(x_inv_)
    fldj = inv.forward_log_det_jacobian(x_, event_ndims=2)
    ildj = inv.inverse_log_det_jacobian(x_inv_, event_ndims=2)

    y_, x_back_, fldj_, ildj_ = self.evaluate([y, x_back, fldj, ildj])

    self.assertAllClose(x_inv_, y_, atol=0., rtol=1e-5)
    self.assertAllClose(x_, x_back_, atol=0., rtol=1e-5)
    self.assertNear(expected_fldj_, fldj_, err=1e-3)
    self.assertNear(-expected_fldj_, ildj_, err=1e-3)

  def testBatch(self):
    # Test batch computation with input shape (2, 1, 2, 2), i.e. batch shape
    # (2, 1).
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.array([[[[1., 0.],
                     [2., 3.]]],
                   [[[4., 0.],
                     [5., -6.]]]], dtype=np.float32)
    x_inv_ = self._inv(x_)
    expected_fldj_ = -4. * np.sum(
        np.log(np.abs(np.diagonal(x_, axis1=-2, axis2=-1))), axis=-1)

    y = inv.forward(x_)
    x_back = inv.inverse(x_inv_)
    fldj = inv.forward_log_det_jacobian(x_, event_ndims=2)
    ildj = inv.inverse_log_det_jacobian(x_inv_, event_ndims=2)

    y_, x_back_, fldj_, ildj_ = self.evaluate([y, x_back, fldj, ildj])

    self.assertAllClose(x_inv_, y_, atol=0., rtol=1e-5)
    self.assertAllClose(x_, x_back_, atol=0., rtol=1e-5)
    self.assertAllClose(expected_fldj_, fldj_, atol=0., rtol=1e-3)
    self.assertAllClose(-expected_fldj_, ildj_, atol=0., rtol=1e-3)

  def testErrorOnInputRankTooLow(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.array([0.1], dtype=np.float32)
    rank_error_msg = "must have rank at least 2"
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
  ##   square_error_msg = "must be a square matrix"
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
    triangular_error_msg = "must be lower triangular"
    with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError,
                                             triangular_error_msg):
      self.evaluate(inv.forward(x_))
    with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError,
                                             triangular_error_msg):
      self.evaluate(inv.inverse(x_))
    with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError,
                                             triangular_error_msg):
      self.evaluate(inv.forward_log_det_jacobian(x_, event_ndims=2))
    with self.assertRaisesWithPredicateMatch(tf.errors.InvalidArgumentError,
                                             triangular_error_msg):
      self.evaluate(inv.inverse_log_det_jacobian(x_, event_ndims=2))

  def testErrorOnInputSingular(self):
    inv = tfb.MatrixInverseTriL(validate_args=True)
    x_ = np.array([[1., 0.],
                   [0., 0.]], dtype=np.float32)
    nonsingular_error_msg = "must have all diagonal entries nonzero"
    with self.assertRaisesOpError(nonsingular_error_msg):
      self.evaluate(inv.forward(x_))
    with self.assertRaisesOpError(nonsingular_error_msg):
      self.evaluate(inv.inverse(x_))
    with self.assertRaisesOpError(nonsingular_error_msg):
      self.evaluate(inv.forward_log_det_jacobian(x_, event_ndims=2))
    with self.assertRaisesOpError(nonsingular_error_msg):
      self.evaluate(inv.inverse_log_det_jacobian(x_, event_ndims=2))


if __name__ == "__main__":
  tf.test.main()
