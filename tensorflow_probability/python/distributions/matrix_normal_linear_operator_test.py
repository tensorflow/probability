# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for MatrixNormalLinearOperator."""

# Dependency imports

import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class _MatrixNormalTest(object):

  def _random_tril_matrix(self, shape, seed):
    mat = tf.random.normal(shape=shape, seed=seed, dtype=self.dtype)
    chol = tfb.TransformDiagonal(
        tfb.Shift(shift=self.dtype(1.))(tfb.Softplus()))(mat)
    return tf.linalg.band_part(chol, -1, 0)

  def _random_loc_and_scale(
      self, batch_shape, matrix_shape, seed_stream=None):
    # This ensures covariance is positive def.
    seed_stream = (
        test_util.test_seed_stream() if seed_stream is None else seed_stream)
    row_shape = batch_shape + [matrix_shape[0], matrix_shape[0]]
    col_shape = batch_shape + [matrix_shape[1], matrix_shape[1]]
    scale_row = tf.linalg.LinearOperatorLowerTriangular(
        self._random_tril_matrix(row_shape, seed_stream()),
        is_non_singular=True)
    scale_col = tf.linalg.LinearOperatorLowerTriangular(
        self._random_tril_matrix(col_shape, seed_stream()),
        is_non_singular=True)
    loc = tf.math.abs(tf.random.normal(
        shape=batch_shape + matrix_shape,
        stddev=2.,
        seed=seed_stream(),
        dtype=self.dtype)) + 1.
    return loc, scale_row, scale_col

  def testLogPDFScalarBatch(self):
    seed_stream = test_util.test_seed_stream()
    loc, scale_row, scale_col = self._random_loc_and_scale(
        [], [2, 3], seed_stream)
    matrix_normal = tfd.MatrixNormalLinearOperator(
        loc, scale_row, scale_col, validate_args=True)
    x = tf.random.normal(shape=[2, 3], seed=seed_stream(), dtype=self.dtype)

    log_pdf = matrix_normal.log_prob(x)
    pdf = matrix_normal.prob(x)

    loc_, row_cov_, col_cov_, log_pdf_, pdf_, x_ = self.evaluate(
        [loc,
         tf.linalg.matmul(
             scale_row.to_dense(), scale_row.to_dense(), adjoint_b=True),
         tf.linalg.matmul(
             scale_col.to_dense(), scale_col.to_dense(), adjoint_b=True),
         log_pdf, pdf, x])
    scipy_matrix_normal = stats.matrix_normal(
        mean=loc_, rowcov=row_cov_, colcov=col_cov_)

    self.assertEqual((), log_pdf_.shape)
    self.assertEqual((), pdf_.shape)
    self.assertAllClose(scipy_matrix_normal.logpdf(x_), log_pdf_)
    self.assertAllClose(scipy_matrix_normal.pdf(x_), pdf_)

  def testLogPDFBatch(self):
    seed_stream = test_util.test_seed_stream()
    loc, scale_row, scale_col = self._random_loc_and_scale(
        batch_shape=[3], matrix_shape=[3, 5], seed_stream=seed_stream)
    matrix_normal = tfd.MatrixNormalLinearOperator(
        loc, scale_row, scale_col, validate_args=True)
    x = tf.random.normal(shape=[3, 3, 5], seed=seed_stream(), dtype=self.dtype)

    log_pdf = matrix_normal.log_prob(x)
    pdf = matrix_normal.prob(x)

    loc_, row_cov_, col_cov_, log_pdf_, pdf_, x_ = self.evaluate(
        [loc,
         tf.linalg.matmul(
             scale_row.to_dense(), scale_row.to_dense(), adjoint_b=True),
         tf.linalg.matmul(
             scale_col.to_dense(), scale_col.to_dense(), adjoint_b=True),
         log_pdf, pdf, x])
    self.assertEqual((3,), log_pdf_.shape)
    self.assertEqual((3,), pdf_.shape)
    for i in range(3):
      scipy_matrix_normal = stats.matrix_normal(
          mean=loc_[i], rowcov=row_cov_[i], colcov=col_cov_[i])

      expected_log_pdf = scipy_matrix_normal.logpdf(x_[i])
      expected_pdf = scipy_matrix_normal.pdf(x_[i])
      self.assertAllClose(expected_log_pdf, log_pdf_[i], rtol=3e-5)
      self.assertAllClose(expected_pdf, pdf_[i], rtol=3e-5)

  def testShapes(self):
    seed_stream = test_util.test_seed_stream()
    loc = tf.random.normal(
        shape=[3, 2, 3], seed=seed_stream(), dtype=self.dtype)
    scale_row = tf.linalg.LinearOperatorLowerTriangular(
        self._random_tril_matrix([5, 1, 2, 2], seed_stream()),
        is_non_singular=True)
    scale_col = tf.linalg.LinearOperatorLowerTriangular(
        self._random_tril_matrix([7, 1, 1, 3, 3], seed_stream()),
        is_non_singular=True)

    matrix_normal = tfd.MatrixNormalLinearOperator(
        loc, scale_row, scale_col, validate_args=True)

    self.assertAllEqual((2, 3), matrix_normal.event_shape)
    self.assertAllEqual((7, 5, 3), matrix_normal.batch_shape)

    self.assertAllEqual(
        (2, 3), self.evaluate(matrix_normal.event_shape_tensor()))
    self.assertAllEqual(
        (7, 5, 3), self.evaluate(matrix_normal.batch_shape_tensor()))

  def testMeanAndVariance(self):
    loc, scale_row, scale_col = self._random_loc_and_scale(
        batch_shape=[3, 4], matrix_shape=[2, 5])
    matrix_normal = tfd.MatrixNormalLinearOperator(loc, scale_row, scale_col)

    cov_row = scale_row.matmul(scale_row.adjoint())
    cov_col = scale_col.matmul(scale_col.adjoint())
    # Compute diagonal of Kronecker product
    expected_variance = (cov_col.to_dense()[..., :, tf.newaxis, :, tf.newaxis] *
                         cov_row.to_dense()[..., tf.newaxis, :, tf.newaxis, :])
    expected_variance = tf.linalg.diag_part(
        tf.reshape(expected_variance, [3, 4, 10, 10]))
    expected_variance = tf.linalg.matrix_transpose(
        tf.reshape(expected_variance, [3, 4, 5, 2]))
    mean_, loc_, variance_, expected_variance_ = self.evaluate([
        matrix_normal.mean(), loc, matrix_normal.variance(), expected_variance])
    self.assertAllClose(loc_, mean_)
    self.assertAllClose(expected_variance_, variance_)

  def testSampleMean(self):
    seed_stream = test_util.test_seed_stream()
    loc, scale_row, scale_col = self._random_loc_and_scale(
        batch_shape=[5, 2], matrix_shape=[2, 3], seed_stream=seed_stream)
    matrix_normal = tfd.MatrixNormalLinearOperator(loc, scale_row, scale_col)
    samples = matrix_normal.sample(int(1e6), seed=seed_stream())
    mean_, samples_ = self.evaluate([matrix_normal.mean(), samples])
    self.assertAllClose(mean_, np.mean(samples_, axis=0), rtol=1e-2)

  def testSampleVariance(self):
    seed_stream = test_util.test_seed_stream()
    loc, scale_row, scale_col = self._random_loc_and_scale(
        batch_shape=[5, 2], matrix_shape=[2, 3], seed_stream=seed_stream)
    matrix_normal = tfd.MatrixNormalLinearOperator(loc, scale_row, scale_col)
    samples = matrix_normal.sample(int(2e6), seed=seed_stream())
    variance_, samples_ = self.evaluate([matrix_normal.variance(), samples])
    self.assertAllClose(variance_, np.var(samples_, axis=0), rtol=1e-2)

  @test_util.tf_tape_safety_test
  def testVariableLocation(self):
    loc = tf.Variable([[1., 1.]])
    scale_row = tf.linalg.LinearOperatorLowerTriangular(
        tf.eye(1), is_non_singular=True)
    scale_column = tf.linalg.LinearOperatorLowerTriangular(
        tf.eye(2), is_non_singular=True)
    d = tfd.MatrixNormalLinearOperator(
        loc, scale_row, scale_column, validate_args=True)
    self.evaluate(loc.initializer)
    with tf.GradientTape() as tape:
      lp = d.log_prob([[0., 0.]])
    self.assertIsNotNone(tape.gradient(lp, loc))

  @test_util.jax_disable_variable_test
  def testVariableScaleAssertions(self):
    loc = tf.constant([[1., 1.]])
    scale_tensor = tf.Variable(tf.eye(2, dtype=np.float32))
    scale_row = tf.linalg.LinearOperatorLowerTriangular(
        tf.eye(1), is_non_singular=True)
    scale_column = tf.linalg.LinearOperatorLowerTriangular(
        scale_tensor,
        is_non_singular=True)
    d = tfd.MatrixNormalLinearOperator(
        loc, scale_row, scale_column, validate_args=True)
    self.evaluate(scale_tensor.initializer)
    with self.assertRaises(Exception):
      with tf.control_dependencies([scale_tensor.assign([[1., 0.], [1., 0.]])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))


@test_util.test_all_tf_execution_regimes
class MatrixNormalTestFloat32Test(test_util.TestCase, _MatrixNormalTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class MatrixNormalFloat64Test(test_util.TestCase, _MatrixNormalTest):
  dtype = np.float64


if __name__ == '__main__':
  test_util.main()
