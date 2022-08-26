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
"""Tests for MatrixTLinearOperator."""

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.bijectors import transform_diagonal
from tensorflow_probability.python.distributions import matrix_t_linear_operator as mtlo
from tensorflow_probability.python.distributions import multivariate_student_t as mst
from tensorflow_probability.python.internal import test_util


def _vec(x):
  return tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], axis=0))


@test_util.test_all_tf_execution_regimes
class _MatrixTTest(object):

  def _random_tril_matrix(self, shape, seed):
    mat = tf.random.normal(shape=shape, seed=seed, dtype=self.dtype)
    chol = transform_diagonal.TransformDiagonal(
        shift.Shift(shift=self.dtype(1.))(softplus.Softplus()))(
            mat)
    return tf.linalg.band_part(chol, -1, 0)

  def _random_df_loc_and_scale(
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
    df = tf.math.abs(tf.random.normal(
        shape=batch_shape, seed=seed_stream(), dtype=self.dtype)) + 5.
    return df, loc, scale_row, scale_col

  def testLogPDFScalarBatch(self):
    seed_stream = test_util.test_seed_stream()
    df, loc, scale_row, scale_col = self._random_df_loc_and_scale(
        [], [2, 3], seed_stream)
    matrix_t = mtlo.MatrixTLinearOperator(
        df, loc, scale_row, scale_col, validate_args=True)
    x = tf.random.normal(shape=[2, 3], seed=seed_stream(), dtype=self.dtype)

    log_pdf = matrix_t.log_prob(x)
    pdf = matrix_t.prob(x)

    # Check that this matches using a Multivariate-T.
    mvt = mst.MultivariateStudentTLinearOperator(
        df,
        loc=_vec(loc),
        scale=tf.linalg.LinearOperatorKronecker([scale_row, scale_col]))

    log_pdf_, pdf_, mvt_log_pdf_, mvt_pdf_ = self.evaluate([
        log_pdf, pdf, mvt.log_prob(_vec(x)), mvt.prob(_vec(x))])

    self.assertEqual((), log_pdf.shape)
    self.assertEqual((), pdf.shape)
    self.assertAllClose(mvt_log_pdf_, log_pdf_)
    self.assertAllClose(mvt_pdf_, pdf_)

  def testLogPDFBatch(self):
    seed_stream = test_util.test_seed_stream()
    df, loc, scale_row, scale_col = self._random_df_loc_and_scale(
        batch_shape=[7], matrix_shape=[3, 5], seed_stream=seed_stream)
    matrix_t = mtlo.MatrixTLinearOperator(
        df, loc, scale_row, scale_col, validate_args=True)
    x = tf.random.normal(shape=[3, 5], seed=seed_stream(), dtype=self.dtype)

    log_pdf = matrix_t.log_prob(x)
    pdf = matrix_t.prob(x)

    # Check that this matches using a Multivariate-T.
    mvt = mst.MultivariateStudentTLinearOperator(
        df,
        loc=_vec(loc),
        scale=tf.linalg.LinearOperatorKronecker([scale_row, scale_col]))

    log_pdf_, pdf_, mvt_log_pdf_, mvt_pdf_ = self.evaluate([
        log_pdf, pdf, mvt.log_prob(_vec(x)), mvt.prob(_vec(x))])

    self.assertEqual((7,), log_pdf.shape)
    self.assertEqual((7,), pdf.shape)
    self.assertAllClose(mvt_log_pdf_, log_pdf_)
    self.assertAllClose(mvt_pdf_, pdf_)

  def testShapes(self):
    seed_stream = test_util.test_seed_stream()
    df = np.array([1., 2., 3.], dtype=self.dtype)
    loc = tf.random.normal(
        shape=[5, 1, 2, 3], seed=seed_stream(), dtype=self.dtype)
    scale_row = tf.linalg.LinearOperatorLowerTriangular(
        self._random_tril_matrix([7, 1, 1, 2, 2], seed_stream()),
        is_non_singular=True)
    scale_col = tf.linalg.LinearOperatorLowerTriangular(
        self._random_tril_matrix([9, 1, 1, 1, 3, 3], seed_stream()),
        is_non_singular=True)

    matrix_t = mtlo.MatrixTLinearOperator(
        df, loc, scale_row, scale_col, validate_args=True)

    self.assertAllEqual((2, 3), matrix_t.event_shape)
    self.assertAllEqual((9, 7, 5, 3), matrix_t.batch_shape)

    self.assertAllEqual(
        (2, 3), self.evaluate(matrix_t.event_shape_tensor()))
    self.assertAllEqual(
        (9, 7, 5, 3), self.evaluate(matrix_t.batch_shape_tensor()))

  def testMeanAndVariance(self):
    df, loc, scale_row, scale_col = self._random_df_loc_and_scale(
        batch_shape=[3, 4], matrix_shape=[2, 5])
    matrix_t = mtlo.MatrixTLinearOperator(df, loc, scale_row, scale_col)

    cov_row = scale_row.matmul(scale_row.adjoint())
    cov_col = scale_col.matmul(scale_col.adjoint())
    # Compute diagonal of Kronecker product
    expected_variance = (cov_col.to_dense()[..., :, tf.newaxis, :, tf.newaxis] *
                         cov_row.to_dense()[..., tf.newaxis, :, tf.newaxis, :])
    expected_variance = tf.linalg.diag_part(
        tf.reshape(expected_variance, [3, 4, 10, 10]))
    expected_variance = tf.linalg.matrix_transpose(
        tf.reshape(expected_variance, [3, 4, 5, 2]))
    expected_variance = expected_variance * (df / (df - 2.))[
        ..., tf.newaxis, tf.newaxis]
    mean_, loc_, variance_, expected_variance_ = self.evaluate([
        matrix_t.mean(), loc, matrix_t.variance(), expected_variance])
    self.assertAllClose(loc_, mean_)
    self.assertAllClose(expected_variance_, variance_)

  def testSampleMean(self):
    seed_stream = test_util.test_seed_stream()
    df, loc, scale_row, scale_col = self._random_df_loc_and_scale(
        batch_shape=[5, 2], matrix_shape=[2, 3], seed_stream=seed_stream)
    matrix_t = mtlo.MatrixTLinearOperator(df, loc, scale_row, scale_col)
    samples = matrix_t.sample(int(1e6), seed=seed_stream())
    mean_, samples_ = self.evaluate([matrix_t.mean(), samples])
    self.assertAllClose(np.mean(samples_, axis=0), mean_, rtol=2e-2)

  def testSampleVariance(self):
    seed_stream = test_util.test_seed_stream()
    df, loc, scale_row, scale_col = self._random_df_loc_and_scale(
        batch_shape=[5, 2], matrix_shape=[2, 3], seed_stream=seed_stream)
    matrix_t = mtlo.MatrixTLinearOperator(df, loc, scale_row, scale_col)
    samples = matrix_t.sample(int(2e6), seed=seed_stream())
    variance_, samples_ = self.evaluate([matrix_t.variance(), samples])
    self.assertAllClose(np.var(samples_, axis=0), variance_, rtol=4e-2)

  @test_util.tf_tape_safety_test
  def testVariableLocation(self):
    df = tf.constant(1.)
    loc = tf.Variable([[1., 1.]])
    scale_row = tf.linalg.LinearOperatorLowerTriangular(
        tf.eye(1), is_non_singular=True)
    scale_column = tf.linalg.LinearOperatorLowerTriangular(
        tf.eye(2), is_non_singular=True)
    d = mtlo.MatrixTLinearOperator(
        df, loc, scale_row, scale_column, validate_args=True)
    self.evaluate(loc.initializer)
    with tf.GradientTape() as tape:
      lp = d.log_prob([[0., 0.]])
    self.assertIsNotNone(tape.gradient(lp, loc))

  @test_util.jax_disable_variable_test
  def testVariableDfAssertions(self):
    df = tf.Variable(1.)
    loc = tf.constant([[1., 1.]])
    scale_row = tf.linalg.LinearOperatorLowerTriangular(
        tf.eye(1), is_non_singular=True)
    scale_column = tf.linalg.LinearOperatorLowerTriangular(
        tf.eye(2), is_non_singular=True)
    d = mtlo.MatrixTLinearOperator(
        df, loc, scale_row, scale_column, validate_args=True)
    self.evaluate(df.initializer)
    with self.assertRaises(Exception):
      with tf.control_dependencies([df.assign(-1.)]):
        self.evaluate(d.sample(seed=test_util.test_seed()))


@test_util.test_all_tf_execution_regimes
class MatrixTTestFloat32Test(test_util.TestCase, _MatrixTTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class MatrixTFloat64Test(test_util.TestCase, _MatrixTTest):
  dtype = np.float64


if __name__ == '__main__':
  test_util.main()
