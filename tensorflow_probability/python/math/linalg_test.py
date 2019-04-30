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
"""Tests for linear algebra."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
from hypothesis.extra import numpy as hpnp

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util as tfp_test_util

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


class _PinvTest(object):

  def expected_pinv(self, a, rcond):
    """Calls `np.linalg.pinv` but corrects its broken batch semantics."""
    if a.ndim < 3:
      return np.linalg.pinv(a, rcond)
    if rcond is None:
      rcond = 10. * max(a.shape[-2], a.shape[-1]) * np.finfo(a.dtype).eps
    s = np.concatenate([a.shape[:-2], [a.shape[-1], a.shape[-2]]])
    a_pinv = np.zeros(s, dtype=a.dtype)
    for i in np.ndindex(a.shape[:(a.ndim - 2)]):
      a_pinv[i] = np.linalg.pinv(
          a[i],
          rcond=rcond if isinstance(rcond, float) else rcond[i])
    return a_pinv

  def test_symmetric(self):
    a_ = self.dtype([[1., .4, .5],
                     [.4, .2, .25],
                     [.5, .25, .35]])
    a_ = np.stack([a_ + 1., a_], axis=0)  # Batch of matrices.
    a = tf.compat.v1.placeholder_with_default(
        input=a_, shape=a_.shape if self.use_static_shape else None)
    if self.use_default_rcond:
      rcond = None
    else:
      rcond = self.dtype([0., 0.01])  # Smallest 1 component is forced to zero.
    expected_a_pinv_ = self.expected_pinv(a_, rcond)
    a_pinv = tfp.math.pinv(a, rcond, validate_args=True)
    a_pinv_ = self.evaluate(a_pinv)
    self.assertAllClose(expected_a_pinv_, a_pinv_,
                        atol=1e-5, rtol=1e-5)
    if not self.use_static_shape:
      return
    self.assertAllEqual(expected_a_pinv_.shape, a_pinv.shape)

  def test_nonsquare(self):
    a_ = self.dtype([[1., .4, .5, 1.],
                     [.4, .2, .25, 2.],
                     [.5, .25, .35, 3.]])
    a_ = np.stack([a_ + 0.5, a_], axis=0)  # Batch of matrices.
    a = tf.compat.v1.placeholder_with_default(
        input=a_, shape=a_.shape if self.use_static_shape else None)
    if self.use_default_rcond:
      rcond = None
    else:
      # Smallest 2 components are forced to zero.
      rcond = self.dtype([0., 0.25])
    expected_a_pinv_ = self.expected_pinv(a_, rcond)
    a_pinv = tfp.math.pinv(a, rcond, validate_args=True)
    a_pinv_ = self.evaluate(a_pinv)
    self.assertAllClose(expected_a_pinv_, a_pinv_,
                        atol=1e-5, rtol=1e-4)
    if not self.use_static_shape:
      return
    self.assertAllEqual(expected_a_pinv_.shape, a_pinv.shape)


@test_util.run_all_in_graph_and_eager_modes
class PinvTestDynamic32DefaultRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float32
  use_static_shape = False
  use_default_rcond = True


@test_util.run_all_in_graph_and_eager_modes
class PinvTestStatic64DefaultRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float64
  use_static_shape = True
  use_default_rcond = True


@test_util.run_all_in_graph_and_eager_modes
class PinvTestDynamic32CustomtRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float32
  use_static_shape = False
  use_default_rcond = False


@test_util.run_all_in_graph_and_eager_modes
class PinvTestStatic64CustomRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float64
  use_static_shape = True
  use_default_rcond = False


class _CholeskyExtend(tf.test.TestCase):

  def testCholeskyExtension(self):
    xs = np.random.random(7).astype(self.dtype)[:, tf.newaxis]
    xs = tf.compat.v1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)
    k = tfp.positive_semidefinite_kernels.MaternOneHalf()
    mat = k.matrix(xs, xs)
    chol = tf.linalg.cholesky(mat)

    ys = np.random.random(3).astype(self.dtype)[:, tf.newaxis]
    ys = tf.compat.v1.placeholder_with_default(
        ys, shape=ys.shape if self.use_static_shape else None)

    xsys = tf.concat([xs, ys], 0)
    new_chol_expected = tf.linalg.cholesky(k.matrix(xsys, xsys))

    new_chol = tfp.math.cholesky_concat(chol, k.matrix(xsys, ys))
    self.assertAllClose(new_chol_expected, new_chol)

  @hp.given(hps.data())
  @hp.settings(deadline=None, max_examples=10,
               derandomize=tfp_test_util.derandomize_hypothesis())
  def testCholeskyExtensionRandomized(self, data):
    jitter = lambda n: tf.linalg.eye(n, dtype=self.dtype) * 1e-5
    target_bs = data.draw(hpnp.array_shapes())
    prev_bs, new_bs = data.draw(tfp_test_util.broadcasting_shapes(target_bs, 2))
    ones = tf.TensorShape([1] * len(target_bs))
    smallest_shared_shp = tuple(np.min(
        [tf.broadcast_static_shape(ones, shp).as_list()
         for shp in [prev_bs, new_bs]],
        axis=0))

    z = data.draw(hps.integers(min_value=1, max_value=12))
    n = data.draw(hps.integers(min_value=0, max_value=z - 1))
    m = z - n

    np.random.seed(data.draw(hps.integers(min_value=0, max_value=2**32 - 1)))
    xs = np.random.uniform(size=smallest_shared_shp + (n,))
    data.draw(hps.just(xs))
    xs = (xs + np.zeros(prev_bs.as_list() + [n]))[..., np.newaxis]
    xs = xs.astype(self.dtype)
    xs = tf.compat.v1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)

    k = tfp.positive_semidefinite_kernels.MaternOneHalf()
    mat = k.matrix(xs, xs) + jitter(n)
    chol = tf.linalg.cholesky(mat)

    ys = np.random.uniform(size=smallest_shared_shp + (m,))
    data.draw(hps.just(ys))
    ys = (ys + np.zeros(new_bs.as_list() + [m]))[..., np.newaxis]
    ys = ys.astype(self.dtype)
    ys = tf.compat.v1.placeholder_with_default(
        ys, shape=ys.shape if self.use_static_shape else None)

    xsys = tf.concat([xs + tf.zeros(target_bs + (n, 1), dtype=self.dtype),
                      ys + tf.zeros(target_bs + (m, 1), dtype=self.dtype)],
                     axis=-2)
    new_chol_expected = tf.linalg.cholesky(k.matrix(xsys, xsys) + jitter(z))

    new_chol = tfp.math.cholesky_concat(
        chol, k.matrix(xsys, ys) + jitter(z)[:, n:])
    self.assertAllClose(new_chol_expected, new_chol, rtol=1e-5, atol=1e-5)


@test_util.run_all_in_graph_and_eager_modes
class CholeskyExtend32Static(_CholeskyExtend):
  dtype = np.float32
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class CholeskyExtend64Dynamic(_CholeskyExtend):
  dtype = np.float64
  use_static_shape = False

del _CholeskyExtend


class _PivotedCholesky(tf.test.TestCase, parameterized.TestCase):

  def _random_batch_psd(self, dim):
    matrix = np.random.random([2, dim, dim])
    matrix = np.matmul(matrix, np.swapaxes(matrix, -2, -1))
    matrix = (matrix + np.diag(np.arange(dim) * .1)).astype(self.dtype)
    masked_shape = (
        matrix.shape if self.use_static_shape else [None] * len(matrix.shape))
    matrix = tf.compat.v1.placeholder_with_default(matrix, shape=masked_shape)
    return matrix

  def testPivotedCholesky(self):
    dim = 11
    matrix = self._random_batch_psd(dim)
    true_diag = tf.linalg.diag_part(matrix)

    pchol = tfp.math.pivoted_cholesky(matrix, max_rank=1)
    mat = tf.matmul(pchol, pchol, transpose_b=True)
    diag_diff_prev = self.evaluate(tf.abs(tf.linalg.diag_part(mat) - true_diag))
    diff_norm_prev = self.evaluate(
        tf.linalg.norm(tensor=mat - matrix, ord='fro', axis=[-1, -2]))
    for rank in range(2, dim + 1):
      # Specifying diag_rtol forces the full max_rank decomposition.
      pchol = tfp.math.pivoted_cholesky(matrix, max_rank=rank, diag_rtol=-1)
      zeros_per_col = dim - tf.math.count_nonzero(pchol, axis=-2)
      mat = tf.matmul(pchol, pchol, transpose_b=True)
      pchol_shp, diag_diff, diff_norm, zeros_per_col = self.evaluate([
          tf.shape(input=pchol),
          tf.abs(tf.linalg.diag_part(mat) - true_diag),
          tf.linalg.norm(tensor=mat - matrix, ord='fro', axis=[-1, -2]),
          zeros_per_col
      ])
      self.assertAllEqual([2, dim, rank], pchol_shp)
      self.assertAllEqual(
          np.ones([2, rank], dtype=np.bool), zeros_per_col >= np.arange(rank))
      self.assertAllLessEqual(diag_diff - diag_diff_prev,
                              np.finfo(self.dtype).resolution)
      self.assertAllLessEqual(diff_norm - diff_norm_prev,
                              np.finfo(self.dtype).resolution)
      diag_diff_prev, diff_norm_prev = diag_diff, diff_norm

  def testGradient(self):
    dim = 11
    matrix = self._random_batch_psd(dim)
    _, dmatrix = tfp.math.value_and_gradient(
        lambda matrix: tfp.math.pivoted_cholesky(matrix, max_rank=dim // 3),
        matrix)
    self.assertIsNotNone(dmatrix)
    self.assertAllGreater(
        tf.linalg.norm(tensor=dmatrix, ord='fro', axis=[-1, -2]), 0.)

  @test_util.enable_control_flow_v2
  def testGradientTapeCFv2(self):
    dim = 11
    matrix = self._random_batch_psd(dim)
    with tf.GradientTape() as tape:
      tape.watch(matrix)
      pchol = tfp.math.pivoted_cholesky(matrix, max_rank=dim // 3)
    dmatrix = tape.gradient(
        pchol, matrix, output_gradients=tf.ones_like(pchol) * .01)
    self.assertIsNotNone(dmatrix)
    self.assertAllGreater(
        tf.linalg.norm(tensor=dmatrix, ord='fro', axis=[-1, -2]), 0.)

  # pyformat: disable
  @parameterized.parameters(
      # Inputs are randomly shuffled arange->tril; outputs from gpytorch.
      (
          np.array([
              [7., 0, 0, 0, 0, 0],
              [9, 13, 0, 0, 0, 0],
              [4, 10, 6, 0, 0, 0],
              [18, 1, 2, 14, 0, 0],
              [5, 11, 20, 3, 17, 0],
              [19, 12, 16, 15, 8, 21]
          ]),
          np.array([
              [3.4444, -1.3545, 4.084, 1.7674, -1.1789, 3.7562],
              [8.4685, 1.2821, 3.1179, 12.9197, 0.0000, 0.0000],
              [7.5621, 4.8603, 0.0634, 7.3942, 4.0637, 0.0000],
              [15.435, -4.8864, 16.2137, 0.0000, 0.0000, 0.0000],
              [18.8535, 22.103, 0.0000, 0.0000, 0.0000, 0.0000],
              [38.6135, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
          ])),
      (
          np.array([
              [1, 0, 0],
              [2, 3, 0],
              [4, 5, 6.]
          ]),
          np.array([
              [0.4558, 0.3252, 0.8285],
              [2.6211, 2.4759, 0.0000],
              [8.7750, 0.0000, 0.0000]
          ])),
      (
          np.array([
              [6, 0, 0],
              [3, 2, 0],
              [4, 1, 5.]
          ]),
          np.array([
              [3.7033, 4.7208, 0.0000],
              [2.1602, 2.1183, 1.9612],
              [6.4807, 0.0000, 0.0000]
          ])))
  # pyformat: enable
  def testOracleExamples(self, mat, oracle_pchol):
    mat = np.matmul(mat, mat.T)
    for rank in range(1, mat.shape[-1] + 1):
      self.assertAllClose(
          oracle_pchol[..., :rank],
          tfp.math.pivoted_cholesky(mat, max_rank=rank, diag_rtol=-1),
          atol=1e-4)


@test_util.run_all_in_graph_and_eager_modes
class PivotedCholesky32Static(_PivotedCholesky):
  dtype = np.float32
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class PivotedCholesky64Dynamic(_PivotedCholesky):
  dtype = np.float64
  use_static_shape = False


del _PivotedCholesky


def make_tensor_hiding_attributes(value, hide_shape, hide_value=True):
  if not hide_value:
    return tf.convert_to_tensor(value=value)

  shape = None if hide_shape else getattr(value, 'shape', None)
  return tf.compat.v1.placeholder_with_default(input=value, shape=shape)


class _LUReconstruct(object):
  dtype = np.float32
  use_static_shape = True

  def test_non_batch(self):
    x_ = np.array(
        [[3, 4], [1, 2]],
        dtype=self.dtype)
    x = tf.compat.v1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = tfp.math.lu_reconstruct(*tf.linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(x_, y_, atol=0., rtol=1e-3)

  def test_batch(self):
    x_ = np.array(
        [
            [[3, 4], [1, 2]],
            [[7, 8], [3, 4]],
        ],
        dtype=self.dtype)
    x = tf.compat.v1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = tfp.math.lu_reconstruct(*tf.linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(x_, y_, atol=0., rtol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class LUReconstructStatic(tf.test.TestCase, _LUReconstruct):
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class LUReconstructDynamic(tf.test.TestCase, _LUReconstruct):
  use_static_shape = False


class _LUMatrixInverse(object):
  dtype = np.float32
  use_static_shape = True

  def test_non_batch(self):
    x_ = np.array([[1, 2], [3, 4]], dtype=self.dtype)
    x = tf.compat.v1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = tfp.math.lu_matrix_inverse(*tf.linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(np.linalg.inv(x_), y_, atol=0., rtol=1e-3)

  def test_batch(self):
    x_ = np.array(
        [
            [[1, 2],
             [3, 4]],
            [[7, 8],
             [3, 4]],
            [[0.25, 0.5],
             [0.75, -2.]],
        ],
        dtype=self.dtype)
    x = tf.compat.v1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = tfp.math.lu_matrix_inverse(*tf.linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(np.linalg.inv(x_), y_, atol=0., rtol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class LUMatrixInverseStatic(tf.test.TestCase, _LUMatrixInverse):
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class LUMatrixInverseDynamic(tf.test.TestCase, _LUMatrixInverse):
  use_static_shape = False


class _LUSolve(object):
  dtype = np.float32
  use_static_shape = True

  def test_non_batch(self):
    x_ = np.array(
        [[1, 2],
         [3, 4]],
        dtype=self.dtype)
    x = tf.compat.v1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    rhs_ = np.array([[1, 1]], dtype=self.dtype).T
    rhs = tf.compat.v1.placeholder_with_default(
        rhs_, shape=rhs_.shape if self.use_static_shape else None)

    lower_upper, perm = tf.linalg.lu(x)
    y = tfp.math.lu_solve(lower_upper, perm, rhs, validate_args=True)
    y_, perm_ = self.evaluate([y, perm])

    self.assertAllEqual([1, 0], perm_)
    expected_ = np.linalg.solve(x_, rhs_)
    if self.use_static_shape:
      self.assertAllEqual(expected_.shape, y.shape)
    self.assertAllClose(expected_, y_, atol=0., rtol=1e-3)

  def test_batch_broadcast(self):
    x_ = np.array(
        [
            [[1, 2],
             [3, 4]],
            [[7, 8],
             [3, 4]],
            [[0.25, 0.5],
             [0.75, -2.]],
        ],
        dtype=self.dtype)
    x = tf.compat.v1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    rhs_ = np.array([[1, 1]], dtype=self.dtype).T
    rhs = tf.compat.v1.placeholder_with_default(
        rhs_, shape=rhs_.shape if self.use_static_shape else None)

    lower_upper, perm = tf.linalg.lu(x)
    y = tfp.math.lu_solve(lower_upper, perm, rhs, validate_args=True)
    y_, perm_ = self.evaluate([y, perm])

    self.assertAllEqual([[1, 0],
                         [0, 1],
                         [1, 0]], perm_)
    expected_ = np.linalg.solve(x_, rhs_[np.newaxis])
    if self.use_static_shape:
      self.assertAllEqual(expected_.shape, y.shape)
    self.assertAllClose(expected_, y_, atol=0., rtol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class LUSolveStatic(tf.test.TestCase, _LUSolve):
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class LUSolveDynamic(tf.test.TestCase, _LUSolve):
  use_static_shape = False


class _SparseOrDenseMatmul(object):
  dtype = np.float32
  use_static_shape = True
  use_sparse_tensor = False

  def _make_placeholder(self, x):
    return tf.compat.v1.placeholder_with_default(
        input=x, shape=(x.shape if self.use_static_shape else None))

  def _make_sparse_placeholder(self, x):
    indices_placeholder = self._make_placeholder(x.indices)
    values_placeholder = self._make_placeholder(x.values)

    if self.use_static_shape:
      dense_shape_placeholder = x.dense_shape
    else:
      dense_shape_placeholder = self._make_placeholder(x.dense_shape)

    return tf.SparseTensor(
        indices=indices_placeholder,
        values=values_placeholder,
        dense_shape=dense_shape_placeholder)

  def verify_sparse_dense_matmul(self, x_, y_):
    if self.use_sparse_tensor:
      x = self._make_sparse_placeholder(tfp.math.dense_to_sparse(x_))
    else:
      x = self._make_placeholder(x_)

    y = self._make_placeholder(y_)

    z = tfp.math.sparse_or_dense_matmul(x, y)
    z_ = self.evaluate(z)

    if self.use_static_shape:
      batch_shape = x_.shape[:-2]
      self.assertAllEqual(z_.shape, batch_shape + (x_.shape[-2], y_.shape[-1]))

    self.assertAllClose(z_, np.matmul(x_, y_), atol=0., rtol=1e-3)

  def verify_sparse_dense_matvecmul(self, x_, y_):
    if self.use_sparse_tensor:
      x = self._make_sparse_placeholder(tfp.math.dense_to_sparse(x_))
    else:
      x = self._make_placeholder(x_)

    y = self._make_placeholder(y_)

    z = tfp.math.sparse_or_dense_matvecmul(x, y)
    z_ = self.evaluate(z)

    if self.use_static_shape:
      batch_shape = x_.shape[:-2]
      self.assertAllEqual(z_.shape, batch_shape + (x_.shape[-2],))

    self.assertAllClose(
        z_[..., np.newaxis],
        np.matmul(x_, y_[..., np.newaxis]),
        atol=0.,
        rtol=1e-3)

  def test_non_batch_matmul(self):
    x_ = np.array([[3, 4, 0], [1, 0, 3]], dtype=self.dtype)
    y_ = np.array([[1, 0], [9, 0], [3, 1]], dtype=self.dtype)
    self.verify_sparse_dense_matmul(x_, y_)

  def test_non_batch_matvecmul(self):
    x_ = np.array([[3, 0, 5], [0, 2, 3]], dtype=self.dtype)
    y_ = np.array([1, 0, 9], dtype=self.dtype)
    self.verify_sparse_dense_matvecmul(x_, y_)

  def test_batch_matmul(self):
    x_ = np.array([
        [[3, 4, 0], [1, 0, 3]],
        [[6, 0, 0], [0, 0, 0]],
    ],
                  dtype=self.dtype)
    y_ = np.array([
        [[1, 0], [9, 0], [3, 1]],
        [[2, 2], [5, 6], [0, 1]],
    ],
                  dtype=self.dtype)
    self.verify_sparse_dense_matmul(x_, y_)

  def test_batch_matvecmul(self):
    x_ = np.array([
        [[3, 0, 5], [0, 2, 3]],
        [[1, 1, 0], [6, 0, 0]],
    ],
                  dtype=self.dtype)

    y_ = np.array([
        [1, 0, 9],
        [0, 0, 2],
    ], dtype=self.dtype)

    self.verify_sparse_dense_matvecmul(x_, y_)


@test_util.run_all_in_graph_and_eager_modes
class SparseOrDenseMatmulStatic(tf.test.TestCase, _SparseOrDenseMatmul):
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class SparseOrDenseMatmulDynamic(tf.test.TestCase, _SparseOrDenseMatmul):
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class SparseOrDenseMatmulStaticSparse(tf.test.TestCase, _SparseOrDenseMatmul):
  use_static_shape = True
  use_sparse_tensor = True


@test_util.run_all_in_graph_and_eager_modes
class SparseOrDenseMatmulDynamicSparse(tf.test.TestCase, _SparseOrDenseMatmul):
  use_static_shape = False
  use_sparse_tensor = True


class _MatrixRankTest(object):

  def test_batch_default_tolerance(self):
    x_ = np.array([[[2, 3, -2],  # = row2+row3
                    [-1, 1, -2],
                    [3, 2, 0]],
                   [[0, 2, 0],   # = 2*row2
                    [0, 1, 0],
                    [0, 3, 0]],  # = 3*row2
                   [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]],
                  self.dtype)
    x = tf.compat.v1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    self.assertAllEqual([2, 1, 3], self.evaluate(tfp.math.matrix_rank(x)))

  def test_custom_tolerance_broadcasts(self):
    q = tf.linalg.qr(tf.random.uniform([3, 3], dtype=self.dtype))[0]
    e = tf.constant([0.1, 0.2, 0.3], dtype=self.dtype)
    a = tf.linalg.solve(q, tf.transpose(a=e * q), adjoint=True)
    self.assertAllEqual([3, 2, 1, 0], self.evaluate(tfp.math.matrix_rank(
        a, tol=[[0.09], [0.19], [0.29], [0.31]])))

  def test_nonsquare(self):
    x_ = np.array([[[2, 3, -2, 2],   # = row2+row3
                    [-1, 1, -2, 4],
                    [3, 2, 0, -2]],
                   [[0, 2, 0, 6],    # = 2*row2
                    [0, 1, 0, 3],
                    [0, 3, 0, 9]]],  # = 3*row2
                  self.dtype)
    x = tf.compat.v1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    self.assertAllEqual([2, 1], self.evaluate(tfp.math.matrix_rank(x)))


@test_util.run_all_in_graph_and_eager_modes
class MatrixRankStatic32Test(tf.test.TestCase, _MatrixRankTest):
  dtype = np.float32
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class MatrixRankDynamic64Test(tf.test.TestCase, _MatrixRankTest):
  dtype = np.float64
  use_static_shape = False


if __name__ == '__main__':
  tf.test.main()
