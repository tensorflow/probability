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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow.python.framework import test_util as tf_test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

JAX_MODE = False


class _CholeskyExtend(test_util.TestCase):

  def testCholeskyExtension(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    xs = tf1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)
    k = tfp.math.psd_kernels.MaternOneHalf()
    mat = k.matrix(xs, xs)
    chol = tf.linalg.cholesky(mat)

    ys = rng.random_sample(3).astype(self.dtype)[:, tf.newaxis]
    ys = tf1.placeholder_with_default(
        ys, shape=ys.shape if self.use_static_shape else None)

    xsys = tf.concat([xs, ys], 0)
    new_chol_expected = tf.linalg.cholesky(k.matrix(xsys, xsys))

    new_chol = tfp.math.cholesky_concat(chol, k.matrix(xsys, ys))
    self.assertAllClose(new_chol_expected, new_chol, rtol=1e-5, atol=2e-5)

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testCholeskyExtensionRandomized(self, data):
    jitter = lambda n: tf.linalg.eye(n, dtype=self.dtype) * 5e-5
    target_bs = data.draw(hpnp.array_shapes())
    prev_bs, new_bs = data.draw(tfp_hps.broadcasting_shapes(target_bs, 2))
    ones = tf.TensorShape([1] * len(target_bs))
    smallest_shared_shp = tuple(np.min(
        [tensorshape_util.as_list(tf.broadcast_static_shape(ones, shp))
         for shp in [prev_bs, new_bs]],
        axis=0))

    z = data.draw(hps.integers(min_value=1, max_value=12))
    n = data.draw(hps.integers(min_value=0, max_value=z - 1))
    m = z - n

    rng_seed = data.draw(hps.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.RandomState(seed=rng_seed)
    xs = rng.uniform(size=smallest_shared_shp + (n,))
    hp.note(xs)
    xs = (xs + np.zeros(tensorshape_util.as_list(prev_bs) +
                        [n]))[..., np.newaxis]
    xs = xs.astype(self.dtype)
    xs = tf1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)

    k = tfp.math.psd_kernels.MaternOneHalf()
    mat = k.matrix(xs, xs) + jitter(n)
    chol = tf.linalg.cholesky(mat)

    ys = rng.uniform(size=smallest_shared_shp + (m,))
    hp.note(ys)
    ys = (ys + np.zeros(tensorshape_util.as_list(new_bs)
                        + [m]))[..., np.newaxis]
    ys = ys.astype(self.dtype)
    ys = tf1.placeholder_with_default(
        ys, shape=ys.shape if self.use_static_shape else None)

    xsys = tf.concat([xs + tf.zeros(target_bs + (n, 1), dtype=self.dtype),
                      ys + tf.zeros(target_bs + (m, 1), dtype=self.dtype)],
                     axis=-2)
    new_chol_expected = tf.linalg.cholesky(k.matrix(xsys, xsys) + jitter(z))

    new_chol = tfp.math.cholesky_concat(
        chol, k.matrix(xsys, ys) + jitter(z)[:, n:])
    self.assertAllClose(new_chol_expected, new_chol, rtol=1e-5, atol=2e-5)


@test_util.test_all_tf_execution_regimes
class CholeskyExtend32Static(_CholeskyExtend):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class CholeskyExtend64Dynamic(_CholeskyExtend):
  dtype = np.float64
  use_static_shape = False

del _CholeskyExtend


def push_apart(xs, axis, shift=1e-3):
  """Push values of `xs` apart from each other by `shift`, along `axis`."""
  # The method is to scale the displacement by each item's sort position, so the
  # 10th item in sorted order gets moved by 10 * shift, the 11th by 11 * shift,
  # etc.  This way, each item moves `shift` away from each of its neighbors.
  inv_perm = np.argsort(xs, axis=axis)
  perm = np.argsort(inv_perm, axis=axis)
  return xs + perm * shift


class PushApartTest(test_util.TestCase):

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testPreservesSortOrder(self, data):
    dtype = data.draw(hpnp.floating_dtypes())
    xs = data.draw(hpnp.arrays(dtype, 10, unique=True))
    pushed = push_apart(xs, axis=-1)
    hp.note(pushed)
    self.assertAllEqual(np.argsort(xs, axis=-1), np.argsort(pushed, axis=-1))


class _CholeskyUpdate(test_util.TestCase):

  def testCholeskyUpdate(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    xs = tf1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)
    k = tfp.math.psd_kernels.MaternOneHalf()
    mat = k.matrix(xs, xs)
    chol = tf.linalg.cholesky(mat)

    u = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    u = tf1.placeholder_with_default(
        u, shape=u.shape if self.use_static_shape else None)

    new_chol_expected = tf.linalg.cholesky(
        mat + tf.linalg.matmul(u, u, transpose_b=True))
    new_chol = tfp.math.cholesky_update(chol, tf.squeeze(u, axis=-1))
    self.assertAllClose(new_chol_expected, new_chol, rtol=1e-5, atol=2e-5)

  def testCholeskyUpdateBatches(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample((3, 1, 7)).astype(self.dtype)[..., tf.newaxis]
    xs = tf1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)
    k = tfp.math.psd_kernels.MaternOneHalf()
    mat = k.matrix(xs, xs)
    chol = tf.linalg.cholesky(mat)

    u = rng.random_sample((1, 5, 7)).astype(self.dtype)[..., tf.newaxis]
    u = tf1.placeholder_with_default(
        u, shape=u.shape if self.use_static_shape else None)
    multiplier = rng.random_sample((2, 1, 1)).astype(self.dtype)

    new_chol_expected = tf.linalg.cholesky(
        mat + multiplier[..., np.newaxis, np.newaxis] * tf.linalg.matmul(
            u, u, transpose_b=True))
    new_chol = tfp.math.cholesky_update(
        chol, tf.squeeze(u, axis=-1), multiplier=multiplier)
    self.assertAllClose(new_chol_expected, new_chol, rtol=1e-5, atol=2e-5)

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testCholeskyUpdateRandomized(self, data):
    target_bs = data.draw(hpnp.array_shapes())
    chol_bs, u_bs, multiplier_bs = data.draw(
        tfp_hps.broadcasting_shapes(target_bs, 3))
    l = data.draw(hps.integers(min_value=1, max_value=12))

    rng_seed = data.draw(hps.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.RandomState(seed=rng_seed)
    xs = push_apart(rng.uniform(size=chol_bs + (l, 1)), axis=-2)
    hp.note(xs)
    xs = xs.astype(self.dtype)
    xs = tf1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)

    k = tfp.math.psd_kernels.MaternOneHalf()
    jitter = lambda n: tf.linalg.eye(n, dtype=self.dtype) * 5e-5

    mat = k.matrix(xs, xs) + jitter(l)
    chol = tf.linalg.cholesky(mat)

    u = rng.uniform(size=u_bs + (l,))
    hp.note(u)
    u = u.astype(self.dtype)
    u = tf1.placeholder_with_default(
        u, shape=u.shape if self.use_static_shape else None)

    multiplier = rng.uniform(size=multiplier_bs)
    hp.note(multiplier)
    multiplier = multiplier.astype(self.dtype)
    multiplier = tf1.placeholder_with_default(
        multiplier, shape=multiplier.shape if self.use_static_shape else None)

    new_chol_expected = tf.linalg.cholesky(
        mat + multiplier[..., tf.newaxis, tf.newaxis] * tf.linalg.matmul(
            u[..., tf.newaxis], u[..., tf.newaxis, :]))

    new_chol = tfp.math.cholesky_update(chol, u, multiplier=multiplier)
    self.assertAllClose(new_chol_expected, new_chol, rtol=1e-5, atol=2e-5)


@test_util.test_all_tf_execution_regimes
class CholeskyUpdate32Static(_CholeskyUpdate):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class CholeskyUpdate64Dynamic(_CholeskyUpdate):
  dtype = np.float64
  use_static_shape = False

del _CholeskyUpdate


class _PivotedCholesky(test_util.TestCase):

  def _random_batch_psd(self, dim):
    rng = test_util.test_np_rng()
    matrix = rng.random_sample([2, dim, dim])
    matrix = np.matmul(matrix, np.swapaxes(matrix, -2, -1))
    matrix = (matrix + np.diag(np.arange(dim) * .1)).astype(self.dtype)
    masked_shape = (
        matrix.shape if self.use_static_shape else [None] * len(matrix.shape))
    matrix = tf1.placeholder_with_default(matrix, shape=masked_shape)
    return matrix

  def testPivotedCholesky(self):
    dim = 11
    matrix = self._random_batch_psd(dim)
    true_diag = tf.linalg.diag_part(matrix)

    pchol = tfp.math.pivoted_cholesky(matrix, max_rank=1)
    mat = tf.matmul(pchol, pchol, transpose_b=True)
    diag_diff_prev = self.evaluate(tf.abs(tf.linalg.diag_part(mat) - true_diag))
    diff_norm_prev = self.evaluate(
        tf.linalg.norm(mat - matrix, ord='fro', axis=[-1, -2]))
    for rank in range(2, dim + 1):
      # Specifying diag_rtol forces the full max_rank decomposition.
      pchol = tfp.math.pivoted_cholesky(matrix, max_rank=rank, diag_rtol=-1)
      zeros_per_col = dim - tf.math.count_nonzero(pchol, axis=-2)
      mat = tf.matmul(pchol, pchol, transpose_b=True)
      pchol_shp, diag_diff, diff_norm, zeros_per_col = self.evaluate([
          tf.shape(pchol),
          tf.abs(tf.linalg.diag_part(mat) - true_diag),
          tf.linalg.norm(mat - matrix, ord='fro', axis=[-1, -2]), zeros_per_col
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
    self.assertAllGreater(tf.linalg.norm(dmatrix, ord='fro', axis=[-1, -2]), 0.)

  @test_util.tf_tape_safety_test
  @tf_test_util.enable_control_flow_v2
  def testGradientTapeCFv2(self):
    dim = 11
    matrix = self._random_batch_psd(dim)
    with tf.GradientTape() as tape:
      tape.watch(matrix)
      pchol = tfp.math.pivoted_cholesky(matrix, max_rank=dim // 3)
    dmatrix = tape.gradient(
        pchol, matrix, output_gradients=tf.ones_like(pchol) * .01)
    self.assertIsNotNone(dmatrix)
    self.assertAllGreater(tf.linalg.norm(dmatrix, ord='fro', axis=[-1, -2]), 0.)

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

  def testLinopKernel(self):
    x = tf.random.uniform([10, 2], dtype=self.dtype, seed=test_util.test_seed())
    masked_shape = x.shape if self.use_static_shape else [None] * len(x.shape)
    x = tf1.placeholder_with_default(x, shape=masked_shape)
    k = tfp.math.psd_kernels.ExponentiatedQuadratic()
    expected = tfp.math.pivoted_cholesky(k.matrix(x, x), max_rank=3)
    actual = tfp.math.pivoted_cholesky(
        tfp.experimental.linalg.LinearOperatorPSDKernel(k, x), max_rank=3)
    expected, actual = self.evaluate([expected, actual])
    self.assertAllClose(expected, actual)


if not JAX_MODE:
  # TODO(b/147693911): Enable these tests once pivoted_cholesky
  # no longer relies on dynamic slices

  @test_util.test_all_tf_execution_regimes
  class PivotedCholesky32Static(_PivotedCholesky):
    dtype = np.float32
    use_static_shape = True

  @test_util.test_all_tf_execution_regimes
  class PivotedCholesky64Dynamic(_PivotedCholesky):
    dtype = np.float64
    use_static_shape = False


del _PivotedCholesky


def make_tensor_hiding_attributes(value, hide_shape, hide_value=True):
  if not hide_value:
    return tf.convert_to_tensor(value)

  shape = None if hide_shape else getattr(value, 'shape', None)
  return tf1.placeholder_with_default(value, shape=shape)


class _LUReconstruct(object):
  dtype = np.float32
  use_static_shape = True

  def test_non_batch(self):
    x_ = np.array(
        [[3, 4], [1, 2]],
        dtype=self.dtype)
    x = tf1.placeholder_with_default(
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
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = tfp.math.lu_reconstruct(*tf.linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(x_, y_, atol=0., rtol=1e-3)


@test_util.test_all_tf_execution_regimes
class LUReconstructStatic(test_util.TestCase, _LUReconstruct):
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class LUReconstructDynamic(test_util.TestCase, _LUReconstruct):
  use_static_shape = False


class _LUMatrixInverse(object):
  dtype = np.float32
  use_static_shape = True

  def test_non_batch(self):
    x_ = np.array([[1, 2], [3, 4]], dtype=self.dtype)
    x = tf1.placeholder_with_default(
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
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)

    y = tfp.math.lu_matrix_inverse(*tf.linalg.lu(x), validate_args=True)
    y_ = self.evaluate(y)

    if self.use_static_shape:
      self.assertAllEqual(x_.shape, y.shape)
    self.assertAllClose(np.linalg.inv(x_), y_, atol=0., rtol=1e-3)


@test_util.test_all_tf_execution_regimes
class LUMatrixInverseStatic(test_util.TestCase, _LUMatrixInverse):
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class LUMatrixInverseDynamic(test_util.TestCase, _LUMatrixInverse):
  use_static_shape = False


class _LUSolve(object):
  dtype = np.float32
  use_static_shape = True

  def test_non_batch(self):
    x_ = np.array(
        [[1, 2],
         [3, 4]],
        dtype=self.dtype)
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    rhs_ = np.array([[1, 1]], dtype=self.dtype).T
    rhs = tf1.placeholder_with_default(
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
    x = tf1.placeholder_with_default(
        x_, shape=x_.shape if self.use_static_shape else None)
    rhs_ = np.array([[1, 1]], dtype=self.dtype).T
    rhs = tf1.placeholder_with_default(
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


@test_util.test_all_tf_execution_regimes
class LUSolveStatic(test_util.TestCase, _LUSolve):
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class LUSolveDynamic(test_util.TestCase, _LUSolve):
  use_static_shape = False


class _SparseOrDenseMatmul(test_util.TestCase):
  dtype = np.float32
  use_static_shape = True
  use_sparse_tensor = False

  def _make_placeholder(self, x):
    return tf1.placeholder_with_default(
        x, shape=(x.shape if self.use_static_shape else None))

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

if not JAX_MODE:
  # TODO(b/147683793): Enable tests when JAX backend supports SparseTensor.

  @test_util.test_all_tf_execution_regimes
  class SparseOrDenseMatmulStatic(_SparseOrDenseMatmul):
    use_static_shape = True

  @test_util.test_all_tf_execution_regimes
  class SparseOrDenseMatmulDynamic(_SparseOrDenseMatmul):
    use_static_shape = False

  @test_util.test_all_tf_execution_regimes
  class SparseOrDenseMatmulStaticSparse(_SparseOrDenseMatmul):
    use_static_shape = True
    use_sparse_tensor = True

  @test_util.test_all_tf_execution_regimes
  class SparseOrDenseMatmulDynamicSparse(_SparseOrDenseMatmul):
    use_static_shape = False
    use_sparse_tensor = True


del _SparseOrDenseMatmul


@test_util.test_all_tf_execution_regimes
class FillTriangularTest(test_util.TestCase):

  def _fill_triangular(self, x, upper=False):
    """Numpy implementation of `fill_triangular`."""
    x = np.asarray(x)
    # Formula derived by solving for n: m = n(n+1)/2.
    m = np.int32(x.shape[-1])
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
      raise ValueError('Invalid shape.')
    n = np.int32(n)
    # We can't do: `x[..., -(n**2-m):]` because this doesn't correctly handle
    # `m == n == 1`. Hence, we do absolute indexing.
    x_tail = x[..., (m - (n * n - m)):]
    y = np.concatenate(
        [x, x_tail[..., ::-1]] if upper else [x_tail, x[..., ::-1]],
        axis=-1)
    y = y.reshape(np.concatenate([
        np.int32(x.shape[:-1]),
        np.int32([n, n]),
    ], axis=0))
    return np.triu(y) if upper else np.tril(y)

  def _run_test(self, x_, use_deferred_shape=False, **kwargs):
    x_ = np.asarray(x_)
    static_shape = None if use_deferred_shape else x_.shape
    x_pl = tf1.placeholder_with_default(x_, shape=static_shape)
    # Add `zeros_like(x)` such that x's value and gradient are identical. We
    # do this so we can ensure each gradient value is mapped to the right
    # gradient location.  (Not doing this means the gradient wrt `x` is simple
    # `ones_like(x)`.)
    # Note:
    #   zeros_like_x_pl == zeros_like(x_pl)
    #   gradient(zeros_like_x_pl, x_pl) == x_pl - 1
    def _zeros_like(x):
      return x * tf.stop_gradient(x - 1.) - tf.stop_gradient(x * (x - 1.))
    actual, grad_actual = tfp.math.value_and_gradient(
        lambda x: tfp.math.fill_triangular(  # pylint: disable=g-long-lambda
            x + _zeros_like(x), **kwargs),
        x_pl)
    actual_, grad_actual_ = self.evaluate([actual, grad_actual])
    expected = self._fill_triangular(x_, **kwargs)
    if use_deferred_shape and not tf.executing_eagerly():
      self.assertEqual(None, actual.shape)
    else:
      self.assertAllEqual(expected.shape, actual.shape)
    self.assertAllClose(expected, actual_, rtol=1e-8, atol=1e-9)
    self.assertAllClose(x_, grad_actual_, rtol=1e-8, atol=1e-9)

  def testCorrectlyMakes1x1TriLower(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(3, int(1*2/2)))

  def testCorrectlyMakesNoBatchTriLower(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(int(4*5/2)))

  def testCorrectlyMakesBatchTriLower(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(3*4/2)))

  def testCorrectlyMakesBatchTriLowerUnknownShape(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(3*4/2)), use_deferred_shape=True)

  def testCorrectlyMakesBatch7x7TriLowerUnknownShape(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(7*8/2)), use_deferred_shape=True)

  def testCorrectlyMakesBatch7x7TriLower(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(7*8/2)))

  def testCorrectlyMakes1x1TriUpper(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(3, int(1*2/2)), upper=True)

  def testCorrectlyMakesNoBatchTriUpper(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(int(4*5/2)), upper=True)

  def testCorrectlyMakesBatchTriUpper(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 2, int(3*4/2)), upper=True)

  def testCorrectlyMakesBatchTriUpperUnknownShape(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 2, int(3*4/2)),
                   use_deferred_shape=True,
                   upper=True)

  def testCorrectlyMakesBatch7x7TriUpperUnknownShape(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(7*8/2)),
                   use_deferred_shape=True,
                   upper=True)

  def testCorrectlyMakesBatch7x7TriUpper(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(7*8/2)), upper=True)


@test_util.test_all_tf_execution_regimes
class FillTriangularInverseTest(FillTriangularTest):

  def _run_test(self, x_, use_deferred_shape=False, **kwargs):
    x_ = np.asarray(x_)
    static_shape = None if use_deferred_shape else x_.shape
    x_pl = tf1.placeholder_with_default(x_, shape=static_shape)

    zeros_like_x_pl = (x_pl * tf.stop_gradient(x_pl - 1.)
                       - tf.stop_gradient(x_pl * (x_pl - 1.)))
    x = x_pl + zeros_like_x_pl
    actual = tfp.math.fill_triangular(x, **kwargs)
    inverse_actual = tfp.math.fill_triangular_inverse(actual, **kwargs)

    inverse_actual_ = self.evaluate(inverse_actual)

    if use_deferred_shape and not tf.executing_eagerly():
      self.assertEqual(None, inverse_actual.shape)
    else:
      self.assertAllEqual(x_.shape, inverse_actual.shape)
    self.assertAllEqual(x_, inverse_actual_)


if __name__ == '__main__':
  tf.test.main()
