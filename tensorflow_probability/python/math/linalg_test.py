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

import functools

# Dependency imports
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
from hypothesis.extra import numpy as hpnp

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import math
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import fill_scale_tril
from tensorflow_probability.python.experimental import linalg as experimental_linalg
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math import linalg
from tensorflow_probability.python.math.psd_kernels import matern

JAX_MODE = False
NUMPY_MODE = False
TF_MODE = not (JAX_MODE or NUMPY_MODE)


class _CholeskyExtend(test_util.TestCase):

  def testCholeskyExtension(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    xs = tf1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    chol = tf.linalg.cholesky(mat)

    ys = rng.random_sample(3).astype(self.dtype)[:, tf.newaxis]
    ys = tf1.placeholder_with_default(
        ys, shape=ys.shape if self.use_static_shape else None)

    xsys = tf.concat([xs, ys], 0)
    new_chol_expected = tf.linalg.cholesky(k.matrix(xsys, xsys))

    new_chol = linalg.cholesky_concat(chol, k.matrix(xsys, ys))
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

    k = matern.MaternOneHalf()
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

    new_chol = linalg.cholesky_concat(chol,
                                      k.matrix(xsys, ys) + jitter(z)[:, n:])
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

  def testCholeskyUpdateXLA(self):
    self.skip_if_no_xla()
    if not (tf1.control_flow_v2_enabled() or self.use_static_shape):
      self.skipTest('TF1 broken')

    cholesky_update_fun = tf.function(linalg.cholesky_update, jit_compile=True)
    self._testCholeskyUpdate(cholesky_update_fun)

  def testCholeskyUpdate(self):
    self._testCholeskyUpdate(linalg.cholesky_update)

  def _testCholeskyUpdate(self, cholesky_update_fun):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    xs = tf1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    chol = tf.linalg.cholesky(mat)

    u = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    u = tf1.placeholder_with_default(
        u, shape=u.shape if self.use_static_shape else None)

    new_chol_expected = tf.linalg.cholesky(
        mat + tf.linalg.matmul(u, u, transpose_b=True))
    new_chol = cholesky_update_fun(chol, tf.squeeze(u, axis=-1))
    self.assertAllClose(new_chol_expected, new_chol, rtol=1e-5, atol=2e-5)
    self.assertAllEqual(tf.linalg.band_part(new_chol, -1, 0), new_chol)

  def testCholeskyUpdateBatches(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample((3, 1, 7)).astype(self.dtype)[..., tf.newaxis]
    xs = tf1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    chol = tf.linalg.cholesky(mat)

    u = rng.random_sample((1, 5, 7)).astype(self.dtype)[..., tf.newaxis]
    u = tf1.placeholder_with_default(
        u, shape=u.shape if self.use_static_shape else None)
    multiplier = rng.random_sample((2, 1, 1)).astype(self.dtype)

    new_chol_expected = tf.linalg.cholesky(
        mat + multiplier[..., np.newaxis, np.newaxis] * tf.linalg.matmul(
            u, u, transpose_b=True))
    new_chol = linalg.cholesky_update(
        chol, tf.squeeze(u, axis=-1), multiplier=multiplier)
    self.assertAllClose(new_chol_expected, new_chol, rtol=1e-5, atol=2e-5)
    self.assertAllEqual(tf.linalg.band_part(new_chol, -1, 0), new_chol)

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testCholeskyUpdateRandomized(self, data):
    target_bs = data.draw(hpnp.array_shapes())
    chol_bs, u_bs, multiplier_bs = data.draw(
        tfp_hps.broadcasting_shapes(target_bs, 3))
    l = data.draw(hps.integers(min_value=1, max_value=12))

    rng_seed = data.draw(hps.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.RandomState(seed=rng_seed)
    xs = push_apart(
        rng.uniform(size=tensorshape_util.concatenate(chol_bs, (l, 1))),
        axis=-2)
    hp.note(xs)
    xs = xs.astype(self.dtype)
    xs = tf1.placeholder_with_default(
        xs, shape=xs.shape if self.use_static_shape else None)

    k = matern.MaternOneHalf()
    jitter = lambda n: tf.linalg.eye(n, dtype=self.dtype) * 5e-5

    mat = k.matrix(xs, xs) + jitter(l)
    chol = tf.linalg.cholesky(mat)

    u = rng.uniform(size=tensorshape_util.concatenate(u_bs, (l,)))
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

    new_chol = linalg.cholesky_update(chol, u, multiplier=multiplier)
    self.assertAllClose(new_chol_expected, new_chol, rtol=1e-5, atol=2e-5)
    self.assertAllEqual(tf.linalg.band_part(new_chol, -1, 0), new_chol)


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

    pchol = linalg.pivoted_cholesky(matrix, max_rank=1)
    mat = tf.matmul(pchol, pchol, transpose_b=True)
    diag_diff_prev = self.evaluate(tf.abs(tf.linalg.diag_part(mat) - true_diag))
    diff_norm_prev = self.evaluate(
        tf.linalg.norm(mat - matrix, ord='fro', axis=[-1, -2]))
    for rank in range(2, dim + 1):
      # Specifying diag_rtol forces the full max_rank decomposition.
      pchol = linalg.pivoted_cholesky(matrix, max_rank=rank, diag_rtol=-1)
      zeros_per_col = dim - tf.math.count_nonzero(pchol, axis=-2)
      mat = tf.matmul(pchol, pchol, transpose_b=True)
      pchol_shp, diag_diff, diff_norm, zeros_per_col = self.evaluate([
          tf.shape(pchol),
          tf.abs(tf.linalg.diag_part(mat) - true_diag),
          tf.linalg.norm(mat - matrix, ord='fro', axis=[-1, -2]), zeros_per_col
      ])
      self.assertAllEqual([2, dim, rank], pchol_shp)
      self.assertAllEqual(
          np.ones([2, rank], dtype=np.bool_), zeros_per_col >= np.arange(rank))
      self.assertAllLessEqual(diag_diff - diag_diff_prev,
                              np.finfo(self.dtype).resolution)
      self.assertAllLessEqual(diff_norm - diff_norm_prev,
                              np.finfo(self.dtype).resolution)
      diag_diff_prev, diff_norm_prev = diag_diff, diff_norm

  @test_util.numpy_disable_gradient_test
  def testGradient(self):
    dim = 11
    matrix = self._random_batch_psd(dim)
    _, dmatrix = gradient.value_and_gradient(
        lambda matrix: linalg.pivoted_cholesky(matrix, max_rank=dim // 3),
        matrix)
    self.assertIsNotNone(dmatrix)
    self.assertAllGreater(tf.linalg.norm(dmatrix, ord='fro', axis=[-1, -2]), 0.)

  @test_util.tf_tape_safety_test
  def testGradientTapeCFv2(self):
    if not tf1.control_flow_v2_enabled():
      self.skipTest('Requires v2 control flow')
    dim = 11
    matrix = self._random_batch_psd(dim)
    with tf.GradientTape() as tape:
      tape.watch(matrix)
      pchol = linalg.pivoted_cholesky(matrix, max_rank=dim // 3)
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
          linalg.pivoted_cholesky(mat, max_rank=rank, diag_rtol=-1),
          atol=1e-4)

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='LinearOperatorPSDKernel not available in numpy backend.')
  def testLinopKernel(self):
    x = tf.random.uniform([10, 2], dtype=self.dtype, seed=test_util.test_seed())
    masked_shape = x.shape if self.use_static_shape else [None] * len(x.shape)
    x = tf1.placeholder_with_default(x, shape=masked_shape)
    k = matern.MaternThreeHalves()
    expected = linalg.pivoted_cholesky(k.matrix(x, x), max_rank=3)
    actual = linalg.pivoted_cholesky(
        experimental_linalg.LinearOperatorPSDKernel(k, x), max_rank=3)
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


class _LowRankCholesky(test_util.TestCase):

  def _random_batch_psd(self, dim):
    rng = test_util.test_np_rng()
    matrix = rng.random_sample([2, dim, dim])
    matrix = np.matmul(matrix, np.swapaxes(matrix, -2, -1))
    matrix = (matrix + np.diag(np.arange(dim) * .1)).astype(self.dtype)
    masked_shape = (
        matrix.shape if self.use_static_shape else [None] * len(matrix.shape))
    matrix = tf1.placeholder_with_default(matrix, shape=masked_shape)
    return matrix

  def testLowRankCholesky(self):
    if not tf.executing_eagerly() and not self.use_static_shape:
      return

    # TODO(thomaswc): Jit this.
    dim = 11
    matrix = self._random_batch_psd(dim)
    true_diag = tf.linalg.diag_part(matrix)

    pchol, r, residual_diag = linalg.low_rank_cholesky(matrix, max_rank=1)
    self.assertEqual(1, self.evaluate(r))
    self.assertEqual((2, 11), residual_diag.shape)
    mat = tf.matmul(pchol, pchol, transpose_b=True)
    diag_diff_prev = self.evaluate(tf.abs(tf.linalg.diag_part(mat) - true_diag))
    diff_norm_prev = self.evaluate(
        tf.linalg.norm(mat - matrix, ord='fro', axis=[-1, -2]))
    old_residual_trace = None
    for rank in range(2, dim + 1):
      # Specifying trace_rtol forces the full max_rank decomposition.
      pchol, r, residual_diag = linalg.low_rank_cholesky(
          matrix, max_rank=rank, trace_rtol=-1)
      self.assertEqual(rank, self.evaluate(r))
      residual_trace = tf.math.reduce_sum(residual_diag, axis=-1)
      if old_residual_trace is not None:
        self.assertTrue(self.evaluate(tf.reduce_all(
            residual_trace < old_residual_trace)))
      old_residual_trace = residual_trace
      zeros_per_col = dim - tf.math.count_nonzero(pchol, axis=-2)
      mat = tf.matmul(pchol, pchol, transpose_b=True)
      pchol_shp, diag_diff, diff_norm, zeros_per_col = self.evaluate([
          tf.shape(pchol),
          tf.abs(tf.linalg.diag_part(mat) - true_diag),
          tf.linalg.norm(mat - matrix, ord='fro', axis=[-1, -2]), zeros_per_col
      ])
      self.assertAllEqual([2, dim, rank], pchol_shp)
      self.assertAllEqual(
          np.ones([2, rank], dtype=np.bool_), zeros_per_col >= np.arange(rank),
          msg=f'For matrix {matrix}, low rank cholesky {pchol} with max_rank '
              f'{rank} has zeros per column of {zeros_per_col}'
      )
      self.assertAllLessEqual(diag_diff - diag_diff_prev,
                              2.0 * np.finfo(self.dtype).resolution)
      self.assertAllLessEqual(diff_norm - diff_norm_prev,
                              np.finfo(self.dtype).resolution)
      diag_diff_prev, diff_norm_prev = diag_diff, diff_norm

  @test_util.numpy_disable_gradient_test
  def testGradient(self):
    if not tf.executing_eagerly():
      return

    dim = 11

    def fn(matrix):
      chol, _, _ = linalg.low_rank_cholesky(matrix, max_rank=dim // 3)
      return chol
    def grad(matrix):
      _, dmatrix = gradient.value_and_gradient(fn, matrix)
      return dmatrix
    if self.jit:
      self.skip_if_no_xla()
      grad = tf.function(grad, jit_compile=True)

    matrix = self._random_batch_psd(dim)
    _, dmatrix = grad(matrix)
    self.assertIsNotNone(dmatrix)
    self.assertAllGreater(tf.linalg.norm(dmatrix, ord='fro', axis=[-1, -2]), 0.)

  @test_util.tf_tape_safety_test
  def testGradientTapeCFv2(self):
    if not tf1.control_flow_v2_enabled():
      self.skipTest('Requires v2 control flow')
    dim = 11

    def grad(matrix):
      with tf.GradientTape() as tape:
        tape.watch(matrix)
        pchol, _, _ = linalg.low_rank_cholesky(matrix, max_rank=dim // 3)
      dmatrix = tape.gradient(
          pchol, matrix, output_gradients=tf.ones_like(pchol) * .01)
      return dmatrix
    if self.jit:
      self.skip_if_no_xla()
      grad = tf.function(grad, jit_compile=True)

    matrix = self._random_batch_psd(dim)
    dmatrix = grad(matrix)
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
    max_rank = mat.shape[-1] + 1
    fns = [functools.partial(
        linalg.low_rank_cholesky, max_rank=rank, trace_rtol=-1)
           for rank in range(max_rank)]
    if self.jit:
      self.skip_if_no_xla()
      fns = [tf.function(fn, jit_compile=True) for fn in fns]

    mat = np.matmul(mat, mat.T)
    for rank in range(1, max_rank):
      lr_chol, r, _ = fns[rank](mat)
      self.assertEqual(self.evaluate(r), rank)
      self.assertAllClose(
          oracle_pchol[..., :rank], lr_chol[..., :rank], atol=1e-4)

  @test_util.disable_test_for_backend(
      disable_jax=True,
      disable_numpy=True,
      reason='LinearOperatorPSDKernel not available in jax or numpy backends.')
  def testLinopKernel(self):
    if not tf.executing_eagerly() and not self.use_static_shape:
      return
    x = tf.random.uniform([10, 2], dtype=self.dtype, seed=test_util.test_seed())
    masked_shape = x.shape if self.use_static_shape else [None] * len(x.shape)
    x = tf1.placeholder_with_default(x, shape=masked_shape)
    k = matern.MaternThreeHalves()
    expected = linalg.low_rank_cholesky(k.matrix(x, x), max_rank=3)
    actual = linalg.low_rank_cholesky(
        experimental_linalg.LinearOperatorPSDKernel(k, x), max_rank=3)
    expected, actual = self.evaluate([expected, actual])
    self.assertAllClose(expected, actual)


@test_util.test_all_tf_execution_regimes
class LowRankCholesky32Static(_LowRankCholesky):
  dtype = np.float32
  use_static_shape = True
  jit = False


@test_util.test_all_tf_execution_regimes
class LowRankCholesky64Dynamic(_LowRankCholesky):
  dtype = np.float64
  use_static_shape = False
  jit = False


@test_util.test_all_tf_execution_regimes
class LowRankCholeskyJit(_LowRankCholesky):
  dtype = np.float32
  use_static_shape = True
  jit = True


del _LowRankCholesky


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

    y = linalg.lu_reconstruct(*tf.linalg.lu(x), validate_args=True)
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

    y = linalg.lu_reconstruct(*tf.linalg.lu(x), validate_args=True)
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

    y = linalg.lu_matrix_inverse(*tf.linalg.lu(x), validate_args=True)
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

    y = linalg.lu_matrix_inverse(*tf.linalg.lu(x), validate_args=True)
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
    y = linalg.lu_solve(lower_upper, perm, rhs, validate_args=True)
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
    y = linalg.lu_solve(lower_upper, perm, rhs, validate_args=True)
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
      x = self._make_sparse_placeholder(math.dense_to_sparse(x_))
    else:
      x = self._make_placeholder(x_)

    y = self._make_placeholder(y_)

    z = linalg.sparse_or_dense_matmul(x, y)
    z_ = self.evaluate(z)

    if self.use_static_shape:
      batch_shape = x_.shape[:-2]
      self.assertAllEqual(z_.shape, batch_shape + (x_.shape[-2], y_.shape[-1]))

    self.assertAllClose(z_, np.matmul(x_, y_), atol=0., rtol=1e-3)

  def verify_sparse_dense_matvecmul(self, x_, y_):
    if self.use_sparse_tensor:
      x = self._make_sparse_placeholder(math.dense_to_sparse(x_))
    else:
      x = self._make_placeholder(x_)

    y = self._make_placeholder(y_)

    z = linalg.sparse_or_dense_matvecmul(x, y)
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

if TF_MODE:
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

    actual, grad_actual = gradient.value_and_gradient(
        lambda x: linalg.fill_triangular(  # pylint: disable=g-long-lambda
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

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakes1x1TriLower(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(3, int(1*2/2)))

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesNoBatchTriLower(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(int(4*5/2)))

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesBatchTriLower(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(3*4/2)))

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesBatchTriLowerUnknownShape(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(3*4/2)), use_deferred_shape=True)

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesBatch7x7TriLowerUnknownShape(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(7*8/2)), use_deferred_shape=True)

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesBatch7x7TriLower(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(7*8/2)))

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakes1x1TriUpper(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(3, int(1*2/2)), upper=True)

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesNoBatchTriUpper(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(int(4*5/2)), upper=True)

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesBatchTriUpper(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 2, int(3*4/2)), upper=True)

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesBatchTriUpperUnknownShape(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 2, int(3*4/2)),
                   use_deferred_shape=True,
                   upper=True)

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesBatch7x7TriUpperUnknownShape(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(7*8/2)),
                   use_deferred_shape=True,
                   upper=True)

  @test_util.numpy_disable_gradient_test
  def testCorrectlyMakesBatch7x7TriUpper(self):
    rng = test_util.test_np_rng()
    self._run_test(rng.randn(2, 3, int(7*8/2)), upper=True)


@test_util.test_all_tf_execution_regimes
class FillTriangularInverseTest(FillTriangularTest):

  def testFillTriangularJitIssue1210(self):
    self.skip_if_no_xla()

    @tf.function(jit_compile=True)
    def transform(x):
      return fill_scale_tril.FillScaleTriL(
          diag_bijector=exp.Exp(), diag_shift=None).inverse(x)

    self.evaluate(transform(tf.linalg.eye(3, dtype=tf.float32)))

  def _run_test(self, x_, use_deferred_shape=False, **kwargs):
    x_ = np.asarray(x_)
    static_shape = None if use_deferred_shape else x_.shape
    x_pl = tf1.placeholder_with_default(x_, shape=static_shape)

    zeros_like_x_pl = (x_pl * tf.stop_gradient(x_pl - 1.)
                       - tf.stop_gradient(x_pl * (x_pl - 1.)))
    x = x_pl + zeros_like_x_pl
    actual = linalg.fill_triangular(x, **kwargs)
    inverse_actual = linalg.fill_triangular_inverse(actual, **kwargs)

    inverse_actual_ = self.evaluate(inverse_actual)

    if use_deferred_shape and not tf.executing_eagerly():
      # TensorShape(None) is not None.
      self.assertEqual(None, inverse_actual.shape)  # pylint: disable=g-generic-assert
    else:
      self.assertAllEqual(x_.shape, inverse_actual.shape)
    self.assertAllEqual(x_, inverse_actual_)


class _HPSDLogDetTest(test_util.TestCase):

  def testEqualsHPSDLogDet(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    self.assertAllClose(
        self.evaluate(linalg.hpsd_logdet(mat)),
        self.evaluate(tf.linalg.logdet(mat)),
        rtol=3e-6)

  def testUsesOverridenCholeskyFactor(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    my_cholesky = tf.linalg.eye(num_rows=7, dtype=mat.dtype)
    # Log Det should be zero.
    logdet_ = self.evaluate(
        linalg.hpsd_logdet(mat, cholesky_matrix=my_cholesky))
    self.assertAllClose(logdet_, np.zeros_like(logdet_))

  @test_util.numpy_disable_gradient_test
  def testHPSDLogDetGradient(self):
    rng = test_util.test_np_rng()
    # Test that this respects batch shapes.
    xs = rng.random_sample((6, 10)).astype(self.dtype)[..., tf.newaxis]
    k = matern.MaternThreeHalves()
    mat = k.matrix(xs, xs)
    # Ensure that the matrix is more well conditioned.
    mat = tf.linalg.set_diag(mat, tf.linalg.diag_part(mat) + 1e-1)

    # Use the square of logdet to test that the upstream gradients are properly
    # incorporated.
    _, naive_gradient = gradient.value_and_gradient(
        lambda x: tf.linalg.logdet(x)**2, mat)
    _, custom_gradient = gradient.value_and_gradient(
        lambda x: linalg.hpsd_logdet(x)**2, mat)
    naive_gradient, custom_gradient = self.evaluate(
        [naive_gradient, custom_gradient])
    if self.dtype == np.float32:
      rtol = 9e-3
    else:
      rtol = 5e-5
    self.assertAllClose(naive_gradient, custom_gradient, rtol=rtol)

  @test_util.numpy_disable_gradient_test
  def testHPSDLogDetGradientWithCholesky(self):
    rng = test_util.test_np_rng()
    # Test that this respects batch shapes.
    xs = rng.random_sample((6, 10)).astype(self.dtype)[..., tf.newaxis]
    k = matern.MaternThreeHalves()
    mat = k.matrix(xs, xs)

    # Check that `logdet` gradient uses the cholesky factor.
    identity = self.evaluate(
        tf.eye(num_rows=10, batch_shape=[6], dtype=self.dtype))
    cholesky_factor = 3. * identity

    _, cholesky_gradient = gradient.value_and_gradient(
        lambda x: linalg.hpsd_logdet(x, cholesky_matrix=cholesky_factor)**2,
        mat)
    cholesky_gradient = self.evaluate(cholesky_gradient)
    # Since the cholesky factors are rescaled identity matrices c * I,
    # using the chain rule we have
    # grad logdet(A)**2 -> 2 * logdet(A) * grad logdet(A), so
    # this should be 4 * n * log(c) * (1 / c**2) I
    # where `n` is dimension of I.
    self.assertAllClose(
        cholesky_gradient, 4. * 10 * np.log(3.) / 9. * identity)


@test_util.test_all_tf_execution_regimes
class HPSDLogDet32Test(_HPSDLogDetTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class HPSDLogDet64Test(_HPSDLogDetTest):
  dtype = np.float64


del _HPSDLogDetTest


class _HPSDSolveTest(test_util.TestCase):

  def testEqualsHPSDSolve(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    y = rng.random_sample((7, 3)).astype(self.dtype)
    self.assertAllClose(
        self.evaluate(linalg.hpsd_solve(mat, y)),
        self.evaluate(tf.linalg.solve(mat, y)),
        rtol=6e-4)

  def testEqualsHPSDSolvevec(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    y = rng.random_sample((7)).astype(self.dtype)
    self.assertAllClose(
        self.evaluate(linalg.hpsd_solvevec(mat, y)),
        self.evaluate(linalg.hpsd_solve(mat, y[..., tf.newaxis])[..., 0]),
        rtol=6e-4)

  def testUsesOverridenCholeskyFactor(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    y = rng.random_sample((7, 3)).astype(self.dtype)
    my_cholesky = tf.linalg.eye(num_rows=7, dtype=mat.dtype)
    solve_ = self.evaluate(
        linalg.hpsd_solve(
            mat, y, cholesky_matrix=my_cholesky))
    self.assertAllClose(solve_, y)

  @test_util.numpy_disable_gradient_test
  def testSolveGradient(self):
    rng = test_util.test_np_rng()
    # Test that this respects batch shapes.
    xs = rng.random_sample((6, 10)).astype(self.dtype)[..., tf.newaxis]
    k = matern.MaternThreeHalves()
    mat = k.matrix(xs, xs)
    # Ensure that the matrix is more well conditioned.
    mat = tf.linalg.set_diag(mat, tf.linalg.diag_part(mat) + 2.)
    rhs = tf.convert_to_tensor(
        rng.random_sample((3, 1, 10, 2)).astype(self.dtype))

    def naive_hpsd_solve(matrix, rhs):
      cholesky_matrix = tf.linalg.cholesky(matrix)
      chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
      return chol_linop.solve(chol_linop.solve(rhs), adjoint=True)

    # Use the reduce_sum of hpsd_solve to test that the upstream
    # gradients are properly incorporated.

    _, naive_gradient = gradient.value_and_gradient(
        lambda x: tf.reduce_sum(naive_hpsd_solve(x, rhs)), mat)
    _, custom_gradient = gradient.value_and_gradient(
        lambda x: tf.reduce_sum(linalg.hpsd_solve(x, rhs)), mat)
    naive_gradient, custom_gradient = self.evaluate(
        [naive_gradient, custom_gradient])
    self.assertAllClose(naive_gradient, custom_gradient, rtol=2e-4)

    _, naive_gradient = gradient.value_and_gradient(
        lambda x: tf.reduce_sum(naive_hpsd_solve(mat, x)), rhs)
    _, custom_gradient = gradient.value_and_gradient(
        lambda x: tf.reduce_sum(linalg.hpsd_solve(mat, x)), rhs)
    naive_gradient, custom_gradient = self.evaluate(
        [naive_gradient, custom_gradient])
    self.assertAllClose(naive_gradient, custom_gradient, rtol=3e-6)

  @test_util.numpy_disable_gradient_test
  def testSolveGradientWithCholesky(self):
    rng = test_util.test_np_rng()
    # Test that this respects batch shapes.
    xs = rng.random_sample((6, 10)).astype(self.dtype)[..., tf.newaxis]
    k = matern.MaternThreeHalves()
    mat = k.matrix(xs, xs)
    rhs = tf.convert_to_tensor(
        rng.random_sample((3, 1, 10, 3)).astype(self.dtype))

    identity = self.evaluate(
        tf.eye(num_rows=10, batch_shape=[6], dtype=self.dtype))
    cholesky_factor = (3. * identity).astype(self.dtype)
    mat_for_factor = tf.convert_to_tensor((9. * identity).astype(self.dtype))

    hpsd_solve = functools.partial(
        linalg.hpsd_solve, cholesky_matrix=cholesky_factor)

    _, [actual_mat_gradient, actual_rhs_gradient] = self.evaluate((
        gradient.value_and_gradient(
            lambda x, y: tf.reduce_sum(hpsd_solve(x, y)), mat, rhs)))

    _, [expected_mat_gradient, expected_rhs_gradient] = self.evaluate((
        gradient.value_and_gradient(
            lambda x, y: tf.reduce_sum(linalg.hpsd_solve(x, y)),
            mat_for_factor, rhs)))

    self.assertAllClose(actual_mat_gradient, expected_mat_gradient)
    self.assertAllClose(actual_rhs_gradient, expected_rhs_gradient)

  def testNoJaxTracerLeak(self):
    if not JAX_MODE:
      return
    import jax  # pylint:disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint:disable=g-import-not-at-top
    def f(carry, x):
      return (carry / 2 + linalg.hpsd_solve(jnp.eye(3), x,
                                            cholesky_matrix=jnp.eye(3)),
              ())

    def g(t):
      carry, _ = jax.lax.scan(f, jnp.zeros([3, 3]), xs=jnp.zeros([7, 3, 3]) + t)
      return carry.sum()

    with jax.checking_leaks():
      jax.grad(g)(jnp.eye(3))


@test_util.test_all_tf_execution_regimes
class HPSDSolve32Test(_HPSDSolveTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class HPSDSolve64Test(_HPSDSolveTest):
  dtype = np.float64


del _HPSDSolveTest


class _QuadraticFormSolveTest(test_util.TestCase):

  def testEqualsQuadraticFormSolve(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    y = rng.random_sample((7, 3)).astype(self.dtype)
    self.assertAllClose(
        self.evaluate(linalg.hpsd_quadratic_form_solve(mat, y)),
        self.evaluate(y.T  @ tf.linalg.solve(mat, y)),
        rtol=6e-5)

  def testUsesOverridenCholeskyFactor(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    y = rng.random_sample((7, 3)).astype(self.dtype)
    my_cholesky = tf.linalg.eye(num_rows=7, dtype=mat.dtype)
    # Symmetric Solve should be just the norm squared.
    symsolve_ = self.evaluate(
        linalg.hpsd_quadratic_form_solve(
            mat, y, cholesky_matrix=my_cholesky))
    self.assertAllClose(symsolve_, y.T @ y)

  @test_util.numpy_disable_gradient_test
  def testQuadraticFormSolveGradient(self):
    rng = test_util.test_np_rng()
    # Test that this respects batch shapes.
    xs = rng.random_sample((6, 10)).astype(self.dtype)[..., tf.newaxis]
    k = matern.MaternThreeHalves()
    mat = k.matrix(xs, xs)
    # Ensure that the matrix is more well conditioned.
    mat = tf.linalg.set_diag(mat, tf.linalg.diag_part(mat) + 1.)
    rhs = tf.convert_to_tensor(
        rng.random_sample((3, 1, 10, 2)).astype(self.dtype))

    def naive_hpsd_quadratic_form_solve(matrix, rhs):
      cholesky_matrix = tf.linalg.cholesky(matrix)
      chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
      result = chol_linop.solve(rhs)
      return tf.linalg.matmul(result, result, transpose_a=True)

    # Use the square of hpsd_quadratic_form_solve to test that the upstream
    # gradients are properly incorporated.
    _, naive_gradient = gradient.value_and_gradient(
        lambda x: tf.reduce_sum(naive_hpsd_quadratic_form_solve(x, rhs)), mat)
    _, custom_gradient = gradient.value_and_gradient(
        lambda x: tf.reduce_sum(linalg.hpsd_quadratic_form_solve(x, rhs)), mat)
    naive_gradient, custom_gradient = self.evaluate(
        [naive_gradient, custom_gradient])
    self.assertAllClose(naive_gradient, custom_gradient, rtol=2e-4)

    _, naive_gradient = gradient.value_and_gradient(
        lambda x: tf.reduce_sum(naive_hpsd_quadratic_form_solve(x, rhs)), mat)
    _, custom_gradient = gradient.value_and_gradient(
        lambda x: tf.reduce_sum(linalg.hpsd_quadratic_form_solve(x, rhs)), mat)
    naive_gradient, custom_gradient = self.evaluate(
        [naive_gradient, custom_gradient])
    self.assertAllClose(naive_gradient, custom_gradient, rtol=3e-6)

  @test_util.numpy_disable_gradient_test
  def testQuadraticFormSolveGradientWithCholesky(self):
    rng = test_util.test_np_rng()
    # Test that this respects batch shapes.
    xs = rng.random_sample((6, 10)).astype(self.dtype)[..., tf.newaxis]
    k = matern.MaternThreeHalves()
    mat = k.matrix(xs, xs)
    rhs = tf.convert_to_tensor(
        rng.random_sample((3, 1, 10, 3)).astype(self.dtype))

    identity = self.evaluate(
        tf.eye(num_rows=10, batch_shape=[6], dtype=self.dtype))
    cholesky_factor = (3. * identity).astype(self.dtype)
    mat_for_factor = tf.convert_to_tensor((9. * identity).astype(self.dtype))

    hpsd_quadratic_form_solve = functools.partial(
        linalg.hpsd_quadratic_form_solve, cholesky_matrix=cholesky_factor)

    _, [actual_mat_gradient, actual_rhs_gradient] = self.evaluate((
        gradient.value_and_gradient(
            lambda x, y: hpsd_quadratic_form_solve(x, y)**2, mat, rhs)))

    _, [expected_mat_gradient, expected_rhs_gradient] = self.evaluate((
        gradient.value_and_gradient(
            lambda x, y: linalg.hpsd_quadratic_form_solve(x, y)**2,
            mat_for_factor, rhs)))

    self.assertAllClose(actual_mat_gradient, expected_mat_gradient)
    self.assertAllClose(actual_rhs_gradient, expected_rhs_gradient)

  def testEqualsQuadraticFormSolveVec(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    y = rng.random_sample(7).astype(self.dtype)
    self.assertAllClose(
        self.evaluate(linalg.hpsd_quadratic_form_solvevec(mat, y)),
        tf.reduce_sum(y.T  * tf.squeeze(
            tf.linalg.solve(mat, y[..., tf.newaxis]), axis=-1), axis=-1),
        rtol=5e-5)

  def testUsesOverridenCholeskyFactorSolveVec(self):
    rng = test_util.test_np_rng()
    xs = rng.random_sample(7).astype(self.dtype)[:, tf.newaxis]
    k = matern.MaternOneHalf()
    mat = k.matrix(xs, xs)
    y = rng.random_sample(7).astype(self.dtype)
    my_cholesky = tf.linalg.eye(num_rows=7, dtype=mat.dtype)
    # Symmetric Solve should be just the norm squared.
    symsolve_ = self.evaluate(
        linalg.hpsd_quadratic_form_solvevec(
            mat, y, cholesky_matrix=my_cholesky))
    self.assertAllClose(symsolve_, np.linalg.norm(y)**2)

  @test_util.numpy_disable_gradient_test
  def testQuadraticFormSolveVecGradient(self):
    rng = test_util.test_np_rng()
    # Test that this respects batch shapes.
    xs = rng.random_sample((6, 10)).astype(self.dtype)[..., tf.newaxis]
    k = matern.MaternThreeHalves()
    mat = k.matrix(xs, xs)
    # Ensure that the matrix is more well conditioned.
    mat = tf.linalg.set_diag(mat, tf.linalg.diag_part(mat) + 1e-1)
    rhs = tf.convert_to_tensor(rng.random_sample((3, 1, 10)).astype(self.dtype))

    def naive_hpsd_quadratic_form_solvevec(matrix, rhs):
      cholesky_matrix = tf.linalg.cholesky(matrix)
      chol_linop = tf.linalg.LinearOperatorLowerTriangular(cholesky_matrix)
      return tf.reduce_sum(
          tf.math.square(chol_linop.solvevec(rhs)), axis=-1)

    # Use the square of hpsd_quadratic_form_solvevec to test that the upstream
    # gradients are properly incorporated.
    _, naive_gradient = gradient.value_and_gradient(
        lambda x: naive_hpsd_quadratic_form_solvevec(x, rhs)**2, mat)
    _, custom_gradient = gradient.value_and_gradient(
        lambda x: linalg.hpsd_quadratic_form_solvevec(x, rhs)**2, mat)
    naive_gradient, custom_gradient = self.evaluate(
        [naive_gradient, custom_gradient])
    self.assertAllClose(naive_gradient, custom_gradient, rtol=5e-4)

    _, naive_gradient = gradient.value_and_gradient(
        lambda x: naive_hpsd_quadratic_form_solvevec(mat, x)**2, rhs)
    _, custom_gradient = gradient.value_and_gradient(
        lambda x: linalg.hpsd_quadratic_form_solvevec(mat, x)**2, rhs)
    naive_gradient, custom_gradient = self.evaluate(
        [naive_gradient, custom_gradient])
    self.assertAllClose(naive_gradient, custom_gradient, rtol=3e-6)

  @test_util.numpy_disable_gradient_test
  def testQuadraticFormSolveVecGradientWithCholesky(self):
    rng = test_util.test_np_rng()
    # Test that this respects batch shapes.
    xs = rng.random_sample((6, 10)).astype(self.dtype)[..., tf.newaxis]
    k = matern.MaternThreeHalves()
    mat = k.matrix(xs, xs)
    rhs = tf.convert_to_tensor(rng.random_sample((3, 1, 10)).astype(self.dtype))

    identity = self.evaluate(
        tf.eye(num_rows=10, batch_shape=[6], dtype=self.dtype))
    cholesky_factor = (3. * identity).astype(self.dtype)
    mat_for_factor = tf.convert_to_tensor((9. * identity).astype(self.dtype))

    hpsd_quadratic_form_solvevec = functools.partial(
        linalg.hpsd_quadratic_form_solvevec, cholesky_matrix=cholesky_factor)

    _, [actual_mat_gradient, actual_rhs_gradient] = self.evaluate((
        gradient.value_and_gradient(
            lambda x, y: hpsd_quadratic_form_solvevec(x, y)**2, mat, rhs)))

    _, [expected_mat_gradient, expected_rhs_gradient] = self.evaluate((
        gradient.value_and_gradient(
            lambda x, y: linalg.hpsd_quadratic_form_solvevec(x, y)**2,
            mat_for_factor, rhs)))

    self.assertAllClose(actual_mat_gradient, expected_mat_gradient)
    self.assertAllClose(actual_rhs_gradient, expected_rhs_gradient)

  def testNoJaxTracerLeak(self):
    if not JAX_MODE:
      return
    import jax  # pylint:disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint:disable=g-import-not-at-top
    def f(carry, x):
      return (carry / 2 + linalg.hpsd_quadratic_form_solve(
          jnp.eye(3), x, cholesky_matrix=jnp.eye(3)), ())

    def g(t):
      carry, _ = jax.lax.scan(f, jnp.zeros([3, 3]), xs=jnp.zeros([7, 3, 3]) + t)
      return carry.sum()

    with jax.checking_leaks():
      jax.grad(g)(jnp.eye(3))


@test_util.test_all_tf_execution_regimes
class QuadraticFormSolve32Test(_QuadraticFormSolveTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class QuadraticFormSolve64Test(_QuadraticFormSolveTest):
  dtype = np.float64


del _QuadraticFormSolveTest


if __name__ == '__main__':
  test_util.main()
