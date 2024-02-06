# Copyright 2024 The TensorFlow Probability Authors.
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
"""Tests for partial_lanczos.py."""

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.fastgp import mbcg
from tensorflow_probability.python.experimental.fastgp import partial_lanczos
from absl.testing import absltest

# pylint: disable=invalid-name


class _PartialLanczosTest(absltest.TestCase):
  def test_gram_schmidt(self):
    w = jnp.ones((5, 1), dtype=self.dtype)
    v = partial_lanczos.gram_schmidt(
        jnp.array([[[1.0, 0, 0, 0, 0],
                    [0, 1.0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]], dtype=self.dtype),
        w)
    self.assertEqual((5, 1), v.shape)
    self.assertEqual(0.0, v[0][0])
    self.assertEqual(0.0, v[1][0])
    self.assertGreater(jnp.linalg.norm(v[:, 0]), 1e-6)

  def test_partial_lanczos_identity(self):
    A = jnp.identity(10).astype(self.dtype)
    v = jnp.ones((10, 1)).astype(self.dtype)
    Q, T = partial_lanczos.partial_lanczos(
        lambda x: A @ x, v, jax.random.PRNGKey(2), 10)
    np.testing.assert_allclose(jnp.identity(10), Q[0] @ Q[0].T, atol=1e-6)
    np.testing.assert_allclose(
        mbcg.tridiagonal_det(T.diag[0, :], T.off_diag[0, :]),
        1.0, rtol=1e-5)

  def test_diagonal_matrix_heavily_imbalanced(self):
    A = jnp.diag(jnp.array([
        1e-3, 1., 2., 3., 4., 10000.], dtype=self.dtype))
    v = jnp.ones((6, 1)).astype(self.dtype)
    Q, T = partial_lanczos.partial_lanczos(
        lambda x: A @ x, v, jax.random.PRNGKey(9), 6)
    atol = 1e-6
    det_rtol = 1e-6
    if self.dtype == np.float32:
      atol = 2e-3
      det_rtol = 0.26
    np.testing.assert_allclose(jnp.identity(6), Q[0] @ Q[0].T, atol=atol)
    np.testing.assert_allclose(
        mbcg.tridiagonal_det(T.diag[0, :], T.off_diag[0, :]),
        240., rtol=det_rtol)

  def test_partial_lanczos_full_lanczos(self):
    A = jnp.array([[1.0, 1.0], [1.0, 4.0]], dtype=self.dtype)
    v = jnp.array([[-1.0], [1.0]], dtype=self.dtype)
    Q, T = partial_lanczos.partial_lanczos(
        lambda x: A @ x, v, jax.random.PRNGKey(3), 2)
    np.testing.assert_allclose(jnp.identity(2), Q[0] @ Q[0].T, atol=1e-6)
    np.testing.assert_allclose(
        mbcg.tridiagonal_det(T.diag[0, :], T.off_diag[0, :]),
        3.0, rtol=1e-5)

  def test_partial_lanczos_with_jit(self):
    def partial_lanczos_pure_tensor(A, v):
      return partial_lanczos.partial_lanczos(
          lambda x: A @ x, v, jax.random.PRNGKey(4), 2)

    partial_lanczos_jit = jax.jit(partial_lanczos_pure_tensor)

    A = jnp.array([[1.0, 1.0], [1.0, 4.0]], dtype=self.dtype)
    v = jnp.array([[-1.0], [1.0]], dtype=self.dtype)
    Q, T = partial_lanczos_jit(A, v)
    np.testing.assert_allclose(jnp.identity(2), Q[0] @ Q[0].T, atol=1e-6)
    np.testing.assert_allclose(
        mbcg.tridiagonal_det(T.diag[0, :], T.off_diag[0, :]),
        3.0, rtol=1e-5)

  def test_partial_lanczos_with_batching(self):
    v = jnp.zeros((10, 3), dtype=self.dtype)
    v = v.at[:, 0].set(jnp.ones(10))
    v = v.at[:, 1].set(jnp.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1]))
    v = v.at[:, 2].set(jnp.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    Q, T = partial_lanczos.partial_lanczos(
        lambda x: x, v, jax.random.PRNGKey(5), 10)
    self.assertEqual(Q.shape, (3, 10, 10))
    self.assertEqual(T.diag.shape, (3, 10))
    np.testing.assert_allclose(
        Q[0, 0, :], jnp.ones(10) / jnp.sqrt(10.0))
    np.testing.assert_allclose(
        Q[1, 0, :], v[:, 1] / jnp.sqrt(10.0))

  def test_make_lanczos_preconditioner(self):
    kernel = jnp.identity(10).astype(self.dtype)
    preconditioner = partial_lanczos.make_lanczos_preconditioner(
        kernel, jax.random.PRNGKey(5))
    log_det = preconditioner.log_abs_determinant()
    self.assertAlmostEqual(0.0, log_det, places=4)
    out = preconditioner.solve(jnp.identity(10))
    np.testing.assert_allclose(out, jnp.identity(10), atol=9e-2)
    kernel = jnp.identity(100).astype(self.dtype)
    preconditioner = partial_lanczos.make_lanczos_preconditioner(
        kernel, jax.random.PRNGKey(6))
    # TODO(thomaswc): Investigate ways to improve the numerical stability
    # here so that log_det is closer to zero than this.
    log_det = preconditioner.log_abs_determinant()
    self.assertLess(jnp.abs(log_det), 10.0)
    out = preconditioner.solve(jnp.identity(100))
    np.testing.assert_allclose(out, jnp.identity(100), atol=0.2)

  def test_preconditioner_preserves_psd(self):
    M = jnp.array([[2.6452732, -1.4553788, -0.5272188, 0.524349],
                   [-1.4553788, 4.4274387, 0.21998158, 1.8666775],
                   [-0.5272188, 0.21998158, 2.4756536, -0.5257966],
                   [0.524349, 1.8666775, -0.5257966, 2.889879]]).astype(
                       self.dtype)
    orig_eigenvalues = jnp.linalg.eigvalsh(M)
    self.assertFalse((orig_eigenvalues < 0).any())

    preconditioner = partial_lanczos.make_lanczos_preconditioner(
        M, jax.random.PRNGKey(7))
    preconditioned_M = preconditioner.solve(M)
    after_eigenvalues = jnp.linalg.eigvalsh(preconditioned_M)
    self.assertFalse((after_eigenvalues < 0).any())

  def test_my_tridiagonal_solve(self):
    empty = jnp.array([]).astype(self.dtype)
    self.assertEqual(
        0,
        partial_lanczos.my_tridiagonal_solve(
            empty, empty, empty, empty).size)

    np.testing.assert_allclose(
        jnp.array([2.5]),
        partial_lanczos.my_tridiagonal_solve(
            jnp.array([0.0], dtype=self.dtype),
            jnp.array([2.0], dtype=self.dtype),
            jnp.array([0.0], dtype=self.dtype),
            jnp.array([5.0], dtype=self.dtype)))

    np.testing.assert_allclose(
        jnp.array([-4.5, 3.5]),
        partial_lanczos.my_tridiagonal_solve(
            jnp.array([0.0, 1.0], dtype=self.dtype),
            jnp.array([2.0, 3.0], dtype=self.dtype),
            jnp.array([4.0, 0.0], dtype=self.dtype),
            jnp.array([5.0, 6.0], dtype=self.dtype)))

    np.testing.assert_allclose(
        jnp.array([-33.0/2.0, 115.0/12.0, -11.0/6.0])[:, jnp.newaxis],
        partial_lanczos.my_tridiagonal_solve(
            jnp.array([0.0, 1.0, 2.0], dtype=self.dtype),
            jnp.array([3.0, 4.0, 5.0], dtype=self.dtype),
            jnp.array([6.0, 7.0, 0.0], dtype=self.dtype),
            jnp.array([8.0, 9.0, 10.0], dtype=self.dtype)[:, jnp.newaxis]),
        atol=1e-6
    )

  def test_psd_solve_multishift(self):
    v = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=self.dtype)
    solutions = partial_lanczos.psd_solve_multishift(
        lambda x: x,
        v[:, jnp.newaxis],
        jnp.array([0.0, 2.0, -1.0], dtype=self.dtype),
        jax.random.PRNGKey(8))
    np.testing.assert_allclose(
        solutions[:, 0, :],
        [[1.0, 1.0, 1.0, 1.0],
         [-1.0, -1.0, -1.0, -1.0],
         [0.5, 0.5, 0.5, 0.5]])


class PartialLanczosTestFloat32(_PartialLanczosTest):
  dtype = np.float32


class PartialLanczosTestFloat64(_PartialLanczosTest):
  dtype = np.float64


del _PartialLanczosTest


if __name__ == "__main__":
  config.update("jax_enable_x64", True)
  absltest.main()
