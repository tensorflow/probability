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
"""Tests for mbcg.py."""

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import scipy
from tensorflow_probability.python.experimental.fastgp import mbcg
from absl.testing import absltest

# pylint: disable=invalid-name


class _MbcgTest(absltest.TestCase):

  def test_modified_batched_conjugate_gradients(self):
    A = jnp.array([[1.0, 1.0], [1.0, 4.0]], dtype=self.dtype)
    def multiplier(B):
      return A @ B
    v = jnp.array([4.0, 6.0], dtype=self.dtype)
    w = v[:, jnp.newaxis]
    z = multiplier(w)
    def identity(x):
      return x
    inverse, t = mbcg.modified_batched_conjugate_gradients(
        multiplier, z, identity)
    np.testing.assert_allclose(v, inverse[:, 0], rtol=1e-6)
    # The tridiagonal matrix should have approximately the same determinant
    # as A.
    np.testing.assert_allclose(
        mbcg.tridiagonal_det(t.diag[0], t.off_diag[0]),
        3.0, rtol=1e-6)

    # Now, with jit.
    def mbcg_pure_tensor(M, B):
      return mbcg.modified_batched_conjugate_gradients(
          lambda x: M @ x, B, identity)
    mbcg_jit = jax.jit(mbcg_pure_tensor)
    inverse2, t2 = mbcg_jit(A, z)
    np.testing.assert_allclose(v, inverse2[:, 0], rtol=1e-6)
    # The tridiagonal matrix should have approximately the same determinant
    # as A.
    np.testing.assert_allclose(
        mbcg.tridiagonal_det(t2.diag[0], t2.off_diag[0]),
        3.0, rtol=1e-6)

  def test_mbcg_scalar(self):
    A = jnp.array([[2.0]], dtype=self.dtype)
    def multiplier(B):
      return A @ B
    v = jnp.array([5.0], dtype=self.dtype)
    w = v[:, jnp.newaxis]
    z = multiplier(w)
    def identity(x):
      return x
    inverse, t = mbcg.modified_batched_conjugate_gradients(
        multiplier, z, identity)
    np.testing.assert_allclose(v, inverse[:, 0], rtol=1e-6)
    np.testing.assert_allclose(t.diag[0][0], 2.0, rtol=1e-6)

  def test_mbcg_three_by_three(self):
    A = jnp.array(
        [[1.0, 0.0, 1.0], [0.0, 4.0, 0.0], [1.0, 0.0, 6.0]],
        dtype=self.dtype)
    w = jnp.array([7.0, 8.0, 9.0], dtype=self.dtype)[:, jnp.newaxis]
    inverse, t = mbcg.modified_batched_conjugate_gradients(
        lambda B: A @ B, A @ w, lambda x: x)
    np.testing.assert_allclose(w[:, 0], inverse[:, 0], rtol=1e-6)
    np.testing.assert_allclose(
        mbcg.tridiagonal_det(t.diag[0], t.off_diag[0]),
        20.0, rtol=1e-6)

  def test_mbcg_identity(self):
    W = np.random.rand(10, 5).astype(self.dtype)
    inverse, tridiagonals = mbcg.modified_batched_conjugate_gradients(
        lambda B: B, W, lambda x: x)
    np.testing.assert_allclose(inverse, W)
    np.testing.assert_allclose(
        tridiagonals.off_diag[0],
        jnp.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=self.dtype))
    np.testing.assert_allclose(
        tridiagonals.diag[0],
        jnp.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=self.dtype))

  def test_mbcg_diagonal(self):
    A = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=self.dtype))
    w = jnp.array([1.0, 0.5, 1.5, 2.0, -1.0], dtype=self.dtype)[:, jnp.newaxis]
    _, tridiagonals = mbcg.modified_batched_conjugate_gradients(
        lambda B: A @ B, A @ w, lambda x: x
    )
    evalues, _ = scipy.linalg.eigh_tridiagonal(
        tridiagonals.diag[0], tridiagonals.off_diag[0]
    )
    np.testing.assert_allclose(jnp.diag(A), sorted(evalues), rtol=1e-6)

  def test_mbcg_diagonal2(self):
    A = jnp.diag(
        jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=self.dtype))
    w = jnp.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
                  dtype=self.dtype)[:, jnp.newaxis]
    _, tridiagonals = mbcg.modified_batched_conjugate_gradients(
        lambda B: A @ B, A @ w, lambda x: x
    )
    evalues, _ = scipy.linalg.eigh_tridiagonal(
        tridiagonals.diag[0], tridiagonals.off_diag[0]
    )
    np.testing.assert_allclose(jnp.diag(A), sorted(evalues), rtol=1e-6)

  def test_mbcg_batching(self):
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]], dtype=self.dtype)
    W = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=self.dtype)
    P = jnp.array([[0.5, 0.0], [0.0, 1.0/3.0]], dtype=self.dtype)
    preconditioner_fn = lambda B: P @ B
    inverses, tridiagonals = mbcg.modified_batched_conjugate_gradients(
        lambda B: A @ B, A @ W, preconditioner_fn)
    np.testing.assert_allclose(inverses, W, atol=0.1)
    np.testing.assert_allclose(
        jnp.array([[1.29, 0.74], [1.35, 0.66], [1.37, 0.63]], dtype=self.dtype),
        tridiagonals.diag, rtol=0.1)
    np.testing.assert_allclose(
        jnp.array([[0.34], [0.23], [0.18]], dtype=self.dtype),
        tridiagonals.off_diag, rtol=0.1)

  def test_mbcg_max_iters(self):
    A = jax.random.normal(
        jax.random.PRNGKey(2), (30, 30)).astype(self.dtype)
    A = A @ A.T
    # Ensure it is diagonally dominant.
    i, j = jnp.diag_indices(30)
    A = A.at[..., i, j].set(0.)
    A = A.at[..., i, j].set(10. * jnp.sum(jnp.abs(A), axis=-1))
    # Finally divide by the diagonal to ensure small condition number.
    A = A / jnp.diag(A)[..., jnp.newaxis]

    _, t = mbcg.modified_batched_conjugate_gradients(
        lambda t: jnp.matmul(A, t),
        B=jnp.ones([A.shape[-1], 1], dtype=A.dtype),
        max_iters=20,
        preconditioner_fn=lambda a: a)
    # Ensure that the shape is at most max iterations.
    self.assertEqual(t.diag[0].shape[-1], 20)
    np.testing.assert_allclose(
        mbcg.tridiagonal_det(t.diag[0], t.off_diag[0]),
        jnp.exp(jnp.linalg.slogdet(A)[1]), rtol=6.3e-2)

  def test_value_and_grad(self):
    def norm1_of_inverse(mat):
      return jnp.sum(mbcg.modified_batched_conjugate_gradients(
          lambda t: jnp.matmul(mat, t),
          B=jnp.ones([mat.shape[-1], 1]),
          preconditioner_fn=lambda a: a)[0])

    v, g = jax.value_and_grad(norm1_of_inverse)(jnp.eye(3, dtype=self.dtype))
    np.testing.assert_allclose(v, [3.0])
    np.testing.assert_allclose(g, jnp.full((3, 3), -1.0))

  def test_tridiagonal_det(self):
    diag = jnp.array([-2, -2, -2, -2], dtype=self.dtype)
    off_diag = jnp.array([1, 1, 1], dtype=self.dtype)
    np.testing.assert_allclose(mbcg.tridiagonal_det(diag, off_diag), 5)

    # Verified by on-line determinant calculator
    np.testing.assert_allclose(
        26.3374,
        mbcg.tridiagonal_det(jnp.array([5.724855, 4.1276183, 1.1475269]),
                             jnp.array([0.6466101, 0.22849184])),
        rtol=1e+2
    )


class MbcgTestFloat32(_MbcgTest):
  dtype = np.float32


class MbcgTestFloat64(_MbcgTest):
  dtype = np.float64


del _MbcgTest


if __name__ == '__main__':
  config.update('jax_enable_x64', True)
  absltest.main()
