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
"""Test for linalg.py."""

from absl.testing import parameterized
import jax
from jax import config
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.fastgp import linalg
import tensorflow_probability.substrates.jax as tfp
from absl.testing import absltest

jtf = tfp.tf2jax


# pylint: disable=invalid-name


class _LinalgTest(parameterized.TestCase):

  def test_largest_eigenvector(self):
    M = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
    evalue, evector = linalg.largest_eigenvector(M, jax.random.PRNGKey(0))
    self.assertAlmostEqual(5.415476, evalue, delta=0.1)
    np.testing.assert_allclose(
        jnp.array([0.4926988, 0.8701998]), evector, atol=0.1
    )

  @parameterized.parameters(2, 3, 5, 10, 15)
  def test_randomized_truncated_svd(self, n):
    # Check that this matches the non-randomized SVD.
    A = jax.random.uniform(
        jax.random.PRNGKey(1), shape=(n, n),
        minval=-1.0, maxval=1.0).astype(self.dtype)
    M = A.T @ A + 0.6 * jnp.eye(n).astype(self.dtype)
    low_rank = linalg.make_randomized_truncated_svd(
        jax.random.PRNGKey(2), M, rank=n, num_iters=2)
    self.assertEqual(low_rank.shape, (n, n))

  @parameterized.parameters(2, 3, 5, 10, 15)
  def test_pivoted_cholesky_exact(self, n):
    A = jax.random.uniform(
        jax.random.PRNGKey(3), shape=(n, n),
        minval=-1.0, maxval=1.0).astype(self.dtype)
    M = A.T @ A + 0.6 * jnp.eye(n).astype(self.dtype)

    low_rank = linalg.make_partial_pivoted_cholesky(M, rank=n)
    self.assertEqual(low_rank.shape, (n, n))
    np.testing.assert_allclose(M, low_rank @ low_rank.T, rtol=5e-6)

  @parameterized.parameters(2, 3, 5, 10, 15)
  def test_pivoted_cholesky_approx(self, n):
    A = jax.random.uniform(
        jax.random.PRNGKey(3), shape=(n, n),
        minval=-1.0, maxval=1.0).astype(self.dtype)
    M = A.T @ A + 0.6 * jnp.eye(n).astype(self.dtype)

    low_rank = linalg.make_partial_pivoted_cholesky(M, rank=n // 2)
    self.assertEqual(low_rank.shape, (n, n // 2))


class LinalgTestFloat32(_LinalgTest):
  dtype = np.float32


class LinalgTestFloat64(_LinalgTest):
  dtype = np.float64


del _LinalgTest


if __name__ == "__main__":
  config.update("jax_enable_x64", True)
  absltest.main()
