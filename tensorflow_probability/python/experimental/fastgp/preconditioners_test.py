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
"""Test for preconditioners.py."""

from absl.testing import parameterized
import jax
from jax import config
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.experimental.fastgp import preconditioners
import tensorflow_probability.substrates.jax as tfp
from absl.testing import absltest

jtf = tfp.tf2jax


# pylint: disable=invalid-name


class _PreconditionersTest(parameterized.TestCase):

  def test_identity_preconditioner(self):
    idp = preconditioners.IdentityPreconditioner(
        jnp.identity(3).astype(self.dtype))
    x = jnp.array([1.0, 2.0, 3.0], dtype=self.dtype)
    np.testing.assert_allclose(x, idp.full_preconditioner().solvevec(x))
    np.testing.assert_allclose(x, idp.preconditioned_operator().matvec(x))
    self.assertAlmostEqual(0.0, idp.log_det())

  def test_diagonal_preconditioner(self):
    dp = preconditioners.DiagonalPreconditioner(
        jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
    )
    self.assertAlmostEqual(jnp.log(4.0), dp.log_det(), places=5)
    np.testing.assert_allclose(
        jnp.array([1.0, 0.25]),
        dp.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.array([1.0, 4.0]),
        dp.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-6,
    )

  def test_diagonal_split_preconditioner(self):
    dp = preconditioners.DiagonalSplitPreconditioner(
        jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
    )
    self.assertAlmostEqual(jnp.log(4.0), dp.log_det(), places=5)
    np.testing.assert_allclose(
        jnp.array([1.0, 0.25]),
        dp.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.array([1.0, 4.0]),
        dp.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-6,
    )

  def test_rank_one_preconditioner(self):
    r1p = preconditioners.RankOnePreconditioner(
        jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype),
        jax.random.PRNGKey(0))
    # The true rank one approximation to M here has v ~ [1.30, 2.02],
    # which leads to the full matrix ~ [[1.3, 0.0], [2.02, 0.28]]
    # which has det ~ 0.37 and log det ~ -1.  Which then implies that
    # the full preconditioner has log det ~ -2.
    self.assertAlmostEqual(1.7164037, r1p.log_det(), delta=4)
    np.testing.assert_allclose(
        jnp.array([0.4, 0.1]),
        r1p.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1,
    )
    np.testing.assert_allclose(
        jnp.array([3.5, 5.6]),
        r1p.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-1,
    )

  def test_partial_cholesky_preconditioner(self):
    pcp = preconditioners.PartialCholeskyPreconditioner(
        jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype)
    )
    self.assertAlmostEqual(jnp.log(7.0), pcp.log_det(), places=5)
    atol = 1e-6
    if self.dtype == np.float32:
      atol = 2e-2

    np.testing.assert_allclose(
        jnp.array([3/7., 1/7.]),
        pcp.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=atol,
    )
    np.testing.assert_allclose(
        jnp.array([3.0, 5.0]),
        pcp.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=atol,
    )

  def test_partial_lanczos_preconditioner(self):
    plp = preconditioners.PartialLanczosPreconditioner(
        jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype),
        jax.random.PRNGKey(5)
    )
    self.assertAlmostEqual(jnp.log(7.0), plp.log_det(), places=5)
    atol = 1e-6
    if self.dtype == np.float32:
      atol = 3e-1
    np.testing.assert_allclose(
        jnp.array([3/7., 1/7.]),
        plp.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=atol,
    )
    np.testing.assert_allclose(
        jnp.array([3.0, 5.0]),
        plp.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=atol,
    )

  def test_truncated_svd_preconditioner(self):
    tsvd = preconditioners.TruncatedSvdPreconditioner(
        jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype),
        jax.random.PRNGKey(1)
    )
    self.assertAlmostEqual(jnp.log(7.0), tsvd.log_det(), delta=0.2)
    np.testing.assert_allclose(
        jnp.array([0.5, 0.25]),
        tsvd.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-1,
    )
    np.testing.assert_allclose(
        jnp.array([2.0, 4.0]),
        tsvd.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-1,
    )

  def test_rank_one_split_preconditioner(self):
    r1p = preconditioners.RankOneSplitPreconditioner(
        jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype),
        jax.random.PRNGKey(0)
    )
    # The true rank one approximation to M here has v ~ [1.30, 2.02],
    # which leads to the full matrix ~ [[1.3, 0.0], [2.02, 0.28]]
    # which has det ~ 0.37 and log det ~ -1.  Which then implies that
    # the full preconditioner has log det ~ -2.
    self.assertAlmostEqual(1.7164037, r1p.log_det(), delta=4)
    np.testing.assert_allclose(
        jnp.array([16, -6]),
        r1p.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1,
    )
    np.testing.assert_allclose(
        jnp.array([2.2, 5.6]),
        r1p.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-1,
    )

  def test_partial_cholesky_split_preconditioner(self):
    pcp = preconditioners.PartialCholeskySplitPreconditioner(
        jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype)
    )
    self.assertAlmostEqual(jnp.log(7.0), pcp.log_det(), places=5)
    np.testing.assert_allclose(
        jnp.array([3/7., 1/7.]),
        pcp.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-1,
    )
    np.testing.assert_allclose(
        jnp.array([3.0, 5.0]),
        pcp.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-1,
    )

  def test_partial_lanczos_split_preconditioner(self):
    plp = preconditioners.PartialLanczosSplitPreconditioner(
        jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype),
        jax.random.PRNGKey(1)
    )
    self.assertAlmostEqual(jnp.log(7.0), plp.log_det(), places=5)
    np.testing.assert_allclose(
        jnp.array([3/7., 1/7.]),
        plp.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=3e-6,
    )
    np.testing.assert_allclose(
        jnp.array([3.0, 5.0]),
        plp.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=3e-6,
    )

  def test_truncated_svd_split_preconditioner(self):
    tsvd = preconditioners.TruncatedSvdSplitPreconditioner(
        jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype),
        jax.random.PRNGKey(1)
    )
    self.assertAlmostEqual(jnp.log(7.0), tsvd.log_det(), delta=0.2)
    np.testing.assert_allclose(
        jnp.array([0.5, 0.25]),
        tsvd.full_preconditioner().solvevec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-1,
    )
    np.testing.assert_allclose(
        jnp.array([2.0, 4.0]),
        tsvd.full_preconditioner().matvec(
            jnp.array([1.0, 1.0], dtype=self.dtype)),
        atol=1e-1,
    )

  @parameterized.parameters(
      ("auto", 1.2e-7, 1e-6),
      ("identity", 2.0, 1.0),
      ("diagonal", 0.2, 0.2),
      ("rank_one", 0.3, 0.1),
      ("partial_cholesky", 1e-6, 0.02),
      ("partial_lanczos", 1e-6, 0.1),
      ("truncated_svd", 0.2, 0.2),
      ("truncated_randomized_svd", 0.2, 0.2),
      ("diagonal_split", 0.2, 0.2),
      ("rank_one_split", 4.0, 20.0),
      ("partial_cholesky_split", 1.2e-7, 1e-6),
      ("partial_lanczos_split", 4e-7, 1e-6),
      ("truncated_svd_split", 0.2, 0.2),
  )
  def test_get_preconditioner(self, preconditioner, log_det_delta, solve_atol):
    m = jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype)
    v = jnp.array([1.0, 1.0], dtype=self.dtype)

    p = preconditioners.get_preconditioner(
        preconditioner, m, key=jax.random.PRNGKey(3))
    self.assertAlmostEqual(jnp.log(7.0), p.log_det(), delta=log_det_delta)
    np.testing.assert_allclose(
        jnp.array([0.42857143, 0.14285714]),
        p.full_preconditioner().solvevec(v),
        atol=solve_atol
    )

  @parameterized.parameters(
      ("partial_cholesky_plus_scaling", 4e-7, 1e-7),
      ("partial_lanczos_plus_scaling", 3e-7, 2e-7),
      ("partial_pivoted_cholesky_plus_scaling", 2e-7, 2e-7),
      # TODO(srvasude): Test this on larger matrices, since
      # the low rank decomposition for small matrices is just a zero matrix.
      # ("truncated_svd_plus_scaling", 3e-7, 1e-7),
  )
  def test_get_preconditioner_with_identity(
      self, preconditioner, log_det_delta, solve_atol):
    m = jnp.array([[1.0, 1.0], [1.0, 3.0]], dtype=self.dtype)
    v = jnp.array([1.0, 1.0], dtype=self.dtype)

    p = preconditioners.get_preconditioner(
        preconditioner,
        m,
        key=jax.random.PRNGKey(3),
        scaling=self.dtype(1.))
    self.assertAlmostEqual(jnp.log(7.0), p.log_det(), delta=log_det_delta)
    np.testing.assert_allclose(
        jnp.array([0.42857143, 0.14285714]),
        p.full_preconditioner().solvevec(v),
        atol=solve_atol
    )

  @parameterized.parameters(
      ("partial_cholesky_plus_scaling", 0.5, 0.3),
      ("partial_lanczos_plus_scaling", 1.3, 0.9),
      ("partial_pivoted_cholesky_plus_scaling", 0.5, 0.3),
      ("truncated_randomized_svd_plus_scaling", 0.5, 0.3),
      # TODO(srvasude): Test this on larger matrices, since
      # the low rank decomposition for small matrices is just a zero matrix.
      # ("truncated_svd_plus_scaling", 1.2e-7, 1e-7),
  )
  def test_get_preconditioner_with_scaling_kwargs(
      self, preconditioner, log_det_delta, solve_atol):
    m = jnp.array([[1.0, 1.0], [1.0, 3.0]], dtype=self.dtype)
    v = jnp.array([1.0, 1.0], dtype=self.dtype)

    p = preconditioners.get_preconditioner(
        preconditioner,
        m,
        rank=1,
        num_iters=5,
        scaling=self.dtype(1.),
        key=jax.random.PRNGKey(4)
    )
    self.assertAlmostEqual(jnp.log(7.0), p.log_det(), delta=log_det_delta)
    np.testing.assert_allclose(
        jnp.array([0.42857143, 0.14285714]),
        p.full_preconditioner().solvevec(v),
        atol=solve_atol
    )

  @parameterized.parameters(
      ("identity", 2.0, 1.0),
      ("diagonal", 0.2, 0.2),
      ("rank_one", 0.3, 0.1),
      ("partial_cholesky", 1e-6, 0.2),
      ("partial_lanczos", 0.6, 0.3),
      ("truncated_svd", 0.2, 0.2),
      ("truncated_randomized_svd", 0.3, 0.3),
      ("diagonal_split", 0.2, 0.2),
      ("rank_one_split", 4.0, 16.0),
      # partial_cholesky_split is basically broken until index permutations
      # are added to LowRankSplitPreconditioner.
      # ("partial_cholesky_split", 20.0, 0),
      ("partial_lanczos_split", 0.9, 1.3),
      ("truncated_svd_split", 0.2, 0.2),
  )
  def test_get_preconditioner_with_kwargs(
      self, preconditioner, log_det_delta, solve_atol):
    m = jnp.array([[2.0, 1.0], [1.0, 4.0]], dtype=self.dtype)
    v = jnp.array([1.0, 1.0], dtype=self.dtype)

    p = preconditioners.get_preconditioner(
        preconditioner, m, rank=1, num_iters=5, key=jax.random.PRNGKey(4)
    )
    self.assertAlmostEqual(jnp.log(7.0), p.log_det(), delta=log_det_delta)
    np.testing.assert_allclose(
        jnp.array([0.42857143, 0.14285714]),
        p.full_preconditioner().solvevec(v),
        atol=solve_atol
    )

  @parameterized.parameters(
      ("identity", 9000),
      ("diagonal", 9000),
      ("rank_one", 3100),
      ("partial_cholesky", 5.0),
      ("partial_lanczos", 9000.),
      ("truncated_svd", 3100.0),
      ("truncated_randomized_svd", 3100.0),
      ("diagonal_split", 9000.),
      ("rank_one_split", 3100),
      # partial_cholesky_split is basically broken until index permutations
      # are added to LowRankSplitPreconditioner.
      # ("partial_cholesky_split", 2.0),
      ("partial_lanczos_split", 2e9),
      ("truncated_svd_split", 3100.0),
  )
  def test_post_conditioned(self, preconditioner, condition_number_bound):
    M = jnp.array([
        [
            1.001,
            0.88311934,
            0.9894911,
            0.9695768,
            0.9987461,
            0.98577714,
            0.97863793,
            0.9880289,
            0.7110599,
            0.7718459,
        ],
        [
            0.88311934,
            1.001,
            0.9395206,
            0.7564426,
            0.86025584,
            0.94721663,
            0.7791884,
            0.8075757,
            0.9478641,
            0.9758552,
        ],
        [
            0.9894911,
            0.9395206,
            1.001,
            0.92534095,
            0.98108065,
            0.9997143,
            0.93953925,
            0.95583755,
            0.79332554,
            0.84795874,
        ],
        [
            0.9695768,
            0.7564426,
            0.92534095,
            1.001,
            0.98049456,
            0.91640615,
            0.9991695,
            0.99564964,
            0.5614807,
            0.6257758,
        ],
        [
            0.9987461,
            0.86025584,
            0.98108065,
            0.98049456,
            1.001,
            0.97622854,
            0.98763895,
            0.99449164,
            0.6813891,
            0.74358207,
        ],
        [
            0.98577714,
            0.94721663,
            0.9997143,
            0.91640615,
            0.97622854,
            1.001,
            0.9313745,
            0.9487237,
            0.80610526,
            0.859435,
        ],
        [
            0.97863793,
            0.7791884,
            0.93953925,
            0.9991695,
            0.98763895,
            0.9313745,
            1.001,
            0.99861676,
            0.5861309,
            0.65042824,
        ],
        [
            0.9880289,
            0.8075757,
            0.95583755,
            0.99564964,
            0.99449164,
            0.9487237,
            0.99861676,
            1.001,
            0.61803514,
            0.68201244,
        ],
        [
            0.7110599,
            0.9478641,
            0.79332554,
            0.5614807,
            0.6813891,
            0.80610526,
            0.5861309,
            0.61803514,
            1.001,
            0.9943819,
        ],
        [
            0.7718459,
            0.9758552,
            0.84795874,
            0.6257758,
            0.74358207,
            0.859435,
            0.65042824,
            0.68201244,
            0.9943819,
            1.001,
        ],
    ], dtype=self.dtype)

    pc = preconditioners.get_preconditioner(
        preconditioner, M, rank=5, key=jax.random.PRNGKey(5)
    )
    post_conditioned = pc.preconditioned_operator().to_dense()

    # For split operators, post conditioned should be symmetric.
    if "_split" in preconditioner:
      np.testing.assert_allclose(
          post_conditioned, post_conditioned.T, rtol=1e-2, atol=1e-4)

    # log det post_conditioned = log det M - pc.log_det
    _, post_log_det = jnp.linalg.slogdet(post_conditioned)
    _, M_log_det = jnp.linalg.slogdet(M)
    # TODO(thomaswc): Figure out why the precoditioner log det calculations
    # are so wrong for partial_cholesky and partial_lanczos.
    if preconditioner not in [
        "partial_cholesky", "partial_lanczos", "truncated_randomized_svd"]:
      self.assertAlmostEqual(post_log_det, M_log_det - pc.log_det(), delta=1e-3)

    evalues = jnp.linalg.eigvalsh(post_conditioned)
    post_cond_number = evalues[-1] / jnp.abs(evalues[0])

    self.assertLess(post_cond_number, condition_number_bound)

    # For split operators, check that the post conditioned matrix is still
    # positive definite.
    lower_bound = 0.0
    if "_split" in preconditioner:
      self.assertGreater(evalues[0], lower_bound)

  def test_are_pytrees(self):
    M = jnp.array([[5.0, 1.0], [1.0, 9.0]], dtype=self.dtype)
    for pc, expected_leaves in {
        "auto": 2,
        "identity": 1,
        "diagonal": 1,
        "rank_one": 2,
        "partial_cholesky": 2,
        "partial_lanczos": 2,
        "truncated_svd": 2,
        "diagonal_split": 1,
        "rank_one_split": 2,
        "partial_cholesky_split": 2,
        "partial_lanczos_split": 2,
        "truncated_svd_split": 2,
    }.items():
      p = preconditioners.get_preconditioner(
          pc, M, rank=5, key=jax.random.PRNGKey(6)
      )
      self.assertLen(jax.tree_util.tree_leaves(p), expected_leaves,
                     f"Expected {expected_leaves} leaves for {pc}")

  def test_flatten_unflatten(self):
    M = jnp.array([[5.0, 1.0], [1.0, 9.0]], dtype=self.dtype)
    ip = preconditioners.IdentityPreconditioner(M)
    leaves, treedef = jax.tree_util.tree_flatten(ip)
    rep = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertIsInstance(rep, preconditioners.IdentityPreconditioner)

    dp = preconditioners.DiagonalSplitPreconditioner(M)
    leaves, treedef = jax.tree_util.tree_flatten(dp)
    rep = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertIsInstance(rep, preconditioners.DiagonalSplitPreconditioner)
    np.testing.assert_allclose(
        dp.right_half().to_dense(), rep.right_half().to_dense()
    )

    r1p = preconditioners.RankOneSplitPreconditioner(
        M, key=jax.random.PRNGKey(7)
    )
    leaves, treedef = jax.tree_util.tree_flatten(r1p)
    rep = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertIsInstance(rep, preconditioners.RankOneSplitPreconditioner)
    np.testing.assert_allclose(
        r1p.right_half().to_dense(), rep.right_half().to_dense()
    )

    pcp = preconditioners.PartialCholeskySplitPreconditioner(
        M, key=jax.random.PRNGKey(7)
    )
    leaves, treedef = jax.tree_util.tree_flatten(pcp)
    rep = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertIsInstance(
        rep, preconditioners.PartialCholeskySplitPreconditioner
    )
    np.testing.assert_allclose(
        pcp.right_half().to_dense(), rep.right_half().to_dense()
    )

    plp = preconditioners.PartialLanczosSplitPreconditioner(
        M, key=jax.random.PRNGKey(7)
    )
    leaves, treedef = jax.tree_util.tree_flatten(plp)
    rep = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertIsInstance(
        rep, preconditioners.PartialLanczosSplitPreconditioner
    )
    np.testing.assert_allclose(
        plp.right_half().to_dense(), rep.right_half().to_dense()
    )

    tsvd = preconditioners.TruncatedSvdSplitPreconditioner(
        M, key=jax.random.PRNGKey(7)
    )
    leaves, treedef = jax.tree_util.tree_flatten(tsvd)
    rep = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertIsInstance(
        rep, preconditioners.TruncatedSvdSplitPreconditioner
    )
    np.testing.assert_allclose(
        tsvd.right_half().to_dense(), rep.right_half().to_dense()
    )

  @parameterized.parameters(
      ("auto", 1),
      ("auto", 2),
      ("auto", 10),
      ("identity", 1),
      ("identity", 10),
      ("diagonal", 1),
      ("diagonal", 10),
      ("rank_one", 1),
      ("rank_one", 2),
      ("rank_one", 10),
      ("partial_cholesky", 1),
      ("partial_cholesky", 10),
      ("partial_lanczos", 1),
      ("partial_lanczos", 2),
      ("partial_lanczos", 10),
      ("truncated_svd", 1),
      ("truncated_svd", 2),
      ("truncated_svd", 10),
      ("truncated_randomized_svd", 1),
      ("truncated_randomized_svd", 2),
      ("truncated_randomized_svd", 10),
      ("diagonal_split", 1),
      ("diagonal_split", 10),
      ("rank_one_split", 1),
      ("rank_one_split", 2),
      ("rank_one_split", 10),
      ("partial_cholesky_split", 1),
      ("partial_cholesky_split", 10),
      ("partial_lanczos_split", 1),
      ("partial_lanczos_split", 2),
      # TODO(srvasude): Re-enable this when partial-lanczos has
      # reorthogonalization added to it.
      # ("partial_lanczos_split", 10),
      ("truncated_svd_split", 1),
      ("truncated_svd_split", 2),
      ("truncated_svd_split", 10),
  )
  def test_trace_of_inverse_product(self, preconditioner, n):
    # Make a random symmetric positive definite matrix M.
    A = jax.random.uniform(jax.random.PRNGKey(8), shape=(n, n),
                           minval=-1.0, maxval=1.0).astype(self.dtype)
    M = A.T @ A + 0.6 * jnp.eye(n).astype(self.dtype)
    # Make a random, not necessarily symmetric or positive definite matrix B.
    B = jax.random.uniform(jax.random.PRNGKey(10), shape=(n, n),
                           minval=-1.0, maxval=1.0).astype(self.dtype)
    p = preconditioners.get_preconditioner(
        preconditioner, M, key=jax.random.PRNGKey(9), rank=5)
    true_trace = jnp.trace(p.full_preconditioner().solve(B))
    error = abs(true_trace - p.trace_of_inverse_product(B))
    relative_error = error / true_trace
    self.assertLess(relative_error, 0.001)
    # AlmostEqual(true_trace, p.trace_of_inverse_product(B), places=2)

  @parameterized.parameters(
      "auto",
      "diagonal",
      "identity",
      "partial_cholesky",
      "partial_lanczos",
      "rank_one",
      "truncated_svd",
      "truncated_randomized_svd",
      "diagonal_split",
      "partial_cholesky_split",
      "partial_lanczos_split",
      "rank_one_split",
      "truncated_svd_split"
  )
  def test_preconditioner_with_linop(self, preconditioner):
    # Make a random symmetric positive definite matrix M.
    A = jax.random.uniform(jax.random.PRNGKey(8), shape=(2, 2),
                           minval=-1.0, maxval=1.0).astype(self.dtype)
    M = A.T @ A + 0.6 * jnp.eye(2).astype(self.dtype)
    M = jtf.linalg.LinearOperatorFullMatrix(M)
    # There are no errors.
    _ = preconditioners.get_preconditioner(
        preconditioner, M, key=jax.random.PRNGKey(9), rank=5)


class PreconditionersTestFloat32(_PreconditionersTest):
  dtype = np.float32


class PreconditionersTestFloat64(_PreconditionersTest):
  dtype = np.float64


del _PreconditionersTest


if __name__ == "__main__":
  config.update("jax_enable_x64", True)
  absltest.main()
