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
"""Library of popular matrix preconditioners.

Given a matrix M, a preconditioner P of M is an easy to compute approximation
of M with the following properties:

  1) Pv and P^(-1)v are easy to compute, preferably in time closer to O(n)
  than O(n^2),
  2) det P is easy to compute, and
  3) one of M P^(-1), P^(-1) M, or A^(-1) M B^(-1) (with P = A B) is closer
  to the identity than M is.  The specific distance metric we are most often
  interested in is condition number, and the three options are referred to
  as "right", "left" or "split" preconditioning respectively.

For more details, see chapter 9 of "Iterative Methods for Sparse Linear
Systems", available online at
https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf

The preconditioners here are intended to be used to improve the convergence
of the log det and yt_inv_y operations needed by Gaussian Processes.  The
preconditioner class you will want to use will depend on the GP kernel type.
https://arxiv.org/pdf/2107.00243.pdf suggests that partial Cholesky or QFF
work well for RBF kernels, and that RFF or truncated SVD might be good when
nothing is known about the kernel structure.
"""

import jax
import jax.numpy as jnp
from tensorflow_probability.python.experimental.fastgp import linalg
from tensorflow_probability.python.experimental.fastgp import linear_operator_sum
from tensorflow_probability.python.internal.backend import jax as tf2jax
from tensorflow_probability.substrates.jax.math import linalg as tfp_math


# pylint: disable=invalid-name


@jax.named_call
def promote_to_operator(M) -> tf2jax.linalg.LinearOperator:
  if isinstance(M, tf2jax.linalg.LinearOperator):
    return M
  return tf2jax.linalg.LinearOperatorFullMatrix(M, is_non_singular=True)


def _diag_part(M) -> jax.Array:
  if isinstance(M, tf2jax.linalg.LinearOperator):
    return M.diag_part()
  return tf2jax.linalg.diag_part(M)


class Preconditioner:
  """Base class for preconditioners."""

  def __init__(self, M: tf2jax.linalg.LinearOperator):
    self.M = M

  def full_preconditioner(self) -> tf2jax.linalg.LinearOperator:
    """Returns the preconditioner."""
    raise NotImplementedError('Base classes must override full_preconditioner.')

  def preconditioned_operator(self) -> tf2jax.linalg.LinearOperator:
    """Returns the combined action of M and the preconditioner."""
    raise NotImplementedError(
        'Base classes must override preconditioned_operator.')

  def log_det(self) -> tf2jax.linalg.LinearOperator:
    """The log absolute value of the determinant of the preconditioner."""
    return self.full_preconditioner().log_abs_determinant()

  def trace_of_inverse_product(self, A: jax.Array):
    """Returns tr( P^(-1) A ) for a n x n, non-batched A."""
    result = self.full_preconditioner().solve(A)
    if isinstance(result, tf2jax.linalg.LinearOperator):
      return result.trace()
    return jnp.trace(result)


@jax.tree_util.register_pytree_node_class
class IdentityPreconditioner(Preconditioner):
  """The do-nothing preconditioner."""

  def __init__(self, M: tf2jax.linalg.LinearOperator, **unused_kwargs):
    n = M.shape[-1]
    self.id = tf2jax.linalg.LinearOperatorIdentity(n, dtype=M.dtype)
    super().__init__(M)

  def full_preconditioner(self) -> tf2jax.linalg.LinearOperator:
    return self.id

  def preconditioned_operator(self) -> tf2jax.linalg.LinearOperator:
    return promote_to_operator(self.M)

  def log_det(self):
    return 0.0

  def trace_of_inverse_product(self, A: jax.Array):
    return jnp.trace(A)

  def tree_flatten(self):
    return ((self.M,), None)

  @classmethod
  def tree_unflatten(cls, unused_aux_data, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class DiagonalPreconditioner(Preconditioner):
  """The best diagonal preconditioner; aka the Jacobi preconditioner."""

  def __init__(self, M: tf2jax.linalg.LinearOperator, **unused_kwargs):
    self.d = jnp.maximum(_diag_part(M), 1e-6)
    super().__init__(M)

  def full_preconditioner(self) -> tf2jax.linalg.LinearOperator:
    return tf2jax.linalg.LinearOperatorDiag(
        self.d, is_non_singular=True, is_positive_definite=True
    )

  def preconditioned_operator(self) -> tf2jax.linalg.LinearOperator:
    return tf2jax.linalg.LinearOperatorComposition(
        [promote_to_operator(self.M), self.full_preconditioner().inverse()]
    )

  def log_det(self):
    return jnp.sum(jnp.log(self.d))

  def trace_of_inverse_product(self, A: jax.Array):
    return jnp.sum(jnp.diag(A) / self.d)

  def tree_flatten(self):
    return ((self.M,), None)

  @classmethod
  def tree_unflatten(cls, unused_aux_data, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class LowRankPreconditioner(Preconditioner):
  """Turns M ~ A A^t for low rank A into a preconditioner."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      low_rank: jax.Array,
      residual_diag: jax.Array = None,
  ):
    self.low_rank = low_rank
    n, self.r = self.low_rank.shape
    assert n == M.shape[-1], (
        f'Low Rank has shape {self.low_rank.shape}; should have shape'
        f' ({M.shape[-1]}, r)'
    )

    if residual_diag is None:
      self.residual_diag = _diag_part(M) - jnp.einsum(
          'ij,ij->i', self.low_rank, self.low_rank
      )
    else:
      self.residual_diag = residual_diag

    self.residual_diag = jnp.maximum(1e-6, self.residual_diag)

    diag_op = tf2jax.linalg.LinearOperatorDiag(
        self.residual_diag, is_non_singular=True, is_positive_definite=True
    )
    self.pre = tf2jax.linalg.LinearOperatorLowRankUpdate(
        diag_op, self.low_rank, is_positive_definite=True
    )
    super().__init__(M)

  def full_preconditioner(self) -> tf2jax.linalg.LinearOperator:
    return self.pre

  def preconditioned_operator(self) -> tf2jax.linalg.LinearOperator:
    return tf2jax.linalg.LinearOperatorComposition(
        [promote_to_operator(self.M), self.pre.inverse()]
    )

  @classmethod
  def from_lowrank(cls, M, low_rank):
    """Alternate constructor when low_rank is already made."""
    x = LowRankPreconditioner(M, low_rank)
    x.__class__ = cls
    return x

  def tree_flatten(self):
    return ((self.M, self.low_rank), None)

  @classmethod
  def tree_unflatten(cls, unused_aux_data, children):
    return cls.from_lowrank(*children)


@jax.tree_util.register_pytree_node_class
class RankOnePreconditioner(LowRankPreconditioner):
  """Preconditioner based on M ~ v v^t using M's largest eigenvector v."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      key: jax.Array,
      num_iters: int = 10,
      **unused_kwargs,
  ):
    evalue, evector = linalg.largest_eigenvector(M, key, num_iters)
    v = jnp.sqrt(evalue) * evector
    low_rank = v[:, jnp.newaxis]
    super().__init__(M, low_rank)


@jax.tree_util.register_pytree_node_class
class PartialCholeskyPreconditioner(LowRankPreconditioner):
  """https://en.wikipedia.org/wiki/Incomplete_Cholesky_factorization ."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      rank: int = 20,
      **unused_kwargs,
  ):
    n = M.shape[-1]
    rank = min(n, rank)
    low_rank, _, residual_diag = tfp_math.low_rank_cholesky(M, rank)
    super().__init__(M, low_rank, residual_diag)


@jax.tree_util.register_pytree_node_class
class PartialLanczosPreconditioner(LowRankPreconditioner):
  """https://www.sciencedirect.com/science/article/pii/S0307904X13002382 ."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      key: jax.Array,
      rank: int = 20,
      **unused_kwargs,
  ):
    low_rank = linalg.make_partial_lanczos(key, M, rank)
    super().__init__(M, low_rank)


@jax.tree_util.register_pytree_node_class
class TruncatedSvdPreconditioner(LowRankPreconditioner):
  """https://www.math.kent.edu/~reichel/publications/tsvd.pdf .

  Note that 5 * num_iters must be less than n.
  """

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      key: jax.Array,
      rank: int = 20,
      num_iters: int = 10,
      **unused_kwargs,
  ):
    low_rank = linalg.make_truncated_svd(key, M, rank, num_iters)
    super().__init__(M, low_rank)


@jax.tree_util.register_pytree_node_class
class TruncatedRandomizedSvdPreconditioner(LowRankPreconditioner):
  """https://www.math.kent.edu/~reichel/publications/tsvd.pdf .

  Note that 5 * num_iters must be less than n.
  """

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      key: jax.Array,
      rank: int = 20,
      num_iters: int = 10,
      **unused_kwargs,
  ):
    low_rank = linalg.make_randomized_truncated_svd(key, M, rank, num_iters)
    super().__init__(M, low_rank)


@jax.tree_util.register_pytree_node_class
class LowRankPlusScalingPreconditioner(Preconditioner):
  """Turns M ~ a * I + A A^t for low rank A into a preconditioner."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      low_rank: jax.Array,
      scaling: jax.Array,
  ):
    self.low_rank = low_rank
    n, self.r = self.low_rank.shape
    assert n == M.shape[-1], (
        f'Low Rank has shape {self.low_rank.shape}; should have shape'
        f' ({M.shape[-1]}, r)'
    )
    self.scaling = scaling
    identity_op = tf2jax.linalg.LinearOperatorScaledIdentity(
        num_rows=M.shape[-1],
        multiplier=self.scaling,
        is_non_singular=True,
        is_positive_definite=True,
    )
    self.pre = tf2jax.linalg.LinearOperatorLowRankUpdate(
        identity_op,
        self.low_rank,
        is_positive_definite=True,
        is_self_adjoint=True,
        is_non_singular=True,
    )
    super().__init__(M)

  def full_preconditioner(self) -> tf2jax.linalg.LinearOperator:
    return self.pre

  def preconditioned_operator(self) -> tf2jax.linalg.LinearOperator:
    linop = promote_to_operator(self.M)
    operator = linear_operator_sum.LinearOperatorSum([
        linop,
        tf2jax.linalg.LinearOperatorScaledIdentity(
            num_rows=self.M.shape[-1], multiplier=self.scaling
        ),
    ])
    return tf2jax.linalg.LinearOperatorComposition(
        [self.pre.inverse(), operator]
    )

  @classmethod
  def from_lowrank(cls, M, low_rank, scaling):
    """Alternate constructor when low_rank is already made."""
    x = LowRankPlusScalingPreconditioner(M, low_rank, scaling)
    x.__class__ = cls
    return x

  def tree_flatten(self):
    return ((self.M, self.low_rank, self.scaling), None)

  @classmethod
  def tree_unflatten(cls, unused_aux_data, children):
    return cls.from_lowrank(*children)


@jax.tree_util.register_pytree_node_class
class PartialCholeskyPlusScalingPreconditioner(
    LowRankPlusScalingPreconditioner):
  """https://en.wikipedia.org/wiki/Incomplete_Cholesky_factorization ."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      scaling: jax.Array,
      rank: int = 20,
      **unused_kwargs,
  ):
    n = M.shape[-1]
    rank = min(n, rank)
    low_rank, _, _ = tfp_math.low_rank_cholesky(M, rank)
    super().__init__(M, low_rank, scaling)


@jax.tree_util.register_pytree_node_class
class PartialPivotedCholeskyPlusScalingPreconditioner(
    LowRankPlusScalingPreconditioner):
  """https://en.wikipedia.org/wiki/Incomplete_Cholesky_factorization ."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      scaling: jax.Array,
      rank: int = 20,
      **unused_kwargs,
  ):
    n = M.shape[-1]
    rank = min(n, rank)
    low_rank = linalg.make_partial_pivoted_cholesky(M, rank)
    super().__init__(M, low_rank, scaling)


@jax.tree_util.register_pytree_node_class
class PartialLanczosPlusScalingPreconditioner(
    LowRankPlusScalingPreconditioner):
  """https://www.sciencedirect.com/science/article/pii/S0307904X13002382 ."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      scaling: jax.Array,
      key: jax.Array,
      rank: int = 20,
      **unused_kwargs,
  ):
    low_rank = linalg.make_partial_lanczos(key, M, rank)
    super().__init__(M, low_rank, scaling)


@jax.tree_util.register_pytree_node_class
class TruncatedSvdPlusScalingPreconditioner(LowRankPlusScalingPreconditioner):
  """https://www.math.kent.edu/~reichel/publications/tsvd.pdf .

  Note that 5 * num_iters must be less than n.
  """

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      scaling: jax.Array,
      key: jax.Array,
      rank: int = 20,
      num_iters: int = 10,
      **unused_kwargs,
  ):
    low_rank = linalg.make_truncated_svd(key, M, rank, num_iters)
    super().__init__(M, low_rank, scaling)


@jax.tree_util.register_pytree_node_class
class TruncatedRandomizedSvdPlusScalingPreconditioner(
    LowRankPlusScalingPreconditioner):
  """https://www.math.kent.edu/~reichel/publications/tsvd.pdf .

  Note that 5 * num_iters must be less than n.
  """

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      scaling: jax.Array,
      key: jax.Array,
      rank: int = 20,
      num_iters: int = 10,
      **unused_kwargs,
  ):
    low_rank = linalg.make_randomized_truncated_svd(key, M, rank, num_iters)
    super().__init__(M, low_rank, scaling)


class SplitPreconditioner(Preconditioner):
  """Base class for symmetric split preconditioners."""

  # pylint: disable-next=useless-parent-delegation
  def __init__(self, M: tf2jax.linalg.LinearOperator):
    super().__init__(M)

  def right_half(self) -> tf2jax.linalg.LinearOperator:
    """Returns R, where the preconditioner is P = R^T R."""
    raise NotImplementedError('Base classes must override right_half method.')

  def full_preconditioner(self) -> tf2jax.linalg.LinearOperator:
    """Returns P = R^T R, the preconditioner's approximation to M."""
    rh = self.right_half()
    lh = rh.adjoint()
    return tf2jax.linalg.LinearOperatorComposition(
        [lh, rh],
        is_self_adjoint=True,
        is_positive_definite=True,
    )

  def preconditioned_operator(self) -> tf2jax.linalg.LinearOperator:
    """Returns R^(-T) M R^(-1)."""
    rhi = self.right_half().inverse()
    lhi = rhi.adjoint()
    return tf2jax.linalg.LinearOperatorComposition(
        [lhi, promote_to_operator(self.M), rhi],
        is_self_adjoint=True,
        is_positive_definite=True,
    )

  def log_det(self):
    """Returns log det(R^T R) = 2 log det R."""
    return 2 * self.right_half().log_abs_determinant()

  def trace_of_inverse_product(self, A: jax.Array):
    """Returns tr( (R^T R)^(-1) A ) for a n x n, non-batched A."""
    raise NotImplementedError(
        'Base classes must override trace_of_inverse_product.')


@jax.tree_util.register_pytree_node_class
class DiagonalSplitPreconditioner(SplitPreconditioner):
  """The split conditioner which pre and post multiplies by a diagonal."""

  def __init__(self, M: tf2jax.linalg.LinearOperator, **unused_kwargs):
    self.d = jnp.maximum(_diag_part(M), 1e-6)
    self.sqrt_d = jnp.sqrt(self.d)
    super().__init__(M)

  def right_half(self) -> tf2jax.linalg.LinearOperator:
    return tf2jax.linalg.LinearOperatorDiag(
        self.sqrt_d, is_non_singular=True, is_positive_definite=True
    )

  def full_preconditioner(self) -> tf2jax.linalg.LinearOperator:
    return tf2jax.linalg.LinearOperatorDiag(
        self.d, is_non_singular=True, is_positive_definite=True
    )

  def log_det(self):
    return jnp.sum(jnp.log(self.d))

  def trace_of_inverse_product(self, A: jax.Array):
    return jnp.sum(jnp.diag(A) / self.d)

  def tree_flatten(self):
    return ((self.M,), None)

  @classmethod
  def tree_unflatten(cls, unused_aux_data, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class LowRankSplitPreconditioner(SplitPreconditioner):
  """Turns M ~ A A^t for low rank A into a split preconditioner."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      low_rank: jax.Array,
      residual_diag: jax.Array = None,
  ):
    self.low_rank = low_rank
    n, self.r = self.low_rank.shape
    assert n == M.shape[-1], (
        f'Low Rank has shape {self.low_rank.shape}; should have shape'
        f' ({M.shape[-1]}, r)'
    )

    if residual_diag is None:
      self.residual_diag = _diag_part(M) - jnp.einsum(
          'ij,ij->i', self.low_rank, self.low_rank
      )
    else:
      self.residual_diag = residual_diag

    self.residual_diag = jnp.maximum(1e-6, self.residual_diag)

    # self.low_rank isn't invertible, so we need to make it part of a block
    # matrix that is.  self.low_rank will be the first r columns of that
    # matrix; the other n-r columns will be zero for the first r rows and
    # a diagonal matrix in the remaining (n-r, n-r) bottom right block.
    # TODO(thomaswc): Iteratively add a small constant to the diagonal of B if
    # it is singular.
    # TODO(thomaswc): Permute the indices so that the lowest r values of
    # self.residual_diag are in the first r entries (so that the diagonal
    # matrix in the bottom right block can kill the largest n-r residual_diag
    # values).
    # Turn off Pyformat because it puts spaces between a slice bound and its
    # colon.
    # fmt: off
    self.B = tf2jax.linalg.LinearOperatorFullMatrix(
        self.low_rank[:self.r, :], is_non_singular=True
    )
    self.C = tf2jax.linalg.LinearOperatorFullMatrix(self.low_rank[self.r:, :])
    sqrt_d = jnp.sqrt(self.residual_diag[self.r:])
    # fmt: on
    self.D = tf2jax.linalg.LinearOperatorDiag(sqrt_d, is_non_singular=True)
    P = tf2jax.linalg.LinearOperatorBlockLowerTriangular(
        [[self.B], [self.C, self.D]], is_non_singular=True
    )
    # We started from M ~ low_rank low_rank^t (because low_rank is n by r),
    # & we need the preconditioner P to satisfy M ~ P^t P, so P = low_rank^t.
    self.P = P.adjoint()

    super().__init__(M)

  def right_half(self) -> tf2jax.linalg.LinearOperator:
    return self.P

  def trace_of_inverse_product(self, A: jax.Array):
    # We want the trace of (P^T P)^(-1) A
    # = P^(-1) P^(-t) A
    # = [[ B^(-1), - B^(-1) C D^(-1)], [0, D^(-1)]]
    #   @ [[ B^(-t), 0], [- D^(-1) C^t B^(-t), D^(-1)]]
    #   @ [[ A11, A12 ], [ A21, A22 ]]
    # = [[ B^(-1) B^(-t) A11 - B^(-1) C D^(-2) (A21 - C^t B^(-t) A11), *],
    #    [*, D^(-2) ( A22 - C^t B^(-t) A12 ) ]]
    # (But actually, because of the self.P = P.adjoint at the end of
    # __init__, B is actually always B^t in the above and C is always
    # C^t.
    n = A.shape[-1]
    if self.r == n:
      return jnp.trace(self.full_preconditioner().solvevec(A))
    A11 = A[:self.r, :self.r]
    A12 = A[:self.r, self.r:]
    A21 = A[self.r:, :self.r]
    A22 = A[self.r:, self.r:]
    D2 = self.residual_diag[self.r:]
    # TODO(thomaswc): Compute the LU decomposition of B, and use that in
    # place of all of the self.B.solvevec's.
    Binvt_A11 = self.B.solvevec(A11)
    first_term = jnp.trace(self.B.H.solvevec(Binvt_A11))
    inner_factor = (A21 - self.C @ Binvt_A11) / D2[:, jnp.newaxis]
    second_term = jnp.trace(self.B.H.solvevec(self.C.H @ inner_factor))
    A22_term = jnp.sum(jnp.diag(A22) / D2)
    Binvt_A12 = self.B.solve(A12)
    diag_Ct_Binvt_A12 = jnp.einsum('ij,ji->i', self.C.to_dense(), Binvt_A12)
    A12_term = jnp.sum(diag_Ct_Binvt_A12 / D2)
    return first_term - second_term + A22_term - A12_term

  @classmethod
  def from_lowrank(cls, M, low_rank):
    """Alternate constructor when low_rank is already made."""
    x = LowRankSplitPreconditioner(M, low_rank)
    x.__class__ = cls
    return x

  def tree_flatten(self):
    return ((self.M, self.low_rank), None)

  @classmethod
  def tree_unflatten(cls, unused_aux_data, children):
    return cls.from_lowrank(*children)


@jax.tree_util.register_pytree_node_class
class RankOneSplitPreconditioner(LowRankSplitPreconditioner):
  """Split preconditioner based on M ~ v v^t using M's largest eigenvector v."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      key: jax.Array,
      num_iters: int = 10,
      **unused_kwargs,
  ):
    evalue, evector = linalg.largest_eigenvector(M, key, num_iters)
    v = jnp.sqrt(evalue) * evector
    low_rank = v[:, jnp.newaxis]
    super().__init__(M, low_rank)


@jax.tree_util.register_pytree_node_class
class PartialCholeskySplitPreconditioner(LowRankSplitPreconditioner):
  """https://en.wikipedia.org/wiki/Incomplete_Cholesky_factorization ."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      rank: int = 20,
      **unused_kwargs,
  ):
    n = M.shape[-1]
    rank = min(n, rank)
    low_rank, _, residual_diag = tfp_math.low_rank_cholesky(M, rank)
    super().__init__(M, low_rank, residual_diag)


@jax.tree_util.register_pytree_node_class
class PartialLanczosSplitPreconditioner(LowRankSplitPreconditioner):
  """https://www.sciencedirect.com/science/article/pii/S0307904X13002382 ."""

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      key: jax.Array,
      rank: int = 20,
      **unused_kwargs,
  ):
    low_rank = linalg.make_partial_lanczos(key, M, rank)
    super().__init__(M, low_rank)


@jax.tree_util.register_pytree_node_class
class TruncatedSvdSplitPreconditioner(LowRankSplitPreconditioner):
  """https://www.math.kent.edu/~reichel/publications/tsvd.pdf .

  Note that 5 * num_iters must be less than n.
  """

  def __init__(
      self,
      M: tf2jax.linalg.LinearOperator,
      key: jax.Array,
      rank: int = 20,
      num_iters: int = 10,
      **unused_kwargs,
  ):
    low_rank = linalg.make_truncated_svd(key, M, rank, num_iters)
    super().__init__(M, low_rank)


# TODO(thomaswc): RFF and QFF preconditioners.


PRECONDITIONER_REGISTRY = {
    'identity': IdentityPreconditioner,
    'diagonal': DiagonalPreconditioner,
    'rank_one': RankOnePreconditioner,
    'partial_cholesky': PartialCholeskyPreconditioner,
    'partial_lanczos': PartialLanczosPreconditioner,
    'truncated_svd': TruncatedSvdPreconditioner,
    'truncated_randomized_svd': TruncatedRandomizedSvdPreconditioner,
    'diagonal_split': DiagonalSplitPreconditioner,
    'rank_one_split': RankOneSplitPreconditioner,
    'partial_cholesky_split': PartialCholeskySplitPreconditioner,
    'partial_lanczos_split': PartialLanczosSplitPreconditioner,
    'truncated_svd_split': TruncatedSvdSplitPreconditioner,
    'partial_pivoted_cholesky_plus_scaling': (
        PartialPivotedCholeskyPlusScalingPreconditioner),
    'partial_cholesky_plus_scaling': PartialCholeskyPlusScalingPreconditioner,
    'partial_lanczos_plus_scaling': PartialLanczosPlusScalingPreconditioner,
    'truncated_svd_plus_scaling': TruncatedSvdPlusScalingPreconditioner,
    'truncated_randomized_svd_plus_scaling': (
        TruncatedRandomizedSvdPlusScalingPreconditioner),
}


def resolve_preconditioner(
    preconditioner_name: str,
    M: tf2jax.linalg.LinearOperator,
    rank: int) -> str:
  """Return the resolved preconditioner_name."""
  if preconditioner_name == 'auto':
    n = M.shape[-1]
    if 5 * rank >= n:
      return 'partial_cholesky_split'
    else:
      return 'truncated_randomized_svd_plus_scaling'
  return preconditioner_name


@jax.named_call
def get_preconditioner(
    preconditioner_name: str, M: tf2jax.linalg.LinearOperator, **kwargs
) -> SplitPreconditioner:
  """Return the preconditioner of the given type for the given matrix."""
  preconditioner_name = resolve_preconditioner(
      preconditioner_name, M, kwargs.get('rank', 20))
  try:
    return PRECONDITIONER_REGISTRY[preconditioner_name](M, **kwargs)
  except KeyError as key_error:
    raise ValueError(
        'Unknown preconditioner name {}, known preconditioners are {}'.format(
            preconditioner_name, PRECONDITIONER_REGISTRY.keys())) from key_error
