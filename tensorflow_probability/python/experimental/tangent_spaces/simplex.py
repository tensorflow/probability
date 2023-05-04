# Copyright 2023 The TensorFlow Probability Authors.
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

"""Tangent Spaces related to simplices."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.linalg import linear_operator_row_block as lorb
from tensorflow_probability.python.experimental.tangent_spaces import spaces
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps


class ProbabilitySimplexSpace(spaces.TangentSpace):
  """Tangent space of M for Simplex distributions in R^n."""

  def compute_basis(self, x):
    """Returns a `TangentSpace` of a n-simplex."""
    # The tangent space of the simplex satisfies `{x | <1, x> = 0}`, where `1`
    # is the vector of all `1`s. This can be seen by the fact that `1` is
    # orthogonal to the unit simplex.
    # We can do this by using the basis:  e_i - e_n, 1 <= i <= n - 1. For n = 4,
    # this looks like:
    # [[1, 0., 0., -1],
    #  [0, 1., 0., -1],
    #  [0, 0., 1., -1]]
    dim = ps.shape(x)[-1]
    block1 = tf.linalg.LinearOperatorIdentity(num_rows=dim - 1, dtype=x.dtype)
    block2 = tf.linalg.LinearOperatorFullMatrix(
        -tf.ones([dim - 1, 1], dtype=x.dtype))
    simplex_basis_linop = lorb.LinearOperatorRowBlock([block1, block2])
    return spaces.LinearOperatorBasis(simplex_basis_linop)

  def _transform_general(self, x, f, **kwargs):
    basis = self.compute_basis(x)
    # Note that B @ B.T results in the matrix I + 11^T, where 1 is the vector of
    # all ones. By the matrix determinant lemma we have det(I + 11^T) = n + 1,
    # or the dimension of the ambient space.
    dim = ps.shape(x)[-1]
    result = dtype_util.as_numpy_dtype(x.dtype)(0.5 * np.log(dim))
    new_basis_tensor = spaces.compute_new_basis_tensor(f, x, basis)
    new_log_volume = spaces.volume_coefficient(
        distribution_util.move_dimension(new_basis_tensor, 0, -2))
    result = new_log_volume - result
    return result, spaces.GeneralSpace(
        spaces.DenseBasis(new_basis_tensor), computed_log_volume=new_log_volume)

  def _transform_coordinatewise(self, x, f, **kwargs):
    # Compute the diagonal. New matrix is Linop that we can easily write.
    dim = ps.shape(x)[-1]
    diag_jacobian = spaces.coordinatewise_jvp(f, x)
    # Multiplying the basis written in block form as [I, 1] by the diagonal
    # results in this operator:
    block1 = tf.linalg.LinearOperatorDiag(diag_jacobian[..., :-1])
    block2 = tf.linalg.LinearOperatorFullMatrix(
        diag_jacobian[..., -1:, tf.newaxis] *
        tf.ones([dim - 1, 1], dtype=x.dtype))
    linop = lorb.LinearOperatorRowBlock([block1, block2])

    # The volume can be calculated again by the matrix determinant lemma:
    # det(D**2 + d_n**2 11^T) = (1 + d_n**2 1(D^-1)**21^T) * det(D**2)
    # = (\sum d_i**-2) * \prod d_i**2
    log_diag_jacobian = tf.math.log(tf.math.abs(diag_jacobian))
    log_volume = tf.math.reduce_sum(log_diag_jacobian, axis=-1)
    log_volume = log_volume + 0.5 * tf.math.reduce_logsumexp(
        -2. * log_diag_jacobian, axis=-1) - 0.5 * np.log(dim)
    return log_volume, spaces.GeneralSpace(
        spaces.LinearOperatorBasis(linop), computed_log_volume=log_volume)


