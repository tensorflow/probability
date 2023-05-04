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

"""Tangent Spaces related to n-spheres."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.linalg import linear_operator_row_block as lorb
from tensorflow_probability.python.experimental.tangent_spaces import spaces
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static as ps


class SphericalSpace(spaces.TangentSpace):
  """Tangent space of M for Spherical distributions in R^n."""

  def compute_spherical_basis(self, x):
    """Returns a `Basis` representing the tangent space of the n-sphere."""
    # TODO(b/197680518): Is there a cheaper way to represent this basis /
    # compute JVPs with it (perhaps we don't need to represent this dense
    # basis when doing transformations)?

    # Given the Hairy Ball Theorem, we can't find a non-zero smooth vector field
    # for an n-sphere, for even n.
    # Thus we exclude the north pole of the sphere, and then add a basis for
    # this at the end.

    # Choose the zero vector when `x` is the north pole. This is not the north
    # pole, but it is safe for all further calculations.
    is_north_pole = tf.math.equal(x[..., -1], 1.)
    safe_x = tf.where(is_north_pole[..., tf.newaxis], tf.zeros_like(x), x)

    # The stereographic projection is a diffeomorphism from the n-sphere without
    # the north pole to R^{n}. We can find a basis `B` at `stereographic_x`
    # (the unit basis), and then compute B* = f_*^-1(B), where f_*^-1 is the
    # pushforward of f^-1, and f^-1 is the inverse stereographic projection.
    # The pushforward is
    # just the differential, or in otherwords the the Jacobian matrix of f_*^-1.

    # Using the derivation in `tfb.UnitVector`, we have that
    # J_{i, i} = 2 * (s^2 - 2z_i^2 + 1) / (s^2 + 1)^2
    # J_{i, j} = 4z_iz_j/(s^2 + 1)^2 for 0 <= i, j <= n - 1
    # J_{n, i} = 4z_i/(s^2 + 1)^2
    # where z_i are the stereographic coordinates, and s^2 is the sum of those
    # coordinates squared.
    # Factoring out 1 / (s^2 + 1)^2, we have that the last row is just 4z, and
    # that the rest of the matrix is 2(s^2 + 1)I - 4zz^T.

    # Finally rewriting all this in the original coordinate system,
    # we have (1 - x_n) I - yy^T for the first n - 1 rows, and (1 - x_n) * y for
    # the last row, where y omits the last coordinate of x. We then transpose
    # this to get the number of bases vectors as the first dimension, and
    # rescale by 1 / (1 - x_n) in order to get bases with unit volume.
    # Note: When we are at the north pole, due to computing safe_x, we have
    # x_n = 0 and y = 0, which results in the identity matrix, which is the
    # correct basis.
    y = safe_x[..., :-1]

    base_operator = tf.linalg.LinearOperatorIdentity(
        ps.shape(y)[-1], batch_shape=ps.shape(x)[:-1], dtype=x.dtype)
    block1 = tf.linalg.LinearOperatorLowRankUpdate(
        base_operator=base_operator,
        u=y[..., tf.newaxis],
        diag_update=-tf.math.reciprocal(1. - safe_x[..., -1:]))
    block2 = tf.linalg.LinearOperatorFullMatrix(y[..., tf.newaxis])
    sphere_basis_linop = lorb.LinearOperatorRowBlock([block1, block2])
    return spaces.LinearOperatorBasis(sphere_basis_linop)

  def _transform_general(self, x, f, **kwargs):
    basis = self.compute_spherical_basis(x)
    new_basis = spaces.compute_new_basis_tensor(f, x, basis)
    new_log_volume = spaces.volume_coefficient(
        distribution_util.move_dimension(new_basis, 0, -2))
    # The original basis has 0 log_volume.
    return new_log_volume, spaces.GeneralSpace(
        spaces.DenseBasis(new_basis), computed_log_volume=new_log_volume)
