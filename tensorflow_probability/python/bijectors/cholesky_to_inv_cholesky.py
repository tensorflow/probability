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
"""CholeskyToInvCholesky bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors.cholesky_outer_product import CholeskyOuterProduct


__all__ = [
    "CholeskyToInvCholesky",
]


class CholeskyToInvCholesky(bijector.Bijector):
  """Maps the Cholesky factor of `M` to the Cholesky factor of `M^{-1}`.

  The `forward` and `inverse` calculations are conceptually identical to:

  ```python
  def forward(x):
    return tf.cholesky(tf.linalg.inv(tf.matmul(x, x, adjoint_b=True)))

  inverse = forward
  ```

  or, similarly,

  ```python
  tfb = tfp.bijectors
  CholeskyToInvCholesky = tfb.Chain([
      tfb.Invert(tfb.CholeskyOuterProduct()),
      tfb.MatrixInverse(),
      tfb.CholeskyOuterProduct(),
  ])
  ```

  However, the actual calculations exploit the triangular structure of the
  matrices.
  """

  def __init__(self, validate_args=False, name=None):
    super(CholeskyToInvCholesky, self).__init__(
        forward_min_event_ndims=2,
        validate_args=validate_args,
        name=name or "cholesky_to_inv_cholesky")
    self._cholesky = CholeskyOuterProduct()

  def _forward(self, x):
    with tf.control_dependencies(self._assertions(x)):
      x_shape = tf.shape(input=x)
      identity_matrix = tf.eye(
          x_shape[-1], batch_shape=x_shape[:-2], dtype=x.dtype.base_dtype)
      # Note `matrix_triangular_solve` implicitly zeros upper triangular of `x`.
      y = tf.linalg.triangular_solve(x, identity_matrix)
      y = tf.matmul(y, y, adjoint_a=True)
      return tf.linalg.cholesky(y)

  _inverse = _forward

  def _forward_log_det_jacobian(self, x):
    # CholeskyToInvCholesky.forward(X) is equivalent to
    # 1) M = CholeskyOuterProduct.forward(X)
    # 2) N = invert(M)
    # 3) Y = CholeskyOuterProduct.inverse(N)
    #
    # For step 1,
    #   |Jac(outerprod(X))| = 2^p prod_{j=0}^{p-1} X[j,j]^{p-j}.
    # For step 2,
    #   |Jac(inverse(M))| = |M|^{-(p+1)} (because M is symmetric)
    #                     = |X|^{-2(p+1)} = (prod_{j=0}^{p-1} X[j,j])^{-2(p+1)}
    #   (see http://web.mit.edu/18.325/www/handouts/handout2.pdf sect 3.0.2)
    # For step 3,
    #   |Jac(Cholesky(N))| = -|Jac(outerprod(Y)|
    #                      = 2^p prod_{j=0}^{p-1} Y[j,j]^{p-j}
    n = tf.cast(tf.shape(input=x)[-1], x.dtype)
    y = self._forward(x)
    return ((self._cholesky.forward_log_det_jacobian(x, event_ndims=2) -
             (n + 1.) * tf.reduce_sum(
                 input_tensor=tf.math.log(tf.linalg.diag_part(x)), axis=-1)) -
            (self._cholesky.forward_log_det_jacobian(y, event_ndims=2) -
             (n + 1.) * tf.reduce_sum(
                 input_tensor=tf.math.log(tf.linalg.diag_part(y)), axis=-1)))

  _inverse_log_det_jacobian = _forward_log_det_jacobian

  def _assertions(self, x):
    if not self.validate_args:
      return []
    x_shape = tf.shape(input=x)
    is_matrix = tf.compat.v1.assert_rank_at_least(
        x, 2, message="Input must have rank at least 2.")
    is_square = tf.compat.v1.assert_equal(
        x_shape[-2], x_shape[-1], message="Input must be a square matrix.")
    diag_part_x = tf.linalg.diag_part(x)
    is_lower_triangular = tf.compat.v1.assert_equal(
        tf.linalg.band_part(x, 0, -1),  # Preserves triu, zeros rest.
        tf.linalg.diag(diag_part_x),
        message="Input must be lower triangular.")
    is_positive_diag = tf.compat.v1.assert_positive(
        diag_part_x, message="Input must have all positive diagonal entries.")
    return [is_matrix, is_square, is_lower_triangular, is_positive_diag]
