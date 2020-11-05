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
"""CholeskyOuterProduct bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    "CholeskyOuterProduct",
]


class CholeskyOuterProduct(bijector.Bijector):
  """Compute `g(X) = X @ X.T`; X is lower-triangular, positive-diagonal matrix.

  Note: the upper-triangular part of X is ignored (whether or not its zero).

  The surjectivity of g as a map from  the set of n x n positive-diagonal
  lower-triangular matrices to the set of SPD matrices follows immediately from
  executing the Cholesky factorization algorithm on an SPD matrix A to produce a
  positive-diagonal lower-triangular matrix L such that `A = L @ L.T`.

  To prove the injectivity of g, suppose that L_1 and L_2 are lower-triangular
  with positive diagonals and satisfy `A = L_1 @ L_1.T = L_2 @ L_2.T`. Then
    `inv(L_1) @ A @ inv(L_1).T = [inv(L_1) @ L_2] @ [inv(L_1) @ L_2].T = I`.
  Setting `L_3 := inv(L_1) @ L_2`, that L_3 is a positive-diagonal
  lower-triangular matrix follows from `inv(L_1)` being positive-diagonal
  lower-triangular (which follows from the diagonal of a triangular matrix being
  its spectrum), and that the product of two positive-diagonal lower-triangular
  matrices is another positive-diagonal lower-triangular matrix.

  A simple inductive argument (proceeding one column of L_3 at a time) shows
  that, if `I = L_3 @ L_3.T`, with L_3 being lower-triangular with positive-
  diagonal, then `L_3 = I`. Thus, `L_1 = L_2`, proving injectivity of g.

  #### Examples

  ```python
  bijector.CholeskyOuterProduct().forward(x=[[1., 0], [2, 1]])
  # Result: [[1., 2], [2, 5]], i.e., x @ x.T

  bijector.CholeskyOuterProduct().inverse(y=[[1., 2], [2, 5]])
  # Result: [[1., 0], [2, 1]], i.e., cholesky(y).
  ```

  """

  def __init__(self, validate_args=False, name="cholesky_outer_product"):
    """Instantiates the `CholeskyOuterProduct` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(CholeskyOuterProduct, self).__init__(
          forward_min_event_ndims=2,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  def _forward(self, x):
    with tf.control_dependencies(self._assertions(x)):
      # For safety, explicitly zero-out the upper triangular part.
      x = tf.linalg.band_part(x, -1, 0)
      return tf.matmul(x, x, adjoint_b=True)

  def _inverse(self, y):
    return tf.linalg.cholesky(y)

  def _forward_log_det_jacobian(self, x):
    # Let Y be a symmetric, positive definite matrix and write:
    #   Y = X X.T
    # where X is lower-triangular.
    #
    # Observe that,
    #   dY[i,j]/dX[a,b]
    #   = d/dX[a,b] { X[i,:] X[j,:] }
    #   = sum_{d=1}^p { I[i=a] I[d=b] X[j,d] + I[j=a] I[d=b] X[i,d] }
    #
    # To compute the Jacobian dX/dY we must represent X,Y as vectors. Since Y is
    # symmetric and X is lower-triangular, we need vectors of dimension:
    #   d = p (p + 1) / 2
    # where X, Y are p x p matrices, p > 0. We use a row-major mapping, i.e.,
    #   k = { i (i + 1) / 2 + j   i>=j
    #       { undef               i<j
    # and assume zero-based indexes. When k is undef, the element is dropped.
    # Example:
    #           j      k
    #        0 1 2 3  /
    #    0 [ 0 . . . ]
    # i  1 [ 1 2 . . ]
    #    2 [ 3 4 5 . ]
    #    3 [ 6 7 8 9 ]
    # Write vec[.] to indicate transforming a matrix to vector via k(i,j). (With
    # slight abuse: k(i,j)=undef means the element is dropped.)
    #
    # We now show d vec[Y] / d vec[X] is lower triangular. Assuming both are
    # defined, observe that k(i,j) < k(a,b) iff (1) i<a or (2) i=a and j<b.
    # In both cases dvec[Y]/dvec[X]@[k(i,j),k(a,b)] = 0 since:
    # (1) j<=i<a thus i,j!=a.
    # (2) i=a>j  thus i,j!=a.
    #
    # Since the Jacobian is lower-triangular, we need only compute the product
    # of diagonal elements:
    #   d vec[Y] / d vec[X] @[k(i,j), k(i,j)]
    #   = X[j,j] + I[i=j] X[i,j]
    #   = 2 X[j,j].
    # Since there is a 2 X[j,j] term for every lower-triangular element of X we
    # conclude:
    #   |Jac(d vec[Y]/d vec[X])| = 2^p prod_{j=0}^{p-1} X[j,j]^{p-j}.
    diag = tf.linalg.diag_part(x)

    # We now ensure diag is columnar. Eg, if `diag = [1, 2, 3]` then the output
    # is `[[1], [2], [3]]` and if `diag = [[1, 2, 3], [4, 5, 6]]` then the
    # output is unchanged.
    diag = self._make_columnar(diag)

    with tf.control_dependencies(self._assertions(x)):
      # Create a vector equal to: [p, p-1, ..., 2, 1].
      if tf.compat.dimension_value(x.shape[-1]) is None:
        p_int = tf.shape(x)[-1]
        p_float = tf.cast(p_int, dtype=x.dtype)
      else:
        p_int = tf.compat.dimension_value(x.shape[-1])
        p_float = dtype_util.as_numpy_dtype(x.dtype)(p_int)
      exponents = tf.linspace(p_float, 1., p_int)

      sum_weighted_log_diag = tf.squeeze(
          tf.matmul(tf.math.log(diag), exponents[..., tf.newaxis]), axis=-1)
      fldj = p_float * np.log(2.) + sum_weighted_log_diag

      # We finally need to undo adding an extra column in non-scalar cases
      # where there is a single matrix as input.
      if tensorshape_util.rank(x.shape) is not None:
        if tensorshape_util.rank(x.shape) == 2:
          fldj = tf.squeeze(fldj, axis=-1)
        return fldj

      shape = ps.shape(fldj)
      maybe_squeeze_shape = ps.concat([
          shape[:-1],
          distribution_util.pick_vector(
              ps.equal(ps.rank(x), 2),
              np.array([], dtype=np.int32), shape[-1:])], 0)
      return tf.reshape(fldj, maybe_squeeze_shape)

  def _make_columnar(self, x):
    """Ensures non-scalar input has at least one column.

    Example:
      If `x = [1, 2, 3]` then the output is `[[1], [2], [3]]`.

      If `x = [[1, 2, 3], [4, 5, 6]]` then the output is unchanged.

      If `x = 1` then the output is unchanged.

    Args:
      x: `Tensor`.

    Returns:
      columnar_x: `Tensor` with at least two dimensions.
    """
    if tensorshape_util.rank(x.shape) is not None:
      if tensorshape_util.rank(x.shape) == 1:
        x = x[tf.newaxis, :]
      return x
    shape = tf.shape(x)
    maybe_expanded_shape = tf.concat([
        shape[:-1],
        distribution_util.pick_vector(
            tf.equal(tf.rank(x), 1), [1], np.array([], dtype=np.int32)),
        shape[-1:],
    ], 0)
    return tf.reshape(x, maybe_expanded_shape)

  def _assertions(self, t):
    if not self.validate_args:
      return []
    is_matrix = assert_util.assert_rank_at_least(t, 2)
    is_square = assert_util.assert_equal(tf.shape(t)[-2], tf.shape(t)[-1])
    is_positive_definite = assert_util.assert_positive(
        tf.linalg.diag_part(t), message="Input must be positive definite.")
    return [is_matrix, is_square, is_positive_definite]

