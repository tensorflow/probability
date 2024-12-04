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
"""CorrelationCholesky bijector."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import fill_triangular
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'CorrelationCholesky',
]


class CorrelationCholesky(bijector.AutoCompositeTensorBijector):
  """Maps unconstrained reals to Cholesky-space correlation matrices.

  #### Mathematical Details

  This bijector provides a change of variables from unconstrained reals to a
  parameterization of the CholeskyLKJ distribution. The CholeskyLKJ distribution
  [1] is a distribution on the set of Cholesky factors of positive definite
  correlation matrices. The CholeskyLKJ probability density function is
  obtained from the LKJ density on n x n matrices as follows:

    1 = int p(A | eta) dA
      = int Z(eta) * det(A) ** (eta - 1) dA
      = int Z(eta) L_ii ** {(n - i - 1) + 2 * (eta - 1)} ^dL_ij (0 <= i < j < n)

  where Z(eta) is the normalizer; the matrix L is the Cholesky factor of the
  correlation matrix A; and ^dL_ij denotes the wedge product (or differential)
  of the strictly lower triangular entries of L. The entries L_ij are
  constrained such that each entry lies in [-1, 1] and the norm of each row is
  1. The norm includes the diagonal; which is not included in the wedge product.
  To preserve uniqueness, we further specify that the diagonal entries are
  positive.

  The image of unconstrained reals under the `CorrelationCholesky` bijector is
  the set of correlation matrices which are positive definite. A [correlation
  matrix](https://en.wikipedia.org/wiki/Correlation_and_dependence#Correlation_matrices)
  can be characterized as a symmetric positive semidefinite matrix with 1s on
  the main diagonal.

  For a lower triangular matrix `L` to be a valid Cholesky-factor of a positive
  definite correlation matrix, it is necessary and sufficient that each row of
  `L` have unit Euclidean norm [1]. To see this, observe that if `L_i` is the
  `i`th row of the Cholesky factor corresponding to the correlation matrix `R`,
  then the `i`th diagonal entry of `R` satisfies:

    1 = R_i,i = L_i . L_i = ||L_i||^2

  where '.' is the dot product of vectors and `||...||` denotes the Euclidean
  norm.

  Furthermore, observe that `R_i,j` lies in the interval `[-1, 1]`. By the
  Cauchy-Schwarz inequality:

    |R_i,j| = |L_i . L_j| <= ||L_i|| ||L_j|| = 1

  This is a consequence of the fact that `R` is symmetric positive definite with
  1s on the main diagonal.

  We choose the mapping from x in `R^{m}` to `R^{n^2}` where `m` is the
  `(n - 1)`th triangular number; i.e. `m = 1 + 2 + ... + (n - 1)`.

    L_ij = x_i,j / s_i (for i < j)
    L_ii = 1 / s_i

  where s_i = sqrt(1 + x_i,0^2 + x_i,1^2 + ... + x_(i,i-1)^2). We can check that
  the required constraints on the image are satisfied.

  #### Examples

  ```python
  bijector.CorrelationCholesky().forward([2., 2., 1.])
  # Result: [[ 1.        ,  0.        ,  0.        ],
             [ 0.70710678,  0.70710678,  0.        ],
             [ 0.66666667,  0.66666667,  0.33333333]]

  bijector.CorrelationCholesky().inverse(
      [[ 1.        ,  0.        ,  0. ],
       [ 0.70710678,  0.70710678,  0.        ],
       [ 0.66666667,  0.66666667,  0.33333333]])
  # Result: [2., 2., 1.]
  ```

  #### References
  [1] Stan Manual. Section 24.2. Cholesky LKJ Correlation Distribution.
  https://mc-stan.org/docs/2_18/functions-reference/cholesky-lkj-correlation-distribution.html
  [2] Daniel Lewandowski, Dorota Kurowicka, and Harry Joe,
  "Generating random correlation matrices based on vines and extended
  onion method," Journal of Multivariate Analysis 100 (2009), pp
  1989-2001.

  """

  def __init__(self, validate_args=False, name='correlation_cholesky'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(CorrelationCholesky, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=1,
          inverse_min_event_ndims=2,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  def _forward_event_shape(self, input_shape):
    if tensorshape_util.rank(input_shape) is None:
      return input_shape
    tril_shape = fill_triangular.FillTriangular().forward_event_shape(
        input_shape)
    n = tril_shape[-1]
    if n is not None:
      n += 1
    return tril_shape[:-2].concatenate([n, n])

  def _forward_event_shape_tensor(self, input_shape):
    tril_shape = fill_triangular.FillTriangular().forward_event_shape_tensor(
        input_shape)
    n = tril_shape[-1] + 1
    return tf.concat([tril_shape[:-2], [n, n]], axis=-1)

  def _inverse_event_shape(self, input_shape):
    if not input_shape.rank:
      return input_shape
    n = input_shape[-1]
    if n is not None:
      n -= 1
    y_shape = input_shape[:-2].concatenate([n, n])
    return fill_triangular.FillTriangular().inverse_event_shape(y_shape)

  def _inverse_event_shape_tensor(self, input_shape):
    n = input_shape[-1] - 1
    y_shape = tf.concat([input_shape[:-2], [n, n]], axis=-1)
    return fill_triangular.FillTriangular().inverse_event_shape_tensor(y_shape)

  def _forward(self, x):
    x = tf.convert_to_tensor(x, name='x')
    batch_shape = ps.shape(x)[:-1]

    # Pad zeros on the top row and right column.
    y = fill_triangular.FillTriangular().forward(x)
    rank = ps.rank(y)
    paddings = ps.concat(
        [ps.zeros([rank - 2, 2], dtype=tf.int32),
         [[1, 0], [0, 1]]],
        axis=0)
    y = tf.pad(y, paddings)

    # Set diagonal to 1s.
    n = ps.shape(y)[-1]
    diag = tf.ones(ps.concat([batch_shape, [n]], axis=-1), dtype=x.dtype)
    y = tf.linalg.set_diag(y, diag)

    # Normalize each row to have Euclidean (L2) norm 1.
    y /= tf.norm(y, axis=-1)[..., tf.newaxis]
    return y

  def _inverse(self, y):
    n = ps.shape(y)[-1]
    batch_shape = ps.shape(y)[:-2]

    # Extract the reciprocal of the row norms from the diagonal.
    diag = tf.linalg.diag_part(y)[..., tf.newaxis]

    # Set the diagonal to 0s.
    y = tf.linalg.set_diag(
        y, tf.zeros(ps.concat([batch_shape, [n]], axis=-1), dtype=y.dtype))

    # Multiply with the norm (or divide by its reciprocal) to recover the
    # unconstrained reals in the (strictly) lower triangular part.
    x = y / diag

    # Remove the first row and last column before inverting the FillTriangular
    # transformation.
    return fill_triangular.FillTriangular().inverse(x[..., 1:, :-1])

  def _forward_log_det_jacobian(self, x):
    # TODO(b/133442896): It should be possible to use the fallback
    # implementation of _forward_log_det_jacobian in terms of
    # _inverse_log_det_jacobian in the base Bijector class.
    return -self._inverse_log_det_jacobian(self.forward(x))

  def _inverse_log_det_jacobian(self, y):
    # The inverse log det jacobian (ILDJ) of the entire mapping is the sum of
    # the ILDJs of each row's mapping.
    #
    # To compute the ILDJ for each row's mapping, consider the forward mapping
    # `f_k` restricted to the `k`th (0-indexed) row. It maps unconstrained reals
    # in `R^k` to the unit disk in `R^k`. `f_k : R^k -> R^k` is:
    #
    #   f(x_1, x_2, ... x_k) = (x_1/s, x_2/s, ..., x_k/s)
    #
    # where `s = norm(x_1, x_2, ..., x_k, 1)`.
    #
    # The change in infinitesimal `k`-dimensional volume is given by
    # |det(J)|; where J is the `k x k` Jacobian matrix.
    #
    # Claim: |det(J)| = s^{-(k + 2)}.
    #
    # Proof: We compute the entries of the Jacobian matrix J:
    #
    #     J_ij =  (s^2 - x_i^2) / s^3  if i == j
    #     J_ij = -(x_i * x_j) / s^3    if i != j
    #
    #   We multiply each row by s^3, which contributes a factor of s^{-3k} to
    #   det(J). The remaining matrix can be written as s^2 I - xx^T. By the
    #   matrix determinant lemma
    #   (https://en.wikipedia.org/wiki/Matrix_determinant_lemma),
    #   det(s^2 I - xx^T) = s^{2k} (1 - (x^Tx / s^2)) = s^{2k - 2}. The last
    #   equality follows from s^2 - x^Tx = s^2 - sum x_i^2 = 1. Hence,
    #   det(J) = s^{-3k} s^{2k - 2} = s^{-(k + 2)}.
    #
    n = ps.shape(y)[-1]
    return -tf.reduce_sum(
        tf.range(2, n + 2, dtype=y.dtype) * tf.math.log(tf.linalg.diag_part(y)),
        axis=-1)
