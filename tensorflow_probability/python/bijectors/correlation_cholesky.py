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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import fill_triangular
from tensorflow_probability.python.internal import prefer_static

__all__ = [
    "CorrelationCholesky",
]


class CorrelationCholesky(bijector.Bijector):
  """Maps unconstrained reals to Cholesky-space correlation matrices.

  This bijector is a mapping between `R^{n}` and the `n`-dimensional manifold of
  Cholesky-space correlation matrices embedded in `R^{m^2}`, where `n` is the
  `(m - 1)`th triangular number; i.e. `n = 1 + 2 + ... + (m - 1)`.

  #### Mathematical Details

  The image of unconstrained reals under the `CorrelationCholesky` bijector is
  the set of correlation matrices which are positive definite. A [correlation
  matrix](https://en.wikipedia.org/wiki/Correlation_and_dependence#Correlation_matrices)
  can be characterized as a symmetric positive semidefinite matrix with 1s on
  the main diagonal. However, the correlation matrix is positive definite if no
  component can be expressed as a linear combination of the other components.

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

  The LKJ distribution with `input_output_cholesky=True` generates samples from
  (and computes log-densities on) the set of Cholesky factors of positive
  definite correlation matrices. [2] The `CorrelationCholesky` bijector provides
  a bijective mapping from unconstrained reals to the support of the LKJ
  distribution.

  #### Examples

  ```python
  bijector.CorrelationCholesky().forward([2., 2., 1.])
  # Result: [[ 1.        ,  0.        ,  0.        ],
             [ 0.70710678,  0.70710678,  0.        ],
             [ 0.66666667,  0.66666667,  0.33333333]]

  # bijector.CorrelationCholesky().inverse(
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

  def __init__(self, validate_args=False, name="correlation_cholesky"):
    with tf.name_scope(name) as name:
      super(CorrelationCholesky, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=1,
          inverse_min_event_ndims=2,
          name=name)

  def _forward_event_shape(self, input_shape):
    if not input_shape.rank:
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
    x = tf.convert_to_tensor(x, name="x")
    batch_shape = prefer_static.shape(x)[:-1]

    # Pad zeros on the top row and right column.
    y = fill_triangular.FillTriangular().forward(x)
    rank = prefer_static.rank(y)
    paddings = tf.concat([
        tf.zeros(shape=(rank - 2, 2), dtype=tf.int32),
        tf.constant([[1, 0], [0, 1]], dtype=tf.int32)
    ],
                         axis=0)
    y = tf.pad(y, paddings)

    # Set diagonal to 1s.
    n = prefer_static.shape(y)[-1]
    diag = tf.ones(tf.concat([batch_shape, [n]], axis=-1), dtype=x.dtype)
    y = tf.linalg.set_diag(y, diag)

    # Normalize each row to have Euclidean (L2) norm 1.
    y /= tf.norm(y, axis=-1)[..., tf.newaxis]
    return y

  def _inverse(self, y):
    n = prefer_static.shape(y)[-1]
    batch_shape = prefer_static.shape(y)[:-2]

    # Extract the reciprocal of the row norms from the diagonal.
    diag = tf.linalg.diag_part(y)[..., tf.newaxis]

    # Set the diagonal to 0s.
    y = tf.linalg.set_diag(
        y, tf.zeros(tf.concat([batch_shape, [n]], axis=-1), dtype=y.dtype))

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
    # `f_k` restricted to the `k`th (1-indexed) row. It maps unconstrained reals
    # in `R^{k-1}` to unit vectors in `R^k`. `f_k : R^{k-1} -> R^k` is given by:
    #
    #   f(x_1, x_2, ... x_{k-1}) = (x_1/s, x_2/s, ..., x_{k-1}/s, 1/s)
    #
    # where `s = norm(x_1, x_2, ..., x_{k-1}, 1)`.
    #
    # The change in infinitesimal `k-1`-dimensional volume (or surface area) is
    # given by sqrt(|det J^T J|); where J is the `k x (k-1)` Jacobian matrix.
    #
    # Claim: sqrt(|det(J^T J)|) = s^{-k}.
    #
    # Proof: We compute the entries of the Jacobian matrix J:
    #
    #     J_{i, j} =  -x_j / s^3           if i == k
    #     J_{i, j} =  (s^2 - x_i^2) / s^3  if i == j and i < k
    #     J_{i, j} = -(x_i * x_j) / s^3    if i != j and i < k
    #
    #   By spherical symmetry, the volume element depends only on `s`; w.l.o.g.
    #   we can assume that `x_1 = r` and `x_2, ..., x_n = 0`; where
    #   `r^2 + 1 = s^2`.
    #
    #   We can write `J^T = [A|B]` where `A` is a diagonal matrix of rank `k-1`
    #   with diagonal `(1/s^3, 1/s, 1/s, ..., 1/s)`; and `B` is a column vector
    #   of size `k-1`, with entries (-r/s^3, 0, 0, ..., 0). Hence,
    #
    #     det(J^T J) = det(diag((r^2 + 1) / s^6, 1/s^2, ..., s^2))
    #                = s^{-2k}.
    #
    #   Or, sqrt(|det(J^T J)|) = s^{-k}.
    #
    # Hence, the forward log det jacobian (FLDJ) for the `k`th row is given by
    # `-k * log(s)`. The ILDJ is equal to negative FLDJ at the pre-image, or,
    # `k * log(s)`; where `s` is the reciprocal of the `k`th diagonal entry.
    #
    n = prefer_static.shape(y)[-1]
    return -tf.reduce_sum(
        tf.range(1, n + 1, dtype=y.dtype) * tf.math.log(tf.linalg.diag_part(y)),
        axis=-1)
