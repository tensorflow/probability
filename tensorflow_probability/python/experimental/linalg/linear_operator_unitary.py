# Copyright 2022 The TensorFlow Probability Authors.
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
"""A kernel covariance matrix LinearOperator."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import tensor_util


class LinearOperatorUnitary(tf.linalg.LinearOperator):
  """Encapsulates a Unitary Linear Operator."""

  def __init__(self,
               matrix,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name='LinearOperatorUnitary'):
    r"""Initialize a `LinearOperatorUnitary`.

    A Unitary Operator is one for which U* = U^-1. That is, the inverse of this
    operator is equivalent to the conjugate transpose of the operator. In the
    case that this operator is of real dtype, this corresponds to an orthogonal
    operator.

    This is useful as it reduces the complexity of `solve` to that of a
    `matmul` with the transpose operator.

    Args:
      matrix:  Shape `[B1,...,Bb, N, N]` `Tensor` with `b >= 0` `N >= 0`.
        The orthogonal matrix.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `diag.dtype` is real, this is auto-set to `True`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.
    """
    parameters = dict(
        matrix=matrix,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    with tf.name_scope(name):
      self._matrix = tensor_util.convert_nonref_to_tensor(matrix, name='matrix')

      if is_square is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError('Unitary operators are square.')
      is_square = True

      # Add checks for unitary matrix
      if (self._matrix.shape[-1] is not None and
          self._matrix.shape[-2] is not None):
        if self._matrix.shape[-2] != self._matrix.shape[-1]:
          raise ValueError(
              'Expected square matrix, got mismatched dimensions {} {}'.format(
                  self._matrix.shape[-2], self._matrix.shape[-1]))

      super(LinearOperatorUnitary, self).__init__(
          dtype=self._matrix.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _shape(self):
    return self._matrix.shape

  def _shape_tensor(self):
    return tf.shape(self._matrix)

  @property
  def matrix(self):
    return self._matrix

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    return tf.linalg.matmul(
        self._matrix, x, adjoint_a=adjoint, adjoint_b=adjoint_arg)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    return tf.linalg.matmul(
        self._matrix, rhs, adjoint_a=(not adjoint), adjoint_b=adjoint_arg)

  def _to_dense(self):
    return self._matrix

  def _log_abs_determinant(self):
    # A unitary operator has eigenvalues with unit norm, and hence log|det(U)|
    # is 1.
    return tf.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _cond(self):
    return tf.ones(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _assert_non_singular(self):
    return tf.no_op('assert_non_singular')

  @property
  def _composite_tensor_fields(self):
    return ('matrix',)
