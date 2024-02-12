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
"""Expresses a sum of operators."""

import jax
from tensorflow_probability.python.internal.backend import jax as tf2jax


@jax.tree_util.register_pytree_node_class
class LinearOperatorSum(tf2jax.linalg.LinearOperator):
  """Encapsulates a sum of linear operators."""

  def __init__(self,
               operators,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorSum"):
    r"""Initialize a `LinearOperatorSum`.

    A Sum Operator, expressing `(A[0] + A[1] + A[2] + ... A[N])`, where
    `A` is a list of operators.

    This is useful to encapsulate a sum of structured operators without
    densifying them.

    Args:
      operators: `List` of `LinearOperator`s.
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
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    if not operators:
      raise ValueError("Expected a non-empty list of `operators`.")
    self._operators = operators
    dtype = operators[0].dtype
    for operator in operators:
      if operator.dtype != dtype:
        raise TypeError(
            "Expected every operation in `operators` to have the same "
            "dtype.")

    if all(operator.is_self_adjoint for operator in operators):
      if is_self_adjoint is False:  # pylint: disable=g-bool-id-comparison
        raise ValueError(
            f"The sum of self-adjoint operators is always "
            f"self-adjoint. Expected argument `is_self_adjoint` to be True. "
            f"Received: {is_self_adjoint}.")
      is_self_adjoint = True

    if all(operator.is_positive_definite for operator in operators):
      if is_positive_definite is False:  # pylint: disable=g-bool-id-comparison
        raise ValueError(
            f"The sum of positive-definite operators is always "
            f"positive-definite. Expected argument `is_positive_definite` to "
            f"be True. Received: {is_positive_definite}.")
      is_positive_definite = True

    super(LinearOperatorSum, self).__init__(
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        parameters=parameters,
        name=name)

  def _shape(self):
    return self._operators[0].shape

  def _shape_tensor(self):
    return self._operators[0].shape_tensor()

  @property
  def operators(self):
    return self._operators

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    return sum(o.matmul(
        x, adjoint=adjoint, adjoint_arg=adjoint_arg) for o in self.operators)

  def _to_dense(self):
    return sum(o.to_dense() for o in self.operators)

  @property
  def _composite_tensor_fields(self):
    return ("operators",)

  def tree_flatten(self):
    return ((self.operators,), None)

  @classmethod
  def tree_unflatten(cls, unused_aux_data, children):
    return cls(*children)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    return {"operators": [0] * len(self.operators)}
