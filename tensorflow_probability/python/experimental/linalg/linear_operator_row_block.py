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

"""`LinearOperator` equivalent to concatenating several `LinearOperator`s by row."""

import tensorflow.compat.v2 as tf


class LinearOperatorRowBlock(tf.linalg.LinearOperator):
  """Represents a single row of a block matrix."""

  def __init__(self,
               operators,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name='LinearOperatorRowBlock'):
    parameters = dict(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    if not operators:
      raise ValueError('Expect a list of length at least one.')
    self._operators = operators
    super(LinearOperatorRowBlock, self).__init__(
        dtype=self._operators[0].dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        parameters=parameters,
        name=name)

  @property
  def operators(self):
    return self._operators

  def _shape(self):
    row_dim = self.operators[0].range_dimension
    col_dim = sum(o.domain_dimension for o in self.operators)
    return tf.TensorShape([row_dim, col_dim])

  def _shape_tensor(self):
    row_dim = self.operators[0].range_dimension_tensor()
    col_dim = sum(o.domain_dimension_tensor() for o in self.operators)
    return tf.stack([row_dim, col_dim])

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x = tf.linalg.adjoint(x) if adjoint_arg else x
    if adjoint:
      results = [o.matmul(x, adjoint=True) for o in self.operators]
      return tf.concat(results, axis=-2)
    xs = tf.split(
        x,
        num_or_size_splits=[
            o.domain_dimension_tensor() for o in self.operators],
        axis=-2)
    return sum(o.matmul(x_i) for o, x_i in zip(self.operators, xs))
