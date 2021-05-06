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
"""MatrixInverseTriL bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import dtype_util


__all__ = [
    'MatrixInverseTriL',
]


@auto_composite_tensor.auto_composite_tensor(
    omit_kwargs=('name',), module_name='tfp.bijectors')
class MatrixInverseTriL(bijector.AutoCompositeTensorBijector):
  """Computes `g(L) = inv(L)`, where `L` is a lower-triangular matrix.

  `L` must be nonsingular; equivalently, all diagonal entries of `L` must be
  nonzero.

  The input must have `rank >= 2`.  The input is treated as a batch of matrices
  with batch shape `input.shape[:-2]`, where each matrix has dimensions
  `input.shape[-2]` by `input.shape[-1]` (hence `input.shape[-2]` must equal
  `input.shape[-1]`).

  #### Examples

  ```python
  tfp.bijectors.MatrixInverseTriL().forward(x=[[1., 0], [2, 1]])
  # Result: [[1., 0], [-2, 1]], i.e., inv(x)

  tfp.bijectors.MatrixInverseTriL().inverse(y=[[1., 0], [-2, 1]])
  # Result: [[1., 0], [2, 1]], i.e., inv(y).
  ```

  """

  def __init__(self, validate_args=False, name='matrix_inverse_tril'):
    """Instantiates the `MatrixInverseTriL` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(MatrixInverseTriL, self).__init__(
          forward_min_event_ndims=2,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  def _forward(self, x):
    with tf.control_dependencies(self._assertions(x)):
      shape = tf.shape(x)
      return tf.linalg.triangular_solve(
          x,
          tf.eye(shape[-1], batch_shape=shape[:-2], dtype=x.dtype),
          lower=True)

  def _inverse(self, y):
    return self._forward(y)

  def _forward_log_det_jacobian(self, x):
    # For a discussion of this (non-obvious) result, see Note 7.2.2 (and the
    # sections leading up to it, for context) in
    # http://neutrino.aquaphoenix.com/ReactionDiffusion/SERC5chap7.pdf
    with tf.control_dependencies(self._assertions(x)):
      matrix_dim = tf.cast(tf.shape(x)[-1],
                           dtype_util.base_dtype(x.dtype))
      return -(matrix_dim + 1) * tf.reduce_sum(
          tf.math.log(tf.abs(tf.linalg.diag_part(x))), axis=-1)

  def _assertions(self, x):
    if not self.validate_args:
      return []
    shape = tf.shape(x)
    is_matrix = assert_util.assert_rank_at_least(
        x, 2, message='Input must have rank at least 2.')
    is_square = assert_util.assert_equal(
        shape[-2], shape[-1], message='Input must be a square matrix.')
    above_diagonal = tf.linalg.band_part(
        tf.linalg.set_diag(x, tf.zeros(shape[:-1], dtype=tf.float32)), 0, -1)
    is_lower_triangular = assert_util.assert_equal(
        above_diagonal,
        tf.zeros_like(above_diagonal),
        message='Input must be lower triangular.')
    # A lower triangular matrix is nonsingular iff all its diagonal entries are
    # nonzero.
    diag_part = tf.linalg.diag_part(x)
    is_nonsingular = assert_util.assert_none_equal(
        diag_part,
        tf.zeros_like(diag_part),
        message='Input must have all diagonal entries nonzero.')
    return [is_matrix, is_square, is_lower_triangular, is_nonsingular]
