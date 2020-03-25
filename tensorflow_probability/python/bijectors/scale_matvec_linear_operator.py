# Copyright 2019 The TensorFlow Probability Authors.
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
"""ScaleMatvecLinearOperator bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'ScaleMatvecLinearOperator',
]


class ScaleMatvecLinearOperator(bijector.Bijector):
  """Compute `Y = g(X; scale) = scale @ X`.

  `scale` is a `LinearOperator`.

  If `X` is a scalar then the forward transformation is: `scale * X`
  where `*` denotes broadcasted elementwise product.

  Example Use:

  ```python
  x = [1., 2, 3]

  diag = [1., 2, 3]
  scale = tf.linalg.LinearOperatorDiag(diag)
  bijector = ScaleMatvecLinearOperator(scale)
  # In this case, `forward` is equivalent to:
  # y = scale @ x
  y = bijector.forward(x)  # [1., 4, 9]

  tril = [[1., 0, 0],
          [2, 1, 0],
          [3, 2, 1]]
  scale = tf.linalg.LinearOperatorLowerTriangular(tril)
  bijector = ScaleMatvecLinearOperator(scale)
  # In this case, `forward` is equivalent to:
  # np.squeeze(np.matmul(tril, np.expand_dims(x, -1)), -1)
  y = bijector.forward(x)  # [1., 4, 10]
  ```

  """

  def __init__(self,
               scale,
               adjoint=False,
               validate_args=False,
               parameters=None,
               name='scale_matvec_linear_operator'):
    """Instantiates the `ScaleMatvecLinearOperator` bijector.

    Args:
      scale:  Subclass of `LinearOperator`. Represents the (batch, non-singular)
        linear transformation by which the `Bijector` transforms inputs.
      adjoint: Python `bool` indicating whether to use the `scale` matrix as
        specified or its adjoint.
        Default value: `False`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      parameters: Locals dict captured by subclass constructor, to be used for
        copy/slice re-instantiation operators.
      name: Python `str` name given to ops managed by this object.

    Raises:
      TypeError: if `scale` is not a `LinearOperator`.
      ValueError: if not `scale.is_non_singular`.
    """
    parameters = dict(locals()) if parameters is None else parameters
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([scale], dtype_hint=tf.float32)
      if not isinstance(scale, tf.linalg.LinearOperator):
        raise TypeError('scale is not an instance of tf.LinearOperator')
      if validate_args and not scale.is_non_singular:
        raise ValueError('Scale matrix must be non-singular.')
      self._scale = scale
      self._adjoint = adjoint
      super(ScaleMatvecLinearOperator, self).__init__(
          forward_min_event_ndims=1,
          is_constant_jacobian=True,
          dtype=dtype,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @property
  def scale(self):
    """The `scale` `LinearOperator` in `Y = scale @ X`."""
    return self._scale

  @property
  def adjoint(self):
    """`bool` indicating whether this class uses `self.scale` or its adjoint."""
    return self._adjoint

  def _forward(self, x):
    return self.scale.matvec(x, adjoint=self.adjoint)

  def _inverse(self, y):
    return self.scale.solvevec(y, adjoint=self.adjoint)

  def _forward_log_det_jacobian(self, x):
    # is_constant_jacobian = True for this bijector, hence the
    # `log_det_jacobian` need only be specified for a single input, as this will
    # be tiled to match `event_ndims`.
    return self.scale.log_abs_determinant()

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    if is_init != any(tensor_util.is_ref(v) for v in self.scale.variables):
      return [self.scale.assert_non_singular()]
    return []
