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
"""AffineLinearOperator bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import dtype_util


__all__ = [
    "AffineLinearOperator",
]


class AffineLinearOperator(bijector.Bijector):
  """Compute `Y = g(X; shift, scale) = scale @ X + shift`.

  `shift` is a numeric `Tensor` and `scale` is a `LinearOperator`.

  If `X` is a scalar then the forward transformation is: `scale * X + shift`
  where `*` denotes broadcasted elementwise product.

  Example Use:

  ```python
  linalg = tf.linalg

  x = [1., 2, 3]

  shift = [-1., 0., 1]
  diag = [1., 2, 3]
  scale = tf.linalg.LinearOperatorDiag(diag)
  affine = AffineLinearOperator(shift, scale)
  # In this case, `forward` is equivalent to:
  # y = scale @ x + shift
  y = affine.forward(x)  # [0., 4, 10]

  shift = [2., 3, 1]
  tril = [[1., 0, 0],
          [2, 1, 0],
          [3, 2, 1]]
  scale = tf.linalg.LinearOperatorLowerTriangular(tril)
  affine = AffineLinearOperator(shift, scale)
  # In this case, `forward` is equivalent to:
  # np.squeeze(np.matmul(tril, np.expand_dims(x, -1)), -1) + shift
  y = affine.forward(x)  # [3., 7, 11]
  ```

  """

  def __init__(self,
               shift=None,
               scale=None,
               adjoint=False,
               validate_args=False,
               name="affine_linear_operator"):
    """Instantiates the `AffineLinearOperator` bijector.

    Args:
      shift: Floating-point `Tensor`.
      scale:  Subclass of `LinearOperator`. Represents the (batch) positive
        definite matrix `M` in `R^{k x k}`.
      adjoint: Python `bool` indicating whether to use the `scale` matrix as
        specified or its adjoint.
        Default value: `False`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.

    Raises:
      TypeError: if `scale` is not a `LinearOperator`.
      TypeError: if `shift.dtype` does not match `scale.dtype`.
      ValueError: if not `scale.is_non_singular`.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args
    graph_parents = []
    with self._name_scope("init"):
      # In the absence of `loc` and `scale`, we'll assume `dtype` is `float32`.
      dtype = tf.float32

      if shift is not None:
        shift = tf.convert_to_tensor(value=shift, name="shift")
        graph_parents += [shift]
        dtype = dtype_util.base_dtype(shift.dtype)
      self._shift = shift

      if scale is not None:
        if (shift is not None and
            not dtype_util.base_equal(shift.dtype, scale.dtype)):
          raise TypeError(
              "shift.dtype({}) is incompatible with scale.dtype({}).".format(
                  shift.dtype, scale.dtype))
        if not isinstance(scale, tf.linalg.LinearOperator):
          raise TypeError("scale is not an instance of tf.LinearOperator")
        if validate_args and not scale.is_non_singular:
          raise ValueError("Scale matrix must be non-singular.")
        graph_parents += scale.graph_parents
        if scale.dtype is not None:
          dtype = dtype_util.base_dtype(scale.dtype)
      self._scale = scale
      self._adjoint = adjoint
      super(AffineLinearOperator, self).__init__(
          forward_min_event_ndims=1,
          graph_parents=graph_parents,
          is_constant_jacobian=True,
          dtype=dtype,
          validate_args=validate_args,
          name=name)

  @property
  def shift(self):
    """The `shift` `Tensor` in `Y = scale @ X + shift`."""
    return self._shift

  @property
  def scale(self):
    """The `scale` `LinearOperator` in `Y = scale @ X + shift`."""
    return self._scale

  @property
  def adjoint(self):
    """`bool` indicating `scale` should be used as conjugate transpose."""
    return self._adjoint

  def _forward(self, x):
    y = x
    if self.scale is not None:
      with tf.control_dependencies(self._maybe_collect_assertions()
                                   if self.validate_args else []):
        y = self.scale.matvec(y, adjoint=self.adjoint)
    if self.shift is not None:
      y += self.shift
    return y

  def _inverse(self, y):
    x = y
    if self.shift is not None:
      x -= self.shift
    if self.scale is not None:
      # Solve fails if the op is singular so we may safely skip this assertion.
      x = self.scale.solvevec(x, adjoint=self.adjoint)
    return x

  def _forward_log_det_jacobian(self, x):
    # is_constant_jacobian = True for this bijector, hence the
    # `log_det_jacobian` need only be specified for a single input, as this will
    # be tiled to match `event_ndims`.
    if self.scale is None:
      return tf.constant(0., dtype=dtype_util.base_dtype(x.dtype))

    with tf.control_dependencies(self._maybe_collect_assertions()
                                 if self.validate_args else []):
      return self.scale.log_abs_determinant()

  def _maybe_collect_assertions(self):
    try:
      return [self.scale.assert_non_singular()]
    except NotImplementedError:
      pass
    return []
