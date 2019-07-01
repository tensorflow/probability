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
"""Affine bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'AffineScalar',
]


class AffineScalar(bijector.Bijector):
  """Compute `Y = g(X; shift, scale) = scale * X + shift`.

  Examples:

  ```python
  # Y = X
  b = AffineScalar()

  # Y = X + shift
  b = AffineScalar(shift=[1., 2, 3])

  # Y = 2 * X + shift
  b = AffineScalar(
    shift=[1., 2, 3],
    scale=2.)
  ```

  """

  def __init__(self,
               shift=None,
               scale=None,
               log_scale=None,
               validate_args=False,
               name='affine_scalar'):
    """Instantiates the `AffineScalar` bijector.

    This `Bijector` is initialized with `shift` `Tensor` and `scale` arguments,
    giving the forward operation:

    ```none
    Y = g(X) = scale * X + shift
    ```

    Alternatively, you can specify `log_scale` instead of `scale` for slighly
    better numerics with tiny scales. Note that when using `log_scale` it is
    currently impossible to specify a negative scale.

    If `scale` or `log_scale` are not specified, then the bijector has the
    semantics of `scale = 1.`. Similarly, if `shift` is not specified, then the
    bijector has the semantics of `shift = 0.`.

    Args:
      shift: Floating-point `Tensor`. If this is set to `None`, no shift is
        applied.
      scale: Floating-point `Tensor`. If this is set to `None`, no scale is
        applied. This should not be set if `log_scale` is set.
      log_scale: Floating-point `Tensor`. Logarithm of the scale. If this is set
        to `None`, no scale is applied. This should not be set if `scale` is
        set.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.

    Raises:
      ValueError: If both `scale` and `log_scale` are specified.
    """
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [shift, scale, log_scale], dtype_hint=tf.float32)

      if scale is not None and log_scale is not None:
        raise ValueError('At most one of `scale` and `log_scale` should be '
                         'specified')

      self._shift = tensor_util.convert_immutable_to_tensor(
          shift, dtype=dtype, name='shift')
      self._scale = tensor_util.convert_immutable_to_tensor(
          scale, dtype=dtype, name='scale')
      self._log_scale = tensor_util.convert_immutable_to_tensor(
          log_scale, dtype=dtype, name='log_scale')

      super(AffineScalar, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=True,
          validate_args=validate_args,
          dtype=dtype,
          name=name)

  @property
  def shift(self):
    """The `shift` term in `Y = scale * X + shift`."""
    return self._shift

  @property
  def scale(self):
    """The `scale` term in `Y = scale * X + shift`."""
    return self._scale

  @property
  def log_scale(self):
    """The `log_scale` term in `Y = exp(log_scale) * X + shift`."""
    return self._log_scale

  def _forward(self, x):
    y = tf.identity(x)
    if self.scale is not None:
      y = y * self.scale
    if self.log_scale is not None:
      y = y * tf.exp(self.log_scale)
    if self.shift is not None:
      y = y + self.shift
    return y

  def _inverse(self, y):
    x = tf.identity(y)
    if self.shift is not None:
      x = x - self.shift
    if self.scale is not None:
      x = x / self.scale
    if self.log_scale is not None:
      x = x / tf.exp(self.log_scale)
    return x

  def _forward_log_det_jacobian(self, x):
    if self.log_scale is not None:
      return self.log_scale
    elif self.scale is not None:
      return tf.math.log(tf.abs(self.scale))
    else:
    # is_constant_jacobian = True for this bijector, hence the
    # `log_det_jacobian` need only be specified for a single input, as this will
    # be tiled to match `event_ndims`.
      return tf.constant(0., dtype=dtype_util.base_dtype(x.dtype))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if (self.scale is not None and
        is_init != tensor_util.is_mutable(self.scale)):
      assertions.append(
          assert_util.assert_none_equal(
              self.scale,
              tf.zeros([], dtype=self._scale.dtype),
              message='Argument `scale` must be non-zero.'))
    return assertions
