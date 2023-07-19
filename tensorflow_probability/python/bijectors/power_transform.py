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
"""PowerTransform bijector."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static as ps


__all__ = [
    'PowerTransform',
]


class PowerTransform(
    bijector.CoordinatewiseBijectorMixin,
    bijector.AutoCompositeTensorBijector):
  """Compute `Y = g(X) = (1 + X * c)**(1 / c), X >= -1 / c`.

  The [power transform](https://en.wikipedia.org/wiki/Power_transform) maps
  inputs from `[0, inf]` to `[-1/c, inf]`; this is equivalent to the `inverse`
  of this bijector.

  This bijector is equivalent to the `Exp` bijector when `c=0`.
  """

  def __init__(self,
               power=0.,
               validate_args=False,
               parameters=None,
               name='power_transform'):
    """Instantiates the `PowerTransform` bijector.

    Args:
      power: Python `float` scalar indicating the transform power, i.e.,
        `Y = g(X) = (1 + X * c)**(1 / c)` where `c` is the `power`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      parameters: Locals dict captured by subclass constructor, to be used for
        copy/slice re-instantiation operators.
      name: Python `str` name given to ops managed by this object.

    Raises:
      ValueError: if `power < 0` or is not known statically.
    """
    parameters = dict(locals()) if parameters is None else parameters
    with tf.name_scope(name) as name:
      power = tf.get_static_value(ps.constant(power, name='power'))
      if power is None or power < 0:
        raise ValueError('`power` must be a non-negative TF constant.')
      self._power = power
      super(PowerTransform, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  @property
  def power(self):
    """The `c` in: `Y = g(X) = (1 + X * c)**(1 / c)`."""
    return self._power

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      if self.power == 0.:
        return tf.exp(x)
      # If large x accuracy is an issue, consider using:
      # (1. + x * self.power)**(1. / self.power) when x >> 1.
      power = tf.cast(self.power, x.dtype)
      return tf.exp(tf.math.log1p(x * power) / power)

  def _inverse(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      if self.power == 0.:
        return tf.math.log(y)
      # If large y accuracy is an issue, consider using:
      # (y**self.power - 1.) / self.power when y >> 1.
      power = tf.cast(self.power, y.dtype)
      return tf.math.expm1(tf.math.log(y) * power) / power

  def _inverse_log_det_jacobian(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      power = tf.cast(self.power, y.dtype)
      return (power - tf.ones([], y.dtype)) * tf.math.log(y)

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      if self.power == 0.:
        return x
      power = tf.cast(self.power, x.dtype)
      return (tf.math.reciprocal(power) - tf.ones([], x.dtype)
              ) * tf.math.log1p(x * power)

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args or self.power == 0.:
      return []
    return [assert_util.assert_non_negative(
        1. + self.power * x,
        message='Forward transformation input must be at least {}.'.format(
            -1. / self.power))]

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return []
    return [assert_util.assert_non_negative(
        y, message=('Inverse transformation input must be greater than or '
                    'equal to 0.'))]
