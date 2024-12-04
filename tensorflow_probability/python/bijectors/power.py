# Copyright 2020 The TensorFlow Probability Authors.
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
"""Bijector to raise input to a given power."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'Power',
]


def _is_odd_integer(x):
  return ps.equal(x, ps.round(x)) & ps.not_equal(2. * ps.floor(x / 2.), x)


class Power(
    bijector.CoordinatewiseBijectorMixin,
    bijector.AutoCompositeTensorBijector):
  """Compute `g(X) = X ** power`; where X is a non-negative real number.

  When `power` is an odd integer, this bijector has domain the whole real line,
  and otherwise is constrained to non-negative numbers.

  Note: Powers that are reciprocal of odd integers like `1. / 3` are not
  supported because of numerical precision issues that make this property
  difficult to test. In order to simulate this behavior, we recommend using
  the `Invert` bijector instead (i.e. instead of `tfb.Power(power=1./3)`
  use `tfb.Invert(tfb.Power(power=3.))`).
  """

  def __init__(self, power, validate_args=False, name='power'):
    """Instantiates the `Power` bijector.

    Args:
      power: float `Tensor` power to raise the input to.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._power = tensor_util.convert_nonref_to_tensor(
          power, name='power', dtype_hint=tf.float32)
      super(Power, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(power=parameter_properties.ParameterProperties())

  @property
  def power(self):
    return self._power

  def _is_increasing(self):
    return self.power > 0

  def _forward(self, x):
    power = tf.cast(self.power, x.dtype)
    with tf.control_dependencies(self._assertions(x, power=power)):
      return tf.math.pow(x, power)

  def _inverse(self, y):
    power = tf.cast(self.power, y.dtype)
    with tf.control_dependencies(self._assertions(y, power=power)):
      return tf.where(
          _is_odd_integer(power),
          tf.math.sign(y) * tf.math.pow(tf.math.abs(y), 1. / power),
          tf.math.pow(y, 1. / power))

  def _forward_log_det_jacobian(self, x):
    power = tf.cast(self.power, x.dtype)
    with tf.control_dependencies(self._assertions(x, power=power)):
      return tf.math.log(tf.abs(power)) + tf.math.xlogy(
          power - 1., tf.math.abs(x))

  def _assertions(self, t, power=None):
    if not self.validate_args:
      return []
    power = power if power is not None else tf.convert_to_tensor(self.power)
    return [tf.debugging.Assert(
        tf.reduce_all((t >= 0.) | _is_odd_integer(power)),
        ['Elements must be non-negative, except for odd-integer powers.'])]

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.power):
      assertions.append(assert_util.assert_none_equal(
          self.power,
          0.,
          message='Argument `power` must be non-zero.'))
    return assertions
