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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'Power',
]


class Power(bijector.Bijector):
  """Compute `g(X) = X ** power`; where X is a positive real number."""

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

  @property
  def power(self):
    return self._power

  def _is_increasing(self):
    return self.power > 0

  def _forward(self, x):
    with tf.control_dependencies(self._assertions(x)):
      power = tf.cast(self.power, x.dtype)
      return tf.math.pow(x, power)

  def _inverse(self, y):
    with tf.control_dependencies(self._assertions(y)):
      power = tf.cast(self.power, y.dtype)
      return tf.math.pow(y, 1. / power)

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._assertions(x)):
      power = tf.cast(self.power, x.dtype)
      return tf.math.log(tf.abs(power)) + tf.math.xlogy(power - 1., x)

  def _assertions(self, t):
    if not self.validate_args:
      return []
    return [assert_util.assert_non_negative(
        t, message='All elements must be non-negative.')]

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
