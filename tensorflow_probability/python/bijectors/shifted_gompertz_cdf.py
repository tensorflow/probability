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
"""Shifted Gompertz CDF bijector."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.math import special


__all__ = [
    'ShiftedGompertzCDF',
]


class ShiftedGompertzCDF(
    bijector.CoordinatewiseBijectorMixin,
    bijector.AutoCompositeTensorBijector):
  """Compute `Y = g(X) = (1 - exp(-rate * X)) * exp(-c * exp(-rate * X))`.

  This bijector maps inputs from `[-inf, inf]` to `[0, inf]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
  random variable with the
  [Shifted Gompertz distribution](
    https://en.wikipedia.org/wiki/Shifted_Gompertz_distribution):

  ```none
  Y ~ ShiftedGompertzCDF(concentration, rate)
  pdf(y; c, r) = r * exp(-r * y - exp(-r * y) / c) * (1 + (1 - exp(-r * y)) / c)
  ```

  Note: Even though this is called `ShiftedGompertzCDF`, when applied to the
  `Uniform` distribution, this is not the same as applying a `GompertzCDF` with
  a `Shift` bijector (i.e. the Shifted Gompertz distribution is not the same as
  a Gompertz distribution with a location parameter).

  Note: Because the Shifted Gompertz distribution concentrates its mass close
  to zero, for larger rates or larger concentrations, `bijector.forward` will
  quickly saturate to 1.
  """

  def __init__(self,
               concentration,
               rate,
               validate_args=False,
               name='shifted_gompertz_cdf'):
    """Instantiates the `ShiftedGompertzCDF` bijector.

    Args:
      concentration: Positive Float-like `Tensor` that is the same dtype and is
        broadcastable with `concentration`.
        This is `c` in
        `Y = g(X) = (1 - exp(-rate * X)) * exp(-exp(-rate * X) / c)`.
      rate: Positive Float-like `Tensor` that is the same dtype and is
        broadcastable with `concentration`.
        This is `rate` in
        `Y = g(X) = (1 - exp(-rate * X)) * exp(-exp(-rate * X) / c)`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [concentration, rate], dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate, dtype=dtype, name='rate')
      super(ShiftedGompertzCDF, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        rate=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def concentration(self):
    """The `c` in `Y = g(X) = (1 - exp(-r * X)) * exp(-exp(-r * X) / c)`."""
    return self._concentration

  @property
  def rate(self):
    """The `r` in `Y = g(X) = (1 - exp(-r * X)) * exp(-exp(-r * X) / c)`."""
    return self._rate

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      rate = tf.convert_to_tensor(self.rate)
      log1mexpx = generic.log1mexp(-rate * x)
      return tf.math.exp(
          log1mexpx - tf.math.exp(-rate * x) / self.concentration)

  def _inverse(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      concentration = tf.convert_to_tensor(self.concentration)
      reciprocal_concentration = tf.math.reciprocal(concentration)
      z = -special.lambertw(
          reciprocal_concentration * tf.math.exp(
              reciprocal_concentration + tf.math.log(y))) * concentration
      # Due to numerical instability, when y approaches 1, this expression
      # can be less than -1. We clip the value to prevent that.
      z = tf.clip_by_value(z, -1., np.inf)
      return -tf.math.log1p(z) / self.rate

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      rate = tf.convert_to_tensor(self.rate)
      concentration = tf.convert_to_tensor(self.concentration)
      z = rate * x
      return (-z - tf.math.exp(-z) / concentration + tf.math.log1p(
          -tf.math.expm1(-z) / concentration) + tf.math.log(rate))

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args:
      return []
    return [assert_util.assert_non_negative(
        x, message='Forward transformation input must be non-negative.')]

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return []
    is_positive = assert_util.assert_non_negative(
        y, message='Inverse transformation input must be greater than 0.')
    less_than_one = assert_util.assert_less_equal(
        y,
        tf.constant(1., y.dtype),
        message='Inverse transformation input must be less than or equal to 1.')
    return [is_positive, less_than_one]

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.rate):
      assertions.append(assert_util.assert_positive(
          self.rate,
          message='Argument `rate` must be positive.'))

    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
          self.concentration,
          message='Argument `concentration` must be positive.'))
    return assertions
