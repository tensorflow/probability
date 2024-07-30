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
"""Frechet bijector."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'FrechetCDF',
]


class FrechetCDF(
    bijector.CoordinatewiseBijectorMixin,
    bijector.AutoCompositeTensorBijector):
  """The Frechet cumulative density function.

  Computes `Y = g(X) = exp(-((X - loc) / scale)**(-concentration))`, the Frechet
  CDF.

  This bijector maps inputs from `[loc, inf]` to `[0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
  random variable with the
  [Frechet distribution](https://en.wikipedia.org/wiki/Frechet_distribution):

  ```none
  Y ~ Frechet(loc, scale, concentration)
  pdf(y; loc, scale, concentration) = (concentration / scale)
      * exp(- ((x - loc)/scale)**(-concentration))
      * ((y - loc) / scale)**(-(1 + concentration))
  ```
  """

  def __init__(self,
               loc=0.,
               scale=1.,
               concentration=1.,
               validate_args=False,
               name='frechet_cdf'):
    """Instantiates the `FrechetCDF` bijector.

    Args:
      loc: Float-like `Tensor` that is the same dtype and is
        broadcastable with `scale`. This is `loc` in
        `Y = g(X) = exp(-((X - loc) / scale)**(-concentration))`.
      scale: Positive Float-like `Tensor` that is the same dtype and is
        broadcastable with `loc`. This is `scale` in
        `Y = g(X) = exp(-((X - loc) / scale)**(-concentration))`.
      concentration: Positive Float-like `Tensor` that is the same dtype and is
        broadcastable with `loc`. This is `concentration` in
        `Y = g(X) = exp(-((X - loc) / scale)**(-concentration))`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale, concentration],
                                      dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      super(FrechetCDF, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          parameters=parameters,
          dtype=dtype,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def loc(self):
    """The lower bound of the Frechet distribution."""
    return self._loc

  @property
  def scale(self):
    """The scale of the Frechet distribution."""
    return self._scale

  @property
  def concentration(self):
    """The concentration of the Frechet distribution."""
    return self._concentration

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      z = (x - self.loc) / self.scale
      return tf.math.exp(-z**(-self.concentration))

  def _inverse(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      return self.loc + self.scale * tf.math.exp(
          -tf.math.log(-tf.math.log(y)) / self.concentration)

  def _inverse_log_det_jacobian(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      concentration = tf.convert_to_tensor(self.concentration)
      return (tf.math.log(self.scale)
              - tf.math.log(concentration)
              - tf.math.log(y)
              - (1. + 1. / concentration) * tf.math.log(-tf.math.log(y)))

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      scale = tf.convert_to_tensor(self.scale)
      concentration = tf.convert_to_tensor(self.concentration)
      z = (x - self.loc) / scale
      return (tf.math.log(concentration) - tf.math.log(scale)
              - (1. + concentration) * tf.math.log(z) - z**(-concentration))

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args:
      return []
    return [assert_util.assert_greater_equal(
        x, self.loc,
        message='Forward transformation input must be greater than `loc`.')]

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return []
    is_positive = assert_util.assert_non_negative(
        y, message='Inverse transformation input must be greater than 0.')
    less_than_one = assert_util.assert_less_equal(
        y, tf.ones([], dtype=y.dtype),
        message='Inverse transformation input must be less than or equal to 1.')
    return [is_positive, less_than_one]

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale,
          message='Argument `scale` must be positive.'))
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
          self.concentration,
          message='Argument `concentration` must be positive.'))
    return assertions
