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
"""RayleighCDF bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'RayleighCDF',
]


class RayleighCDF(bijector.Bijector):
  """Compute `Y = g(X) = 1 - exp( -(X/scale)**2 / 2 ), X >= 0`.

  This bijector maps inputs from `[0, inf]` to `[0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
  random variable with the
  [Rayleigh distribution](https://en.wikipedia.org/wiki/Rayleigh_distribution):

  ```none
  Y ~ Rayleigh(scale)
  pdf(y; scale, y >= 0) =
      (1 / scale) * (y / scale) *
      exp(-(y / scale)**2 / 2)
  ```

  Likwewise, the forward of this bijector is the Rayleigh distribution CDF.
  """

  def __init__(self,
               scale=1.,
               validate_args=False,
               name='rayleigh_cdf'):
    """Instantiates the `RayleighCDF` bijector.

    Args:
      scale: Positive Float-type `Tensor`.
        This is `l` in `Y = g(X) = 1 - exp( -(X/l)**2 / 2 ), X >= 0`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([scale], dtype_hint=tf.float32)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      super(RayleighCDF, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          parameters=parameters,
          name=name,
          dtype=dtype)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def scale(self):
    return self._scale

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      return -tf.math.expm1(-tf.math.square(x / self.scale) / 2.)

  def _inverse(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      return self.scale * tf.math.sqrt(-2. * tf.math.log1p(-y))

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      scale = tf.convert_to_tensor(self.scale)
      return (-tf.math.square(x / scale) / 2 +
              tf.math.log(x) -
              2 * tf.math.log(scale))

  def _inverse_log_det_jacobian(self, y):
    # The derivative of inverse is given by:
    #   d/dy(s sqrt(-2 log(-y + 1)))
    #     = -(s / sqrt(2)) / ((y - 1) sqrt(-log(1 - y)))
    #     = (s / sqrt(2)) / ((1 - y) sqrt(-log(1 - y)))
    #
    # Denote z = (s / sqrt(2)) / ((1 - y) sqrt(-log(1 - y))).
    #
    # Taking the log of z yields:
    #   log(z)  = log(s / sqrt(2)) - log(1 - y) - log(sqrt(-log(1 - y)))
    #           = log(s / sqrt(2)) - log1p(-y) - log(-log1p(-y)) / 2.
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      scale = tf.convert_to_tensor(self.scale)
      return (-tf.math.log1p(-y) +
              -tf.math.log(-tf.math.log1p(-y)) / 2. +
              tf.math.log(scale / tf.math.sqrt(2.)))

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args:
      return []
    return [assert_util.assert_non_negative(
        x, message='Forward transformation input must be at least 0.')]

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
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale,
          message='Argument `scale` must be positive.'))
    return assertions
