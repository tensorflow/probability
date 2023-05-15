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
"""Moyal bijector."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import special as tfp_math


__all__ = [
    'MoyalCDF',
]


class MoyalCDF(
    bijector.CoordinatewiseBijectorMixin,
    bijector.AutoCompositeTensorBijector):
  """Compute `Y = g(X) = erfc(exp(- 1/2 * (X - loc) / scale) / sqrt(2))`.

  This bijector maps inputs from `[-inf, inf]` to `[0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
  random variable with the Moyal distribution:

  ```none
  Y ~ MoyalCDF(loc, scale)
  pdf(y; loc, scale) = exp(
    - 1/2 * ( (y - loc) / scale + exp(- (y - loc) / scale) ) ) /
    (sqrt(2 * pi) * scale)
  ```
  """

  def __init__(self,
               loc=0.,
               scale=1.,
               validate_args=False,
               name='moyal_cdf'):
    """Instantiates the `MoyalCDF` bijector.

    Args:
      loc: Float-like `Tensor` that is the same dtype and is
        broadcastable with `scale`.
        This is `loc` in
        `Y = g(X) = erfc(exp(- 1/2 * (X - loc) / scale) / sqrt(2))`.
      scale: Positive Float-like `Tensor` that is the same dtype and is
        broadcastable with `loc`.
        This is `scale` in
        `Y = g(X) = erfc(exp(- 1/2 * (X - loc) / scale) / sqrt(2))`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      super(MoyalCDF, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def loc(self):
    """The loc of the Moyal distribution."""
    return self._loc

  @property
  def scale(self):
    """The scale of the Moyal distribution."""
    return self._scale

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    z = (x - self.loc) / self.scale
    return tf.math.erfc(tf.exp(-z / 2) / np.sqrt(2.))

  def _inverse(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      np_dtype = dtype_util.as_numpy_dtype(y.dtype)
      return (self.loc - self.scale *
              (np.log(np_dtype(2.)) + 2. * tf.math.log(tfp_math.erfcinv(y))))

  def _inverse_log_det_jacobian(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      np_dtype = dtype_util.as_numpy_dtype(y.dtype)
      return (tf.math.square(tfp_math.erfcinv(y)) + tf.math.log(self.scale) +
              0.5 * np_dtype(np.log(np.pi)) - tf.math.log(tfp_math.erfcinv(y)))

  def _forward_log_det_jacobian(self, x):
    scale = tf.convert_to_tensor(self.scale)
    z = (x - self.loc) / scale
    return -0.5 * (tf.exp(-z) + z + np.log(2 * np.pi) + 2. * tf.math.log(scale))

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
