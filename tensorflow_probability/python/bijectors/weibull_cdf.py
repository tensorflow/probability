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
"""WeibullCDF bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'WeibullCDF',
]


@auto_composite_tensor.auto_composite_tensor(
    omit_kwargs=('name',), module_name='tfp.bijectors')
class WeibullCDF(bijector.AutoCompositeTensorBijector):
  """Compute `Y = g(X) = 1 - exp( -( X / scale) ** concentration), X >= 0`.

  This bijector maps inputs from `[0, inf]` to `[0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
  random variable with the
  [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution):

  ```none
  Y ~ Weibull(scale, concentration)
  pdf(y; scale, concentration, y >= 0) =
      (concentration / scale) * (y / scale)**(concentration - 1) *
      exp(-(y / scale)**concentration)
  ```

  Likwewise, the forward of this bijector is the Weibull distribution CDF.
  """

  def __init__(self,
               scale=1.,
               concentration=1.,
               validate_args=False,
               name='weibull_cdf'):
    """Instantiates the `WeibullCDF` bijector.

    Args:
      scale: Positive Float-type `Tensor` that is the same dtype and is
        broadcastable with `concentration`.
        This is `l` in `Y = g(X) = 1 - exp( -( x / l) ** k)`.
      concentration: Positive Float-type `Tensor` that is the same dtype and is
        broadcastable with `scale`.
        This is `k` in `Y = g(X) = 1 - exp( -( x / l) ** k)`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [scale, concentration], dtype_hint=tf.float32)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      super(WeibullCDF, self).__init__(
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
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def scale(self):
    """The `l` in `Y = g(X) = 1 - exp( -( x / l) ** k)`."""
    return self._scale

  @property
  def concentration(self):
    """The `k` in `Y = g(X) = 1 - exp( -( x / l) ** k)`."""
    return self._concentration

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      return -tf.math.expm1(-((x / self.scale) ** self.concentration))

  def _inverse(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      return self.scale * (-tf.math.log1p(-y)) ** (1 / self.concentration)

  def _inverse_log_det_jacobian(self, y):
    with tf.control_dependencies(self._maybe_assert_valid_y(y)):
      scale = tf.convert_to_tensor(self.scale)
      concentration = tf.convert_to_tensor(self.concentration)
      return (-tf.math.log1p(-y) +
              tf.math.xlogy(1 / concentration - 1, -tf.math.log1p(-y)) +
              tf.math.log(scale / concentration))

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._maybe_assert_valid_x(x)):
      scale = tf.convert_to_tensor(self.scale)
      concentration = tf.convert_to_tensor(self.concentration)
      one = tf.constant(1., dtype=self.dtype)
      return (-(x / scale)**concentration +
              tf.math.xlogy(concentration - one, x) +
              tf.math.log(concentration) - concentration * tf.math.log(scale))

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
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
          self.concentration,
          message='Argument `concentration` must be positive.'))
    return assertions
