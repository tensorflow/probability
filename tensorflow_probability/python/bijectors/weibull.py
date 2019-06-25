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
"""Weibull bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util


__all__ = [
    "Weibull",
]


class Weibull(bijector.Bijector):
  """Compute `Y = g(X) = 1 - exp((-X / scale) ** concentration), X >= 0`.

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
  """

  def __init__(self,
               scale=1.,
               concentration=1.,
               validate_args=False,
               name="weibull"):
    """Instantiates the `Weibull` bijector.

    Args:
      scale: Positive Float-type `Tensor` that is the same dtype and is
        broadcastable with `concentration`.
        This is `l` in `Y = g(X) = 1 - exp((-x / l) ** k)`.
      concentration: Positive Float-type `Tensor` that is the same dtype and is
        broadcastable with `scale`.
        This is `k` in `Y = g(X) = 1 - exp((-x / l) ** k)`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    with tf.name_scope(name) as name:
      self._scale = tf.convert_to_tensor(scale, name="scale")
      self._concentration = tf.convert_to_tensor(
          concentration, name="concentration")
      dtype_util.assert_same_float_dtype([self._scale, self._concentration])
      if validate_args:
        self._scale = distribution_util.with_dependencies([
            assert_util.assert_positive(
                self._scale, message="Argument scale was not positive")
        ], self._scale)
        self._concentration = distribution_util.with_dependencies([
            assert_util.assert_positive(
                self._concentration,
                message="Argument concentration was not positive")
        ], self._concentration)
      super(Weibull, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          name=name)

  @property
  def scale(self):
    """The `l` in `Y = g(X) = 1 - exp((-x / l) ** k)`."""
    return self._scale

  @property
  def concentration(self):
    """The `k` in `Y = g(X) = 1 - exp((-x / l) ** k)`."""
    return self._concentration

  def _forward(self, x):
    x = self._maybe_assert_valid_x(x)
    return -tf.math.expm1(-((x / self.scale)**self.concentration))

  def _inverse(self, y):
    y = self._maybe_assert_valid_y(y)
    return self.scale * (-tf.math.log1p(-y))**(1 / self.concentration)

  def _inverse_log_det_jacobian(self, y):
    y = self._maybe_assert_valid_y(y)
    return (-tf.math.log1p(-y) +
            tf.math.xlogy(1 / self.concentration - 1, -tf.math.log1p(-y)) +
            tf.math.log(self.scale / self.concentration))

  def _forward_log_det_jacobian(self, x):
    x = self._maybe_assert_valid_x(x)
    return (-(x / self.scale)**self.concentration +
            tf.math.xlogy(self.concentration - 1, x) +
            tf.math.log(self.concentration) -
            self.concentration * tf.math.log(self.scale))

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args:
      return x
    is_valid = assert_util.assert_non_negative(
        x, message="Forward transformation input must be at least 0.")
    return distribution_util.with_dependencies([is_valid], x)

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return y
    is_positive = assert_util.assert_non_negative(
        y, message="Inverse transformation input must be greater than 0.")
    less_than_one = assert_util.assert_less_equal(
        y,
        tf.constant(1., y.dtype),
        message="Inverse transformation input must be less than or equal to 1.")
    return distribution_util.with_dependencies([is_positive, less_than_one], y)
