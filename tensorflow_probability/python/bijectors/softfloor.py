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
"""Softfloor bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'Softfloor',
]


class Softfloor(bijector.Bijector):
  """Compute a differentiable approximation to `tf.math.floor`.

  Given `x`, compute a differentiable approximation to `tf.math.floor(x)`.
  It is parameterized by a temperature parameter `t` to control the closeness
  of the approximation at the cost of numerical stability of the inverse.

  This `Bijector` has the following properties:
    * This `Bijector` is a map between `R` to `R`.
    * For `t` close to `0`, this bijector mimics the identity function.
    * For `t` approaching `infinity`, this bijector converges pointwise
    to `tf.math.floor` (except at integer points).

  Note that for lower temperatures `t`, this bijector becomes more numerically
  unstable. In particular, the inverse for this bijector is not numerically
  stable at lower temperatures, because flooring is not a bijective function (
  and hence any pointwise limit towards the floor function will start to have a
  non-numerically stable inverse).

  #### Mathematical details

  Let `x` be in `[0.5, 1.5]`. We would like to simulate the floor function on
  this interval. We will do this via a shifted and rescaled `sigmoid`.

  `floor(x) = 0` for `x < 1` and `floor(x) = 1` for `x >= 1`.
  If we take `f(x) = sigmoid((x - 1.) / t)`, where `t > 0`, we can see that
  when `t` goes to zero, we get that when `x > 1`, the `f(x)` tends towards `1`
  while `f(x)` tends to `0` when `x < 1`, thus giving us a function that looks
  like the floor function. If we shift `f(x)` by `-sigmoid(-0.5 / t)` and
  rescale by `1 / (sigmoid(0.5 / t) - sigmoid(-0.5 / t))`, we preserve the
  pointwise limit, but also fix `f(0.5) = 0.` and `f(1.5) = 1.`.

  Thus we can define `softfloor(x, t) = a * sigmoid((x - 1.) / t) + b`

  where
    * `a = 1 / (sigmoid(0.5 / t) - sigmoid(-0.5 / t))`
    * `b = -sigmoid(-0.5 / t) / (sigmoid(0.5 / t) - sigmoid(-0.5 / t))`


  The implementation of the `Softfloor` bijector follows this, with the caveat
  that we extend the function to all of the real line, by appropriately shifting
  this function for each integer.

  #### Examples

  Example use:

  ```python
  # High temperature.
  soft_floor = Softfloor(temperature=100.)
  x = [2.1, 3.2, 5.5]
  soft_floor.forward(x)

  # Low temperature. This acts like a floor.
  soft_floor = Softfloor(temperature=0.01)
  soft_floor.forward(x) # Should be close to [2., 3., 5.]

  # Ceiling is just a shifted floor at non-integer points.
  soft_ceiling = tfb.Chain(
    [tfb.AffineScalar(1.),
     tfb.Softfloor(temperature=1.)])
  soft_ceiling.forward(x) # Should be close to [3., 5., 6.]
  ```
  """

  def __init__(self,
               temperature,
               validate_args=False,
               name='softfloor'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [temperature], dtype_hint=tf.float32)
      self._temperature = tensor_util.convert_nonref_to_tensor(
          temperature, name='temperature', dtype=dtype)
      super(Softfloor, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          dtype=dtype,
          parameters=parameters,
          name=name)

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    # This has a well defined derivative with respect to x.
    # This is because in the range [0.5, 1.5] this is just a rescaled
    # logit function and hence has a derivative. At the end points, because
    # the logit function satisfies 1 - sigma(-x) = sigma(x), we have that
    # the derivative is symmetric around the center of the interval=1.,
    # and hence is continuous at the endpoints.
    t = tf.convert_to_tensor(self.temperature)
    fractional_part = x - tf.math.floor(x)
    # First, because our function is defined on the interval [0.5, 1.5]
    # repeated, we need to rescale our input to reflect that. x - floor(x)
    # will map our input to [0, 1]. However, we need to map inputs whose
    # fractional part is < 0.5 to the right hand portion of the interval.
    # We'll also need to adjust the integer part to reflect this.
    integer_part = tf.math.floor(x)
    # We wrap `0.5` in a `tf.constant` with explicit dtype to avoid upcasting
    # in the numpy backed. Declare this once, and use it everywhere.
    one_half = tf.constant(0.5, self.dtype)
    integer_part = tf.where(
        fractional_part < one_half,
        integer_part - tf.ones([], self.dtype), integer_part)
    fractional_part = tf.where(fractional_part < one_half,
                               fractional_part + one_half,
                               fractional_part - one_half)

    # Rescale so the left tail is 0., and the right tail is 1. This
    # will also guarantee us continuity. Differentiability comes from the
    # fact that the derivative of the sigmoid is symmetric, and hence
    # the two endpoints will have the same value for derivatives.
    # The below calculations are just
    # (sigmoid((f - 0.5) / t) - sigmoid(-0.5 / t)) /
    # (sigmoid(0.5 / t) - sigmoid(0.5 / t))
    # We use log_sum_exp and log_sub_exp to make this calculation more
    # numerically stable.

    log_numerator = tfp_math.log_sub_exp(
        (one_half + fractional_part) / t, one_half / t)
    # If fractional_part == 0, then we'll get log(0).
    log_numerator = tf.where(
        tf.equal(fractional_part, 0.),
        tf.constant(-np.inf, self.dtype), log_numerator)
    log_denominator = tfp_math.log_sub_exp(
        (one_half + fractional_part) / t, fractional_part / t)
    # If fractional_part == 0, then we'll get log(0).
    log_denominator = tf.where(
        tf.equal(fractional_part, 0.),
        tf.constant(-np.inf, self.dtype), log_denominator)
    log_denominator = tfp_math.log_add_exp(
        log_denominator,
        tfp_math.log_sub_exp(tf.ones([], self.dtype) / t, one_half / t))
    rescaled_part = tf.math.exp(log_numerator - log_denominator)
    return integer_part + rescaled_part

  def _inverse(self, y):
    # We undo the transformation from [0, 1] -> [0, 1].
    # The inverse of the transformation will look like a shifted and scaled
    # logit function. We rewrite this to be more numerically stable, and will
    # produce a term log(a / b). log_{numerator, denominator} below is log(a)
    # and log(b) respectively.
    t = tf.convert_to_tensor(self.temperature)
    fractional_part = y - tf.math.floor(y)
    log_f = tf.math.log(fractional_part)
    # We wrap `0` and `0.5` in a `tf.constant` with explicit dtype to avoid
    # upcasting in the numpy backed. Declare this once, and use it everywhere.
    zero = tf.zeros([], self.dtype)
    one_half = tf.constant(0.5, self.dtype)

    log_numerator = tfp_math.log_sub_exp(one_half / t + log_f, log_f)
    log_numerator = tfp_math.log_add_exp(zero, log_numerator)
    # When the fractional part is zero, the numerator is 1.
    log_numerator = tf.where(tf.equal(fractional_part, 0.), zero, log_numerator)
    log_denominator = tfp_math.log_sub_exp(one_half / t, log_f + one_half / t)
    log_denominator = tfp_math.log_add_exp(log_f, log_denominator)
    # When the fractional part is zero, the denominator is 0.5 / t.
    log_denominator = tf.where(
        tf.equal(fractional_part, 0.),
        one_half / t, log_denominator)

    new_fractional_part = (t * (log_numerator - log_denominator) + one_half)
    # We finally shift this up since the original transformation was from
    # [0.5, 1.5] to [0, 1].
    new_fractional_part = new_fractional_part + one_half
    return tf.math.floor(y) + new_fractional_part

  def _forward_log_det_jacobian(self, x):
    t = tf.convert_to_tensor(self.temperature)
    fractional_part = x - tf.math.floor(x)
    # Because our function is from [0.5, 1.5], we need to transform our
    # fractional_part to that domain like in the forward transformation.
    fractional_part = tf.where(
        fractional_part < 0.5, fractional_part + 0.5, fractional_part - 0.5)
    inner_part = (fractional_part - 0.5) / t

    offset = (tf.math.log(t) - tf.math.softplus(0.5 / t) +
              tfp_math.softplus_inverse(0.5 / t))

    return (-tf.math.softplus(-inner_part) -
            tf.math.softplus(inner_part) -
            offset)

  @property
  def temperature(self):
    return self._temperature

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if (self.temperature is not None and
        is_init != tensor_util.is_ref(self.temperature)):
      assertions.append(assert_util.assert_positive(
          self._temperature,
          message='Argument `temperature` was not positive.'))
    return assertions
