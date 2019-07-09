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

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util


__all__ = [
    "Softfloor",
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
               name="softfloor"):
    with tf.name_scope(name) as name:
      self._temperature = tf.convert_to_tensor(temperature, name="temperature")
      if validate_args:
        self._temperature = distribution_util.with_dependencies([
            assert_util.assert_positive(
                self._temperature,
                message="Argument temperature was not positive")
        ], self._temperature)
      super(Softfloor, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          dtype=self._temperature.dtype,
          name=name)

  def _forward(self, x):
    # This has a well defined derivative with respect to x.
    # This is because in the range [a, a + 1.] this is just a rescaled
    # logit function and hence has a derivative. At the end points, because
    # the logit function satisfies 1 - sigma(-x) = sigma(x), we have that
    # the derivative is symmetric around the center of the interval (a + 0.5),
    # and hence is continuous at the endpoints.
    x = x - 0.5
    fractional_part = x - tf.math.floor(x)
    cyclic_part = tf.math.sigmoid((fractional_part - 0.5) / self.temperature)
    # Rescale so the left tail is 0., and the right tail is 1. This
    # will also guarantee us continuity. Differentiability comes from the
    # fact that the derivative of the sigmoid is symmetric, and hence
    # the two endpoints will have the same value for derivatives.
    rescaled_part = (
        cyclic_part / tf.math.tanh(1. / (self.temperature * 4)) -
        tf.math.exp(-0.5 / self.temperature) / (
            -tf.math.expm1(-0.5 / self.temperature)))
    return tf.math.floor(x) + rescaled_part

  # TODO(b/134588121): Improve the numerical stability of this function.
  def _inverse(self, y):
    fractional_part = y - tf.math.floor(y)
    # The naive thing to do is affine scale the fractional part, and apply
    # a logit function (to invert the _forward). However that has bad numerics
    # at lower temperatures, whereas this rewriting allows for lower
    # temperature scaling.
    new_fractional_part = (
        tf.math.log1p(fractional_part * -tf.math.expm1(
            -0.5 / self.temperature)) -
        tf.math.log(tf.math.exp(-0.5 / self.temperature) -
                    fractional_part * tf.math.expm1(-0.5 / self.temperature)))
    new_fractional_part = self.temperature * new_fractional_part + 0.5
    return tf.math.floor(y) + new_fractional_part

  def _forward_log_det_jacobian(self, x):
    x = x - 0.5
    fractional_part = x - tf.math.floor(x)
    inner_part = (fractional_part - 0.5) / self.temperature

    offset = (tf.math.log(self.temperature) - tf.math.softplus(
        0.5 / self.temperature) + tfp_math.softplus_inverse(
            0.5 / self.temperature))

    return (-tf.math.softplus(-inner_part) -
            tf.math.softplus(inner_part) -
            offset)

  @property
  def temperature(self):
    return self._temperature
