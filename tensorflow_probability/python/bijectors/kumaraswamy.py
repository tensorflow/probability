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
"""Kumaraswamy bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util

__all__ = [
    "Kumaraswamy",
]


class Kumaraswamy(bijector.Bijector):
  """Compute `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a), X in [0, 1]`.

  This bijector maps inputs from `[0, 1]` to `[0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
  random variable with the [Kumaraswamy distribution](
  https://en.wikipedia.org/wiki/Kumaraswamy_distribution):

  ```none
  Y ~ Kumaraswamy(a, b)
  pdf(y; a, b, 0 <= y <= 1) = a * b * y ** (a - 1) * (1 - y**a) ** (b - 1)
  ```
  """

  def __init__(self,
               concentration1=1.,
               concentration0=1.,
               validate_args=False,
               name="kumaraswamy"):
    """Instantiates the `Kumaraswamy` bijector.

    Args:
      concentration1: Python `float` scalar indicating the transform power,
        i.e., `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)` where `a` is
        `concentration1`.
      concentration0: Python `float` scalar indicating the transform power,
        i.e., `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)` where `b` is
        `concentration0`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args

    with self._name_scope("init"):
      concentration1 = self._maybe_assert_valid_concentration(
          tf.convert_to_tensor(value=concentration1, name="concentration1"),
          validate_args=validate_args)
      concentration0 = self._maybe_assert_valid_concentration(
          tf.convert_to_tensor(value=concentration0, name="concentration0"),
          validate_args=validate_args)

    self._concentration1 = concentration1
    self._concentration0 = concentration0
    super(Kumaraswamy, self).__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  @property
  def concentration1(self):
    """The `a` in: `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)`."""
    return self._concentration1

  @property
  def concentration0(self):
    """The `b` in: `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)`."""
    return self._concentration0

  def _forward(self, x):
    x = self._maybe_assert_valid(x)
    return tf.exp(
        tf.math.log1p(-tf.exp(tf.math.log1p(-x) / self.concentration0)) /
        self.concentration1)

  def _inverse(self, y):
    y = self._maybe_assert_valid(y)
    return tf.exp(
        tf.math.log1p(-(1 - y**self.concentration1)**self.concentration0))

  def _inverse_log_det_jacobian(self, y):
    y = self._maybe_assert_valid(y)
    return (tf.math.log(self.concentration1) +
            tf.math.log(self.concentration0) +
            tf.math.xlogy(self.concentration1 - 1, y) +
            (self.concentration0 - 1) * tf.math.log1p(-y**self.concentration1))

  def _maybe_assert_valid_concentration(self, concentration, validate_args):
    """Checks the validity of a concentration parameter."""
    if not validate_args:
      return concentration
    return distribution_util.with_dependencies([
        assert_util.assert_positive(
            concentration, message="Concentration parameter must be positive."),
    ], concentration)

  def _maybe_assert_valid(self, x):
    if not self.validate_args:
      return x
    return distribution_util.with_dependencies([
        assert_util.assert_non_negative(
            x, message="sample must be non-negative"),
        assert_util.assert_less_equal(
            x,
            tf.ones([], self.concentration0.dtype),
            message="sample must be no larger than `1`."),
    ], x)
