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
"""Gumbel bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.distributions import bijector

__all__ = [
    "Gumbel",
]


class Gumbel(bijector.Bijector):
  """Compute `Y = g(X) = exp(-exp(-(X - loc) / scale))`.

  This bijector maps inputs from `[-inf, inf]` to [0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1) gives back a
  random variable with the
  [Gumbel distribution](https://en.wikipedia.org/wiki/Gumbel_distribution):

  ```none
  Y ~ Gumbel(loc, scale)
  pdf(y; loc, scale) = exp(
    -( (y - loc) / scale + exp(- (y - loc) / scale) ) ) / scale
  ```
  """

  def __init__(self,
               loc=0.,
               scale=1.,
               validate_args=False,
               name="gumbel"):
    """Instantiates the `Gumbel` bijector.

    Args:
      loc: Float-like `Tensor` that is the same dtype and is
        broadcastable with `scale`.
        This is `loc` in `Y = g(X) = exp(-exp(-(X - loc) / scale))`.
      scale: Positive Float-like `Tensor` that is the same dtype and is
        broadcastable with `loc`.
        This is `scale` in `Y = g(X) = exp(-exp(-(X - loc) / scale))`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args
    with self._name_scope("init", values=[loc, scale]):
      self._loc = tf.convert_to_tensor(loc, name="loc")
      self._scale = tf.convert_to_tensor(scale, name="scale")
      tf.assert_same_float_dtype([self._loc, self._scale])
      if validate_args:
        self._scale = control_flow_ops.with_dependencies([
            tf.assert_positive(
                self._scale, message="Argument scale was not positive")
        ], self._scale)

    super(Gumbel, self).__init__(
        validate_args=validate_args,
        forward_min_event_ndims=0,
        name=name)

  @property
  def loc(self):
    """The `loc` in `Y = g(X) = exp(-exp(-(X - loc) / scale))`."""
    return self._loc

  @property
  def scale(self):
    """This is `scale` in `Y = g(X) = exp(-exp(-(X - loc) / scale))`."""
    return self._scale

  def _forward(self, x):
    z = (x - self.loc) / self.scale
    return tf.exp(-tf.exp(-z))

  def _inverse(self, y):
    y = self._maybe_assert_valid_y(y)
    return self.loc - self.scale * tf.log(-tf.log(y))

  def _inverse_log_det_jacobian(self, y):
    y = self._maybe_assert_valid_y(y)
    return tf.log(self.scale / (-tf.log(y) * y))

  def _forward_log_det_jacobian(self, x):
    z = (x - self.loc) / self.scale
    return -z - tf.exp(-z) - tf.log(self.scale)

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return y
    is_positive = tf.assert_non_negative(
        y, message="Inverse transformation input must be greater than 0.")
    less_than_one = tf.assert_less_equal(
        y,
        tf.constant(1., y.dtype),
        message="Inverse transformation input must be less than or equal to 1.")
    return control_flow_ops.with_dependencies([is_positive, less_than_one], y)
