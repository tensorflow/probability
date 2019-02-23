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
"""NormalCDF bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import special_math

__all__ = [
    "NormalCDF",
]


class NormalCDF(bijector.Bijector):
  """Compute `Y = g(X) = NormalCDF(x)`.

  This bijector maps inputs from `[-inf, inf]` to `[0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
  random variable with the
  [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution):

  ```none
  Y ~ Normal(0, 1)
  pdf(y; 0., 1.) = 1 / sqrt(2 * pi) * exp(-y ** 2 / 2)
  ```
  """

  def __init__(self,
               validate_args=False,
               name="normal"):
    """Instantiates the `NormalCDF` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args

    super(NormalCDF, self).__init__(
        validate_args=validate_args,
        forward_min_event_ndims=0,
        name=name)

  def _forward(self, x):
    return special_math.ndtr(x)

  def _inverse(self, y):
    y = self._maybe_assert_valid_y(y)
    return special_math.ndtri(y)

  def _forward_log_det_jacobian(self, x):
    return -0.5 * np.log(2 * np.pi) - tf.square(x) / 2.

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return y
    is_positive = tf.compat.v1.assert_non_negative(
        y, message="Inverse transformation input must be greater than 0.")
    less_than_one = tf.compat.v1.assert_less_equal(
        y,
        tf.constant(1., y.dtype),
        message="Inverse transformation input must be less than or equal to 1.")
    return distribution_util.with_dependencies([is_positive, less_than_one], y)
