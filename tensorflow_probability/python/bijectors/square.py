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
"""Square bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util


__all__ = [
    "Square",
]


class Square(bijector.Bijector):
  """Compute `g(X) = X^2`; X is a positive real number.

  g is a bijection between the non-negative real numbers (R_+) and the
  non-negative real numbers.

  #### Examples

  ```python
  bijector.Square().forward(x=[[1., 0], [2, 1]])
  # Result: [[1., 0], [4, 1]], i.e., x^2

  bijector.Square().inverse(y=[[1., 4], [9, 1]])
  # Result: [[1., 2], [3, 1]], i.e., sqrt(y).
  ```

  """

  def __init__(self, validate_args=False, name="square"):
    """Instantiates the `Square` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    with tf.name_scope(name) as name:
      super(Square, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          name=name)

  def _forward(self, x):
    with tf.control_dependencies(self._assertions(x)):
      return tf.square(x)

  def _inverse(self, y):
    with tf.control_dependencies(self._assertions(y)):
      return tf.sqrt(y)

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._assertions(x)):
      return np.log(2.) + tf.math.log(x)

  def _assertions(self, t):
    if not self.validate_args:
      return []
    return [assert_util.assert_non_negative(
        t, message="All elements must be non-negative.")]
