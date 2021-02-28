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
"""Softsign bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util


__all__ = [
    "Softsign",
]


class Softsign(bijector.Bijector):
  """Bijector which computes `Y = g(X) = X / (1 + |X|)`.

  The softsign `Bijector` has the following two useful properties:

  * The domain is all real numbers
  * `softsign(x) approx sgn(x)`, for large `|x|`.

  #### Examples

  ```python
  # Create the Y = softsign(X) transform.
  softsign = Softsign()
  x = [[[1., 2],
        [3, 4]],
       [[5, 6],
        [7, 8]]]
  x / (1 + abs(x)) == softsign.forward(x)
  x / (1 - abs(x)) == softsign.inverse(x)
  ```
  """

  def __init__(self, validate_args=False, name="softsign"):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(Softsign, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _is_increasing(cls):
    return True

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  def _forward(self, x):
    abs_x = tf.math.abs(x)
    # This is right for finite x, but if x == +-inf, this formula will give nan,
    # whereas the correct answer is sign(x).
    answer = x / (1. + abs_x)
    # So we explicitly fix it, by masking off any x big enough that 1 + abs_x ==
    # abs_x even in float 64.
    return tf.where(abs_x >= 1e20, tf.math.sign(x), answer)

  def _inverse(self, y):
    with tf.control_dependencies(self._assertions(y)):
      # y = +-1 exactly is still ok: the denominator will be 0, so the result
      # will be +-inf according to the sign of y, which is the best answer we
      # can give.
      return y / (1. - tf.math.abs(y))

  def _forward_log_det_jacobian(self, x):
    return -2. * tf.math.log1p(tf.math.abs(x))

  def _inverse_log_det_jacobian(self, y):
    with tf.control_dependencies(self._assertions(y)):
      return -2. * tf.math.log1p(-tf.math.abs(y))

  def _assertions(self, t):
    if not self.validate_args:
      return []
    return [
        assert_util.assert_greater_equal(
            t,
            dtype_util.as_numpy_dtype(t.dtype)(-1),
            message="Inverse transformation input must be >= -1."),
        assert_util.assert_less_equal(
            t,
            dtype_util.as_numpy_dtype(t.dtype)(1),
            message="Inverse transformation input must be <= 1.")
    ]
