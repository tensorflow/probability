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
"""AbsoluteValue bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    "AbsoluteValue",
]


class AbsoluteValue(bijector.Bijector):
  """Computes `Y = g(X) = Abs(X)`, element-wise.

  This non-injective bijector allows for transformations of scalar distributions
  with the absolute value function, which maps `(-inf, inf)` to `[0, inf)`.

  * For `y in (0, inf)`, `AbsoluteValue.inverse(y)` returns the set inverse
    `{x in (-inf, inf) : |x| = y}` as a tuple, `-y, y`.
  * `AbsoluteValue.inverse(0)` returns `0, 0`, which is not the set inverse
    (the set inverse is the singleton `{0}`), but "works" in conjunction with
    `TransformedDistribution` to produce a left semi-continuous pdf.
  * For `y < 0`, `AbsoluteValue.inverse(y)` happily returns the
    wrong thing, `-y, y`.  This is done for efficiency.  If
    `validate_args == True`, `y < 0` will raise an exception.


  ```python
  abs = tfp.bijectors.AbsoluteValue()

  abs.forward([-1., 0., 1.])
  ==> [1., 0.,  1.]

  abs.inverse(1.)
  ==> [-1., 1.]

  # The |dX/dY| is constant, == 1.  So Log|dX/dY| == 0.
  abs.inverse_log_det_jacobian(1.)
  ==> [0., 0.]

  # Special case handling of 0.
  abs.inverse(0.)
  ==> [0., 0.]

  abs.inverse_log_det_jacobian(0.)
  ==> [0., 0.]
  ```

  """

  def __init__(self, validate_args=False, name="absolute_value"):
    """Instantiates the `AbsoluteValue` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness, in particular whether inputs to `inverse` and
        `inverse_log_det_jacobian` are non-negative.
      name: Python `str` name given to ops managed by this object.
    """
    with tf.name_scope(name) as name:
      super(AbsoluteValue, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          name=name)

  def _forward(self, x):
    return tf.math.abs(x)

  def _inverse(self, y):
    with tf.control_dependencies(self._assertions(y)):
      return -y, y

  def _inverse_log_det_jacobian(self, y):
    # If event_ndims = 2,
    # F^{-1}(y) = (-y, y), so DF^{-1}(y) = (-1, 1),
    # so Log|DF^{-1}(y)| = Log[1, 1] = [0, 0].
    with tf.control_dependencies(self._assertions(y)):
      zero = tf.zeros([], dtype=dtype_util.base_dtype(y.dtype))
      return zero, zero

  @property
  def _is_injective(self):
    return False

  def _assertions(self, t):
    if not self.validate_args:
      return []
    return [assert_util.assert_non_negative(
        t, message="Argument y was negative")]
