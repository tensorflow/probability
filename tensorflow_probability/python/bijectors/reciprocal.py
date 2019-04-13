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
"""A `Bijector` that computes `b(x) = 1. / x`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util

__all__ = ["Reciprocal"]


class Reciprocal(bijector.Bijector):
  """A `Bijector` that computes the reciprocal `b(x) = 1. / x` entrywise.

  This bijector accepts any non-zero values for both `forward` and `inverse`.

  #### Examples

  ```python
  bijector.Reciprocal().forward(x=[[1., 2.], [4., 5.]])
  # Result: [[1., .5], [.25, .2]], i.e., 1 / x

  bijector.Reciprocal().forward(x=[[0., 2.], [4., 5.]])
  # Result: AssertionError, doesn't accept zero.

  bijector.Square().inverse(y=[[1., 2.], [4., 5.]])
  # Result: [[1., .5], [.25, .2]], i.e. 1 / x

  ```
  """

  def __init__(self, validate_args=False, name="reciprocal"):
    """Instantiates the `Reciprocal`.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._name = name
    super(Reciprocal, self).__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    x = self._maybe_assert_valid(x)
    return 1. / x

  _inverse = _forward

  def _forward_log_det_jacobian(self, x):
    x = self._maybe_assert_valid(x)
    return -2. * tf.math.log(tf.abs(x))

  _inverse_log_det_jacobian = _forward_log_det_jacobian

  def _maybe_assert_valid(self, t):
    if not self.validate_args:
      return t
    is_valid = assert_util.assert_none_equal(
        t, 0., message="All elements must be non-zero.")
    return distribution_util.with_dependencies([is_valid], t)
