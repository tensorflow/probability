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
"""Cumsum bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import prefer_static

__all__ = [
    "Cumsum",
]


class Cumsum(bijector.Bijector):
  """Computes the cumulative sum of a tensor along a specified axis.

  If `axis` is not provided, the default uses the rightmost dimension, i.e.,
  axis=-1.

  #### Example

  ```python
  x = tfb.Cumsum()

  x.forward([[1., 1.],
             [2., 2.],
             [3., 3.]])
  # ==> [[1., 2.],
         [2., 4.],
         [3., 6.]]

  x = tfb.Cumsum(axis=-2)

  x.forward([[1., 1.],
             [2., 2.],
             [3., 3.]])
  # ==> [[1., 1.],
         [3., 3.],
         [6., 6.]]
  ```

  """

  def __init__(self, axis=-1, validate_args=False, name="cumsum"):
    """Instantiates the `Cumsum` bijector.

    Args:
      axis: Negative Python `int` indicating the axis along which to compute the
        cumulative sum. Note that positive (and zero) values are not supported.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.

    Raises:
      ValueError: If `axis` is not a negative `int`.
    """
    if not isinstance(axis, int) or axis >= 0:
      raise ValueError("`axis` must be a negative integer.")
    self._axis = axis

    super(Cumsum, self).__init__(
        is_constant_jacobian=True,
        forward_min_event_ndims=-axis,  # Positive because we verify `axis < 0`.
        validate_args=validate_args,
        name=name)

  @property
  def axis(self):
    """Returns the axis over which this `Bijector` computes the cumsum."""
    return self._axis

  def _forward(self, x):
    return tf.cumsum(x, axis=self._axis)

  def _inverse(self, y):
    ndims = prefer_static.rank(y)
    shifted_y = tf.pad(
        tensor=tf.slice(
            y, tf.zeros(ndims, dtype=tf.dtypes.int32),
            prefer_static.shape(y) -
            tf.one_hot(ndims + self._axis, ndims, dtype=tf.dtypes.int32)
        ),  # Remove the last entry of y in the chosen dimension.
        paddings=tf.one_hot(
            tf.one_hot(ndims + self._axis, ndims, on_value=0, off_value=-1),
            2,
            dtype=tf.dtypes.int32
        )  # Insert zeros at the beginning of the chosen dimension.
    )

    return y - shifted_y

  def _forward_log_det_jacobian(self, x):
    return tf.constant(0., x.dtype)
