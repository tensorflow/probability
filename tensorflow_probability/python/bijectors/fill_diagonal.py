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
"""FillDiagonal bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    "FillDiagonal",
]


class FillDiagonal(bijector.Bijector):
  """Transforms vectors to diagonal matrices.

  Given input with shape `batch_shape + [d]`, produces output with
  shape `batch_shape + [d, d]`.

  #### Example

  ```python
  b = tfb.FillDiagonal()
  b.forward([1, 2, 3])
  # ==> [[1, 0, 0],
  #      [0, 2, 0],
  #      [0, 0, 3]]

  ```
  """
  def __init__(self, validate_args=False, name="fill_diagonal"):
    """Instantiates the `FillDiagonal` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    super(FillDiagonal, self).__init__(
        forward_min_event_ndims=1,
        inverse_min_event_ndims=2,
        is_constant_jacobian=True,
        validate_args=validate_args,
        name=name,
    )

  def _forward(self, x):
    return tf.linalg.diag(x)

  def _inverse(self, y):
    return tf.linalg.diag_part(y)

  def _forward_log_det_jacobian(self, x):
    return tf.zeros([], dtype=x.dtype)

  def _inverse_log_det_jacobian(self, y):
    return tf.zeros([], dtype=y.dtype)

  def _forward_event_shape(self, input_shape):
    batch_shape, d = input_shape[:-1], tf.compat.dimension_value(
        input_shape[-1])
    return tensorshape_util.concatenate(batch_shape, [d, d])

  def _inverse_event_shape(self, output_shape):
    batch_shape, n1, n2 = (
        output_shape[:-2],
        tf.compat.dimension_value(output_shape[-2]),
        tf.compat.dimension_value(output_shape[-1]),
    )
    if n1 is None or n2 is None:
      m = None
    elif n1 != n2:
      raise ValueError("Matrix must be square. (saw [{}, {}])".format(n1, n2))
    else:
      m = n1
    return tensorshape_util.concatenate(batch_shape, [m])

  def _forward_event_shape_tensor(self, input_shape_tensor):
    batch_shape, d = input_shape_tensor[:-1], input_shape_tensor[-1]
    return tf.concat([batch_shape, [d, d]], axis=0)

  def _inverse_event_shape_tensor(self, output_shape_tensor):
    batch_shape, n = output_shape_tensor[:-2], output_shape_tensor[-1]
    if self.validate_args:
      is_square_matrix = assert_util.assert_equal(
          n, output_shape_tensor[-2], message="Matrix must be square.")
      with tf.control_dependencies([is_square_matrix]):
        n = tf.identity(n)
    return tf.concat([batch_shape, [n]], axis=0)
