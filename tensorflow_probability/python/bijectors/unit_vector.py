# Copyright 2022 The TensorFlow Probability Authors.
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
"""UnitVector bijector."""

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import pad as pad_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps

__all__ = [
    "UnitVector",
]


class UnitVector(bijector.AutoCompositeTensorBijector):
  """Bijector mapping vectors onto the unit sphere.

    This bijector maps points in n-dimensional space to the
    unit sphere in (n + 1)-dimensional space via the inverse
    stereographic projection from the point (0, ..., 0, 1).

    The forward map is:

    f(x_0, ..., x_{n-1})
        = (2x_0/(s^2 + 1), ..., 2x_{n-1}/(s^2 + 1), (s^2 - 1)/(s^2 + 1))

    where s^2 = x_0^2 + ... + x_{n-1}^2.

    And the inverse map is

    f^{-1}(y_0, ..., y_n) = (y_0/(1 - y_n), ..., y_{n-1}/(1 - y_n)).

    Example Use:

    ```python
    UnitVector().forward([2., 1, 2])
    # Result: [0.4, 0.2, 0.4, 0.8]

    UnitVector().inverse([0.4, 0.2, 0.4, 0.8])
    # Result: [2., 1., 2.]
    ```
  """

  def __init__(self, validate_args=False, name="unit_vector"):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._pad = pad_lib.Pad(validate_args=validate_args)
      super(UnitVector, self).__init__(
          forward_min_event_ndims=1,
          validate_args=validate_args,
          parameters=parameters,
          name=name,
      )

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  def _forward_event_shape(self, input_shape):
    return self._pad.forward_event_shape(input_shape)

  def _forward_event_shape_tensor(self, input_shape):
    return self._pad.forward_event_shape_tensor(input_shape)

  def _inverse_event_shape(self, output_shape):
    return self._pad.inverse_event_shape(output_shape)

  def _inverse_event_shape_tensor(self, output_shape):
    return self._pad.inverse_event_shape_tensor(output_shape)

  def _forward(self, x):
    norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    return tf.concat([2 * x, norm_sq - 1], axis=-1) / (norm_sq + 1)

  def _inverse(self, y):
    assertions = []
    if self.validate_args:
      assertions.append(
          assert_util.assert_near(
              tf.reduce_sum(tf.square(y), axis=-1),
              tf.ones([], y.dtype),
              2.0 * np.finfo(dtype_util.as_numpy_dtype(y.dtype)).eps,
              message="`y` must be a unit vector scaled along the last axis.",
          ))
      assertions.append(
          assert_util.assert_none_equal(
              y[..., -1],
              tf.ones([], y.dtype),
              message="Last term of last axis of `y` cannot be 1."))

    # will return nan if validate_args=False and y[..., -1] has 1
    with tf.control_dependencies(assertions):
      x = y / (1 - y[..., -1:])
      x = x[..., :-1]
    return x

  def _inverse_log_det_jacobian(self, y):
    return -self._forward_log_det_jacobian(self._inverse(y))

  def _forward_log_det_jacobian(self, x):
    # Since the forward map maps R^n to the sphere in R^{n+1}, we must
    # find sqrt{det{J^T J}} where J is the (n + 1) x n Jacobian matrix.
    #
    # J_{i, i} = 2 * (s^2 - 2x_i^2 + 1)/(s^2 + 1)^2
    #
    # J_{i, j} = -4x_ix_j/(s^2 + 1)^2, where 0 <= i, j <= n-1
    #
    # J_{n, i} = 4x_i/(s^2 + 1)^2
    #
    # The determinant will only depend on s^2, so assume
    # x_0 = s and x_1 = ... = x_{n-1} = 0.
    # This means sqrt{det{J^T J}} = (2 / (s^2 + 1))^n.

    n = ps.cast(ps.shape(x)[-1], dtype=x.dtype)
    return n * (
        ps.cast(tf.math.log(2.0), dtype=x.dtype) -
        tf.math.log1p(tf.reduce_sum(tf.square(x), axis=-1)))
