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
"""IteratedSigmoidCentered bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.bijectors import bijector


__all__ = [
    "IteratedSigmoidCentered",
]


class IteratedSigmoidCentered(bijector.Bijector):
  """Bijector which applies a Stick Breaking procedure.

  Given a vector `x`, transform it in to a vector `y` such that
  `y[i] > 0, sum_i y[i] = 1.`. In other words, takes a vector in
  `R^{k-1}` (unconstrained space) and maps it to a vector in the
  unit simplex in `R^{k}`.

  This transformation is centered in that it maps the zero vector
  `[0., 0., ... 0.]` to the center of the simplex `[1/k, ... 1/k]`.

  This bijector arises from the stick-breaking procedure for constructing
  a Dirichlet distribution / Dirichlet process as defined in [Stan, 2018][1].


  Example Use:

  ```python

  bijector.IteratedSigmoidCentered().forward([0., 0., 0.])
  # Result: [0.25, 0.25, 0.25, 0.25]
  # Extra result: 0.25

  bijector.IteratedSigmoidCentered().inverse([0.25, 0.25, 0.25, 0.25])
  # Result: [0., 0., 0.]
  # Extra coordinate removed.
  ```

  At first blush it may seem like the [Invariance of domain](
  https://en.wikipedia.org/wiki/Invariance_of_domain) theorem implies this
  implementation is not a bijection. However, the appended dimension
  makes the (forward) image non-open and the theorem does not directly apply.


  #### References

  [1]: Stan Development Team. 2018. Stan Modeling Language Users Guide and
       Reference Manual, Version 2.18.0. http://mc-stan.org
  """

  def __init__(self,
               validate_args=False,
               name="iterated_sigmoid"):
    self._graph_parents = []
    self._name = name
    super(IteratedSigmoidCentered, self).__init__(
        forward_min_event_ndims=1,
        validate_args=validate_args,
        name=name)

  def _forward_event_shape(self, input_shape):
    if not input_shape[-1:].is_fully_defined():
      return input_shape
    return input_shape[:-1].concatenate(input_shape[-1] + 1)

  def _forward_event_shape_tensor(self, input_shape):
    return tf.concat([input_shape[:-1], [input_shape[-1] + 1]], axis=0)

  def _inverse_event_shape(self, output_shape):
    if not output_shape[-1:].is_fully_defined():
      return output_shape
    if output_shape[-1] <= 1:
      raise ValueError("output_shape[-1] = %d <= 1" % output_shape[-1])
    return output_shape[:-1].concatenate(output_shape[-1] - 1)

  def _inverse_event_shape_tensor(self, output_shape):
    if self.validate_args:
      # It is not possible for a negative shape so we need only check <= 1.
      dependencies = [tf.compat.v1.assert_greater(
          output_shape[-1], 1, message="Need last dimension greater than 1.")]
    else:
      dependencies = []
    with tf.control_dependencies(dependencies):
      return tf.concat([output_shape[:-1], [output_shape[-1] - 1]], axis=0)

  def _forward(self, x):
    # As specified in the Stan reference manual, the procedure is as follows:
    # N = x.shape[-1] + 1
    # z_k = sigmoid(x + log(1 / (N - k)))
    # y_1 = z_1
    # y_k = (1 - sum_{i=1 to k-1} y_i) * z_k
    # y_N = 1 - sum_{i=1 to N-1} y_i
    # TODO(b/128857065): The numerics can possibly be improved here with a
    # log-space computation.
    offset = -tf.math.log(tf.cast(
        tf.range(tf.shape(input=x)[-1], 0, delta=-1), dtype=x.dtype.base_dtype))
    z = tf.nn.sigmoid(x + offset)
    y = z * tf.math.cumprod(1 - z, axis=-1, exclusive=True)
    return tf.concat(
        [y, 1. - tf.reduce_sum(input_tensor=y, axis=-1, keepdims=True)],
        axis=-1)

  def _inverse(self, y):
    # As specified in the Stan reference manual, the procedure is as follows:
    # N = y.shape[-1]
    # z_k = y_k / (1 - sum_{i=1 to k-1} y_i)
    # x_k = logit(z_k) - log(1 / (N - k))
    offset = tf.math.log(tf.cast(
        tf.range(
            tf.shape(input=y)[-1] - 1, 0, delta=-1), dtype=y.dtype.base_dtype))
    z = y / (1. - tf.math.cumsum(y, axis=-1, exclusive=True))
    return tf.math.log(z[..., :-1]) - tf.math.log1p(-z[..., :-1]) + offset

  def _inverse_log_det_jacobian(self, y):
    z = y / (1. - tf.math.cumsum(y, axis=-1, exclusive=True))
    return tf.reduce_sum(input_tensor=(
        -tf.math.log(y[..., :-1]) - tf.math.log1p(-z[..., :-1])), axis=-1)
