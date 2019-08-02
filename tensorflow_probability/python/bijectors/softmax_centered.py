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
"""SoftmaxCentered bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    "SoftmaxCentered",
]


class SoftmaxCentered(bijector.Bijector):
  """Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.

  To implement [softmax](https://en.wikipedia.org/wiki/Softmax_function) as a
  bijection, the forward transformation appends a value to the input and the
  inverse removes this coordinate. The appended coordinate represents a pivot,
  e.g., `softmax(x) = exp(x-c) / sum(exp(x-c))` where `c` is the implicit last
  coordinate.

  Example Use:

  ```python
  bijector.SoftmaxCentered().forward(tf.log([2, 3, 4]))
  # Result: [0.2, 0.3, 0.4, 0.1]
  # Extra result: 0.1

  bijector.SoftmaxCentered().inverse([0.2, 0.3, 0.4, 0.1])
  # Result: tf.log([2, 3, 4])
  # Extra coordinate removed.
  ```

  At first blush it may seem like the [Invariance of domain](
  https://en.wikipedia.org/wiki/Invariance_of_domain) theorem implies this
  implementation is not a bijection. However, the appended dimension
  makes the (forward) image non-open and the theorem does not directly apply.
  """

  def __init__(self,
               validate_args=False,
               name="softmax_centered"):
    with tf.name_scope(name) as name:
      super(SoftmaxCentered, self).__init__(
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
      is_greater_one = assert_util.assert_greater(
          output_shape[-1], 1, message="Need last dimension greater than 1.")
      with tf.control_dependencies([is_greater_one]):
        output_shape = tf.identity(output_shape)
    return tf.concat([output_shape[:-1], [output_shape[-1] - 1]], axis=0)

  def _forward(self, x):
    # Pad the last dim with a zeros vector. We need this because it lets us
    # infer the scale in the inverse function.
    y = distribution_util.pad(x, axis=-1, back=True)

    # Set shape hints.
    if tensorshape_util.rank(x.shape) is not None:
      last_dim = tf.compat.dimension_value(x.shape[-1])
      shape = tensorshape_util.concatenate(
          x.shape[:-1],
          None if last_dim is None else last_dim + 1)
      tensorshape_util.set_shape(y, shape)

    return tf.math.softmax(y)

  def _inverse(self, y):
    # To derive the inverse mapping note that:
    #   y[i] = exp(x[i]) / normalization
    # and
    #   y[end] = 1 / normalization.
    # Thus:
    # x[i] = log(exp(x[i])) - log(y[end]) - log(normalization)
    #      = log(exp(x[i])/normalization) - log(y[end])
    #      = log(y[i]) - log(y[end])

    # Do this first to make sure CSE catches that it'll happen again in
    # _inverse_log_det_jacobian.
    x = tf.math.log(y)

    log_normalization = (-x[..., -1])[..., tf.newaxis]
    x = x[..., :-1] + log_normalization

    # Set shape hints.
    if tensorshape_util.rank(y.shape) is not None:
      last_dim = tf.compat.dimension_value(y.shape[-1])
      shape = tensorshape_util.concatenate(
          y.shape[:-1],
          None if last_dim is None else last_dim - 1)
      tensorshape_util.set_shape(x, shape)

    return x

  def _inverse_log_det_jacobian(self, y):
    # Let B be the forward map defined by the bijector. Consider the map
    # F : R^n -> R^n where the image of B in R^{n+1} is restricted to the first
    # n coordinates.
    #
    # Claim: det{ dF(X)/dX } = prod(Y) where Y = B(X).
    # Proof: WLOG, in vector notation:
    #     X = log(Y[:-1]) - log(Y[-1])
    #   where,
    #     Y[-1] = 1 - sum(Y[:-1]).
    #   We have:
    #     det{dF} = 1 / det{ dX/dF(X} }                                      (1)
    #             = 1 / det{ diag(1 / Y[:-1]) + 1 / Y[-1] }
    #             = 1 / det{ inv{ diag(Y[:-1]) - Y[:-1]' Y[:-1] } }
    #             = det{ diag(Y[:-1]) - Y[:-1]' Y[:-1] }
    #             = (1 + Y[:-1]' inv{diag(Y[:-1])} Y[:-1]) det{diag(Y[:-1])} (2)
    #             = Y[-1] prod(Y[:-1])
    #             = prod(Y)
    #
    # Let P be the image of R^n under F. Define the lift G, from P to R^{n+1},
    # which appends the last coordinate, Y[-1] := 1 - \sum_k Y_k. G is linear,
    # so its Jacobian is constant.
    #
    # The differential of G, DG, is eye(n) with a row of -1s appended to the
    # bottom. To compute the Jacobian sqrt{det{(DG)^T(DG)}}, one can see that
    # (DG)^T(DG) = A + eye(n), where A is the n x n matrix of 1s. This has
    # eigenvalues (n + 1, 1,...,1), so the determinant is (n + 1). Hence, the
    # Jacobian of G is sqrt{n + 1} everywhere.
    #
    # Putting it all together, the forward bijective map B can be written as
    # B(X) = G(F(X)) and has Jacobian sqrt{n + 1} * prod(F(X)).
    #
    # (1) - https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    #       or by noting that det{ dX/dY } = 1 / det{ dY/dX } from Bijector
    #       docstring "Tip".
    # (2) - https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    n_plus_one = prefer_static.shape(y)[-1]
    return -tf.reduce_sum(
        tf.math.log(y), axis=-1) - 0.5 * tf.math.log(
            tf.cast(n_plus_one, dtype=y.dtype))

  def _forward_log_det_jacobian(self, x):
    # This code is similar to tf.math.log_softmax but different because we have
    # an implicit zero column to handle. I.e., instead of:
    #   reduce_sum(logits - reduce_sum(exp(logits), dim))
    # we must do:
    #   log_normalization = 1 + reduce_sum(exp(logits))
    #   -log_normalization + reduce_sum(logits - log_normalization)
    n = prefer_static.shape(x)[-1]
    log_normalization = tf.math.softplus(
        tf.reduce_logsumexp(x, axis=-1, keepdims=True))
    return tf.squeeze(
        (-log_normalization +
         tf.reduce_sum(x - log_normalization, axis=-1, keepdims=True)),
        axis=-1) + 0.5 * tf.math.log(tf.cast(n + 1, dtype=x.dtype))
