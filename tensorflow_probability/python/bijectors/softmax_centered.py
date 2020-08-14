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

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import pad as pad_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps


__all__ = [
    'SoftmaxCentered',
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
               name='softmax_centered'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._pad = pad_lib.Pad(validate_args=validate_args)
      super(SoftmaxCentered, self).__init__(
          forward_min_event_ndims=1,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  def _forward_event_shape(self, input_shape):
    return self._pad.forward_event_shape(input_shape)

  def _forward_event_shape_tensor(self, input_shape):
    return self._pad.forward_event_shape_tensor(input_shape)

  def _inverse_event_shape(self, output_shape):
    return self._pad.inverse_event_shape(output_shape)

  def _inverse_event_shape_tensor(self, output_shape):
    return self._pad.inverse_event_shape_tensor(output_shape)

  def _forward(self, x):
    return tf.math.softmax(self._pad.forward(x))

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

    assertions = []
    if self.validate_args:
      assertions.append(assert_util.assert_near(
          tf.reduce_sum(y, axis=-1),
          tf.ones([], y.dtype),
          2. * np.finfo(dtype_util.as_numpy_dtype(y.dtype)).eps,
          message='Last dimension of `y` must sum to `1`.'))
      assertions.append(assert_util.assert_less_equal(
          y, tf.ones([], y.dtype),
          message='Elements of `y` must be less than or equal to `1`.'))
      assertions.append(assert_util.assert_non_negative(
          y, message='Elements of `y` must be non-negative.'))

    with tf.control_dependencies(assertions):
      x = tf.math.log(y)
      x, log_normalization = tf.split(x, num_or_size_splits=[-1, 1], axis=-1)
    return x - log_normalization

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
    np1 = ps.cast(ps.shape(y)[-1], dtype=y.dtype)
    return -(0.5 * ps.log(np1) +
             tf.reduce_sum(tf.math.log(y), axis=-1))

  def _forward_log_det_jacobian(self, x):
    # This code is similar to tf.math.log_softmax but different because we have
    # an implicit zero column to handle. I.e., instead of:
    #   reduce_sum(logits - reduce_sum(exp(logits), dim))
    # we must do:
    #   log_normalization = 1 + reduce_sum(exp(logits))
    #   -log_normalization + reduce_sum(logits - log_normalization)
    np1 = ps.cast(1 + ps.shape(x)[-1], dtype=x.dtype)
    return (0.5 * ps.log(np1) +
            tf.reduce_sum(x, axis=-1) -
            np1 * tf.math.softplus(tf.reduce_logsumexp(x, axis=-1)))
