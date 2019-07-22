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
"""Sigmoid bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector


__all__ = [
    "Sigmoid",
]


class Sigmoid(bijector.Bijector):
  """Bijector which computes `Y = g(X) = 1 / (1 + exp(-X))`."""

  def __init__(self, validate_args=False, name="sigmoid"):
    with tf.name_scope(name) as name:
      super(Sigmoid, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          name=name)

  def _forward(self, x):
    return tf.sigmoid(x)

  def _inverse(self, y):
    return tf.math.log(y) - tf.math.log1p(-y)

  # We implicitly rely on _forward_log_det_jacobian rather than explicitly
  # implement _inverse_log_det_jacobian since directly using
  # `-tf.log(y) - tf.log1p(-y)` has lower numerical precision.

  def _forward_log_det_jacobian(self, x):
    return -tf.math.softplus(-x) - tf.math.softplus(x)
