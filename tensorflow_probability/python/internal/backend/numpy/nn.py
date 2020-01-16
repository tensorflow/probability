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
"""Numpy implementations of TensorFlow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import numpy_array
from tensorflow_probability.python.internal.backend.numpy.numpy_math import l2_normalize
from tensorflow_probability.python.internal.backend.numpy.numpy_math import log_softmax
from tensorflow_probability.python.internal.backend.numpy.numpy_math import reduce_logsumexp
from tensorflow_probability.python.internal.backend.numpy.numpy_math import reduce_mean
from tensorflow_probability.python.internal.backend.numpy.numpy_math import softmax
from tensorflow_probability.python.internal.backend.numpy.numpy_math import softplus
from tensorflow_probability.python.internal.backend.numpy.numpy_math import squared_difference
from tensorflow_probability.python.internal.backend.numpy.numpy_math import top_k
from tensorflow_probability.python.internal.backend.numpy.ops import stop_gradient


__all__ = [
    'l2_normalize',
    'log_softmax',
    'moments',
    'relu',
    'softmax',
    'softplus',
    'sigmoid_cross_entropy_with_logits',
    'sparse_softmax_cross_entropy_with_logits',
    'top_k',
]


def _sigmoid_cross_entropy_with_logits(  # pylint: disable=invalid-name,unused-argument
    _sentinel=None,
    labels=None,
    logits=None,
    name=None):
  return (np.maximum(logits, 0)
          - logits * labels + np.log1p(np.exp(-np.abs(logits))))


def _sparse_softmax_cross_entropy_with_logits(  # pylint: disable=invalid-name,unused-argument
    _sentinel=None,
    labels=None,
    logits=None,
    name=None):
  """Softmax cross entropy with logits."""
  labels_shape = labels.shape
  num_classes = logits.shape[-1]
  logits = np.reshape(logits, [-1, num_classes])
  labels = np.reshape(labels, [-1])

  labels = numpy_array.one_hot(labels, num_classes)

  cost = -np.sum(
      labels * (logits - reduce_logsumexp(logits, axis=-1, keepdims=True)),
      axis=-1)
  cost = np.reshape(cost, labels_shape)
  return cost


# --- Begin Public Functions --------------------------------------------------

l2_normalize = utils.copy_docstring(
    tf.nn.l2_normalize,
    l2_normalize)


def _moments(x, axes, shift=None, keepdims=False, name=None):  # pylint: disable=unused-argument
  # NOTE: If x.dtype is float16, we may want to compute in float32.
  mean = reduce_mean(x, axis=axes, keepdims=True)
  # NOTE: The gradient backpropagated to the mean from the variance calcuation
  # is zero, so we can safely use `stop_gradient(mean)` for efficiency.
  variance = reduce_mean(squared_difference(x, stop_gradient(mean)),
                         axis=axes, keepdims=keepdims)
  if not keepdims:
    mean = numpy_array.squeeze(mean, axes)
  return (mean, variance)

moments = utils.copy_docstring(
    tf.nn.moments,
    _moments)


relu = utils.copy_docstring(
    tf.nn.relu,
    lambda features, name=None: np.max(features, 0))


softplus = utils.copy_docstring(
    tf.nn.softplus,
    lambda features, name=None: np.log(1 + np.exp(features)))


sigmoid_cross_entropy_with_logits = utils.copy_docstring(
    tf.nn.sigmoid_cross_entropy_with_logits,
    _sigmoid_cross_entropy_with_logits)


sparse_softmax_cross_entropy_with_logits = utils.copy_docstring(
    tf.nn.sparse_softmax_cross_entropy_with_logits,
    _sparse_softmax_cross_entropy_with_logits)
