# Copyright 2020 The TensorFlow Probability Authors.
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
"""Contains useful math functions."""

import jax.numpy as np
import jax.scipy as scipy


logsumexp = scipy.special.logsumexp


def sigmoid(x):
  #  See https://github.com/google/jax/pull/1227
  return scipy.special.expit(x)


def softplus(x):
  return np.logaddexp(x, 0.)


def relu(x):
  return np.maximum(x, 0.)


def logsoftmax(x, axis=-1):
  return x - logsumexp(x, axis=axis, keepdims=True)


def softmax(x, axis=-1):
  exp_logits = np.exp(x - x.max(axis, keepdims=True))
  return exp_logits / exp_logits.sum(axis, keepdims=True)
