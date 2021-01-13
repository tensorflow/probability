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
"""Utilities for computing losses."""
# TODO(emilyaf): Add tests.

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow_probability.python.experimental.nn import layers as layers_lib
from tensorflow_probability.python.experimental.nn import variational_base as vi_lib
from tensorflow_probability.python.internal.reparameterization import FULLY_REPARAMETERIZED
from tensorflow_probability.python.monte_carlo import expectation


__all__ = [
    'kl_divergence_monte_carlo',
    'kl_divergence_exact',
    'negloglik',
    'compute_extra_loss',
]


def kl_divergence_monte_carlo(q, r, w):
  """Monte Carlo KL Divergence."""
  return expectation(
      lambda w: q.log_prob(w) - r.log_prob(w),
      samples=w,
      log_prob=q.log_prob,
      use_reparameterization=all(
          rt == FULLY_REPARAMETERIZED
          for rt in tf.nest.flatten(q.reparameterization_type)),
      axis=())


def kl_divergence_exact(q, r, w):  # pylint: disable=unused-argument
  """Exact KL Divergence."""
  return kl_lib.kl_divergence(q, r)


def negloglik(x, y, model_fn, axis=-1):
  """Negative log-likelihood."""
  return -tf.reduce_mean(model_fn(x).log_prob(y), axis=axis)


def compute_extra_loss(
    layer,
    loss_fn=kl_divergence_monte_carlo):
  loss = 0.
  if isinstance(layer, layers_lib.Sequential):
    for x in layer.layers:
      loss += compute_extra_loss(x)
  elif isinstance(layer, vi_lib.VariationalLayer):
    loss += loss_fn(layer.posterior, layer.prior, layer.posterior_value)
  return loss
