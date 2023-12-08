# Copyright 2023 The TensorFlow Probability Authors.
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
"""Utilities for AutoBNN."""

from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
import scipy
from tensorflow_probability.python.experimental.autobnn import bnn
from tensorflow_probability.substrates.jax.distributions import distribution as distribution_lib


def make_transforms(
    net: bnn.BNN,
) -> Tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
  """Returns unconstraining bijectors for all variables in the BNN."""
  jb = jax.tree_map(
      lambda x: x.experimental_default_event_space_bijector(),
      net.get_all_distributions(),
      is_leaf=lambda x: isinstance(x, distribution_lib.Distribution),
  )

  def transform(params):
    return {'params': jax.tree_map(lambda p, b: b(p), params['params'], jb)}

  def inverse_transform(params):
    return {
        'params': jax.tree_map(lambda p, b: b.inverse(p), params['params'], jb)
    }

  def inverse_log_det_jacobian(params):
    return jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_map(
            lambda p, b: jnp.sum(b.inverse_log_det_jacobian(p)),
            params['params'],
            jb,
        ),
        initializer=0.0,
    )

  return transform, inverse_transform, inverse_log_det_jacobian


def suggest_periods(ys) -> List[float]:
  """Suggest a few periods for the time series."""
  f, pxx = scipy.signal.periodogram(ys)

  top5_powers, top5_indices = jax.lax.top_k(pxx, 5)
  top5_power = jnp.sum(top5_powers)
  best_indices = [i for i in top5_indices if pxx[i] > 0.05 * top5_power]
  # Sort in descending order so the best periods are first.
  best_indices.sort(reverse=True, key=lambda i: pxx[i])
  return [1.0 / f[i] for i in best_indices if 1.0 / f[i] < 0.6 * len(ys)]


def load_fake_dataset():
  """Return some fake data for testing purposes."""
  x_train = jnp.arange(0.0, 120.0) / 120.0
  y_train = x_train + jnp.sin(x_train * 10.0) + x_train * x_train
  x_train = x_train[..., jnp.newaxis]
  return x_train, y_train[..., jnp.newaxis]
