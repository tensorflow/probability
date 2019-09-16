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
"""FunMCMC utilities implemented via JAX."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import jax
from jax import lax
from jax import random
from jax.experimental import stax
import jax.numpy as np
# TODO(siege): Switch to a JAX-specific nested structure API.
import tensorflow.compat.v2 as tf
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'assert_same_shallow_tree',
    'flatten_tree',
    'make_dynamic_array',
    'map_tree',
    'map_tree_up_to',
    'random_categorical',
    'random_normal',
    'random_uniform',
    'snapshot_dynamic_array',
    'split_seed',
    'value_and_grad',
    'write_dynamic_array',
]


def map_tree(fn, tree, *args):
  """Maps `fn` over the leaves of a nested structure."""
  return tf.nest.map_structure(fn, tree, *args)


def flatten_tree(tree):
  """Flattens a nested structure to a list."""
  return tf.nest.flatten(tree)


def unflatten_tree(tree, xs):
  """Inverse operation of `flatten_tree`."""
  return tf.nest.pack_sequence_as(tree, xs)


def map_tree_up_to(shallow, fn, tree, *rest):
  """`map_tree` with recursion depth defined by depth of `shallow`."""
  return nest.map_structure_up_to(shallow, fn, tree, *rest)


def assert_same_shallow_tree(shallow, tree):
  """Asserts that `tree` has the same shallow structure as `shallow`."""
  nest.assert_shallow_structure(shallow, tree)


def make_dynamic_array(dtype, size, element_shape):
  """Creates an array that can be written to dynamically."""
  return np.empty((size,) + element_shape, dtype=dtype)


def write_dynamic_array(array, i, e):
  """Writes to an index in a dynamic array."""
  return jax.ops.index_update(array, i, e)


def snapshot_dynamic_array(array):
  """Converts a dynamic array back to a static array."""
  return array


def value_and_grad(fn, args):
  """Given `fn: (args) -> out, extra`, returns `dout/dargs`."""
  output, vjp_fn, extra = jax.vjp(fn, args, has_aux=True)
  grad = vjp_fn(np.ones_like(output))[0]
  return output, extra, grad


def split_seed(seed, count):
  """Splits a seed into `count` seeds."""
  return random.split(seed, count)


def random_uniform(shape, dtype, seed):
  """Generates a sample from uniform distribution over [0., 1)."""
  return random.uniform(shape=tuple(shape), dtype=dtype, key=seed)


def random_normal(shape, dtype, seed):
  """Generates a sample from a standard normal distribution."""
  return random.normal(shape=tuple(shape), dtype=dtype, key=seed)


def _searchsorted(a, v):
  """Returns where `v` can be inserted so that `a` remains sorted."""
  def cond(state):
    low_idx, high_idx = state
    return low_idx < high_idx

  def body(state):
    low_idx, high_idx = state
    mid_idx = (low_idx + high_idx) // 2
    mid_v = a[mid_idx]
    low_idx = np.where(v > mid_v, mid_idx + 1, low_idx)
    high_idx = np.where(v > mid_v, high_idx, mid_idx)
    return low_idx, high_idx

  low_idx, _ = lax.while_loop(cond, body, (0, a.shape[-1]))
  return low_idx


def random_categorical(logits, num_samples, seed):
  """Returns a sample from a categorical distribution. `logits` must be 2D."""
  probs = stax.softmax(logits)
  cum_sum = np.cumsum(probs, axis=-1)

  eta = random.uniform(seed, (num_samples,) + cum_sum.shape[:-1])
  cum_sum = np.broadcast_to(cum_sum, (num_samples,) + cum_sum.shape)

  flat_cum_sum = cum_sum.reshape([-1, cum_sum.shape[-1]])
  flat_eta = eta.reshape([-1])
  return jax.vmap(_searchsorted)(flat_cum_sum, flat_eta).reshape(eta.shape).T
