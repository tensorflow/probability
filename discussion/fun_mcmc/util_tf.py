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
"""FunMCMC utilities implemented via TensorFlow."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'flatten_tree',
    'make_dynamic_array',
    'map_tree',
    'map_tree_up_to',
    'random_normal',
    'random_uniform',
    'snapshot_dynamic_array',
    'split_seed',
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


def make_dynamic_array(dtype, size, element_shape):
  """Creates an array that can be written to dynamically."""
  return tf.TensorArray(dtype=dtype, size=size, element_shape=element_shape)


def write_dynamic_array(array, i, e):
  """Writes to an index in a dynamic array."""
  return array.write(i, e)


def snapshot_dynamic_array(array):
  """Converts a dynamic array back to a static array."""
  return array.stack()


def value_and_grad(fn, args):
  """Given `fn: (args) -> out, extra`, returns `dout/dargs`."""
  with tf.GradientTape() as tape:
    args = map_tree(tf.convert_to_tensor, args)
    tape.watch(args)
    ret, extra = fn(args)
  grads = tape.gradient(ret, args)
  return ret, extra, grads


def split_seed(seed, count):
  """Splits a seed into `count` seeds."""
  # TODO(siege): Switch to stateless RNG ops.
  if seed is None:
    return count * [None]
  return [
      np.random.RandomState(seed + i).randint(0, 2**31)
      for i, seed in enumerate([seed] * count)
  ]


def random_uniform(shape, dtype, seed):
  """Generates a sample from uniform distribution over [0., 1)."""
  # TODO(siege): Switch to stateless RNG ops.
  return tf.random.uniform(shape=shape, dtype=dtype, seed=seed)


def random_normal(shape, dtype, seed):
  """Generates a sample from a standard normal distribution."""
  # TODO(siege): Switch to stateless RNG ops.
  return tf.random.normal(shape=shape, dtype=dtype, seed=seed)
