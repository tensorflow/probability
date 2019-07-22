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
from jax import random
import jax.numpy as np

__all__ = [
    'flatten_tree',
    'make_dynamic_array',
    'map_tree',
    'map_tree_up_to',
    'random_normal',
    'random_uniform',
    'snapshot_dynamic_array',
    'split_seed',
    'value_and_grad',
    'write_dynamic_array',
]


# JAX has an annoying default where it entirely skips running functions on
# `None`s in structures. We at times need to detect their presense, so we need
# to sanitize `None`s before giving them to JAX.
class _NoneSentinel(object):
  pass


_None = _NoneSentinel()  # pylint: disable=invalid-name


def _replace_none_sentinels(a_list):
  return [None if r is _None else r for r in a_list]


def _replace_nones(tree):
  if tree is None:
    return _None
  node_type = jax.tree_util._get_node_type(tree)  # pylint: disable=protected-access
  if node_type:
    children, node_spec = node_type.to_iterable(tree)
    new_children = [_replace_nones(child) for child in children]
    return node_type.from_iterable(node_spec, new_children)
  else:
    return tree


def map_tree(fn, tree, *rest):
  """Maps `fn` over the leaves of a nested structure."""
  rest = (tree,) + rest
  rest = [_replace_nones(r) for r in rest]

  def _wrapper(*args):
    return fn(*_replace_none_sentinels(args))

  return jax.tree_util.tree_multimap(_wrapper, *rest)


def flatten_tree(tree):
  """Flattens a nested structure to a list."""
  tree = _replace_nones(tree)

  ret = jax.tree_util.tree_flatten(tree)[0]
  return _replace_none_sentinels(ret)


def unflatten_tree(tree, xs):
  """Inverse operation of `flatten_tree`."""
  tree = _replace_nones(tree)

  return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tree), xs)


def map_tree_up_to(shallow, fn, tree, *rest):
  """`map_tree` with recursion depth defined by depth of `shallow`."""
  shallow = _replace_nones(shallow)

  def wrapper(_, *rest):
    return fn(*rest)

  return jax.tree_util.tree_multimap(wrapper, shallow, tree, *rest)


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
