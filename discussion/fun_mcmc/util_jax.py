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
"""FunMCMC utilities implemented via JAX."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import jax
from jax import lax
from jax import random
from jax import tree_util
from jax.experimental import stax
import jax.numpy as np

__all__ = [
    'assert_same_shallow_tree',
    'flatten_tree',
    'map_tree',
    'map_tree_up_to',
    'random_categorical',
    'random_normal',
    'random_uniform',
    'split_seed',
    'trace',
    'value_and_grad',
]


def map_tree(fn, tree, *args):
  """Maps `fn` over the leaves of a nested structure."""
  return tree_util.tree_multimap(fn, tree, *args)


def flatten_tree(tree):
  """Flattens a nested structure to a list."""
  return tree_util.tree_flatten(tree)[0]


def unflatten_tree(tree, xs):
  """Inverse operation of `flatten_tree`."""
  return tree_util.tree_unflatten(tree_util.tree_structure(tree), xs)


def map_tree_up_to(shallow, fn, tree, *rest):
  """`map_tree` with recursion depth defined by depth of `shallow`."""

  def wrapper(_, *rest):
    return fn(*rest)

  return tree_util.tree_multimap(wrapper, shallow, tree, *rest)


def assert_same_shallow_tree(shallow, tree):
  """Asserts that `tree` has the same shallow structure as `shallow`."""
  # Do a dummy multimap for the side-effect of verifying that the structures are
  # the same. This doesn't catch all the errors we actually care about, sadly.
  map_tree_up_to(shallow, lambda *args: (), tree)


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


def trace(state, fn, num_steps, **_):
  """Implementation of `trace` operator, without the calling convention."""
  # We need the shapes and dtypes of the outputs of `fn`.
  _, untraced_spec, traced_spec = jax.eval_shape(
      fn, map_tree(lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype), state))
  untraced_init = map_tree(lambda spec: np.zeros(spec.shape, spec.dtype),
                           untraced_spec)

  try:
    num_steps = int(num_steps)
    use_scan = True
  except TypeError:
    use_scan = False
    if flatten_tree(traced_spec):
      raise ValueError(
          'Cannot trace values when `num_steps` is not statically known. Pass '
          'False to `trace_mask` or return an empty structure (e.g. `()`) as '
          'the extra output.')

  if use_scan:

    def wrapper(state_untraced, _):
      state, _ = state_untraced
      state, untraced, traced = fn(state)
      return (state, untraced), traced

    (state, untraced), traced = lax.scan(
        wrapper,
        (state, untraced_init),
        xs=None,
        length=num_steps,
    )
  else:
    trace_arrays = map_tree(
        lambda spec: np.zeros((num_steps,) + spec.shape, spec.dtype),
        traced_spec)

    def wrapper(i, state_untraced_traced):
      state, _, trace_arrays = state_untraced_traced
      state, untraced, traced = fn(state)
      trace_arrays = map_tree(lambda a, e: jax.ops.index_update(a, i, e),
                              trace_arrays, traced)
      return (state, untraced, trace_arrays)
    state, untraced, traced = lax.fori_loop(
        np.asarray(0, num_steps.dtype),
        num_steps,
        wrapper,
        (state, untraced_init, trace_arrays),
    )
  return state, untraced, traced
