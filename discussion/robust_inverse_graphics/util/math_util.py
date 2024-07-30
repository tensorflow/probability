# Copyright 2024 The TensorFlow Probability Authors.
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
"""Some math utilities."""

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

__all__ = [
    'transform_gradients',
    'sanitize_gradients',
    'clip_gradients',
    'is_finite',
]


T = TypeVar('T')


def is_finite(tree: Any) -> jax.Array:
  """Verifies that all the elements in `x` are finite."""
  leaves = jax.tree_util.tree_leaves(
      jax.tree.map(lambda x: jnp.isfinite(x).all(), tree)
  )
  if leaves:
    return jnp.stack(leaves).all()
  else:
    return jnp.array(True)


def transform_gradients(x: T, handler_fn: Callable[[T], T]) -> T:
  """Applies `handler_fn` to gradients flowing through `x`."""
  wrapper = jax.custom_vjp(lambda x: x)

  def fwd(x):
    return x, ()

  def bwd(_, g):
    return (handler_fn(g),)

  wrapper.defvjp(fwd, bwd)

  return wrapper(x)


def sanitize_gradients(x: T) -> T:
  """Zeroes all gradients flowing through `x` if any element is non-finite."""

  def sanitize_fn(x):
    finite = is_finite(x)
    return jax.tree.map(lambda x: jnp.where(finite, x, jnp.zeros_like(x)), x)

  return transform_gradients(x, sanitize_fn)


def clip_gradients(
    x: T,
    global_norm: jax.typing.ArrayLike = 1.0,
    eps: jax.typing.ArrayLike = 1e-20,
) -> T:
  """Clips the norm of gradients flowing through `x`."""

  def clip_fn(x):
    leaves = jax.tree.leaves(jax.tree.map(lambda x: jnp.square(x).sum(), x))
    norm = jnp.sqrt(eps + jnp.sum(jnp.stack(leaves)))
    new_norm = jnp.where(norm > global_norm, global_norm, norm)
    return jax.tree.map(lambda x: x * new_norm / norm, x)

  return transform_gradients(x, clip_fn)
