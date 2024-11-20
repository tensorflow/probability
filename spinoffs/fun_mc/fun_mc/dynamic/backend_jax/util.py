# Copyright 2021 The TensorFlow Probability Authors.
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
"""FunMC utilities implemented via JAX."""

import dataclasses
import functools
from typing import TypeVar, dataclass_transform

import jax
from jax import lax
from jax import random
from jax import tree_util
import jax.numpy as jnp
import jaxtyping
import numpy as np

__all__ = [
    'Array',
    'assert_same_shallow_tree',
    'block_until_ready',
    'convert_to_tensor',
    'dataclass',
    'diff',
    'DType',
    'flatten_tree',
    'get_shallow_tree',
    'get_static_value',
    'inverse_fn',
    'make_tensor_seed',
    'map_tree',
    'map_tree_up_to',
    'move_axis',
    'named_call',
    'new_dynamic_array',
    'random_categorical',
    'random_integer',
    'random_normal',
    'random_permutation',
    'random_uniform',
    'repeat',
    'Seed',
    'split_seed',
    'stack_dynamic_array',
    'trace',
    'value_and_grad',
    'value_and_ldj',
    'write_to_dynamic_array',
]


Array = jaxtyping.Array
DType = jax.typing.DTypeLike
Seed = jaxtyping.PRNGKeyArray


def map_tree(fn, tree, *args):
  """Maps `fn` over the leaves of a nested structure."""
  return tree_util.tree_map(fn, tree, *args)


def flatten_tree(tree):
  """Flattens a nested structure to a list."""
  return tree_util.tree_flatten(tree)[0]


def unflatten_tree(tree, xs):
  """Inverse operation of `flatten_tree`."""
  return tree_util.tree_unflatten(tree_util.tree_structure(tree), xs)


def map_tree_up_to(shallow, fn, tree, *rest):
  """`map_tree` with recursion depth defined by depth of `shallow`."""

  def wrapper(x, *rest):
    return None if x is None else fn(*rest)

  return tree_util.tree_map(
      wrapper, shallow, tree, *rest, is_leaf=lambda x: x is None
  )


def get_shallow_tree(is_leaf, tree):
  """Returns a shallow tree, expanding only when is_leaf(subtree) is False."""
  return tree_util.tree_map(is_leaf, tree, is_leaf=is_leaf)


def assert_same_shallow_tree(shallow, tree):
  """Asserts that `tree` has the same shallow structure as `shallow`."""
  # Do a dummy map for the side-effect of verifying that the structures are
  # the same. This doesn't catch all the errors we actually care about, sadly.
  map_tree_up_to(shallow, lambda *args: (), tree)


def value_and_grad(fn, args):
  """Given `fn: (args) -> out, extra`, returns `dout/dargs`."""
  output, vjp_fn, extra = jax.vjp(fn, args, has_aux=True)
  grad = vjp_fn(jnp.ones_like(output))[0]
  return output, extra, grad


def make_tensor_seed(seed):
  """Converts a seed to a `Tensor` seed."""
  if seed is None:
    raise ValueError('seed must not be None when using JAX')
  if hasattr(seed, 'dtype') and jax.dtypes.issubdtype(
      seed.dtype, jax.dtypes.prng_key
  ):
    return jnp.asarray(seed)
  else:
    return jnp.asarray(seed, jnp.uint32)


def split_seed(seed, count):
  """Splits a seed into `count` seeds."""
  return random.split(make_tensor_seed(seed), count)


def random_uniform(shape, dtype, seed):
  """Generates a sample from uniform distribution over [0., 1)."""
  return random.uniform(
      shape=tuple(shape), dtype=dtype, key=make_tensor_seed(seed)
  )


def random_integer(shape, dtype, minval, maxval, seed):
  """Generates a sample from uniform distribution over [minval, maxval)."""
  return random.randint(
      shape=tuple(shape),
      dtype=dtype,
      minval=minval,
      maxval=maxval,
      key=make_tensor_seed(seed),
  )


def random_normal(shape, dtype, seed):
  """Generates a sample from a standard normal distribution."""
  return random.normal(
      shape=tuple(shape), dtype=dtype, key=make_tensor_seed(seed)
  )


def _searchsorted(a, v):
  """Returns where `v` can be inserted so that `a` remains sorted."""

  def cond(state):
    low_idx, high_idx = state
    return low_idx < high_idx

  def body(state):
    low_idx, high_idx = state
    mid_idx = (low_idx + high_idx) // 2
    mid_v = a[mid_idx]
    low_idx = jnp.where(v > mid_v, mid_idx + 1, low_idx)
    high_idx = jnp.where(v > mid_v, high_idx, mid_idx)
    return low_idx, high_idx

  low_idx, _ = lax.while_loop(cond, body, (0, a.shape[-1]))
  return low_idx


def random_categorical(logits, num_samples, seed):
  """Returns a sample from a categorical distribution. `logits` must be 2D."""
  probs = jax.nn.softmax(logits)
  cum_sum = jnp.cumsum(probs, axis=-1)

  eta = random.uniform(
      make_tensor_seed(seed), (num_samples,) + cum_sum.shape[:-1]
  )
  cum_sum = jnp.broadcast_to(cum_sum, (num_samples,) + cum_sum.shape)

  flat_cum_sum = cum_sum.reshape([-1, cum_sum.shape[-1]])
  flat_eta = eta.reshape([-1])
  return jax.vmap(_searchsorted)(flat_cum_sum, flat_eta).reshape(eta.shape).T


def random_permutation(value, seed):
  """Randomly permutes the array."""
  return random.permutation(seed, value)


def trace(state, fn, num_steps, unroll, max_steps, **_):
  """Implementation of `trace` operator, without the calling convention."""
  # We need the shapes and dtypes of the outputs of `fn`.
  _, untraced_spec, traced_spec, stop_spec = jax.eval_shape(
      fn, map_tree(lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype), state)
  )
  if isinstance(stop_spec, tuple):
    stop = ()
  else:
    stop = False
  untraced_init, traced_init = map_tree(
      lambda spec: jnp.zeros(spec.shape, spec.dtype),
      (untraced_spec, traced_spec),
  )

  try:
    num_steps = int(num_steps)
    use_scan = True
  except TypeError:
    use_scan = False
    if max_steps is None:
      if flatten_tree(traced_spec):
        raise ValueError(  # pylint: disable=raise-missing-from
            'Cannot trace values when `num_steps` is not statically known and '
            '`max_steps` is not specified. Pass `False` to `trace_mask` or '
            'return an empty structure (e.g. `()`) as '
            'the extra output.'
        )
      if unroll:
        raise ValueError(  # pylint: disable=raise-missing-from
            'Cannot unroll when `num_steps` is not statically known and '
            '`max_steps` is not specified.'
        )
  if max_steps is not None or not isinstance(stop_spec, tuple):
    use_scan = False

  if unroll:
    num_outputs = num_steps if max_steps is None else max_steps

    traced_lists = map_tree(lambda _: [], traced_spec)
    untraced = untraced_init
    for step in range(num_outputs):
      if step < num_steps and not stop:
        state, untraced, traced_element, stop = fn(state)
      else:
        traced_element = traced_init
      map_tree_up_to(
          traced_spec, lambda l, e: l.append(e), traced_lists, traced_element
      )
    # Using asarray instead of stack to handle empty arrays correctly.
    traced = map_tree_up_to(
        traced_spec,
        lambda l, s: jnp.asarray(l, dtype=s.dtype),
        traced_lists,
        traced_spec,
    )
  elif use_scan:

    def wrapper(state_untraced, _):
      state, _ = state_untraced
      state, untraced, traced, _ = fn(state)
      return (state, untraced), traced

    (state, untraced), traced = lax.scan(
        wrapper,
        (state, untraced_init),
        xs=None,
        length=num_steps,
    )
  else:
    num_outputs = num_steps if max_steps is None else max_steps
    num_steps = (
        num_steps if max_steps is None else jnp.minimum(num_steps, max_steps)
    )

    trace_arrays = map_tree(
        lambda spec: jnp.zeros((num_outputs,) + spec.shape, spec.dtype),
        traced_spec,
    )
    loop_vars = (
        jnp.zeros_like(num_steps),
        stop,
        state,
        untraced_init,
        trace_arrays,
    )

    def cond(loop_vars):
      i, stop, *_ = loop_vars
      return (i < num_steps) & (isinstance(stop, tuple) or ~stop)

    def body(loop_vars):
      i, _, state, _, trace_arrays = loop_vars
      state, untraced, traced, stop = fn(state)
      trace_arrays = map_tree(lambda a, e: a.at[i].set(e), trace_arrays, traced)

      return i + 1, stop, state, untraced, trace_arrays

    _, _, state, untraced, traced = lax.while_loop(
        cond,
        body,
        loop_vars,
    )
  return state, untraced, traced


# TODO(siege): This is WIP, probably to be replaced by JAX's budding inverse
# function support.
def value_and_ldj(fn, args):
  """Compute the value and log-det jacobian of function evaluated at args.

  This assumes that `fn`'s `extra` output is a 2-tuple, where the first element
  is arbitrary and the the last element is the log determinant of the jacobian
  of the transformation.

  Args:
    fn: Function to evaluate.
    args: Arguments to `fn`.

  Returns:
    ret: First output of `fn`.
    extra: Second output of `fn`.
    ldj: Log-det jacobian of `fn`.

  #### Example

  ```python
  def scale_by_two(x):
    # Return x unchanged as the extra output for illustrative purposes.
    return 2 * x, (x, jnp.log(2))

  y, y_extra, y_ldj = value_and_ldj(scale_by_2, 3.)
  assert y == 6
  assert y_extra == 3
  assert y_ldj == jnp.log(2)
  ```
  """
  value, (extra, ldj) = fn(args)
  return value, (extra, ldj), ldj


def inverse_fn(fn):
  """Compute the inverse of a function.

  This assumes that `fn` has a field called `inverse` which contains the inverse
  of the function.

  Args:
    fn: Function to invert.

  Returns:
    inverse: Inverse of `fn`.

  #### Example

  ```python
  def scale_by_two(x):
    # Return x unchanged as the extra output for illustrative purposes.
    return 2 * x, (x, jnp.log(2))

  def scale_by_half(x):
    return x / 2, (x, -jnp.log(2))

  scale_by_two.inverse = scale_by_half
  scale_by_half.inverse = scale_by_two

  y, y_extra, y_ldj = value_and_ldj(scale_by_2, 3.)
  assert y == 6
  assert y_extra == 3
  assert y_ldj == jnp.log(2)

  inv_scale_by_2 = inverse_fn(scale_by_2)
  assert inv_scale_by_2 == scale_by_half

  x, x_extra, x_ldj = value_and_ldj(inv_scale_by_2, 4.)
  assert x == 2
  assert x_extra == 4
  assert x_ldj == -jnp.log(2)
  ```
  """
  return fn.inverse


def block_until_ready(tensors):
  """Blocks computation until it is ready.

  Args:
    tensors: A nest of Tensors.

  Returns:
    tensors: Tensors that are are guaranteed to be ready to materialize.
  """

  def _block_until_ready(tensor):
    if hasattr(tensor, 'block_until_ready'):
      return tensor.block_until_ready()
    else:
      return tensor

  return map_tree(_block_until_ready, tensors)


def move_axis(x, source, dest):
  """Move axis from source to dest."""
  return jnp.moveaxis(x, source, dest)


def named_call(f=None, name=None):
  """Adds a name to a function for profiling purposes."""
  if f is None:
    return functools.partial(named_call, name=name)

  return jax.named_call(f, name=name)


def diff(x, prepend=None):
  """Like jnp.diff."""
  return jnp.diff(x, prepend=prepend)


def repeat(x, repeats, total_repeat_length):
  """Like jnp.repeat."""
  return jnp.repeat(x, repeats, total_repeat_length=total_repeat_length)


def new_dynamic_array(shape, dtype, size):
  """Creates a new dynamic array."""
  return jnp.zeros((size,) + tuple(shape), dtype)


def write_to_dynamic_array(array, index, element):
  """Writes to the dynamic array."""
  return array.at[index].set(element)


def stack_dynamic_array(array):
  """Stacks the dynamic array."""
  return jnp.asarray(array)


def eval_shape(fn, *args):
  """Evaluates the shape/dtypes of fn statically."""
  return jax.eval_shape(fn, *args)


def convert_to_tensor(x):
  """A looser convert_to_tensor."""
  if x is None:
    return x
  return jnp.asarray(x)


T = TypeVar('T')


@dataclass_transform()
def dataclass(cls: T) -> T:
  """Create a tree-compatible dataclass."""
  cls = dataclasses.dataclass(frozen=True)(cls)
  fields = [f.name for f in dataclasses.fields(cls)]
  jax.tree_util.register_dataclass(cls, fields, [])

  def replace(self, **updates):
    """Returns a new object replacing the specified fields with new values."""
    return dataclasses.replace(self, **updates)

  cls.replace = replace

  return cls


def get_static_value(x):
  """Returns the static value of x, or None if x is dynamic."""
  try:
    return np.array(x)
  except TypeError:
    return None
