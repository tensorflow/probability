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
"""FunMC utilities implemented via TensorFlow."""

import dataclasses
import functools
from typing import TypeVar, dataclass_transform

import numpy as np
import six
import tensorflow.compat.v2 as tf

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

tnp = tf.experimental.numpy

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


Array = tf.Tensor
DType = tf.DType
Seed = tf.Tensor | int


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


def get_shallow_tree(is_leaf, tree):
  """Returns a shallow tree, expanding only when is_leaf(subtree) is False."""
  return nest.get_traverse_shallow_structure(lambda t: not is_leaf(t), tree)


def assert_same_shallow_tree(shallow, tree):
  """Asserts that `tree` has the same shallow structure as `shallow`."""
  nest.assert_shallow_structure(shallow, tree)


def value_and_grad(fn, args):
  """Given `fn: (args) -> out, extra`, returns `dout/dargs`."""
  with tf.GradientTape() as tape:
    args = map_tree(tf.convert_to_tensor, args)
    tape.watch(args)
    ret, extra = fn(args)
  grads = tape.gradient(ret, args)
  return ret, extra, grads


def _is_stateful_seed(seed):
  return seed is None or isinstance(seed, six.integer_types)


def make_tensor_seed(seed):
  """Converts a seed to a `Tensor` seed."""
  if _is_stateful_seed(seed):
    iinfo = np.iinfo(np.int32)
    return tf.random.uniform(
        [2], minval=iinfo.min, maxval=iinfo.max, dtype=tf.int32, name='seed'
    )
  else:
    return tf.convert_to_tensor(seed, dtype=tf.int32, name='seed')


def split_seed(seed, count):
  """Splits a seed into `count` seeds."""
  if _is_stateful_seed(seed):
    if seed is None:
      return count * [None]
    return [
        np.random.RandomState(seed + i).randint(0, 2**31)
        for i, seed in enumerate([seed] * count)
    ]
  else:
    seeds = tf.random.stateless_uniform(
        [count, 2],
        seed=make_tensor_seed(seed),
        minval=None,
        maxval=None,
        dtype=tf.int32,
    )
    return tf.unstack(seeds)


def random_uniform(shape, dtype, seed):
  """Generates a sample from uniform distribution over [0., 1)."""
  if _is_stateful_seed(seed):
    return tf.random.uniform(shape=shape, dtype=dtype, seed=seed)
  else:
    return tf.random.stateless_uniform(shape=shape, dtype=dtype, seed=seed)


def random_integer(shape, dtype, minval, maxval, seed):
  """Generates a sample from uniform distribution over [minval, maxval)."""
  if _is_stateful_seed(seed):
    return tf.random.uniform(
        shape=shape, dtype=dtype, minval=minval, maxval=maxval, seed=seed
    )
  else:
    return tf.random.stateless_uniform(
        shape=shape, dtype=dtype, minval=minval, maxval=maxval, seed=seed
    )


def random_normal(shape, dtype, seed):
  """Generates a sample from a standard normal distribution."""
  if _is_stateful_seed(seed):
    return tf.random.normal(shape=shape, dtype=dtype, seed=seed)
  else:
    return tf.random.stateless_normal(shape=shape, dtype=dtype, seed=seed)


def random_categorical(logits, num_samples, seed):
  """Returns a sample from a categorical distribution. `logits` must be 2D."""
  if _is_stateful_seed(seed):
    return tf.random.categorical(
        logits=logits, num_samples=num_samples, seed=seed
    )
  else:
    return tf.random.stateless_categorical(
        logits=logits, num_samples=num_samples, seed=seed
    )


def random_permutation(value, seed):
  """Randomly permutes the array."""
  if _is_stateful_seed(seed):
    return tf.random.shuffle(value, seed)
  else:
    return tf.random.experimental.stateless_shuffle(value, seed)


def _eval_shape(fn, input_spec):
  """Gets output `TensorSpec`s from `fn` given input `TensorSpec`."""
  raw_compiled_fn = tf.function(fn, autograph=False).get_concrete_function(
      input_spec
  )

  def compiled_fn(x):
    return raw_compiled_fn(*tf.nest.flatten(x))

  output_spec = tf.nest.map_structure(
      tf.TensorSpec,
      raw_compiled_fn.output_shapes,
      raw_compiled_fn.output_dtypes,
  )
  return compiled_fn, output_spec


def trace(state, fn, num_steps, unroll, max_steps, parallel_iterations=10):
  """TF implementation of `trace` operator, without the calling convention."""
  num_outputs = num_steps if max_steps is None else max_steps

  if tf.config.experimental_functions_run_eagerly() or tf.executing_eagerly():
    state, first_untraced, first_traced, stop = fn(state)
    arrays = tf.nest.map_structure(
        lambda v: tf.TensorArray(  # pylint: disable=g-long-lambda
            v.dtype, size=num_outputs, element_shape=v.shape
        ).write(0, v),
        first_traced,
    )
    start_idx = 1
  else:
    # We need the shapes and dtypes of the outputs of `fn` function to create
    # the `TensorArray`s etc., we can get it by pre-compiling the wrapper
    # function.
    input_spec = tf.nest.map_structure(tf.TensorSpec.from_tensor, state)
    fn, (_, untraced_spec, traced_spec, stop_spec) = _eval_shape(fn, input_spec)

    arrays = tf.nest.map_structure(
        lambda spec: tf.TensorArray(  # pylint: disable=g-long-lambda
            spec.dtype, size=num_outputs, element_shape=spec.shape
        ),
        traced_spec,
    )
    first_untraced = tf.nest.map_structure(
        lambda spec: tf.zeros(spec.shape, spec.dtype), untraced_spec
    )
    start_idx = 0
    if isinstance(stop_spec, tuple):
      stop = ()
    else:
      stop = False

  def body(i, stop, state, untraced, arrays):
    del stop, untraced
    state, untraced, traced, stop = fn(state)
    arrays = tf.nest.map_structure(lambda a, e: a.write(i, e), arrays, traced)
    return i + 1, stop, state, untraced, arrays

  def cond(i, stop, *_):
    return (i < num_steps) & (isinstance(stop, tuple) or ~stop)

  static_num_steps = tf.get_static_value(num_steps)
  static_num_outputs = tf.get_static_value(num_outputs)
  loop_vars = (start_idx, stop, state, first_untraced, arrays)

  if unroll:
    if static_num_steps is None:
      raise ValueError(
          'Cannot unroll when `num_steps` is not statically known or '
          '`max_steps` is None.'
      )
    static_num_iters = (
        static_num_steps
        if max_steps is None
        else min(static_num_steps, max_steps)
    )
    # TODO(siege): Investigate if using lists instead of TensorArray's is faster
    # (like is done in the JAX backend).
    for _ in range(start_idx, static_num_iters):
      if loop_vars[1]:
        break
      loop_vars = body(*loop_vars)
    _, _, state, untraced, arrays = loop_vars
  else:
    if static_num_steps is None:
      if max_steps is None:
        maximum_iterations = None
      else:
        maximum_iterations = max_steps - start_idx
    else:
      if max_steps is None:
        maximum_iterations = static_num_steps - start_idx
      else:
        maximum_iterations = min(static_num_steps, max_steps) - start_idx
    _, _, state, untraced, arrays = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars,
        parallel_iterations=parallel_iterations,
        maximum_iterations=maximum_iterations,
    )

  traced = tf.nest.map_structure(lambda a: a.stack(), arrays)

  def _merge_static_length(x):
    x.set_shape(tf.TensorShape(static_num_outputs).concatenate(x.shape[1:]))
    return x

  traced = tf.nest.map_structure(_merge_static_length, traced)

  return state, untraced, traced


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
    return 2 * x, (x, np.log(2))

  y, y_extra, y_ldj = value_and_ldj(scale_by_2, 3.)
  assert y == 6
  assert y_extra == 3
  assert y_ldj == np.log(2)
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
    return 2 * x, (x, np.log(2))

  def scale_by_half(x):
    return x / 2, (x, -np.log(2))

  scale_by_two.inverse = scale_by_half
  scale_by_half.inverse = scale_by_two

  y, y_extra, y_ldj = value_and_ldj(scale_by_2, 3.)
  assert y == 6
  assert y_extra == 3
  assert y_ldj == np.log(2)

  inv_scale_by_2 = inverse_fn(scale_by_2)
  assert inv_scale_by_2 == scale_by_half

  x, x_extra, x_ldj = value_and_ldj(inv_scale_by_2, 4.)
  assert x == 2
  assert x_extra == 4
  assert x_ldj == -np.log(2)
  ```
  """
  return fn.inverse


def block_until_ready(tensors):
  """Blocks computation until it is ready.

  Does nothing on the TensorFlow backend, as the computation is eagerly
  computed.

  Args:
    tensors: A nest of Tensors.

  Returns:
    tensors: Tensors that are are guaranteed to be ready to materialize.
  """
  return tensors


def move_axis(x, source, dest):
  """Move axis from source to dest."""
  return tf.convert_to_tensor(tnp.moveaxis(x, source, dest))


def named_call(f=None, name=None):
  """Adds a name to a function for profiling purposes."""
  if f is None:
    return functools.partial(named_call, name=name)

  if name is None:
    name = f.__name__

  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    with tf.name_scope(name):
      return f(*args, **kwargs)

  return wrapped


def diff(x, prepend=None):
  """Like jnp.diff."""
  if prepend is not None:
    x = tf.concat(
        [tf.convert_to_tensor(prepend, dtype=x.dtype)[tf.newaxis], x], 0
    )
  return x[1:] - x[:-1]


def repeat(x, repeats, total_repeat_length):
  """Like jnp.repeat."""
  # Implementation based on JAX, with some adjustments due to TF's stricted
  # indexing validation.
  exclusive_repeats = tf.concat([[0], repeats[:-1]], axis=0)
  scatter_indices = tf.cumsum(exclusive_repeats)
  scatter_indices = tf.where(
      scatter_indices < total_repeat_length,
      scatter_indices,
      total_repeat_length,
  )
  block_split_indicators = tf.zeros([total_repeat_length + 1], tf.int32)
  block_split_indicators = tf.tensor_scatter_nd_add(
      block_split_indicators,
      scatter_indices[..., tf.newaxis],
      tf.ones_like(scatter_indices),
  )
  gather_indices = tf.cumsum(block_split_indicators[:-1]) - 1
  return tf.gather(x, gather_indices)


def new_dynamic_array(shape, dtype, size):
  """Creates a new dynamic array."""
  return tf.TensorArray(dtype, size=size, element_shape=shape)


def write_to_dynamic_array(array, index, element):
  """Writes to the dynamic array."""
  return array.write(index, element)


def stack_dynamic_array(array):
  """Stacks the dynamic array."""
  return array.stack()


def eval_shape(fn, *args):
  """Evaluates the shape/dtypes of fn statically."""
  args = tf.nest.map_structure(tf.TensorSpec.from_tensor, args)
  _, shape = _eval_shape(lambda args: fn(*args), args)
  return shape


def convert_to_tensor(x):
  """A looser convert_to_tensor."""
  if x is None:
    return x
  if isinstance(x, tf.TensorArray):
    return x
  return tf.convert_to_tensor(x)


T = TypeVar('T')


@dataclass_transform()
def dataclass(cls: T) -> T:
  """Create a tree-compatible dataclass."""
  cls = dataclasses.dataclass(frozen=True)(cls)

  def __tf_flatten__(self):  # pylint: disable=invalid-name
    metadata = ()
    fields = dataclasses.fields(self)
    components = tuple(getattr(self, f.name) for f in fields)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, leaves):  # pylint: disable=invalid-name
    del metadata
    return cls(*leaves)

  def __len__(self):  # pylint: disable=invalid-name
    # This is to work around a bug in TF's tree-prefix matching.
    return len(dataclasses.fields(self))

  cls.__tf_flatten__ = __tf_flatten__
  cls.__tf_unflatten__ = __tf_unflatten__
  cls.__len__ = __len__

  def replace(self, **updates):
    """Returns a new object replacing the specified fields with new values."""
    return dataclasses.replace(self, **updates)

  cls.replace = replace

  return cls


def get_static_value(x):
  """Returns the static value of x, or None if x is dynamic."""
  return tf.get_static_value(x)
