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
"""FunMCMC utilities implemented via TensorFlow."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import numpy as np
import six
import tensorflow.compat.v2 as tf
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'assert_same_shallow_tree',
    'flatten_tree',
    'inverse_fn',
    'make_tensor_seed',
    'map_tree',
    'map_tree_up_to',
    'random_categorical',
    'random_normal',
    'random_uniform',
    'split_seed',
    'trace',
    'value_and_ldj',
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
    return tf.random.uniform([2],
                             minval=iinfo.min,
                             maxval=iinfo.max,
                             dtype=tf.int32,
                             name='seed')
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
        logits=logits, num_samples=num_samples, seed=seed)
  else:
    return tf.random.stateless_categorical(
        logits=logits, num_samples=num_samples, seed=seed)


def _eval_shape(fn, input_spec):
  """Gets output `TensorSpec`s from `fn` given input `TensorSpec`."""
  raw_compiled_fn = tf.function(
      fn, autograph=False).get_concrete_function(input_spec)

  def compiled_fn(x):
    return raw_compiled_fn(*tf.nest.flatten(x))

  output_spec = tf.nest.map_structure(tf.TensorSpec,
                                      raw_compiled_fn.output_shapes,
                                      raw_compiled_fn.output_dtypes)
  return compiled_fn, output_spec


def trace(state, fn, num_steps, unroll, parallel_iterations=10):
  """TF implementation of `trace` operator, without the calling convention."""
  if tf.config.experimental_functions_run_eagerly() or tf.executing_eagerly():
    state, first_untraced, first_traced = fn(state)
    arrays = tf.nest.map_structure(
        lambda v: tf.TensorArray(  # pylint: disable=g-long-lambda
            v.dtype, size=num_steps, element_shape=v.shape).write(0, v),
        first_traced)
    start_idx = 1
  else:
    # We need the shapes and dtypes of the outputs of `fn` function to create
    # the `TensorArray`s etc., we can get it by pre-compiling the wrapper
    # function.
    input_spec = tf.nest.map_structure(tf.TensorSpec.from_tensor, state)
    fn, (_, untraced_spec, traced_spec) = _eval_shape(fn, input_spec)

    arrays = tf.nest.map_structure(
        lambda spec: tf.TensorArray(  # pylint: disable=g-long-lambda
            spec.dtype, size=num_steps, element_shape=spec.shape), traced_spec)
    first_untraced = tf.nest.map_structure(
        lambda spec: tf.zeros(spec.shape, spec.dtype), untraced_spec)
    start_idx = 0

  def body(i, state, _, arrays):
    state, untraced, traced = fn(state)
    arrays = tf.nest.map_structure(lambda a, e: a.write(i, e), arrays, traced)
    return i + 1, state, untraced, arrays

  def cond(i, *_):
    return i < num_steps

  static_num_steps = tf.get_static_value(num_steps)
  loop_vars = (start_idx, state, first_untraced, arrays)

  if unroll:
    if static_num_steps is None:
      raise ValueError(
          'Cannot unroll when `num_steps` is not statically known.')
    # TODO(siege): Investigate if using lists instead of TensorArray's is faster
    # (like is done in the JAX backend).
    for _ in range(start_idx, static_num_steps):
      loop_vars = body(*loop_vars)
    _, state, untraced, arrays = loop_vars
  else:
    if static_num_steps is None:
      maximum_iterations = None
    else:
      maximum_iterations = static_num_steps - start_idx
    _, state, untraced, arrays = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars,
        parallel_iterations=parallel_iterations,
        maximum_iterations=maximum_iterations,
    )

  traced = tf.nest.map_structure(lambda a: a.stack(), arrays)

  def _merge_static_length(x):
    x.set_shape(tf.TensorShape(static_num_steps).concatenate(x.shape[1:]))
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
