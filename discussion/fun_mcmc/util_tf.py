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
import tensorflow.compat.v2 as tf
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

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


def random_categorical(logits, num_samples, seed):
  """Returns a sample from a categorical distribution. `logits` must be 2D."""
  # TODO(siege): Switch to stateless RNG ops.
  return tf.random.categorical(
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


def trace(state, fn, num_steps, parallel_iterations=10):
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

  static_length = tf.get_static_value(num_steps)

  _, state, untraced, arrays = tf.while_loop(
      cond=cond,
      body=body,
      loop_vars=(start_idx, state, first_untraced, arrays),
      parallel_iterations=parallel_iterations,
      maximum_iterations=static_length,
  )

  traced = tf.nest.map_structure(lambda a: a.stack(), arrays)

  def _merge_static_length(x):
    x.set_shape(tf.TensorShape(static_length).concatenate(x.shape[1:]))
    return x

  traced = tf.nest.map_structure(_merge_static_length, traced)

  return state, untraced, traced
