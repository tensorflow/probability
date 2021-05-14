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
"""Utilities for writing distributed log prob functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import samplers

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

JAX_MODE = False

if JAX_MODE:
  from jax import lax  # pylint: disable=g-import-not-at-top


def canonicalize_axis_name(axis_name):
  """Converts an input into a list of axis strings."""
  if not axis_name:
    return []
  if (isinstance(axis_name, str) or
      not isinstance(axis_name, collections.Iterable)):
    return [axis_name]
  return list(axis_name)


def psum(x, axis_name=None):
  axis_name = canonicalize_axis_name(axis_name)
  for name in axis_name:
    x = rwb_psum(x, name)
  return x


def pbroadcast(x, axis_name=None):
  axis_name = canonicalize_axis_name(axis_name)
  for name in axis_name:
    x = rwb_pbroadcast(x, name)
  return x


def pmean(x, axis_name=None):
  if JAX_MODE:
    axis_name = canonicalize_axis_name(axis_name)
    for name in axis_name:
      x = lax.pmean(x, name)
    return x
  ctx = tf.distribute.get_replica_context()
  return ctx.all_reduce('mean', x)


def get_axis_index(axis_name=None):
  if JAX_MODE:
    return lax.axis_index(axis_name)
  ctx = tf.distribute.get_replica_context()
  return ctx.replica_id_in_sync_group


def get_axis_size(axis_name=None):
  if JAX_MODE:
    return lax.psum(1, axis_name)
  ctx = tf.distribute.get_replica_context()
  return ctx.num_replicas_in_sync


def _rwb_psum_fwd(x, axis_name):
  if JAX_MODE:
    axis_name = canonicalize_axis_name(axis_name)
    out = lax.psum(x, axis_name)
  else:
    ctx = tf.distribute.get_replica_context()
    out = ctx.all_reduce('sum', x)
  return out, None


def _rwb_psum_bwd(axis_name, _, cts):
  return (rwb_pbroadcast(cts, axis_name),)


def fold_in_axis_index(seed, axis_name=None):
  """Folds the active axis index into a seed according to its axis name."""
  if axis_name is None:
    return seed
  nest.assert_shallow_structure(seed, axis_name)
  axis_names = nest.map_structure_up_to(
      seed, canonicalize_axis_name, axis_name)
  def fold_in(seed, axes):
    for name in axes:
      axis_index = get_axis_index(name)
      seed = samplers.fold_in(seed, tf.cast(axis_index, tf.int32))
    return seed
  return nest.map_structure_up_to(seed, fold_in, seed, axis_names)


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_rwb_psum_fwd, vjp_bwd=_rwb_psum_bwd, nondiff_argnums=(1,))
def rwb_psum(x, axis_name):
  """Applies a psum with reduce-without-broadcast (RWB) semantics.

  RWB semantics allow taking gradients w.r.t. unmapped variables of functions
  with psums in them.

  Args:
    x: a `Tensor` target for the psum.
    axis_name: A string axis name for the psum.
  Returns:
    A `Tensor` that is the result of applying a psum to an input `Tensor`.
  """
  return _rwb_psum_fwd(x, axis_name)[0]


def _rwb_pbroadcast_fwd(x, _):
  return x, None


def _rwb_pbroadcast_bwd(axis_name, _, cts):
  return (rwb_psum(cts, axis_name),)


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=_rwb_pbroadcast_fwd,
    vjp_bwd=_rwb_pbroadcast_bwd,
    nondiff_argnums=(1,))
def rwb_pbroadcast(x, axis_name):
  """Applies a pbroadcast with reduce-without-broadcast (RWB) semantics."""
  return _rwb_pbroadcast_fwd(x, axis_name)[0]


def make_pbroadcast_function(fn, in_axes, out_axes, out_dtype):
  """Constructs a function that broadcasts inputs over named axes.

  Given a function `fn`, `make_pbroadcast_function` returns a new one that
  applies `pbroadcast` to input terms according to axis names provided in
  `in_axes` and `out_axes`. For each output axis in each term out the output of
  `fn`, inputs that do not have the output axes present are pbroadcasted before
  that term is computed.

  Args:
    fn: a callable to be transformed to have proadcasts at its inputs.
    in_axes: A structure of axis names that should match the structure of the
      input to `fn`. If the set of input axes for an input value does not match
      the output axes of a particular output value, the gradient of that output
      value w.r.t. the input value will be psum-ed over the axes present in the
      output but not the input.
    out_axes: A structure of axis names that should match the structure of the
      output of `fn`. The inputs to `fn` will be pbroadcast-ed before
      computing output terms according to their output axes.
    out_dtype: A structure of dtypes that matches the output of `fn`.

  Returns:
    A new function that applies pbroadcasts to the inputs of the original
    function.
  """

  if not isinstance(in_axes, tuple):
    in_axes = (in_axes,)

  def pbroadcast_fn(*args):
    nest.assert_shallow_structure(args, in_axes)
    nest.assert_shallow_structure(out_dtype, out_axes)
    map_in_axes = nest.map_structure_up_to(args, canonicalize_axis_name,
                                           in_axes)
    map_out_axes = nest.map_structure_up_to(out_dtype, canonicalize_axis_name,
                                            out_axes)

    def _pbroadcast_input(out_axes, x, in_axes):
      psum_axes = [
          axis_name for axis_name in out_axes if axis_name not in in_axes
      ]
      return pbroadcast(x, psum_axes)

    def _flat_fn_index(i, *args):
      out = fn(*args)
      return tf.nest.flatten(out)[i]

    def _flat_fn(*args):
      outputs = []
      for i, out_axis in enumerate(nest.flatten_up_to(out_dtype, map_out_axes)):
        local_args = nest.map_structure_up_to(
            args, functools.partial(_pbroadcast_input, out_axis), args,
            map_in_axes)
        outputs.append(_flat_fn_index(i, *local_args))
      return tf.nest.pack_sequence_as(out_dtype, outputs)

    return _flat_fn(*args)

  return pbroadcast_fn


def make_psum_function(fn, in_axes, out_axes, out_dtype):
  """Constructs a function that broadcasts inputs over named axes.

  Given a function `fn`, `make_psum_function` returns a new one that
  includes psums over terms according to axis names provided in `out_axes`. It
  also adds psums for the vector-Jacobian product of the outputs of `fn` w.r.t.
  its inputs according to `in_axes` if there are axes in the outputs that are
  not present in an input.

  Args:
    fn: a callable to be transformed to have psums at its outputs and on the
      gradients to its inputs.
    in_axes: A structure of axis names that should match the structure of the
      input to `fn`. If the set of input axes for an input value does not match
      the output axes of a particular output value, the gradient of that output
      value w.r.t. the input value will be psum-ed over the axes present in the
      output but not the input.
    out_axes: A structure of axis names that should match the structure of the
      output of `fn`. The outputs of `fn` will be psum-med according to their
      respective output axes.
    out_dtype: A structure of dtypes that matches the output of `fn`.

  Returns:
    A new function that applies psums on to the output of the original
    function and corrects the gradient with respect to its inputs.
  """

  out_axes = nest.map_structure_up_to(out_dtype, canonicalize_axis_name,
                                      out_axes)

  def psum_fn(*args):
    out = make_pbroadcast_function(fn, in_axes, out_axes, out_dtype)(*args)

    def _psum_output(x, out_axis):
      return psum(x, out_axis)

    return nest.map_structure_up_to(out_dtype, _psum_output, out, out_axes)

  return psum_fn


def make_sharded_log_prob_parts(log_prob_parts_fn, axis_names):
  """Constructs a log prob parts function that all-reduces over terms.

  Given a log_prob_parts function, this function will return a new one that
  includes all-reduce sums over terms according to the `is_sharded` property. It
  will also add all-reduce sums for the gradient of sharded terms w.r.t.
  unsharded terms.

  Args:
    log_prob_parts_fn: a callable that takes in a structured value and returns a
      structure of log densities for each of the terms, that when summed returns
      a locally correct log-density.
    axis_names: a structure of values that matches the input and output of
      `log_prob_parts_fn`. Each value in `axis_names` is either `None, a string
      name of a mapped axis in the JAX backend or any non-`None` value in TF
      backend, or an iterable thereof corresponding to multiple sharding axes.
      If the `axis_name` is not `None`, the returned function will add
      all-reduce sum(s) for its term in the log prob calculation. If it is
      `None`, the returned function will have an all-reduce sum over the
      gradient of sharded terms w.r.t. to the unsharded value.

  Returns:
    A new log prob parts function that can be run inside of a strategy.
  """

  def sharded_log_prob_parts_fn(*args):
    out_dtype = tf.nest.map_structure(lambda x: x.dtype, args)
    if len(args) == 1:
      out_dtype = out_dtype[0]
    return make_psum_function(log_prob_parts_fn, (axis_names,), axis_names,
                              out_dtype)(*args)

  return sharded_log_prob_parts_fn
