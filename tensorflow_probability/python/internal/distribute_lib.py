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

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.math import gradient as math_gradient

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
  if JAX_MODE:
    axis_name = canonicalize_axis_name(axis_name)
    for name in axis_name:
      x = lax.psum(x, name)
    return x
  ctx = tf.distribute.get_replica_context()
  return ctx.all_reduce('sum', x)


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


class _DummyGrads(object):
  """Wraps gradients to preserve structure when computing a custom gradient."""

  def __init__(self, grads):
    self.grads = grads

  def tree_flatten(self):
    return (self.grads,), ()

  @classmethod
  def tree_unflatten(cls, _, xs):
    return cls(*xs)

  def __repr__(self):
    return f'_DummyGrads({self.grads})'


if JAX_MODE:
  from jax import tree_util  # pylint: disable=g-import-not-at-top
  tree_util.register_pytree_node_class(_DummyGrads)


def make_psum_function(fn, in_axes, out_axes):
  """Constructs a function that psums over outputs and corrects input gradients.

  Given a function `fn`, this `make_psum_function` returns a new one that
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

  Returns:
    A new function that applies psums on to the output of the original
    function and corrects the gradient with respect to its inputs.
  """

  if not isinstance(in_axes, tuple):
    in_axes = (in_axes,)

  def _psum_fn_fwd(*args):
    nest.assert_shallow_structure(args, in_axes)
    out_parts = fn(*args)
    nest.assert_shallow_structure(out_parts, out_axes)
    map_out_axes = nest.map_structure_up_to(out_parts, canonicalize_axis_name,
                                            out_axes)

    total_out_parts = nest.map_structure_up_to(
        out_parts,
        lambda out_part, axis_name: (  # pylint: disable=g-long-lambda
            psum(out_part, axis_name=axis_name) if axis_name else out_part),
        out_parts,
        map_out_axes)

    return total_out_parts, (args, out_parts)

  def _psum_fn_bwd(args_and_out_parts, gs):
    args, out_parts = args_and_out_parts
    map_in_axes = nest.map_structure_up_to(args, canonicalize_axis_name,
                                           in_axes)
    map_out_axes = nest.map_structure_up_to(out_parts, canonicalize_axis_name,
                                            out_axes)

    def flat_fn(flat_args):
      unflat_args = tf.nest.pack_sequence_as(args, flat_args)
      out_parts = fn(*unflat_args)
      return tf.nest.flatten(out_parts)

    # Operate with flattened lists, to make it easier to tease-out individual
    # outputs for the local grads.
    flat_value = tf.nest.flatten(args)
    flat_gs = tf.nest.flatten(gs)
    local_grads = [
        math_gradient.value_and_gradient(  # pylint: disable=g-complex-comprehension
            lambda *val: flat_fn(val)[out_idx],  # pylint: disable=cell-var-from-loop
            flat_value,
            output_gradients=out_g)[1] for out_idx, out_g in enumerate(flat_gs)
    ]
    # Transpose.
    local_grads = list(zip(*local_grads))
    # Repack.
    local_grads = tf.nest.pack_sequence_as(args, [
        _DummyGrads(tf.nest.pack_sequence_as(out_parts, v)) for v in local_grads
    ])

    def value_grad(v, in_axis_names, term_grads):
      """Computes reductions of output gradients.

      A `log_prob_parts` function takes in a list of values and outputs
      a log density for each input to the function. The vector-Jacobian
      product (VJP) of a `log_prob_parts` function thus needs to compute the
      gradient of each output term w.r.t. each input value. This function
      overrides the default VJP of an output term `j` w.r.t to an input
      value `i` to include an all-reduce-sum when:
      1) The gradient of `j` w.r.t. `i` is connected.
      2) `j` is a sharded term and `i` is an unsharded value.

      If these conditions do not hold, the gradient remains the same and
      either corresponds to:
      1) The gradient of a sharded term w.r.t to a sharded value
      2) The gradient of an unsharded term w.r.t. to an unsharded value.
      3) The gradient of an unsharded term w.r.t. to an sharded value.
      In any of these cases, no all-reduce-sum is necessary.
      Args:
        v: The output term of a `log_prob_part` function.
        in_axis_names: A list of axis names indicating whether or not the output
          term is sharded or not, `None` if no sharding.
        term_grads: The gradient of the output term w.r.t. to each of the input
          values to the `log_prob_part` function.

      Returns:
        The vector Jacobian product of `v` w.r.t. the input parts of the
        `log_prob_parts` function.
      """
      term_grads = term_grads.grads

      def psum_grads(term_grad, out_axis_names):
        if term_grad is not None:
          psum_axes = [
              axis_name for axis_name in out_axis_names
              if axis_name not in in_axis_names
          ]
          if psum_axes:
            term_grad = psum(term_grad, axis_name=psum_axes)
        return term_grad

      total_grad = nest.map_structure_up_to(term_grads, psum_grads, term_grads,
                                            map_out_axes)
      if all([grad is None for grad in tf.nest.flatten(total_grad)]):
        return None
      return tf.add_n([
          v for v in tf.nest.flatten(total_grad)
          if tfp_custom_gradient.is_valid_gradient(v)
      ])

    out = nest.map_structure_up_to(args, value_grad, args, map_in_axes,
                                   local_grads)
    return out

  @tfp_custom_gradient.custom_gradient(
      vjp_fwd=_psum_fn_fwd, vjp_bwd=_psum_fn_bwd)
  def psum_fn(*args):
    return _psum_fn_fwd(*args)[0]

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
  return make_psum_function(log_prob_parts_fn, (axis_names,), axis_names)
