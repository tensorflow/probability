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

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.math import gradient as math_gradient


JAX_MODE = False

if JAX_MODE:
  from jax import lax  # pylint: disable=g-import-not-at-top


def psum(x, axis_name=None):
  if JAX_MODE:
    return lax.psum(x, axis_name)
  ctx = tf.distribute.get_replica_context()
  return ctx.all_reduce('sum', x)


def pmean(x, axis_name=None):
  if JAX_MODE:
    return lax.pmean(x, axis_name)
  ctx = tf.distribute.get_replica_context()
  return ctx.all_reduce('mean', x)


def get_replica_id(axis_name=None):
  if JAX_MODE:
    return lax.axis_index(axis_name)
  ctx = tf.distribute.get_replica_context()
  return ctx.replica_id_in_sync_group


def get_num_replicas(axis_name=None):
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


def make_sharded_log_prob_parts(log_prob_parts_fn, is_sharded, axis_name=None):
  """Constructs a log prob parts function that all-reduces over terms.

  Given a log_prob_parts function, this function will return a new one that
  includes all-reduce sums over terms according to the `is_sharded` property. It
  will also add all-reduce sums for the gradient of sharded terms w.r.t.
  unsharded terms.

  Args:
    log_prob_parts_fn: a callable that takes in a structured value and returns a
      structure of log densities for each of the terms, that when summed returns
      a locally correct log-density.
    is_sharded: a structure of boolean values that matches the input and output
      of `log_prob_parts_fn`. If a value in `log_prob_parts_fn` has a
      corresponding `is_sharded` value set to `True`, the returned function will
      add an all-reduce sum for its term in the log prob calculation. If it is
      `False`, the returned function will have an all-reduce sum over the
      gradient of sharded terms w.r.t. to the unsharded value.
    axis_name: a `str` used for the axis name in the JAX backend. Unused in the
      TensorFlow backend.

  Returns:
    A new log prob parts function that can be run inside of strategy.
  """

  def _sharded_log_prob_parts_fwd(value):
    tf.nest.assert_same_structure(value, is_sharded)
    log_prob_parts = log_prob_parts_fn(value)
    tf.nest.assert_same_structure(log_prob_parts, is_sharded)

    total_log_prob_parts = tf.nest.map_structure(
        lambda log_prob_part, sharded: (  # pylint: disable=g-long-lambda
            psum(log_prob_part, axis_name=axis_name)
            if sharded else log_prob_part),
        log_prob_parts,
        is_sharded)

    return total_log_prob_parts, value

  def _sharded_log_prob_parts_bwd(value, gs):

    def flat_log_prob_parts_fn(flat_args):
      args = tf.nest.pack_sequence_as(is_sharded, flat_args)
      log_prob_parts = log_prob_parts_fn(args)
      return tf.nest.flatten(log_prob_parts)

    # Operate with flattened lists, to make it easier to tease-out individual
    # outputs for the local grads.
    flat_value = tf.nest.flatten(value)
    flat_gs = tf.nest.flatten(gs)
    local_grads = [
        math_gradient.value_and_gradient(  # pylint: disable=g-complex-comprehension
            lambda *val: flat_log_prob_parts_fn(val)[out_idx],  # pylint: disable=cell-var-from-loop
            flat_value,
            output_gradients=flat_gs[out_idx])[1]
        for out_idx, value_part in enumerate(flat_value)
    ]
    # Transpose.
    local_grads = list(zip(*local_grads))
    # Repack.
    local_grads = tf.nest.pack_sequence_as(is_sharded, [
        _DummyGrads(tf.nest.pack_sequence_as(is_sharded, v))
        for v in local_grads
    ])

    def value_grad(v, value_sharded, term_grads):
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
        value_sharded: A boolean indicating whether or not the output term is
          sharded or not.
        term_grads: The gradient of the output term w.r.t. to each of the input
          values to the `log_prob_part` function.

      Returns:
        The vector Jacobian product of `v` w.r.t. the input parts of the
        `log_prob_parts` function.
      """
      term_grads = term_grads.grads
      def psum_grads(term_grad, term_sharded):
        if term_grad is not None:
          if not value_sharded and term_sharded:
            term_grad = psum(term_grad, axis_name=axis_name)
        return term_grad

      total_grad = tf.nest.map_structure(psum_grads, term_grads,
                                         is_sharded)
      if all([grad is None for grad in tf.nest.flatten(total_grad)]):
        return None
      return tf.add_n(
          [v for v in tf.nest.flatten(total_grad) if v is not None])

    out = tf.nest.map_structure(value_grad, value, is_sharded, local_grads)
    return (out,)

  @tfp_custom_gradient.custom_gradient(
      vjp_fwd=_sharded_log_prob_parts_fwd, vjp_bwd=_sharded_log_prob_parts_bwd)
  def sharded_log_prob_parts(value):
    return _sharded_log_prob_parts_fwd(value)[0]

  return sharded_log_prob_parts
