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


def psum(x):
  ctx = tf.distribute.get_replica_context()
  return ctx.all_reduce('sum', x)


def pmean(x):
  ctx = tf.distribute.get_replica_context()
  return ctx.all_reduce('mean', x)


class _DummyGrads(object):

  def __init__(self, grads):
    self.grads = grads


def make_sharded_log_prob_parts(log_prob_parts_fn, is_sharded):
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

  Returns:
    A new log prob parts function that can be run inside of strategy.
  """

  @tf.custom_gradient
  def sharded_log_prob_parts(value):
    if not isinstance(value, (list, tuple)):
      raise NotImplementedError('Can only shard functions that output `list`s.'
                                ' or `tuple`s')
    tf.nest.assert_same_structure(value, is_sharded)
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(value)
      log_prob_parts = log_prob_parts_fn(value)
      tf.nest.assert_same_structure(log_prob_parts, is_sharded)

    total_log_prob_parts = tf.nest.map_structure(
        lambda log_prob_part, sharded: (  # pylint: disable=g-long-lambda
            psum(log_prob_part) if sharded else log_prob_part),
        log_prob_parts,
        is_sharded)

    def vjp(*gs):
      assert len(gs) == len(log_prob_parts)

      def local_grad(v, g):
        return _DummyGrads([
            tape.gradient(log_prob_part, v, output_gradients=g)
            for log_prob_part in log_prob_parts
        ])

      local_grads = tf.nest.map_structure(local_grad, value, list(gs))

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
            is sharded or not.
          term_grads: The gradient of the output term w.r.t. to each of the
            input values to the `log_prob_part` function.
        Returns:
          The vector Jacobian product of `v` w.r.t. the input parts of the
          `log_prob_parts` function.
        """
        term_grads = term_grads.grads
        total_grad = []
        for term_grad, term_sharded in zip(term_grads, is_sharded):
          if term_grad is not None:
            if not value_sharded and term_sharded:
              term_grad = psum(term_grad)
          total_grad.append(term_grad)
        if all([grad is None for grad in tf.nest.flatten(total_grad)]):
          return None
        return tf.add_n(
            [v for v in tf.nest.flatten(total_grad) if v is not None])

      return tf.nest.map_structure(value_grad, value, is_sharded, local_grads)

    return total_log_prob_parts, vjp

  return sharded_log_prob_parts
