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
"""Quadratic approximation for `JointDistributions`.

This is supporting code for Statistical Rethinking, and provides light wrappers
around existing TensorFlow Probability functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

__all__ = ["quap"]


def merge_flat_args(args, defaults):
  idx = 0
  new_args = []
  for val in defaults:
    if val is None:
      new_args.append(args[idx])
      idx += 1
    else:
      new_args.append(val)
  return new_args


def quap(joint_dist, data=None, max_tries=20, initial_position=None, name=None):
  """Compute a quadratic approximation to a ``JointDistributionNamed``.


  Traverses a JointDistribution*, uses bfgs to minimize the negative
  log probability and estimate the hessian, and returns a JointDistribution of
  the same type,  whose distributions are all Gaussians, and covariances are
  set appropriately.

  Args:
    joint_dist: A `JointDistributionNamed` or `JointDistributionSequential`
      model. Also works with auto batched versions of the same.
    data: Optional `dict` of data to condition the joint_dist with. The return
      value will be conditioned o this data. If this is `None`, the return
      value will be a quadratic approximation to the distribution itself.
    max_tries: Optional `int` of number of times to run the optimizer internally
      before raising a `RuntimeError`. Default is 20.
    initial_position: Optional `dict` to initialize the optimizer. Keys should
      correspond to names in the JointDistribution. Defaults to random draws
      from `joint_dist`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., 'quap').

  Returns:
    `JointDistributionNamed` which is a quadratic approximation to the input
    `joint_dist`, conditioned on `data`.

  Raises:
    RuntimeError: In case the optimizer does not converge within `max_tries`.
  """
  with tf.name_scope(name or "quap"):
    max_tries = tf.convert_to_tensor(max_tries)
    structure = joint_dist.sample()

    # A dictionary is the only structure that does not already
    # have None's as placeholders
    if isinstance(data, dict):
      data = {k: data.get(k) for k in structure}

    if data is None:
      data = tf.nest.map_structure(lambda j: None, structure)

    data = tf.nest.map_structure(lambda j: None if j is None else j, data)
    flat_data = tf.nest.flatten(data)

    def try_optimize(idx, opt):  # pylint: disable=unused-argument
      locs = tf.nest.flatten(joint_dist.sample(value=initial_position))
      locs = [j for idx, j in enumerate(locs) if flat_data[idx] is None]
      def neg_logp_and_grad(vals):
        def neg_logp(vals):
          args = merge_flat_args(vals, flat_data)
          return -joint_dist.log_prob(tf.nest.pack_sequence_as(
              structure, tf.unstack(args)))
        return tfp.math.value_and_gradient(neg_logp, vals)
      return idx + 1, tfp.optimizer.bfgs_minimize(neg_logp_and_grad, locs)

    def should_stop(idx, opt):
      return (idx < max_tries) & ~opt.converged

    idx = tf.constant(0, dtype=max_tries.dtype)
    idx, opt = try_optimize(idx, None)
    _, opt = tf.while_loop(should_stop, try_optimize, [idx, opt])

    with tf.control_dependencies([tf.debugging.Assert(
        condition=opt.converged, data=opt)]):
      dists = {}
      stddevs = tf.sqrt(tf.linalg.diag_part(opt.inverse_hessian_estimate))

      gaussians = tf.nest.map_structure(
          tfd.Normal,
          tf.unstack(opt.position),
          tf.unstack(stddevs))
      dists = merge_flat_args(gaussians, flat_data)
      dists = [v if isinstance(v, tfd.Distribution) else
               tfd.Deterministic(v) for v in dists]

      approx = joint_dist.__class__(
          tf.nest.pack_sequence_as(structure, dists), name=name)
      return approx
