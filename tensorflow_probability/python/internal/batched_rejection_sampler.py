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
"""Utilities for batched rejection sampling."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import samplers

__all__ = [
    'batched_las_vegas_algorithm',
    'batched_rejection_sampler',
]


def batched_las_vegas_algorithm(
    batched_las_vegas_trial_fn, seed=None, name=None):
  """Batched Las Vegas Algorithm.

  This utility encapsulates the notion of a 'batched las_vegas_algorithm'
  (BLVA): a batch of independent (but not necessarily identical) randomized
  computations, each of which will eventually terminate after an unknown number
  of trials [(Babai, 1979)][1]. The number of trials will in general vary
  across batch points.

  The computation is parameterized by a callable representing a single trial for
  the entire batch. The utility runs the callable repeatedly, keeping track of
  which batch points have succeeded, until all have succeeded.

  Because we keep running the callable repeatedly until we've generated at least
  one good value for every batch point, we may generate multiple good values for
  many batch point. In this case, the particular good batch point returned is
  deliberately left unspecified.

  Args:
    batched_las_vegas_trial_fn: A callable that takes a Python integer PRNG seed
      and returns two values. (1) A structure of Tensors containing the results
      of the computation, all with a shape broadcastable with (2) a boolean mask
      representing whether each batch point succeeded.
    seed: Python integer or `Tensor`, for seeding PRNG.
    name: A name to prepend to created ops.
      Default value: `'batched_las_vegas_algorithm'`.

  Returns:
    results, num_iters: A structure of Tensors representing the results of a
    successful computation for each batch point, and a scalar int32 tensor, the
    number of calls to `randomized_computation`.

  #### References

  [1]: Laszlo Babai. Monte-Carlo algorithms in graph isomorphism
       testing. Universite de Montreal, D.M.S. No. 79-10.
  """
  with tf.name_scope(name or 'batched_las_vegas_algorithm'):
    init_seed, loop_seed = samplers.split_seed(
        seed, salt='batched_las_vegas_algorithm')
    values, good_values_mask = batched_las_vegas_trial_fn(init_seed)
    num_iters = tf.constant(1)

    def cond(unused_values, good_values_mask, unused_num_iters, unused_seed):
      return tf.math.logical_not(tf.reduce_all(good_values_mask))

    def body(values, good_values_mask, num_iters, seed):
      """Batched Las Vegas Algorithm body."""

      trial_seed, new_seed = samplers.split_seed(seed)
      new_values, new_good_values_mask = batched_las_vegas_trial_fn(trial_seed)

      values = tf.nest.map_structure(
          lambda new, old: tf.where(new_good_values_mask, new, old),
          new_values, values)

      good_values_mask = tf.logical_or(good_values_mask, new_good_values_mask)

      return values, good_values_mask, num_iters + 1, new_seed

    (values, _, num_iters, _) = tf.while_loop(
        cond, body, (values, good_values_mask, num_iters, loop_seed))
    return values, num_iters


def batched_rejection_sampler(
    proposal_fn, target_fn, seed=None, dtype=tf.float32, name=None):
  """Generic batched rejection sampler.

  In each iteration, the sampler generates a batch of proposed samples and
  proposal heights by calling `proposal_fn`. For each such sample point S, a
  uniform random variate U is generated and the sample is accepted if U *
  height(S) <= target(S). The batched rejection sampler keeps track of which
  batch points have been accepted, and continues generating new batches until
  all batch points have been acceped.

  The values returned by `proposal_fn` should satisfy the desiderata of
  rejection sampling: proposed samples should be drawn independently from a
  valid distribution on some domain D that includes the domain of `target_fn`,
  and the proposal must upper bound the target: for all points S in D, height(S)
  >= target(S).

  Args:
    proposal_fn: A callable that takes a Python integer PRNG seed and returns a
      set of proposed samples and the value of the proposal at the samples.
    target_fn: A callable that takes a tensor of samples and returns the value
      of the target at the samples.
    seed: Python integer or `Tensor`, for seeding PRNG.
    dtype: The TensorFlow dtype used internally by `proposal_fn` and
      `target_fn`.  Default value: `tf.float32`.
    name: A name to prepend to created ops.
      Default value: `'batched_rejection_sampler'`.

  Returns:
    results, num_iters: A tensor of samples from the target and a scalar int32
    tensor, the number of calls to `proposal_fn`.
  """
  with tf.name_scope(name or 'batched_rejection_sampler'):
    def randomized_computation(seed):
      """Internal randomized computation."""
      proposal_seed, mask_seed = samplers.split_seed(
          seed, salt='batched_rejection_sampler')
      proposed_samples, proposed_values = proposal_fn(proposal_seed)
      # The comparison needs to be strictly less to avoid spurious acceptances
      # when the uniform samples exactly 0 (or when the product underflows to
      # 0).
      good_samples_mask = tf.less(
          proposed_values * samplers.uniform(
              prefer_static.shape(proposed_samples),
              seed=mask_seed,
              dtype=dtype),
          target_fn(proposed_samples))
      return proposed_samples, good_samples_mask

    return batched_las_vegas_algorithm(randomized_computation, seed)
