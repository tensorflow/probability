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

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers

__all__ = [
    'batched_las_vegas_algorithm',
    'batched_rejection_sampler',
]


def batched_las_vegas_algorithm(
    batched_las_vegas_trial_fn, max_trials=None, seed=None, name=None):
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
  many batch points. In this case, the particular good batch point returned is
  deliberately left unspecified.

  Args:
    batched_las_vegas_trial_fn: A callable that takes a PRNG seed and returns
      two values. (1) A structure of Tensors containing the results of the
      computation, all with a shape broadcastable with (2) a boolean mask
      representing whether each batch point succeeded.
    max_trials: An optional integer Tensor giving an upper bound on
      the number of calls to `batched_las_vegas_trial_fn`.  If not
      supplied, no limit is enforced.
    seed: Python integer or `Tensor`, for seeding PRNG.
    name: A name to prepend to created ops.
      Default value: `'batched_las_vegas_algorithm'`.

  Returns:
    results: A structure of Tensors representing the results of a
      successful computation for each batch point.
    successes: A boolean Tensor indicating which batch points succeeded.
      This will be all `True` if no `max_trials` limit was supplied.
    num_iters: A scalar int32 tensor, the number of calls to
      `batched_las_vegas_algorithm`.

  #### References

  [1]: Laszlo Babai. Monte-Carlo algorithms in graph isomorphism
       testing. Universite de Montreal, D.M.S. No. 79-10.
  """
  with tf.name_scope(name or 'batched_las_vegas_algorithm'):
    init_seed, loop_seed = samplers.split_seed(seed)
    values, good_values_mask = batched_las_vegas_trial_fn(init_seed)
    num_iters = tf.constant(1)

    def cond(unused_values, good_values_mask, num_iters, unused_seed):
      all_done = tf.reduce_all(good_values_mask)
      if max_trials is not None:
        return ~all_done & (num_iters < max_trials)
      else:
        return ~all_done

    def body(values, good_values_mask, num_iters, seed):
      """Batched Las Vegas Algorithm body."""

      trial_seed, new_seed = samplers.split_seed(seed)
      new_values, new_good_values_mask = batched_las_vegas_trial_fn(trial_seed)

      def pick(new, old):
        return bu.where_left_justified_mask(new_good_values_mask, new, old)

      values = tf.nest.map_structure(pick, new_values, values)

      good_values_mask = good_values_mask | new_good_values_mask

      return values, good_values_mask, num_iters + 1, new_seed

    (values, final_successes, num_iters, _) = tf.while_loop(
        cond, body, (values, good_values_mask, num_iters, loop_seed),
        back_prop=False)
    return values, final_successes, num_iters


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
      proposal_seed, mask_seed = samplers.split_seed(seed)

      proposed_samples, proposed_values = proposal_fn(proposal_seed)

      # The comparison needs to be strictly less to avoid spurious acceptances
      # when the uniform samples exactly 0 (or when the product underflows to
      # 0).
      target_values = target_fn(proposed_samples)
      good_samples_mask = tf.less(
          proposed_values * samplers.uniform(
              ps.shape(proposed_samples),
              seed=mask_seed,
              dtype=dtype),
          target_values)

      # If either the `proposed_value` or the corresponding `target_value` is
      # `nan`, force that `proposed_sample` to `nan` and accept.  Why?
      #
      # - A `nan` would never be accepted, because tf.less must return False
      #   when either argument is `nan`.
      #
      # - If `nan` happens every time (e.g., due to `nan` in the parameters of
      #   the distribution we are trying to sample from), then we should clearly
      #   return `nan` after going around the rejection loop only once, rather
      #   than looping forever.
      #
      # - If `nan` happens only some of the time, it would silently skew the
      #   distribution on results to always reject, because some of those `nan`
      #   values may have stood for proposals that would have been accepted if
      #   we had computed more accurately.  Instead we forward the `nan`
      #   upstream, so the client can fix their proposal or evaluation
      #   functions.
      #
      # - We force the `proposed_sample` to `nan` because not doing so would
      #   hide a `nan` that occurred in only the `proposed_value` or
      #   `target_value`, silently skewing the distribution on results.
      #
      # - Corner case: if the `proposed_sample` is `nan` but both the
      #   corresponding `proposed_value` and `proposed_target` are for some
      #   reason not `nan`, we trust the user and proceed normally.
      nans = tf.math.is_nan(proposed_values) | tf.math.is_nan(target_values)
      proposed_samples = tf.where(
          nans, tf.cast(np.nan, proposed_samples.dtype), proposed_samples)
      good_samples_mask |= nans
      return proposed_samples, good_samples_mask

    samples, _, num_iters = batched_las_vegas_algorithm(
        randomized_computation, seed=seed)
    return samples, num_iters
