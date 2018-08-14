# Copyright 2018 The TensorFlow Probability Authors.
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
"""Functions to apply slice sampling update in one dimension."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python import distributions


def _left_doubling_increments(batch_shape, max_doublings, step_size, seed=None,
                              name=None):
  """Computes the doubling increments for the left end point.

  The doubling procedure expands an initial interval to find a superset of the
  true slice. At each doubling iteration, the interval width is doubled to
  either the left or the right hand side with equal probability.
  If, initially, the left end point is at `L(0)` and the width of the
  interval is `w(0)`, then the left end point and the width at the
  k-th iteration (denoted L(k) and w(k) respectively) are given by the following
  recursions:

  ```none
  w(k) = 2 * w(k-1)
  L(k) = L(k-1) - w(k-1) * X_k, X_k ~ Bernoulli(0.5)
  or, L(0) - L(k) = w(0) Sum(2^i * X(i+1), 0 <= i < k)
  ```

  This function computes the sequence of `L(0)-L(k)` and `w(k)` for k between 0
  and `max_doublings` independently for each chain.

  Args:
    batch_shape: Positive int32 `tf.Tensor`. The batch shape.
    max_doublings: Scalar positive int32 `tf.Tensor`. The maximum number of
      doublings to consider.
    step_size: A real `tf.Tensor` with shape compatible with [num_chains].
      The size of the initial interval.
    seed: (Optional) positive int. The random seed. If None, no seed is set.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'find_slice_bounds').

  Returns:
    left_increments: A tensor of shape (max_doublings+1, batch_shape). The
      relative position of the left end point after the doublings.
    widths: A tensor of shape (max_doublings+1, ones_like(batch_shape)). The
      widths of the intervals at each stage of the doubling.
  """
  with tf.name_scope(name, 'left_doubling_increments',
                     [batch_shape, max_doublings, step_size]):

    step_size = tf.convert_to_tensor(step_size)
    dtype = step_size.dtype.base_dtype
    # Output shape of the left increments tensor.
    output_shape = tf.concat(([max_doublings + 1], batch_shape), axis=0)
    # A sample realization of X_k.
    expand_left = distributions.Bernoulli(0.5, dtype=dtype).sample(
        sample_shape=output_shape, seed=seed)

    # The widths of the successive intervals. Starts with 1.0 and ends with
    # 2^max_doublings.
    width_multipliers = tf.cast(2 ** tf.range(0, max_doublings+1), dtype=dtype)
    # Output shape of the `widths` tensor.
    widths_shape = tf.concat(([max_doublings + 1],
                              tf.ones_like(batch_shape)), axis=0)
    width_multipliers = tf.reshape(width_multipliers, shape=widths_shape)
    # Widths shape is [max_doublings + 1, 1, 1, 1...].
    widths = width_multipliers * step_size

    # Take the cumulative sum of the left side increments in slice width to give
    # the resulting distance from the inital lower bound.
    left_increments = tf.cumsum(widths * expand_left, exclusive=True, axis=0)
    return left_increments, widths


def _find_best_interval_idx(x, name=None):
  """Finds the index of the optimal set of bounds for each chain.

  For each chain, finds the smallest set of bounds for which both edges lie
  outside the slice. This is equivalent to the point at which a for loop
  implementation (P715 of Neal (2003)) of the algorithm would terminate.

  Performs the following calculation, where i is the number of doublings that
  have been performed and k is the max number of doublings:

  (2 * k - i) * flag + i

  The argmax of the above returns the earliest index where the bounds were
  outside the slice and if there is no such point, the widest bounds.

  Args:
    x: A tensor of shape (max_doublings+1, batch_shape). Type int32, with value
      0 or 1. Indicates if this set of bounds is outside the slice.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'find_slice_bounds').

  Returns:
    indices: A tensor of shape batch_shape. Type int32, with the index of the
      first set of bounds outside the slice and if there are none, the index of
      the widest set.
  """
  with tf.name_scope(name, 'find_best_interval_idx', [x]):
    # Returns max_doublings + 1. Positive int32.
    k = tf.shape(x)[0]
    dtype = x.dtype.base_dtype
    # Factors by which to multiply the flag. Corresponds to (2 * k - i) above.
    mults = tf.range(2 * k, k, -1, dtype=dtype)[:, tf.newaxis]
    # Factors by which to shift the flag. Corresponds to i above. Ensures the
    # widest bounds are selected if there are no bounds outside the slice.
    shifts = tf.range(k, dtype=dtype)[:, tf.newaxis]
    indices = tf.argmax(mults * x + shifts, axis=0, output_type=dtype)
    return indices


def slice_bounds_by_doubling(x_initial,
                             target_log_prob,
                             log_slice_heights,
                             max_doublings,
                             step_size,
                             seed=None,
                             name=None):
  """Returns the bounds of the slice at each stage of doubling procedure.

  Precomputes the x coordinates of the left (L) and right (R) endpoints of the
  interval `I` produced in the "doubling" algorithm [Neal 2003][1] P713. Note
  that we simultaneously compute all possible doubling values for each chain,
  for the reason that at small-medium densities, the gains from parallel
  evaluation might cause a speed-up, but this will be benchmarked against the
  while loop implementation.

  Args:
    x_initial: `tf.Tensor` of any shape and any real dtype consumable by
      `target_log_prob`. The initial points.
    target_log_prob: A callable taking a `tf.Tensor` of shape and dtype as
      `x_initial` and returning a tensor of the same shape. The log density of
      the target distribution.
    log_slice_heights: `tf.Tensor` with the same shape as `x_initial` and the
      same dtype as returned by `target_log_prob`. The log of the height of the
      slice for each chain. The values must be bounded above by
      `target_log_prob(x_initial)`.
    max_doublings: Scalar positive int32 `tf.Tensor`. The maximum number of
      doublings to consider.
    step_size: `tf.Tensor` with same dtype as and shape compatible with
      `x_initial`. The size of the initial interval.
    seed: (Optional) positive int. The random seed. If None, no seed is set.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'find_slice_bounds').

  Returns:
    upper_bounds: A tensor of same shape and dtype as `x_initial`. Slice upper
      bounds for each chain.
    lower_bounds: A tensor of same shape and dtype as `x_initial`. Slice lower
      bounds for each chain.
    both_ok: A tensor of shape `x_initial` and boolean dtype. Indicates if both
      the chosen upper and lower bound lie outside of the slice.

  #### References

  [1]: Radford M. Neal. Slice Sampling. The Annals of Statistics. 2003, Vol 31,
       No. 3 , 705-767.
       https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
  """
  with tf.name_scope(name, 'slice_bounds_by_doubling',
                     [x_initial, log_slice_heights, max_doublings, step_size]):
    seed_gen = distributions.SeedStream(seed, salt='slice_bounds_by_doubling')
    x_initial = tf.convert_to_tensor(x_initial)
    batch_shape = tf.shape(x_initial)
    dtype = step_size.dtype.base_dtype
    left_endpoints = x_initial + step_size * tf.random_uniform(batch_shape,
                                                               minval=-1.0,
                                                               maxval=0.0,
                                                               dtype=dtype,
                                                               seed=seed_gen())

    # Compute the increments by which we need to step the upper and lower bounds
    # part of the doubling procedure.
    left_increments, widths = _left_doubling_increments(
        batch_shape, max_doublings, step_size, seed=seed_gen())
    # The left and right end points. Shape (max_doublings+1,) + batch_shape.
    left_endpoints -= left_increments
    right_endpoints = left_endpoints + widths

    # Test if these end points lie outside of the slice.
    # Checks if the end points of the slice are outside the graph of the pdf.
    left_ep_values = tf.map_fn(target_log_prob, left_endpoints)
    right_ep_values = tf.map_fn(target_log_prob, right_endpoints)
    left_ok = left_ep_values < log_slice_heights
    right_ok = right_ep_values < log_slice_heights
    both_ok = left_ok & right_ok

    both_ok_f = tf.reshape(both_ok, [max_doublings + 1, -1])

    best_interval_idx = _find_best_interval_idx(tf.to_int32(both_ok_f))

    # Formats the above index as required to use with gather_nd.
    point_index_gather = tf.stack([best_interval_idx,
                                   tf.range(tf.size(best_interval_idx))],
                                  axis=1, name='point_index_gather')
    left_ep_f = tf.reshape(left_endpoints, [max_doublings + 1, -1])
    right_ep_f = tf.reshape(right_endpoints, [max_doublings + 1, -1])
    # The x values of the uppper and lower bounds of the slices for each chain.
    lower_bounds = tf.reshape(tf.gather_nd(left_ep_f, point_index_gather),
                              batch_shape)
    upper_bounds = tf.reshape(tf.gather_nd(right_ep_f, point_index_gather),
                              batch_shape)
    both_ok = tf.reduce_any(both_ok, axis=0)
    return upper_bounds, lower_bounds, both_ok


def _test_acceptance(x_initial, target_log_prob, decided, log_slice_heights,
                     x_proposed, step_size, lower_bounds, upper_bounds,
                     name=None):
  """Ensures the chosen point does not violate reversibility.

    Implements Fig 6 of Neal 2003 page 717, which checks that the path from the
    existing point to the new point would also have been possible in reverse.
    This is done by checking that the algorithm would not have been terminated
    before reaching the old point.

  Args:
    x_initial: A tensor of any shape and real dtype. The initial positions of
      the chains. This function assumes that all the dimensions of `x_initial`
      are batch dimensions (i.e. the event shape is `[]`).
    target_log_prob: Callable accepting a tensor like `x_initial` and returning
      a tensor containing the log density at that point of the same shape.
    decided: A `tf.bool` tensor of the same shape as `x_initial`. Indicates
      whether the acceptance has already been decided. A point is tested only
      if `decided` for that point is False.
    log_slice_heights: Tensor of the same shape and dtype as the return value
      of `target_log_prob` when applied to `x_initial`. The log of the height of
      the chosen slice.
    x_proposed: A tensor of the same shape and dtype as `x_initial`. The
      proposed points.
    step_size: A tensor of shape and dtype compatible with `x_initial`. The min
      interval size in the doubling algorithm.
    lower_bounds: Tensor of same shape and dtype as `x_initial`. Slice lower
      bounds for each chain.
    upper_bounds: Tensor of same shape and dtype as `x_initial`. Slice upper
      bounds for each chain.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'find_slice_bounds').

  Returns:
    acceptable: A boolean tensor of same shape as `x_initial` indicating whether
      the proposed points are acceptable for reversibility or not.
  """
  with tf.name_scope(name, 'test_acceptance',
                     [x_initial, decided, log_slice_heights, x_proposed,
                      step_size, lower_bounds, upper_bounds]):
    d = tf.zeros_like(x_initial, dtype=tf.bool)
    # Keeps track of points for which the loop has "effectively terminated".
    # Termination is when either their interval width has shrunk to the minimum
    # value (step_size) or if the point has already been rejected.
    def cond(_, decided, *ignored_args):  # pylint: disable=unused-argument
      # Continue until all the points have been decided.
      return ~tf.reduce_all(decided)
    acceptable = tf.ones_like(x_initial, dtype=tf.bool)
    def body(acceptable, decided, left, right, d):
      """Checks reversibility as described on P717 of Neal 2003."""
      midpoint = (left + right) / 2
      divided = (((x_initial < midpoint) & (x_proposed >= midpoint)) |
                 ((x_proposed < midpoint) & (x_initial >= midpoint)))
      next_d = d | divided
      next_right = tf.where(x_proposed < midpoint, midpoint, right)
      next_left = tf.where(x_proposed >= midpoint, midpoint, left)
      left_test = (log_slice_heights >= target_log_prob(next_left))
      right_test = (log_slice_heights >= target_log_prob(next_right))
      unacceptable = next_d & left_test & right_test
      # Logic here: For points which have not already been decided,
      # and are unacceptable, set acceptable to False. For others, let them
      # be as they were.
      now_decided = ~decided & unacceptable
      next_acceptable = tf.where(now_decided, ~unacceptable, acceptable)
      # Decided if (a) was already decided, or
      # (b) the new width is less than 1.1 step_size, or
      # (c) was marked unacceptable.
      next_decided = (decided | (next_right - next_left <= 1.1 * step_size) |
                      now_decided)
      return (next_acceptable, next_decided, next_left, next_right, next_d)

    return tf.while_loop(cond, body=body, loop_vars=(acceptable, decided,
                                                     lower_bounds,
                                                     upper_bounds, d))[0]


def _sample_with_shrinkage(x_initial, target_log_prob, log_slice_heights,
                           step_size, lower_bounds, upper_bounds, seed=None,
                           name=None):
  """Samples from the slice by applying shrinkage for rejected points.

  Implements the one dimensional slice sampling algorithm of Neal (2003), with a
  doubling algorithm (Neal 2003 P715 Fig. 4), which doubles the size of the
  interval at each iteration and shrinkage (Neal 2003 P716 Fig. 5), which
  reduces the width of the slice when a selected point is rejected, by setting
  the relevant bound that that value. Randomly sampled points are checked for
  two criteria: that they lie within the slice and that they pass the
  acceptability check (Neal 2003 P717 Fig. 6), which tests that the new state
  could have generated the previous one.

  Args:
    x_initial: A tensor of any shape. The initial positions of the chains. This
      function assumes that all the dimensions of `x_initial` are batch
      dimensions (i.e. the event shape is `[]`).
    target_log_prob: Callable accepting a tensor like `x_initial` and returning
      a tensor containing the log density at that point of the same shape.
    log_slice_heights: Tensor of the same shape and dtype as the return value
      of `target_log_prob` when applied to `x_initial`. The log of the height of
      the chosen slice.
    step_size: A tensor of shape and dtype compatible with `x_initial`. The min
      interval size in the doubling algorithm.
    lower_bounds: Tensor of same shape and dtype as `x_initial`. Slice lower
      bounds for each chain.
    upper_bounds: Tensor of same shape and dtype as `x_initial`. Slice upper
      bounds for each chain.
    seed: (Optional) positive int. The random seed. If None, no seed is set.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'find_slice_bounds').

  Returns:
    x_proposed: A tensor of the same shape and dtype as `x_initial`. The next
      proposed state of the chain.
  """
  with tf.name_scope(name, 'sample_with_shrinkage',
                     [x_initial, log_slice_heights, step_size, lower_bounds,
                      upper_bounds]):
    seed_gen = distributions.SeedStream(seed, salt='_sample_with_shrinkage')
    # Keeps track of whether an acceptable sample has been found for the chain.
    found = tf.zeros_like(x_initial, dtype=tf.bool)
    cond = lambda found, *ignored_args: ~tf.reduce_all(found)
    x_next = tf.identity(x_initial)
    x_initial_shape = tf.shape(x_initial)
    x_initial_dtype = x_initial.dtype.base_dtype
    def _body(found, left, right, x_next):
      """Iterates until every chain has found a suitable next state."""
      proportions = tf.random_uniform(x_initial_shape, dtype=x_initial_dtype,
                                      seed=seed_gen())
      x_proposed = tf.where(~found, left + proportions * (right - left), x_next)
      accept_res = _test_acceptance(x_initial, target_log_prob=target_log_prob,
                                    decided=found,
                                    log_slice_heights=log_slice_heights,
                                    x_proposed=x_proposed, step_size=step_size,
                                    lower_bounds=left, upper_bounds=right)
      boundary_test = log_slice_heights < target_log_prob(x_proposed)
      can_accept = boundary_test & accept_res
      next_found = found | can_accept
      # Note that it might seem that we are moving the left and right end points
      # even if the point has been accepted (which is contrary to the stated
      # algorithm in Neal). However, this does not matter because the endpoints
      # for points that have been already accepted are not used again so it
      # doesn't matter what we do with them.
      next_left = tf.where(x_proposed < x_initial, x_proposed, left)
      next_right = tf.where(x_proposed >= x_initial, x_proposed, right)
      return next_found, next_left, next_right, x_proposed

    return tf.while_loop(cond, body=_body, loop_vars=(found, lower_bounds,
                                                      upper_bounds, x_next))[-1]


def slice_sampler_one_dim(target_log_prob, x_initial, step_size=0.01,
                          max_doublings=30, seed=None, name=None):
  """For a given x position in each Markov chain, returns the next x.

  Applies the one dimensional slice sampling algorithm as defined in Neal (2003)
  to an input tensor x of shape (num_chains,) where num_chains is the number of
  simulataneous Markov chains, and returns the next tensor x of shape
  (num_chains,) when these chains are evolved by the slice sampling algorithm.

  Args:
    target_log_prob: Callable accepting a tensor like `x_initial` and returning
      a tensor containing the log density at that point of the same shape.
    x_initial: A tensor of any shape. The initial positions of the chains. This
      function assumes that all the dimensions of `x_initial` are batch
      dimensions (i.e. the event shape is `[]`).
    step_size: A tensor of shape and dtype compatible with `x_initial`. The min
      interval size in the doubling algorithm.
    max_doublings: Scalar tensor of dtype `tf.int32`. The maximum number of
      doublings to try to find the slice bounds.
    seed: (Optional) positive int. The random seed. If None, no seed is set.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'find_slice_bounds').

  Returns:
    retval: A tensor of the same shape and dtype as `x_initial`. The next state
      of the Markov chain.
    next_target_log_prob: The target log density evaluated at `retval`.
    bounds_satisfied: A tensor of bool dtype and shape batch dimensions.
    upper_bounds: Tensor of the same shape and dtype as `x_initial`. The upper
      bounds for the slice found.
    lower_bounds: Tensor of the same shape and dtype as `x_initial`. The lower
      bounds for the slice found.
  """
  with tf.name_scope(name, 'slice_sampler_one_dim',
                     [x_initial, step_size, max_doublings]):
    x_initial = tf.convert_to_tensor(x_initial)
    # Obtain the input dtype of the array.
    dtype = x_initial.dtype.base_dtype
    # Select the height of the slice. Tensor of shape x_initial.shape.
    log_slice_heights = target_log_prob(x_initial) - tf.random_gamma(
        tf.shape(x_initial), alpha=1, dtype=dtype, seed=seed)
    # Given the above x and slice heights, compute the bounds of the slice for
    # each chain.
    upper_bounds, lower_bounds, bounds_satisfied = slice_bounds_by_doubling(
        x_initial, target_log_prob, log_slice_heights, max_doublings, step_size,
        seed=seed)
    retval = _sample_with_shrinkage(x_initial, target_log_prob=target_log_prob,
                                    log_slice_heights=log_slice_heights,
                                    step_size=step_size,
                                    lower_bounds=lower_bounds,
                                    upper_bounds=upper_bounds, seed=seed)
    return (retval, target_log_prob(retval), bounds_satisfied,
            upper_bounds, lower_bounds)
