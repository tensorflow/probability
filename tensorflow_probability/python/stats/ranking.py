# Copyright 2019 The TensorFlow Probability Authors.
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
"""Functions for computing rank-related statistics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static


def quantile_auc(q0, n0, q1, n1, curve='ROC', name=None):
  """Calculate ranking stats AUROC and AUPRC.

  Computes AUROC and AUPRC from quantiles (one for positive trials and one for
  negative trials).

  We use `pi(x)` to denote a score. We assume that if `pi(x) > k` for some
  threshold `k` then the event is predicted to be "1", and otherwise it is
  predicted to be a "0".  Its actual label is `y`, which may or may not be the
  same.

  Area Under Curve: Receiver Operator Characteristic (AUROC) is defined as:

  / 1
  |  TruePositiveRate(k) d FalsePositiveRate(k)
  / 0

  where,

  ```none
  TruePositiveRate(k) = P(pi(x) > k | y = 1)
  FalsePositiveRate(k) = P(pi(x) > k | y = 0)
  ```

  Area Under Curve: Precision-Recall (AUPRC) is defined as:

  / 1
  |  Precision(k) d Recall(k)
  / 0

  where,

  ```none
    Precision(k) = P(y = 1 | pi(x) > k)
    Recall(k) = TruePositiveRate(k) = P(pi(x) > k | y = 1)
  ```

  Notice that AUROC and AUPRC exchange the role of Recall in the
  integration, i.e.,

              Integrand    Measure
            +------------+-----------+
    AUROC   |  Recall    |  FPR      |
            +------------+-----------+
    AUPRC   |  Precision |  Recall   |
            +------------+-----------+

  To learn more about the relationship between AUROC and AUPRC see [1].

  Args:
    q0: `N-D` `Tensor` of `float`, Quantiles of predicted probabilities given a
      negative trial. The first `N-1` dimensions are batch dimensions, and the
      AUC is calculated over the final dimension.
    n0: `float` or `(N-1)-D Tensor`, Number of negative trials. If `Tensor`,
      dimensions must match the first `N-1` dimensions of `q0`.
    q1: `N-D` `Tensor` of `float`, Quantiles of predicted probabilities given a
      positive trial. The first `N-1` dimensions are batch dimensions, which
      must match those of `q0`.
    n1: `float` or `(N-1)-D Tensor`, Number of positive trials. If `Tensor`,
      dimensions must match the first `N-1` dimensions of `q0`.
    curve: `str`, Specifies the name of the curve to be computed. Must be 'ROC'
      [default] or 'PR' for the Precision-Recall-curve.
    name: `str`, An optional name_scope name.

  Returns:
    auc: `Tensor` of `float`, area under the ROC or PR curve.

  #### Examples

  ```python
    n = 1000
    m = 500
    predictions_given_positive = np.random.rand(n)
    predictions_given_negative = np.random.rand(m)
    q1 = tfp.stats.quantiles(predictions_given_positive, num_quantiles=50)
    q0 = tfp.stats.quantiles(predictions_given_negative, num_quantiles=50)
    auroc = tfp.stats.quantile_auc(q0, m, q1, n, curve='ROC')

  ```

  ### Mathematical Details

  The algorithm proceeds by partitioning the combined quantile data into a
  series of intervals `[a, b)`, approximating the probability mass of
  predictions conditioned on a positive trial (`d1`) and probability mass of
  predictions conditioned on a negative trial (`d0`) in each interval `[a, b)`,
  and accumulating the incremental AUROC/AUPRC as functions of `d0` and `d1`.

  We assume that pi(x) is uniform within a given bucket of each quantile. Thus
  it will also be uniform within an interval [a, b) as long as the interval does
  not cross the quantile's bucket boundaries.

  A consequence of this assumption is that the cdf is piecewise linear.
  That is,

    P( pi(x) > k | y = 0 ), and,
    P( pi(x) > k | y = 1 ),

  are linear in `k`.

  Standard AUROC is fairly easier to calculate. Under the conditional uniformity
  assumptions we have a piece's contribution, [a, b), as:

   / b
   |
   |  P(y = 1 | pi > k) d P(pi > k | y = 0)
   |
   / a

      / b
      |
  = - |  P(pi > k | y = 1) P(pi = k | y = 0) d k
      |
      / a

                         / b
     -1 / (len(q0) - 1)  |
  = -------------------- |  P(pi > k | y = 1) d k
     q0[j + 1] - q0[j]   |
                         / a

                                / 1
     1 / (len(q0) - 1)          |
  = ------------------- (b - a) |  (p1 + u d1) d u
     q0[j + 1] - q0[j]          |
                                / 0

     1 / (len(q0) - 1)
  = ------------------- (b - a) (p1 + d1 / 2)
     q0[j + 1] - q0[j]


  AUPRC is a bit harder to calculate since the integrand,
  `P(y > 0 | pi(x) > k)`, is conditional on `k` rather than a probability over
  `k`.

  We proceed by formulating Precision in terms of the quantiles we have
  available to us.

  Precision(k) = P(y = 1 | pi(x) > k)

                     P(pi(x) > delta | y = 1 ) P(y = 1)
  = -----------------------------------------------------------------------
    P(pi(x) > delta | y = 1 ) P(y = 1) + P(pi(x) > delta | y = 0 ) P(y = 0)


  Since the cdf's are piecewise linear, we calculate this piece's contribution
  to AUPRC by the integral:

   / b
   |
   |  P(y = 1 | pi(x) > delta) d P(pi > delta | y = 1)
   |
   / a

     1 / (len(q1) - 1)
  = ------------------- (b - a) *
     q1[i + 1] - q1[i]

        / 1
        |             n1 * (u d1 + p1)
      * |  -------------------------------------- du
        |   n1 * (u d1 + p1) +  n0 * (u d0 + p0)
        / 0

                                / 1
     1 / (len(q1) - 1)          |            n1 * (u d1 + p1)
  = ------------------- (b - a) |  ------------------------------------ du
     q1[i + 1] - q1[i]          |  n1 * (u d1 + p1) +  n0 * (u d0 + p0)
                                / 0

  where the equality is a consequence of the piecewise uniformity assumption.

  The solution to the integral is given by Mathematica:

  ```
  Integrate[n1 (u d1 + p1) / (n1 (u d1 + p1) + n0 (u d0 + p0)), {u, 0, 1},
            Assumptions -> {p1 > 0, d1 > 0, p0 > 0, d0 > 0, n1 > 0, n0 > 0}]
  ```

  This integral can be solved by hand by noticing that:

    f(x) / (f(x) + g(x)) = 1 / (1 + g(x)/f(x))

  Thus define: u = 1 + g(x)/f(x)
  for which: du = [g'(x)h(x) - g(x)f'(x)] / h(x)^2 dx
  and solving integral 1/u du.


  #### References

    [1]: Jesse Davis and Mark Goadrich. The relationship between
         Precision-Recall and ROC curves. In _International Conference on
         Machine Learning_, 2006. http://dl.acm.org/citation.cfm?id=1143874
  """
  with tf.name_scope(name or 'quantile_auc'):
    dtype = dtype_util.common_dtype([q0, q1, n1, n0], dtype_hint=tf.float32)
    q1 = tf.convert_to_tensor(q1, dtype=dtype)
    q0 = tf.convert_to_tensor(q0, dtype=dtype)
    n1 = tf.convert_to_tensor(n1, dtype=dtype)
    n0 = tf.convert_to_tensor(n0, dtype=dtype)

    # Broadcast `q0` and `q1` to at least 2D. This makes `q1[i]` and `q0[j]` at
    # least 1D, allowing `tf.where` to operate on them.
    q1 = q1 + tf.zeros((1, 1), dtype=dtype)
    q0 = q0 + tf.zeros((1, 1), dtype=dtype)

    q1_shape = prefer_static.shape(q1)
    batch_shape = q1_shape[:-1]
    n_q1 = q1_shape[-1]
    n_q0 = prefer_static.shape(q0)[-1]
    static_batch_shape = q1.shape[:-1]

    n0 = tf.broadcast_to(n0, batch_shape)
    n1 = tf.broadcast_to(n1, batch_shape)

    def loop_body(auc_prev, p0_prev, p1_prev, i, j):
      """Body of the loop to integrate over the ROC/PR curves."""

      # We will align intervals from q0 and q1 by moving from end to beginning,
      # stopping with whatever boundary we encounter first. This boundary is
      # `b`. We find `a` by continuing the search from `b` to the left.
      b, i, j = _get_endpoint_b(i, j, q1, q0, batch_shape)

      # Get q1[i] and q1[i+1]
      ind = tf.where(tf.greater_equal(i, 0))
      q1_i = _get_q_slice(q1, i, ind, batch_shape=batch_shape)
      ip1 = tf.minimum(i + 1, n_q1 - 1)
      q1_ip1 = _get_q_slice(q1, ip1, ind, batch_shape=batch_shape)

      # Get q0[j] and q0[j+1]
      ind = tf.where(tf.greater_equal(j, 0))
      q0_j = _get_q_slice(q0, j, ind, batch_shape=batch_shape)
      jp1 = tf.minimum(j + 1, n_q0 - 1)
      q0_jp1 = _get_q_slice(q0, jp1, ind, batch_shape=batch_shape)

      a = _get_endpoint_a(i, j, q1_i, q0_j, batch_shape)

      # Calculate the proportion [a, b) represents of [q1[i], q1[i+1]).
      d1 = _get_interval_proportion(i, n_q1, q1_i, q1_ip1, a, b, batch_shape)

      # Calculate the proportion [a, b) represents of [q1[i], q1[i+1]).
      d0 = _get_interval_proportion(j, n_q0, q0_j, q0_jp1, a, b, batch_shape)

      # Notice that because we assumed within bucket values are distributed
      # uniformly, we end up with something which is identical to the
      # trapezoidal rule: definite_integral += (b - a) * (f(a) + f(b)) / 2.
      if curve == 'ROC':
        auc = auc_prev + d0 * (p1_prev + d1 / 2.)
      else:
        total_scaled_delta = n0 * d0 + n1 * d1
        total_scaled_cdf_at_b = n0 * p0_prev + n1 * p1_prev

        def get_auprc_update():
          proportion = (total_scaled_cdf_at_b /
                        (total_scaled_cdf_at_b + total_scaled_delta))
          definite_integral = (
              (n1 / tf.square(total_scaled_delta)) *
              (d1 * total_scaled_delta + tf.math.log(proportion) *
               (d1 * total_scaled_cdf_at_b - p1_prev * total_scaled_delta)))
          return d1 * definite_integral

        # Values should be non-negative and we use > 0.0 rather than != 0.0 to
        # catch possible numerical imprecision.
        delta_gt_0 = tf.greater(total_scaled_delta, 0.)
        cdf_gt_0 = tf.greater(total_scaled_cdf_at_b, 0.)
        d1_gt_0 = tf.greater(d1, 0.)
        ind = tf.where(delta_gt_0 & cdf_gt_0 & d1_gt_0)
        auc_update = tf.gather_nd(get_auprc_update(), ind)
        auc = tf.tensor_scatter_nd_add(auc_prev, ind, auc_update)

      # TODO(jvdillon): In calculating AUROC and AUPRC, we probably should
      # resolve ties randomly, making the following eight states possible (where
      # e = 1 means we expected a positive trial and e = 0 means we expected a
      # negative trial):
      #
      #   P(y = 1 | pi(x) > delta)
      #   P(y = 1 | pi(x) = delta, e = 1) 0.5
      #   P(y = 1 | pi(x) = delta, e = 0) 0.5
      #   P(y = 1 | pi(x) < delta)
      #
      #   P(y = 0 | pi(x) > delta)
      #   P(y = 0 | pi(x) = delta, e = 1) 0.5
      #   P(y = 0 | pi(x) = delta, e = 0) 0.5
      #   P(y = 0 | pi(x) < delta)
      #
      # This makes the math a bit harder and its not clear this adds much,
      # especially when we're assuming piecewise uniformity.

      # Accumulate this mass (d1, d0) for the next iteration.
      p1 = p1_prev + d1
      p0 = p0_prev + d0

      return auc, p0, p1, i, j

    init_auc = tf.zeros(batch_shape, dtype=dtype)
    init_p0 = tf.zeros(batch_shape, dtype=dtype)
    init_p1 = tf.zeros(batch_shape, dtype=dtype)
    init_i = tf.zeros(batch_shape, dtype=tf.int64) + n_q1 - 1
    init_j = tf.zeros(batch_shape, dtype=tf.int64) + n_q0 - 1

    loop_cond = lambda auc, p0, p1, i, j: tf.reduce_all(  # pylint: disable=g-long-lambda
        tf.greater_equal(i, 0) | tf.greater_equal(j, 0))

    init_vars = [init_auc, init_p0, init_p1, init_i, init_j]
    auc, _, _, _, _ = tf.while_loop(
        loop_cond, loop_body, init_vars,
        shape_invariants=[static_batch_shape]*5)

    return tf.squeeze(auc)


def _get_q_slice(q, k, ind, b=None, batch_shape=None):
  """Returns `q1[i]` or `q0[j]` for a batch of indices `i` or `j`."""
  q_ind = tf.concat(
      [ind, tf.expand_dims(tf.gather_nd(k, ind), -1)], axis=1)
  b_updates = tf.gather_nd(q, q_ind)
  if b is None:
    return tf.scatter_nd(ind, b_updates, batch_shape)
  return tf.tensor_scatter_nd_update(b, ind, b_updates)


def _update_batch(ind, b_update, b=None, batch_shape=None):
  """Updates a batch of `i`, `j`, `q1[i]` or `q0[j]`."""
  updates = tf.gather_nd(b_update, ind)
  if b is None:
    return tf.scatter_nd(ind, updates, batch_shape)
  return tf.tensor_scatter_nd_update(b, ind, updates)


def _get_interval_proportion(k, n, q_k, q_kp1, a, b, batch_shape):
  """Calculate the proportion `[a, b)` represents of `[q[i], q[i+1])`."""
  k_valid = tf.greater_equal(k, 0) & tf.less(k + 1, n)
  q_equal = tf.equal(q_kp1, q_k)
  d = tf.where(q_equal,
               tf.ones((batch_shape), dtype=q_k.dtype),
               ((b - a) / (q_kp1 - q_k)))
  d = tf.where(k_valid, d, tf.zeros((batch_shape), dtype=q_k.dtype))

  # Turn proportions in to probabilities. This is where we assume that
  # the input vectors are quantiles.
  return d / tf.cast(n - 1, d.dtype)


def _get_endpoint_b(i, j, q1, q0, batch_shape):
  """Determine the end of the interval, `b` as either `q0[j]` or `q1[i]`."""

  # Get `i`-th element of `q1` and the `j`-th of `q0` (for batched data).
  i_geq_0 = tf.where(tf.greater_equal(i, 0))
  q1_i = _get_q_slice(q1, i, i_geq_0, batch_shape=batch_shape)
  j_geq_0 = tf.where(tf.greater_equal(j, 0))
  q0_j = _get_q_slice(q0, j, j_geq_0, batch_shape=batch_shape)

  # Find `b`. If `b==q0[j]`, decrement `j`; if `b==q1[i]`, decrement `i`.
  # Note: we could have just said "b = a;" at the end of this loop but then
  # we'd have to handle corner cases before the loop and within.  Grabbing
  # it each time is minimal more work and saves added code complexity.

  # if i < 0: i, j, b = i, j-1, q0[j]
  i_lt_0 = tf.less(i, 0)
  ind = tf.where(i_lt_0)
  b = _update_batch(ind, q0_j, batch_shape=batch_shape)
  j = _update_batch(ind, j-1, b=j)

  # elif j < 0: i, j, b = i - 1, j, q1[i]
  j_lt_0 = tf.less(j, 0)
  cond_from_previous = ~i_lt_0
  ind = tf.where(j_lt_0 & cond_from_previous)
  b = _update_batch(ind, q1_i, b=b)
  i = _update_batch(ind, i-1, b=i)

  # elif q1[i] == q0[j]: i, j, b = i - 1, j - 1, q1[i]
  q_equal = tf.equal(q1_i, q0_j)
  cond_from_previous = cond_from_previous & ~j_lt_0
  ind = tf.where(q_equal & cond_from_previous)
  b = _update_batch(ind, q1_i, b=b)
  i = _update_batch(ind, i-1, b=i)
  j = _update_batch(ind, j-1, b=j)

  # elif q1[i] > q0[j]: i, j, b = i - 1, j, q1[i]
  q1_gt_q0 = tf.greater(q1_i, q0_j)
  cond_from_previous = cond_from_previous & ~q_equal
  ind = tf.where(q1_gt_q0 & cond_from_previous)
  b = _update_batch(ind, q1_i, b=b)
  i = _update_batch(ind, i-1, b=i)

  # else: i, j, b = i, j - 1, q0[j]
  ind = tf.where(cond_from_previous & ~q1_gt_q0)
  b = _update_batch(ind, q0_j, b=b)
  j = _update_batch(ind, j-1, b=j)

  return b, i, j


def _get_endpoint_a(i, j, q1_i, q0_j, batch_shape):
  """Determine the beginning of the interval, `a`."""
  # if i < 0: a = q0[j]
  i_lt_0 = tf.less(i, 0)
  ind = tf.where(i_lt_0)
  a_update = tf.gather_nd(q0_j, ind)
  a = tf.scatter_nd(ind, a_update, batch_shape)

  # elif j < 0: a = q1[i]
  j_lt_0 = tf.less(j, 0)
  ind = tf.where(j_lt_0)
  a_update = tf.gather_nd(q1_i, ind)
  a = tf.tensor_scatter_nd_update(a, ind, a_update)

  # else: a = max(q0[j], q1[i])
  ind = tf.where(~(i_lt_0 | j_lt_0))
  q_max = tf.maximum(q0_j, q1_i)
  a_update = tf.gather_nd(q_max, ind)
  a = tf.tensor_scatter_nd_update(a, ind, a_update)
  return a
