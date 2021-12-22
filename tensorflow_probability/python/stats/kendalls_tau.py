# Copyright 2020 The TensorFlow Probability Authors. All Rights Reserved.
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
# ==============================================================================
"""Implements Kendall's Tau metric and loss."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

__all__ = ['iterative_mergesort', 'kendalls_tau']


def iterative_mergesort(y, permutation, name=None):
  """Non-recusive mergesort that counts exchanges.

  Args:
    y: a `Tensor` of shape `[n]` containing values to be sorted.
    permutation: `Tensor` of shape `[n]` with original ordering.
    name: Optional Python `str` name for ops created by this method.
      Default value: `None` (i.e., 'iterative_mergesort').

  Returns:
    exchanges: `int32` scalar that counts the number of exchanges required to
      produce a sorted permutation
    permutation: and a `tf.int32` Tensor that contains the ordering of y values
      that are sorted.
  """

  with tf.name_scope(name or 'iterative_mergesort'):
    y = tf.convert_to_tensor(y, name='y')
    permutation = tf.convert_to_tensor(
        permutation, name='permutation', dtype=tf.int32)
    shape = permutation.shape
    tensorshape_util.assert_is_compatible_with(y.shape, shape)
    n = ps.size(y)

    def outer_body(k, exchanges, permutation):
      # The outer body progressively merges lists as k grows by powers of 2,
      # tracking the total swaps required in exchanges as the new permutation is
      # built in place.
      y_ordered = tf.gather(y, permutation)

      def middle_body(left, exchanges, permutation):
        # the middle body advances through the sublists of size k, advancing
        # the left edge until the end of the input is reached.
        right = left + k
        end = tf.minimum(right + k, n)

        # See explanation here
        # https://www.geeksforgeeks.org/counting-inversions/.

        def inner_body(i, j, x, np, p):
          # The [left, right) and [right, end) lists are merged sorted, with
          # i and j tracking the advance through each range. x records the
          # number of order (bubble-sort equivalent) swaps that are happening
          # with each insertion, and np represents the size of the output
          # permutation that's been filled in using the p tensor.
          y_less = y_ordered[i] <= y_ordered[j]
          element = tf.where(y_less, [permutation[i]], [permutation[j]])
          new_p = tf.concat([p[0:np], element, p[np + 1:n]], axis=0)
          tensorshape_util.set_shape(new_p, p.shape)
          return (tf.where(y_less, i + 1, i), tf.where(y_less, j, j + 1),
                  tf.where(y_less, x, x + right - i), np + 1, new_p)

        i_j_x_np_p = (left, right, exchanges, 0, tf.zeros([n], dtype=tf.int32))
        (i, j, exchanges, np, p) = tf.while_loop(
            cond=lambda i, j, x, np, p: tf.math.logical_and(i < right, j < end),
            body=inner_body,
            loop_vars=i_j_x_np_p)
        permutation = tf.concat([
            permutation[0:left], p[0:np], permutation[i:right],
            permutation[j:end], permutation[end:n]
        ],
                                axis=0)
        tensorshape_util.set_shape(permutation, shape)
        return left + 2 * k, exchanges, permutation

      _, exchanges, permutation = tf.while_loop(
          cond=lambda left, exchanges, permutation: left < n - k,
          body=middle_body,
          loop_vars=(0, exchanges, permutation))
      k *= 2
      return k, exchanges, permutation

    _, exchanges, permutation = tf.while_loop(
        cond=lambda k, exchanges, permutation: k < n,
        body=outer_body,
        loop_vars=(1, 0, permutation))
    return exchanges, permutation


def lexicographical_indirect_sort(primary, secondary, name=None):
  """Sorts by primary, then by secondary returning the indices.

  Args:
    primary: a `Tensor` of shape `[n]` containing the primary sort key. the
      primary sort key value.
    secondary: a `Tensor` of shape `[n]` containing the secondary sort key to be
      used when the primary keys are identical.
    name: Optional Python `str` name for ops created by this method.
      Default value: `None` (i.e., 'lexicographical_indirect_sort').

  Returns:
    lexicographic: A permutation of range(n) that provides the sorted primary,
      then secondary values.
  """
  with tf.name_scope(name or 'lexicographical_indirect_sort'):
    n = ps.size0(primary)
    permutation = tf.argsort(primary)
    # scan for ties, and for each range of ties do a argsort on
    # the secondary value. (TF has no lexicographical sorting, although
    # jax can sort complex number lexicographically. Hmm.)
    primary_ordered = tf.gather(primary, permutation)

    def body(left, right, lexicographic):
      # We make a single pass through the list using right and left, where right
      # advances and left chases it looking for spans that are equal in their
      # primary key to then institute a sort on the secondary key.
      not_equal = tf.not_equal(primary_ordered[left], primary_ordered[right])

      def secondary_sort():
        x = tf.concat([
            lexicographic[0:left],
            tf.gather(permutation[left:right],
                      tf.argsort(tf.gather(secondary,
                                           permutation[left:right]))),
            lexicographic[right:n],
        ],
                      axis=0)
        tensorshape_util.set_shape(x, [n])
        return x

      return (tf.where(not_equal, right, left), right + 1,
              tf.cond(not_equal, secondary_sort, lambda: lexicographic))

    left, _, lexicographic = tf.while_loop(
        cond=lambda left, right, lexicographic: right < n,
        body=body,
        loop_vars=(0, 0, tf.zeros_like(permutation, dtype=tf.int32)))
    return tf.concat([
        lexicographic[0:left],
        tf.gather(permutation[left:n],
                  tf.argsort(tf.gather(secondary, permutation[left:n])))
    ],
                     axis=0)


def kendalls_tau(y_true, y_pred, name=None):
  """Computes Kendall's Tau for two ordered lists.

  Kendall's Tau measures the correlation between ordinal rankings. This
  implementation is similar to the one used in scipy.stats.kendalltau.
  The provided values may be of any type that is sortable, with the
  argsort indices indicating the true or proposed ordinal sequence.

  Args:
    y_true: a `Tensor` of shape `[n]` containing the true ordinal ranking.
    y_pred: a `Tensor` of shape `[n]` containing the predicted ordering of the
      same N items.
    name: Optional Python `str` name for ops created by this method.
      Default value: `None` (i.e., 'kendalls_tau').

  Returns:
    kendalls_tau: Kendall's Tau, the 1945 tau-b formulation that ignores
      ordering of ties, as a `float32` scalar Tensor.
  """
  with tf.name_scope(name or 'kendalls_tau'):
    in_type = dtype_util.common_dtype([y_true, y_pred], dtype_hint=tf.float32)
    y_true = tf.convert_to_tensor(y_true, name='y_true', dtype=in_type)
    y_pred = tf.convert_to_tensor(y_pred, name='y_pred', dtype=in_type)
    tensorshape_util.assert_is_compatible_with(y_true.shape, y_pred.shape)
    assertions = [
        assert_util.assert_rank(y_true, 1),
        assert_util.assert_greater(
            ps.size(y_true), 1, 'Ordering requires at least 2 elements.')
    ]
    with tf.control_dependencies(assertions):
      lexa = lexicographical_indirect_sort(y_true, y_pred)

    # See A Computer Method for Calculating Kendall's Tau with Ungrouped Data
    # by William Night, Journal of the American Statistical Association,
    # Jun., 1966, Vol. 61, No. 314, Part 1 (Jun., 1966), pp. 436-439
    # for notation https://www.jstor.org/stable/2282833

    def jointly_tied_pairs_body(first, t, i):
      not_equal = tf.math.logical_or(
          tf.not_equal(y_true[lexa[first]], y_true[lexa[i]]),
          tf.not_equal(y_pred[lexa[first]], y_pred[lexa[i]]))
      return (tf.where(not_equal, i, first),
              tf.where(not_equal, t + ((i - first) * (i - first - 1)) // 2,
                       t), i + 1)

    n = ps.size0(y_true)
    first, t, _ = tf.while_loop(
        cond=lambda first, t, i: i < n,
        body=jointly_tied_pairs_body,
        loop_vars=(0, 0, 1))
    t += ((n - first) * (n - first - 1)) // 2

    def ties_y_true_body(first, v, i):
      not_equal = tf.not_equal(y_true[lexa[first]], y_true[lexa[i]])
      return (tf.where(not_equal, i, first),
              tf.where(not_equal, v + ((i - first) * (i - first - 1)) // 2,
                       v), i + 1)

    first, v, _ = tf.while_loop(
        cond=lambda first, v, i: i < n,
        body=ties_y_true_body,
        loop_vars=(0, 0, 1))
    v += ((n - first) * (n - first - 1)) // 2

    # count exchanges
    exchanges, newperm = iterative_mergesort(y_pred, lexa)

    def ties_in_y_pred_body(first, u, i):
      not_equal = tf.not_equal(y_pred[newperm[first]], y_pred[newperm[i]])
      return (tf.where(not_equal, i, first),
              tf.where(not_equal, u + ((i - first) * (i - first - 1)) // 2,
                       u), i + 1)

    first, u, _ = tf.while_loop(
        cond=lambda first, u, i: i < n,
        body=ties_in_y_pred_body,
        loop_vars=(0, 0, 1))
    u += ((n - first) * (n - first - 1)) // 2
    n0 = (n * (n - 1)) // 2
    assertions = [
        assert_util.assert_less(v, tf.cast(n0, tf.int32),
                                'All ranks are ties for y_true.'),
        assert_util.assert_less(u, tf.cast(n0, tf.int32),
                                'All ranks are ties for y_pred.')
    ]
    with tf.control_dependencies(assertions):
      return (tf.cast(n0 - (u + v - t), tf.float32) -
              2.0 * tf.cast(exchanges, tf.float32)) / tf.math.sqrt(
                  tf.cast(n0 - v, tf.float32) * tf.cast(n0 - u, tf.float32))
