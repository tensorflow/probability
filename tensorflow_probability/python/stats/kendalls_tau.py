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

import warnings

import numpy as np
import tensorflow as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'iterative_mergesort',
    'kendalls_tau'
]


def iterative_mergesort(y, aperm, name=None):
  """Non-recusive mergesort that counts exchanges.

  Args:
    y: values to be sorted.
    aperm: original ordering.

  Returns:
    A tuple consisting of a int32 scalar that counts the number of
    exchanges required to produce a sorted permutation, and a tf.int32
    Tensor that contains the ordering of y values that are sorted.
  """
  with tf.name_scope(name or 'iterative_mergesort'):
    y = tf.convert_to_tensor(y, name='y', dtype=tf.int32)
    aperm = tf.convert_to_tensor(aperm, name='aperm', dtype=tf.int32)
    exchanges = 0
    num = tf.size(y)
    k = 1
    while tf.less(k, num):
      for left in tf.range(0, num - k, 2 * k, dtype=tf.int32):
        rght = left + k
        rend = tf.minimum(rght + k, num)
        tmp = tf.TensorArray(dtype=tf.int32, size=num)
        m, i, j = 0, left, rght
        while tf.less(i, rght) and tf.less(j, rend):
          permij = aperm.gather([i, j])
          yij = tf.gather(y, permij)
          if tf.less_equal(yij[0], yij[1]):
            tmp = tmp.write(m, permij[0])
            i += 1
          else:
            tmp = tmp.write(m, permij[1])
            # Explanation here
            # https://www.geeksforgeeks.org/counting-inversions/.
            exchanges += rght - i
            j += 1
          m += 1
        while tf.less(i, rght):
          tmp = tmp.write(m, aperm.read(i))
          i += 1
          m += 1
        while tf.less(j, rend):
          tmp = tmp.write(m, aperm.read(j))
          j += 1
          m += 1
        aperm = aperm.scatter(tf.range(left, rend), tmp.gather(tf.range(0, m)))
      k *= 2
    return exchanges, aperm.stack()


def kendalls_tau(y_true, y_pred, name=None):
  """Computes Kendall's Tau for two ordered lists.

  Kendall's Tau measures the correlation between ordinal rankings. This
  implementation is similar to the one used in scipy.stats.kendalltau.
  Args:
    y_true: A tensor that provides a true ordinal ranking of N items.
    y_pred: A presumably model provided ordering of the same N items:

  Returns:
    Kendell's Tau, the 1945 tau-b formulation that ignores ordering of
    ties, as a scalar Tensor.
  """
  with tf.name_scope(name or 'kendalls_tau'):
    in_type = dtype_util.common_dtype([y_true, y_pred], dtype_hint=tf.float32)
    y_true = tf.convert_to_tensor(y_true, name='y_true', dtype=in_type)
    y_pred = tf.convert_to_tensor(y_pred, name='y_pred', dtype=in_type)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    tensorshape_util.assert_is_compatible_with(y_true.shape, y_pred.shape)
    if tf.equal(tf.size(y_true), 0) or tf.equal(tf.size(y_pred), 0):
      warnings.warn("y_true and y_pred tensors are not the same size.")
      return np.nan
    perm = tf.argsort(y_true)
    n = tf.shape(perm)[0]
    if tf.less(n, 2):
      warnings.warn("Scalar tensors have no defined ordering.")
      return np.nan

    left = 0
    # scan for ties, and for each range of ties do a argsort on
    # the y_pred value. (TF has no lexicographical sorting, although
    # jax can sort complex number lexicographically. Hmm.)
    lexi = tf.TensorArray(tf.int32, size=n)
    for i in tf.range(n):
      lexi = lexi.write(i, perm[i])
    for right in tf.range(1, n):
      ytruelr = tf.gather(y_true, tf.gather(perm, [left, right]))
      if tf.not_equal(ytruelr[0], ytruelr[1]):
        sub = perm[left:right]
        subperm = tf.argsort(tf.gather(y_pred, sub))
        lexi = lexi.scatter(tf.range(left, right), tf.gather(sub, subperm))
        left = right
    sub = perm[left:n]
    subperm = tf.argsort(tf.gather(y_pred, perm[left:n]))
    lexi.scatter(tf.range(left, n), tf.gather(sub, subperm))

    # This code follows roughly along with scipy/stats/stats.py v. 0.15.1
    # compute joint ties
    first = 0
    t = 0
    for i in tf.range(1, n):
      permfirsti = lexi.gather([first, i])
      y_truefirsti = tf.gather(y_true, permfirsti)
      y_predfirsti = tf.gather(y_pred, permfirsti)
      if (y_truefirsti[0] != y_truefirsti[1] or
          y_predfirsti[0] != y_predfirsti[1]):
        t += ((i - first) * (i - first - 1)) // 2
        first = i
    t += ((n - first) * (n - first - 1)) // 2

    # compute ties in y_true
    first = 0
    u = 0
    for i in tf.range(1, n):
      y_truefirsti = tf.gather(y_true, lexi.gather([first, i]))
      if y_truefirsti[0] != y_truefirsti[1]:
        u += ((i - first) * (i - first - 1)) // 2
        first = i
    u += ((n - first) * (n - first - 1)) // 2

    # count exchanges
    exchanges, newperm = iterative_mergesort(y_pred, lexi)
    # compute ties in y_pred after mergesort with counting
    first = 0
    v = 0
    for i in tf.range(1, n):
      y_predfirsti = tf.gather(y_pred, tf.gather(newperm, [first, i]))
      if y_predfirsti[0] != y_predfirsti[1]:
        v += ((i - first) * (i - first - 1)) // 2
        first = i
    v += ((n - first) * (n - first - 1)) // 2

    tot = (n * (n - 1)) // 2
    if tf.equal(tot, u) or tf.equal(tot, v):
      return np.nan  # Special case for all ties in both ranks

    # Prevent overflow; equal to np.sqrt((tot - u) * (tot - v))
    denom = tf.math.exp(
      0.5
      * (
        tf.math.log(tf.cast(tot - u, tf.float32))
        + tf.math.log(tf.cast(tot - v, tf.float32))
      )
    )
    tau = (
      tf.cast(tot - (v + u - t), tf.float32) - 2.0 * tf.cast(exchanges,
                                                             tf.float32)
    ) / denom

    return tau
