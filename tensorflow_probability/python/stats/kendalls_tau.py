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

__all__ = ['kendalls_tau']


def _tril_indices(n):
  """Emulate np.tril_indices(n, k=-1).

  This method ensures static shapes throughout (ie, XLA compilable).
  This method only works for n <= 30000.

  Args:
    n: number elements to generate all pairs

  Returns:
    A [2, n * (n - 1) / 2] vector of all combinations of range(n).
  """
  n = tf.convert_to_tensor(n, dtype_hint=tf.int32)
  # Number of lower triangular entries in an nxn matrix
  m = (n - 1) * n / 2
  r = tf.cast(tf.range(m), dtype=tf.float64)

  # From Sloane: https://oeis.org/A002024 "k appears k times"
  # e.g., [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, ...]
  e = tf.math.floor(tf.math.sqrt(2 * (r + 1)) + .5)

  # From Sloane: https://oeis.org/A002262 "Triangle read by rows"
  # e.g., [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, ...]
  f = tf.math.floor(tf.math.sqrt(2 * r + .25) - .5)
  g = r - f * (f + 1) / 2

  return tf.cast(tf.stack([e, g]), dtype=tf.int32)


def kendalls_tau(y_true, y_pred, name=None):
  """Computes Kendall's Tau for two ordered lists.

  Kendall's Tau measures the correlation between ordinal rankings.
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
      n = ps.size0(y_true)
      indices = _tril_indices(n)
      dxij = tf.sign(
          tf.gather(y_true, indices[0]) - tf.gather(y_true, indices[1]))
      dyij = tf.sign(
          tf.gather(y_pred, indices[0]) - tf.gather(y_pred, indices[1]))
      # s is sum of concordant pairs minus discordant pairs.
      s = tf.cast(tf.math.reduce_sum(dxij * dyij), tf.float32)
      # t is the number of y_true pairs that are not ties.
      t = tf.math.count_nonzero(dxij, dtype=tf.float32)
      # u is the number of y_pred pairs that are not ties.
      u = tf.math.count_nonzero(dyij, dtype=tf.float32)
      assertions = [
          assert_util.assert_positive(t, 'All ranks are ties for y_true.'),
          assert_util.assert_positive(u, 'All ranks are ties for y_pred.')
      ]
      with tf.control_dependencies(assertions):
        return s / tf.math.sqrt(t * u)
