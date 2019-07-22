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
"""Mutual information estimators and helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

__all__ = [
    'lower_bound_barber_agakov',
]


def lower_bound_barber_agakov(logits, entropy, name=None):
  """Lower bound on mutual information from [Barber and Agakov (2003)][1].

  This method gives a lower bound on the mutual information I(X; Y),
  by replacing the unknown conditional p(x|y) with a variational
  decoder q(x|y), but it requires knowing the entropy of X, h(X).
  The lower bound was introduced in [Barber and Agakov (2003)][1].
  ```none
  I(X; Y) = E_p(x, y)[log( p(x|y) / p(x) )]
          = E_p(x, y)[log( q(x|y) / p(x) )] + E_p(x)[KL[ p(x|y) || q(x|y) ]]
          >= E_p(x, y)[log( q(x|y) )] + H(X) = I_[lower_bound_barbar_agakov]
  ```

  Args:
    logits: `float`-like `Tensor` of size [batch_size] representing
      log(q(x_i | y_i)) for each (x_i, y_i) pair.
    entropy: `float`-like `scalar` representing the entropy of X.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'lower_bound_barber_agakov').
  Returns:
    lower_bound: `float`-like `scalar` for lower bound on mutual information.

  #### References

  [1]: David Barber, Felix V. Agakov. The IM algorithm: a variational
       approach to Information Maximization. In _Conference on Neural
       Information Processing Systems_, 2003.
  """

  with tf.name_scope(name or 'lower_bound_barber_agakov'):
    logits = tf.convert_to_tensor(logits, dtype_hint=tf.float32, name='logits')
    # The first term is 1/K * sum(i=1:K, log(q(x_i | y_i)), where K is
    # the `batch_size` and q(x_i | y_i) is the likelihood from a tractable
    # decoder for the samples from joint distribution.
    # The second term is simply the entropy of p(x), which we assume
    # is tractable.
    return tf.reduce_mean(logits) + entropy
