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
# ============================================================================
"""Compute einsums in log space."""

import numpy as np
import tensorflow as tf
import tensorflow_probability.python.experimental.marginalize.logeinsumexp as logeinsumexp
from tensorflow_probability.python.internal import test_util


# pylint: disable=no-member
# pylint: disable=no-value-for-parameter


def _logmatmulexp(a, b):
  """`matmul` computed in log space."""
  return tf.reduce_logsumexp(a[..., :, :, None] + b[..., None, :, :], axis=-2)


def _random_sparse_tensor(shape, min_value=0., max_value=1.0):
  """Return sparse tensor."""

  elements = np.random.uniform(low=min_value, high=max_value, size=shape)
  # By including sparsity we test elements whose logs are -inf.
  sparsity = np.random.rand(*shape) < 0.5
  return tf.where(sparsity, elements, np.zeros(shape))


# Tests
@test_util.test_all_tf_execution_regimes
class _EinLogSumExpTest(test_util.TestCase):

  def test_sum(self):
    np.random.seed(42)

    a = _random_sparse_tensor([100])
    u = tf.reduce_logsumexp(tf.math.log(a))
    v = logeinsumexp.logeinsumexp('i->', tf.math.log(a))

    self.assertAllClose(u, v)

  def test_batch_sum(self):
    np.random.seed(42)

    a = _random_sparse_tensor([100, 100])
    u = tf.reduce_logsumexp(tf.math.log(a), axis=-1)
    v = logeinsumexp.logeinsumexp('ij->i', tf.math.log(a))

    self.assertAllClose(u, v)

  def test_dot(self):
    np.random.seed(42)

    a = _random_sparse_tensor([100])
    b = _random_sparse_tensor([100])
    u = tf.reduce_logsumexp(tf.math.log(a) + tf.math.log(b))
    v = logeinsumexp.logeinsumexp(
        'i,i->', tf.math.log(a), tf.math.log(b))

    self.assertAllClose(u, v)

  def test_batch_diagonal(self):
    """Batch matrix multiplication test."""

    np.random.seed(42)

    a = _random_sparse_tensor([100, 8, 8])
    u = tf.linalg.diag_part(a)
    v = logeinsumexp.logeinsumexp('ijj->ij', a)

    self.assertAllClose(u, v)

  def test_generalized_trace(self):
    """Batch matrix multiplication test."""

    np.random.seed(42)

    a = _random_sparse_tensor([2, 2, 2, 2, 2, 2, 2, 2])
    u = tf.math.log(a[0, 0, 0, 0, 0, 0, 0, 0] + a[1, 1, 1, 1, 1, 1, 1, 1])
    v = logeinsumexp.logeinsumexp('iiiiiiii->', tf.math.log(a))

    self.assertAllClose(u, v)

  def test_compare_einsum(self):
    """Batch matrix multiplication test."""

    np.random.seed(42)

    # We're comparing with regular `einsum` so we choose a `min_value`
    # to make underflow impossible.
    a = _random_sparse_tensor([2, 2, 2, 2, 2, 2, 2], min_value=0.1)
    b = _random_sparse_tensor([2, 2, 2, 2, 2, 2, 2], min_value=0.1)
    formula = 'abcdcfg,edfcbaa->bd'
    u = tf.math.log(tf.einsum(formula, a, b))
    v = logeinsumexp.logeinsumexp(formula, tf.math.log(a), tf.math.log(b))

    self.assertAllClose(u, v)

  def test_zero_zero_multiplication(self):
    """Batch matrix multiplication test."""

    a = np.array([[0.0, 0.0], [0.0, 0.0]])
    v = tf.exp(logeinsumexp.logeinsumexp('ij,jk->ik',
                                         tf.math.log(a), tf.math.log(a)))

    self.assertAllClose(a, v)

  def test_matmul(self):
    """Batch matrix multiplication test."""

    np.random.seed(42)

    a = _random_sparse_tensor([200, 2, 2])
    b = _random_sparse_tensor([200, 2, 2])
    u = tf.math.log(tf.matmul(a, b))
    v = logeinsumexp._binary_einslogsumexp(
        'hij,hjk->hik', tf.math.log(a), tf.math.log(b))

    self.assertAllClose(u, v)

  def test_permutation(self):
    """Permutation test."""

    np.random.seed(42)

    a = _random_sparse_tensor(8 * [2])
    u = tf.transpose(a, perm=[1, 2, 3, 4, 5, 6, 7, 0])
    v = logeinsumexp.logeinsumexp('abcdefgh->bcdefgha', a)

    self.assertAllClose(u, v)

  # This example produces an erroneous `-inf` when implemented using the
  # method described on stackoverflow:
  # https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-np
  def test_counterexample(self):
    a = tf.convert_to_tensor(np.array([0., -1000.]), dtype=tf.float32)
    b = tf.convert_to_tensor(np.array([-1000., 0.]), dtype=tf.float32)
    v = logeinsumexp.logeinsumexp('i,i->', a, b)

    self.assertAllClose(np.log(2.) - 1000., v)

  def test_matrix_power(self):
    np.random.seed(42)

    # Testing with small values as we want case that would
    # result in underflow when ordinary `einsum` is used.
    a = _random_sparse_tensor([200, 4, 4], max_value=1e-20)
    m = tf.math.log(a)
    m2 = _logmatmulexp(m, m)
    m4 = _logmatmulexp(m2, m2)
    m8 = _logmatmulexp(m4, m4)
    m16 = _logmatmulexp(m8, m8)
    u = _logmatmulexp(m16, m16)

    # Compute product of 25 matrices as a single `logeinsumexp` over
    # 26 dimensions ensuring that:
    # (1) a 26-dimensional intermediate isn't materialized
    # (2) we can work with logs of quantities that would otherwise
    #     cause an underflow.
    # (3) we can in fact do this with a batch of matrices.
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lhs = ','.join(['Z' + letters[i] + letters[i+1] for i in range(32)])
    rhs = 'Z' + letters[0] + letters[32]
    formula = '{}->{}'.format(lhs, rhs)
    log_as = 32 * [m]
    v = logeinsumexp.logeinsumexp(formula, *log_as)

    self.assertAllClose(u, v)

if __name__ == '__main__':
  tf.test.main()
