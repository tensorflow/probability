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

import hypothesis
from hypothesis.extra import numpy as hpnp
import hypothesis.strategies as hps
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_probability.python.experimental.marginalize.logeinsumexp import _binary_einslogsumexp
from tensorflow_probability.python.experimental.marginalize.logeinsumexp import logeinsumexp
from tensorflow_probability.python.internal import test_util
import tensorflow_probability.python.internal.hypothesis_testlib as testlib


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


@hps.composite
def tensor_shape_and_indices(draw, rank, min_dim=1, max_dim=4,
                             dimensions=(2, 2, 2, 2, 3, 3, 4),
                             letters='ijklmno'):
  """Return a shape and a set of indices consistent with its dimensions."""

  # We don't draw dimensions directly but from an array of dimensions
  # so that we can ensure that all appearances of a particular letter
  # in a formula correspond to the same dimension.
  indices = draw(hps.tuples(*(rank * [hps.integers(min_dim, max_dim)])))
  shape = tuple(dimensions[i] for i in indices)
  formula = ''.join(letters[i] for i in indices)
  return formula, shape


@hps.composite
def indexed_tensor(draw, n, min_value=None, max_value=None, shape=None):
  """Return tensor paired with a set of indices suitable for it."""

  formula, shape = draw(tensor_shape_and_indices(n))
  t = draw(hpnp.arrays(dtype=np.float64, shape=shape,
                       elements=hps.floats(min_value=min_value,
                                           max_value=max_value)))
  return formula, t


@hps.composite
def nary_einsum(draw, max_num_tensors):
  """Return an entire `einsum` example."""

  n = draw(hps.integers(min_value=1, max_value=max_num_tensors))
  lhss = []
  tensors = []
  for _ in range(n):
    rank = draw(hps.integers(min_value=0, max_value=4))
    # Deliberately choosing values in a restricted range so as not to cause
    # underflow making results comparable with ordinary `einsum`.
    lhs_t, t = draw(indexed_tensor(rank, min_value=1e-5, max_value=1.))
    lhss.append(lhs_t)
    tensors.append(t)

  # Find all indices in use anywhere on left hand side of formula.
  # We build a right hand side by selecting a list of these
  # letters using each letter at most once.
  letters = list(set(''.join(lhss)))
  rhs_letters = draw(hps.lists(hps.sampled_from(letters),
                               max_size=len(letters),
                               unique=True))
  rhs = ''.join(rhs_letters)

  # Combine pieces on left hand side with right hand side to make
  # a single formula.
  # Some example formulas this strategy can produce:
  #
  #  'kjjm,mllm,j,kmk->jmlk'
  #  'l,lllk,kkll,->'
  #  'mm,jk,l,kll->'
  #  ',km,,jkjl->m'
  #  'jmm,k,m,km->km'
  #  'k,l,kjl,ml->jkl'

  formula = '{}->{}'.format(','.join(lhss), rhs)

  return formula, tensors


# Tests
@test_util.test_all_tf_execution_regimes
class _EinLogSumExpTest(test_util.TestCase):

  def test_sum(self):
    np.random.seed(42)

    a = _random_sparse_tensor([100])
    u = tf.reduce_logsumexp(tf.math.log(a))
    v = logeinsumexp('i->', tf.math.log(a))

    self.assertAllClose(u, v)

  def test_batch_sum(self):
    np.random.seed(42)

    a = _random_sparse_tensor([100, 100])
    u = tf.reduce_logsumexp(tf.math.log(a), axis=-1)
    v = logeinsumexp('ij->i', tf.math.log(a))

    self.assertAllClose(u, v)

  def test_dot(self):
    np.random.seed(42)

    a = _random_sparse_tensor([100])
    b = _random_sparse_tensor([100])
    u = tf.reduce_logsumexp(tf.math.log(a) + tf.math.log(b))
    v = logeinsumexp(
        'i,i->', tf.math.log(a), tf.math.log(b))

    self.assertAllClose(u, v)

  def test_batch_diagonal(self):
    """Batch matrix multiplication test."""

    np.random.seed(42)

    a = _random_sparse_tensor([100, 8, 8])
    u = tf.linalg.diag_part(a)
    v = logeinsumexp('ijj->ij', a)

    self.assertAllClose(u, v)

  def test_generalized_trace(self):
    """Batch matrix multiplication test."""

    np.random.seed(42)

    a = _random_sparse_tensor([2, 2, 2, 2, 2, 2, 2, 2])
    u = tf.math.log(a[0, 0, 0, 0, 0, 0, 0, 0] + a[1, 1, 1, 1, 1, 1, 1, 1])
    v = logeinsumexp('iiiiiiii->', tf.math.log(a))

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
    v = logeinsumexp(formula, tf.math.log(a), tf.math.log(b))

    self.assertAllClose(u, v)

  def test_zero_zero_multiplication(self):
    """Batch matrix multiplication test."""

    a = np.array([[0.0, 0.0], [0.0, 0.0]])
    v = tf.exp(logeinsumexp('ij,jk->ik', tf.math.log(a), tf.math.log(a)))

    self.assertAllClose(a, v)

  def test_matmul(self):
    """Batch matrix multiplication test."""

    np.random.seed(42)

    a = _random_sparse_tensor([200, 2, 2])
    b = _random_sparse_tensor([200, 2, 2])
    u = tf.math.log(tf.matmul(a, b))
    v = _binary_einslogsumexp(
        'hij,hjk->hik', tf.math.log(a), tf.math.log(b))

    self.assertAllClose(u, v)

  def test_permutation(self):
    """Permutation test."""

    np.random.seed(42)

    t = _random_sparse_tensor(8 * [2])
    u = tf.transpose(t, perm=[2, 3, 4, 5, 6, 7, 0, 1])
    v = logeinsumexp('abcdefgh->cdefghab', t)

    self.assertAllClose(u, v)

  # This example produces an erroneous `-inf` when implemented using the
  # method described on stackoverflow:
  # https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-np
  def test_counterexample(self):
    a = tf.convert_to_tensor(np.array([0., -1000.]), dtype=tf.float64)
    b = tf.convert_to_tensor(np.array([-1000., 0.]), dtype=tf.float64)
    v = logeinsumexp('i,i->', a, b)

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

    # Compute product of 32 matrices as a single `logeinsumexp` over
    # 33 dimensions ensuring that:
    # (1) a 33-dimensional intermediate isn't materialized
    # (2) we can work with logs of quantities that would otherwise
    #     cause an underflow.
    # (3) we can in fact do this with a batch of matrices.
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lhs = ','.join(['Z' + letters[i] + letters[i+1] for i in range(32)])
    rhs = 'Z' + letters[0] + letters[32]
    formula = '{}->{}'.format(lhs, rhs)
    log_as = 32 * [m]
    v = logeinsumexp(formula, *log_as)

    self.assertAllClose(u, v)

  @hypothesis.given(nary_einsum(3))
  # Hypothesis complains about the large test size but it's
  # important to use substantial examples to test `logeinsumexp`.
  @testlib.tfp_hp_settings(default_max_examples=10)
  def test_einsum(self, data):
    formula, tensors = data
    tensors = [tf.convert_to_tensor(a, dtype=tf.float64) for a in tensors]
    u = tf.math.log(tf.einsum(formula, *tensors))
    log_tensors = [tf.math.log(t) for t in tensors]
    v = logeinsumexp(formula, *log_tensors)

    self.assertAllClose(u, v)

if __name__ == '__main__':
  tf.test.main()
