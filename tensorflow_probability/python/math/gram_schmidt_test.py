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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import scipy.linalg as scipy_linalg

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


def gs_numpy(mat):
  res = np.array(mat)
  n_vecs = res.shape[-1]
  for i in range(n_vecs):
    u = res[:, i]
    u /= np.linalg.norm(u)
    for j in range(i + 1, n_vecs):
      v = res[:, j]
      res[:, j] -= np.dot(u, v) * u
  return res / np.sqrt(np.sum(np.square(res), -2, keepdims=True))


@test_util.test_all_tf_execution_regimes
class GramSchmidtTest(test_util.TestCase):

  def testGramSchmidt(self):
    """Checks that we get the result of the modified gram-schmidt algorithm."""
    dim, n_vectors = 20, 10
    np_matrix = self.evaluate(
        tf.random.normal([dim, n_vectors], seed=test_util.test_seed()))
    matrix = tf.constant(np_matrix)

    self.assertAllClose(gs_numpy(np_matrix), tfp.math.gram_schmidt(matrix))

    self.assertAllClose(gs_numpy(np_matrix)[:, :7],
                        tfp.math.gram_schmidt(matrix, 7)[:, :7])

  def testGramSchmidtOrthonormal(self):
    """Checks that all vectors form an orthonormal basis."""
    dim, n_vectors = 10, 5
    vectors = tf.random.normal([dim, n_vectors], seed=test_util.test_seed())

    ortho = self.evaluate(tfp.math.gram_schmidt(vectors))

    for i in range(n_vectors):
      self.assertAllClose(
          np.linalg.norm(ortho[..., i]), 1., rtol=1e-7)
      for j in range(i + 1, n_vectors):
        self.assertAllClose(
            np.dot(ortho[..., i], ortho[..., j]), 0., atol=1e-6)

  def testIllConditioned(self):
    # Modified G-S handles ill-conditioned matrices much better (numerically)
    # than original G-S.
    dim = 200
    mat = tf.eye(200, dtype=tf.float64) * 1e-5 + scipy_linalg.hilbert(dim)
    mat = tf.math.l2_normalize(mat, axis=-1)
    ortho = tfp.math.gram_schmidt(mat)
    xtx = tf.matmul(ortho, ortho, transpose_a=True)
    self.assertAllClose(
        0., tf.linalg.norm(tf.eye(dim, dtype=tf.float64) - xtx,
                           ord='fro', axis=(-1, -2)))

  def testXLA(self):
    self.skip_if_no_xla()
    gs = tf.function(tfp.math.gram_schmidt, experimental_compile=True)
    mat = self.evaluate(tf.random.normal([10, 5], seed=test_util.test_seed()))
    self.assertAllClose(gs(mat), tfp.math.gram_schmidt(mat))

  def testBatched(self):
    shp = (5, 13, 7)
    mat_numpy = self.evaluate(tf.random.normal(shp, seed=test_util.test_seed()))
    mat = tf.constant(mat_numpy)
    self.assertAllClose(
        np.vectorize(gs_numpy, signature='(d,n)->(d,n)')(mat_numpy),
        tfp.math.gram_schmidt(mat))

  def testBatch1UnknownInnermost(self):
    shp = (1, 13, 7)
    mat_numpy = self.evaluate(tf.random.normal(shp, seed=test_util.test_seed()))
    mat = tf.constant(mat_numpy)
    @tf.function
    def f(n):
      return tfp.math.gram_schmidt(mat[..., :n])
    self.assertAllClose(
        np.vectorize(gs_numpy, signature='(d,n)->(d,n)')(mat_numpy[..., :4]),
        f(tf.constant(4)))

  def testQrEquivalenceUpToN(self):
    shp = (5, 13, 7)
    # Ensure the matrix is well conditioned.
    mat = tf.random.normal(shp, seed=test_util.test_seed())
    mat = tf.linalg.matmul(mat, mat, transpose_b=True)
    mat = tf.linalg.set_diag(mat, tf.linalg.diag_part(mat) + 1.)
    mat = self.evaluate(tf.math.l2_normalize(mat, axis=-2))
    n_vecs = 4
    # The two are identical up to sign.
    self.assertAllClose(
        tf.math.abs(tf.linalg.qr(mat).q)[..., :n_vecs],
        tf.math.abs(
            tfp.math.gram_schmidt(mat, n_vecs))[..., :n_vecs],
        rtol=1e-5)


if __name__ == '__main__':
  tf.test.main()
