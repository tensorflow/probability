# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for no_pivot_ldl."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.math.no_pivot_ldl import no_pivot_ldl
from tensorflow_probability.python.experimental.math.no_pivot_ldl import simple_robustified_cholesky
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class NoPivotLDLTest(test_util.TestCase):

  def _randomDiag(self, n, batch_shape, low, high, forcemin=None, seed=42):
    np.random.seed(seed)
    shape = batch_shape + [n]
    diag = np.random.uniform(low, high, size=shape)
    if forcemin:
      assert forcemin < low
      diag = np.where(diag == np.min(diag, axis=-1)[..., np.newaxis],
                      forcemin, diag)
    return diag

  def _randomTril(self, n, batch_shape, seed=42):
    np.random.seed(seed)
    unit_tril = np.random.standard_normal(batch_shape + [n, n])
    unit_tril = np.tril(unit_tril)
    unit_tril[..., range(n), range(n)] = 1.
    return unit_tril

  def _randomSymmetricMatrix(self, n, batch_shape, low, high,
                             forcemin=None, seed=42):
    diag = self._randomDiag(n, batch_shape, low, high, forcemin, seed)
    unit_tril = self._randomTril(n, batch_shape, seed)
    return np.einsum('...ij,...j,...kj->...ik', unit_tril, diag, unit_tril)

  def testLDLRandomPSD(self):
    matrix = self._randomSymmetricMatrix(
        10, [2, 1, 3], 1e-6, 10., forcemin=0., seed=42)
    left, diag = self.evaluate(no_pivot_ldl(matrix))
    reconstruct = np.einsum('...ij,...j,...kj->...ik', left, diag, left)
    self.assertAllClose(matrix, reconstruct)

  def testLDLIndefinite(self):
    matrix = [[1., 2.], [2., 1.]]
    left, diag = self.evaluate(no_pivot_ldl(matrix))
    reconstruct = np.einsum('...ij,...j,...kj->...ik', left, diag, left)
    self.assertAllClose(matrix, reconstruct)

  def testSimpleIsCholeskyRandomPD(self):
    matrix = self._randomSymmetricMatrix(10, [2, 1, 3], 1e-6, 10., seed=42)
    chol, left = self.evaluate(
        (tf.linalg.cholesky(matrix),
         simple_robustified_cholesky(matrix)))
    self.assertAllClose(chol, left)

  def testSimpleIndefinite(self):
    matrix = [[1., 2.], [2., 1.]]
    left = self.evaluate(
        simple_robustified_cholesky(matrix, tol=.1))
    reconstruct = np.einsum('...ij,...kj->...ik', left, left)
    eigv, _ = self.evaluate(tf.linalg.eigh(reconstruct))
    self.assertAllTrue(eigv > 0.)


if __name__ == '__main__':
  tf.test.main()
