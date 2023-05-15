# Copyright 2023 The TensorFlow Probability Authors.
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

"""Tests for Row Block."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.linalg import linear_operator_row_block as lorb
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class LinearOperatorRowBlockTest(test_util.TestCase):

  def testShape(self):
    a = tf.linalg.LinearOperatorDiag([1., 2., 3.])
    b = tf.linalg.LinearOperatorFullMatrix([[2., 1.], [3., 2.], [4., 5.]])
    block = lorb.LinearOperatorRowBlock([a, b])
    self.assertAllEqual(block.shape, [3, 5])
    self.assertAllEqual(block.shape_tensor(), [3, 5])

  def testMatmul(self):
    a = tf.linalg.LinearOperatorDiag([1., 2., 3.])
    b = tf.linalg.LinearOperatorFullMatrix([[2., 1.], [3., 2.], [4., 5.]])
    block = lorb.LinearOperatorRowBlock([a, b])
    c = np.array([
        [1., -1.],
        [2., -2.],
        [3., -3.],
        [4., -4.],
        [5., -5.]
    ]).astype(np.float32)
    expected = self.evaluate(block.matmul(c))
    actual = self.evaluate(a.matmul(c[:3, ...]) + b.matmul(c[3:, ...]))
    self.assertAllClose(expected, actual)

  def testMatmulAdjoint(self):
    a = tf.linalg.LinearOperatorDiag([1., 2., 3.])
    b = tf.linalg.LinearOperatorFullMatrix([[2., 1.], [3., 2.], [4., 5.]])
    block = lorb.LinearOperatorRowBlock([a, b])
    c = np.array([
        [1., -1.],
        [2., -2.],
        [3., -3.],
    ]).astype(np.float32)
    expected = self.evaluate(block.matmul(c, adjoint=True))
    self.assertAllClose(
        expected[:3, ...], self.evaluate(a.matmul(c, adjoint=True)))
    self.assertAllClose(
        expected[3:, ...], self.evaluate(b.matmul(c, adjoint=True)))


if __name__ == '__main__':
  test_util.main()
