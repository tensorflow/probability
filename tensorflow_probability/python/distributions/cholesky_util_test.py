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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import cholesky_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class CholeskyUtilsTest(test_util.TestCase):

  def testJitterFn(self):
    cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn(jitter=0.)

    x = tf.random.normal(shape=[2, 4, 4], seed=test_util.test_seed())
    x = tf.linalg.matmul(x, x, transpose_b=True)
    actual_chol, expected_chol = self.evaluate([
        cholesky_fn(x), tf.linalg.cholesky(x)])
    self.assertAllClose(expected_chol, actual_chol)

    cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn(jitter=1.)
    x = 3. * tf.linalg.eye(3)
    self.assertAllClose(
        self.evaluate(2. * tf.linalg.eye(3)),
        self.evaluate(cholesky_fn(x)))

  def testCholeskyFnType(self):
    identity = tf.linalg.LinearOperatorIdentity(3)
    self.assertIsInstance(
        cholesky_util.cholesky_from_fn(identity, tf.linalg.cholesky),
        tf.linalg.LinearOperatorIdentity)

    diag = tf.linalg.LinearOperatorDiag([3., 5., 7.])
    self.assertIsInstance(
        cholesky_util.cholesky_from_fn(diag, tf.linalg.cholesky),
        tf.linalg.LinearOperatorDiag)

    kron = tf.linalg.LinearOperatorKronecker([identity, diag])
    self.assertIsInstance(
        cholesky_util.cholesky_from_fn(kron, tf.linalg.cholesky),
        tf.linalg.LinearOperatorKronecker)

    block_diag = tf.linalg.LinearOperatorBlockDiag([identity, diag])
    self.assertIsInstance(
        cholesky_util.cholesky_from_fn(block_diag, tf.linalg.cholesky),
        tf.linalg.LinearOperatorBlockDiag)


if __name__ == '__main__':
  test_util.main()
