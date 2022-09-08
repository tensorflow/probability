# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for Unitary linop."""


# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.linalg import linear_operator_unitary
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class LinearOperatorUnitaryTest(test_util.TestCase):
  """Tests for linear_operator_unitary.LinearOperatorUnitary."""

  def test_unitary_linop_raises_non_square(self):
    with self.assertRaisesRegex(ValueError, 'Expected square matrix'):
      x = tf.random.stateless_normal(
          [12, 13], seed=test_util.test_seed(sampler_type='stateless'))
      linear_operator_unitary.LinearOperatorUnitary(x)

  def test_shape(self):
    x = tf.random.stateless_normal(
        [5, 13, 13], seed=test_util.test_seed(sampler_type='stateless'))
    q, _ = tf.linalg.qr(x)
    operator = linear_operator_unitary.LinearOperatorUnitary(q)
    self.assertEqual([5], operator.batch_shape)

  def test_log_det(self):
    x = tf.random.stateless_normal(
        [5, 13, 13], seed=test_util.test_seed(sampler_type='stateless'))
    q, _ = tf.linalg.qr(x)
    operator = linear_operator_unitary.LinearOperatorUnitary(q)
    true_logdet, expected_logdet = self.evaluate([
        tf.linalg.slogdet(q)[1], operator.log_abs_determinant()])
    self.assertAllClose(expected_logdet, true_logdet)

  def test_matmul(self):
    x = tf.random.stateless_normal(
        [7, 1, 13, 13], seed=test_util.test_seed(sampler_type='stateless'))
    q, _ = tf.linalg.qr(x)
    operator = linear_operator_unitary.LinearOperatorUnitary(q)
    y = tf.random.stateless_normal(
        [13, 13], seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllClose(
        self.evaluate(tf.linalg.matmul(q, y)),
        self.evaluate(operator.matmul(y)))

    self.assertAllClose(
        self.evaluate(tf.linalg.matmul(q, y, adjoint_b=True)),
        self.evaluate(operator.matmul(y, adjoint_arg=True)))

    self.assertAllClose(
        self.evaluate(tf.linalg.matmul(q, y, adjoint_a=True)),
        self.evaluate(operator.matmul(y, adjoint=True)))

    self.assertAllClose(
        self.evaluate(tf.linalg.matmul(q, y, adjoint_a=True, adjoint_b=True)),
        self.evaluate(operator.matmul(y, adjoint=True, adjoint_arg=True)))

  def test_solve(self):
    x = tf.random.stateless_normal(
        [7, 1, 13, 13], seed=test_util.test_seed(sampler_type='stateless'))
    q, _ = tf.linalg.qr(x)
    operator = linear_operator_unitary.LinearOperatorUnitary(q)
    y = tf.random.stateless_normal(
        [13, 3], seed=test_util.test_seed(sampler_type='stateless'))
    broadcasted_y = tf.broadcast_to(y, [7, 1, 13, 3])
    self.assertAllClose(
        self.evaluate(tf.linalg.solve(q, broadcasted_y)),
        self.evaluate(operator.solve(y)),
        atol=1e-5, rtol=1e-5)

    self.assertAllClose(
        self.evaluate(tf.linalg.solve(q, broadcasted_y, adjoint=True)),
        self.evaluate(operator.solve(y, adjoint=True)),
        atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  test_util.main()
