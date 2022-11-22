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
"""Tests for marginal_fns."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.distributions import marginal_fns
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


def _value_and_grads(f, x, has_aux=False):
  val, grad = gradient.value_and_gradient(f, x, has_aux=has_aux)
  _, grad_of_grad = gradient.value_and_gradient(
      lambda t: gradient.value_and_gradient(f, t)[1], x)
  return val, grad, grad_of_grad


@test_util.test_all_tf_execution_regimes
class MarginalFnsTest(test_util.TestCase):

  def testNoBackoff(self):
    matrix = np.array([[2., 1.], [1., 2.]])
    backoff = marginal_fns.make_backoff_cholesky(lambda x: x)
    test, ans = self.evaluate((backoff(matrix), tf.linalg.cholesky(matrix)))
    self.assertAllClose(test, ans)

  def testBackoff(self):
    matrix = np.array([[1., 2.], [2., 1.]])
    backoff = marginal_fns.make_backoff_cholesky(tf.convert_to_tensor)
    test = self.evaluate(backoff(matrix))
    self.assertAllClose(test, matrix)

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='No gradients available in numpy.')
  def testRetryingCholeskyWithBatchAndXLA(self):
    matrix = tf.convert_to_tensor([
        [[1.0, 0.5], [0.5, 1.0]],
        [[1e-2, 1.002e-2], [1.002e-2, 1e-2]],
        [[3.0, 3.03], [3.03, 3.031]]
    ])

    expected_shift = tf.convert_to_tensor([
        [[0.0, 0.0], [0.0, 0.0]],
        [[1e-4, 0.0], [0.0, 1e-4]],
        [[0.1, 0.0], [0.0, 0.1]]
    ])
    expected, expected_grad, expected_grad_of_grad = self.evaluate(
        _value_and_grads(lambda x: tf.linalg.cholesky(x + expected_shift),
                         matrix))

    (res, shift), grad, grad_of_grad = self.evaluate(
        _value_and_grads(
            lambda x: marginal_fns.retrying_cholesky(x, max_iters=6),
            matrix, has_aux=True))
    self.assertAllEqual(expected, res)
    self.assertAllClose(expected_shift[..., 0, 0], shift)
    self.assertAllEqual(expected_grad, grad)
    self.assertAllEqual(expected_grad_of_grad, grad_of_grad)

    # Test value and gradients of XLA-compiled `retrying_cholesky`.
    xla_retrying_cholesky = tf.function(
        lambda x: marginal_fns.retrying_cholesky(x, max_iters=6),
        autograph=False, jit_compile=True)
    (res, shift), grad, grad_of_grad = self.evaluate(
        _value_and_grads(xla_retrying_cholesky, matrix, has_aux=True))
    self.assertAllClose(expected, res)
    self.assertAllClose(expected_shift[..., 0, 0], shift)
    self.assertAllClose(expected_grad, grad)
    self.assertAllClose(expected_grad_of_grad, grad_of_grad, rtol=1e-3)

    # Test XLA-compilation of `retrying_cholesky` and its gradients.
    @tf.function(autograph=False, jit_compile=True)
    def xla_retrying_cholesky_with_grads(x):
      return _value_and_grads(
          lambda x: marginal_fns.retrying_cholesky(x, max_iters=6),
          x, has_aux=True)
    (res, shift), grad, grad_of_grad = self.evaluate(
        xla_retrying_cholesky_with_grads(matrix))
    self.assertAllClose(expected, res)
    self.assertAllClose(expected_shift[..., 0, 0], shift)
    self.assertAllClose(expected_grad, grad)
    self.assertAllClose(expected_grad_of_grad, grad_of_grad, rtol=1e-3)

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='No gradients available in numpy.')
  def testRetryingCholeskyFloat64(self):
    matrix = tf.convert_to_tensor(
        [[2., 2., 0.], [2., 2., 2e-5], [0., 2e-5, 0.5]], dtype=tf.float64)

    expected_shift = tf.eye(3, dtype=tf.float64) * 1e-9
    expected, expected_grad, expected_grad_of_grad = self.evaluate(
        _value_and_grads(lambda x: tf.linalg.cholesky(x + expected_shift),
                         matrix))

    (res, shift), grad, grad_of_grad = self.evaluate(
        _value_and_grads(
            marginal_fns.retrying_cholesky, matrix, has_aux=True))
    self.assertAllEqual(expected, res)
    self.assertAllClose(expected_shift[..., 0, 0], shift)
    self.assertAllEqual(expected_grad, grad)
    self.assertAllEqual(expected_grad_of_grad, grad_of_grad)

    expected, expected_grad, expected_grad_of_grad = self.evaluate(
        _value_and_grads(lambda x: tf.linalg.cholesky(x + expected_shift),
                         matrix))

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='No gradients available in numpy.')
  def testRetryingCholeskyFailures(self):
    matrix = tf.convert_to_tensor([
        [[1.0, 0.5], [0.5, 1.0]],
        [[1e4, 1e4+1], [1e4+1, 1e4]],
        [[3.0, 3.03], [3.03, 3.031]]
    ])

    expected_eps = tf.convert_to_tensor([
        [[0.0, 0.0], [0.0, 0.0]],
        [[np.nan, np.nan], [np.nan, np.nan]],
        [[0.1, 0.0], [0.0, 0.1]]
    ])
    expected, expected_grad, expected_grad_of_grad = self.evaluate(
        _value_and_grads(lambda x: tf.linalg.cholesky(x + expected_eps),
                         matrix))

    (res, _), grad, grad_of_grad = self.evaluate(
        _value_and_grads(
            lambda x: marginal_fns.retrying_cholesky(x, max_iters=6),
            matrix, has_aux=True))

    self.assertAllEqual([expected[0], expected[2]], [res[0], res[2]])
    self.assertAllEqual([expected_grad[0], expected_grad[2]],
                        [grad[0], grad[2]])
    self.assertAllEqual([expected_grad_of_grad[0], expected_grad_of_grad[2]],
                        [grad_of_grad[0], grad_of_grad[2]])

    # Check that the lower-triangular part of `res[1]` is NaN.
    for i in range(res[1].shape[0]):
      self.assertAllNan(res[1, i, :i+1])

    self.assertAllNan(grad[1])
    self.assertAllNan(grad_of_grad[1])

  def testCholeskyLikeMarginalFn(self):
    fake_input = np.array([[1., 2.], [2., 1.]], dtype=np.float32)
    fake_output = np.array([[2., 1.], [1., 2.]], dtype=np.float32)
    fake_chol = np.linalg.cholesky(fake_output)
    marginal_fn = marginal_fns.make_cholesky_like_marginal_fn(
        lambda x: tf.convert_to_tensor(fake_chol))
    test = self.evaluate(marginal_fn(tf.zeros(2), fake_input).covariance())
    self.assertAllClose(test, fake_output)

  def testEigHMarginalFn(self):
    tol = 1e-5
    v1 = np.sqrt([.5, .5])
    v2 = np.array(v1)
    v2[0] = -v2[0]
    vectors = np.array([v1, v2], dtype=np.float32).T
    values = np.array([-1., .5], dtype=np.float32)
    safe_values = np.where(values < tol, tol, values)
    input_matrix = np.dot(vectors, np.dot(np.diag(values), vectors.T))
    output_matrix = np.dot(vectors, np.dot(np.diag(safe_values), vectors.T))

    marginal_fn = marginal_fns.make_eigh_marginal_fn(tol=tol)
    test = self.evaluate(marginal_fn(tf.zeros(2), input_matrix).covariance())
    self.assertAllClose(test, output_matrix)


if __name__ == '__main__':
  test_util.main()
