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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


marginal_fns = tfp.experimental.distributions.marginal_fns


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
