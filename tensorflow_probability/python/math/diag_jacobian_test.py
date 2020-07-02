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
"""Tests for Jacobian computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class JacobianTest(test_util.TestCase):

  def testJacobianDiagonal3DListInput(self):
    """Tests that the diagonal of the Jacobian matrix computes correctly."""

    dtype = np.float32
    true_mean = dtype([0, 0, 0])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 2, 0.25], [0.25, 0.25, 3]])
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a list of tensors `x` and `y`.
    # Then the target function is defined as follows:
    def target_fn(x, y):
      # Stack the input tensors together
      z = tf.concat([x, y], axis=-1) - true_mean
      return target.log_prob(z)

    sample_shape = [3, 5]
    state = [tf.ones(sample_shape + [2], dtype=dtype),
             tf.ones(sample_shape + [1], dtype=dtype)]
    fn_val, grads = tfp.math.value_and_gradient(target_fn, state)
    grad_fn = lambda *args: tfp.math.value_and_gradient(target_fn, args)[1]

    _, diag_jacobian_shape_passed = tfp.math.diag_jacobian(
        xs=state, ys=grads, fn=grad_fn, sample_shape=tf.shape(fn_val))
    _, diag_jacobian_shape_none = tfp.math.diag_jacobian(
        xs=state, ys=grads, fn=grad_fn)

    true_diag_jacobian_1 = np.zeros(sample_shape + [2])
    true_diag_jacobian_1[..., 0] = -1.05
    true_diag_jacobian_1[..., 1] = -0.52

    true_diag_jacobian_2 = -0.34 * np.ones(sample_shape + [1])

    self.assertAllClose(self.evaluate(diag_jacobian_shape_passed[0]),
                        true_diag_jacobian_1,
                        atol=0.01, rtol=0.01)
    self.assertAllClose(self.evaluate(diag_jacobian_shape_none[0]),
                        true_diag_jacobian_1,
                        atol=0.01, rtol=0.01)

    self.assertAllClose(self.evaluate(diag_jacobian_shape_passed[1]),
                        true_diag_jacobian_2,
                        atol=0.01, rtol=0.01)
    self.assertAllClose(self.evaluate(diag_jacobian_shape_none[1]),
                        true_diag_jacobian_2,
                        atol=0.01, rtol=0.01)

  def testJacobianDiagonal4D(self):
    """Tests that the diagonal of the Jacobian matrix computes correctly."""

    dtype = np.float32
    true_mean = dtype([0, 0, 0, 0])
    true_cov = dtype([[1, 0.25, 0.25, 0.25], [0.25, 2, 0.25, 0.25],
                      [0.25, 0.25, 3, 0.25], [0.25, 0.25, 0.25, 4]])
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a 2x2 matrix of sample_shape = [5, 3]:
    sample_shape = [5, 3]
    def target_fn(x):
      z = tf.reshape(x, sample_shape + [4])
      return target.log_prob(z)

    state = [tf.ones(sample_shape + [2, 2], dtype=dtype)]
    fn_val, grads = tfp.math.value_and_gradient(target_fn, state)
    grad_fn = lambda *args: tfp.math.value_and_gradient(target_fn, args)[1]

    _, diag_jacobian_shape_passed = tfp.math.diag_jacobian(
        xs=state, ys=grads, fn=grad_fn, sample_shape=tf.shape(fn_val))
    _, diag_jacobian_shape_none = tfp.math.diag_jacobian(
        xs=state, ys=grads, fn=grad_fn)

    true_diag_jacobian = np.zeros(sample_shape + [2, 2])
    true_diag_jacobian[..., 0, 0] = -1.06
    true_diag_jacobian[..., 0, 1] = -0.52
    true_diag_jacobian[..., 1, 0] = -0.34
    true_diag_jacobian[..., 1, 1] = -0.26

    self.assertAllClose(self.evaluate(diag_jacobian_shape_passed[0]),
                        true_diag_jacobian,
                        atol=0.01, rtol=0.01)
    self.assertAllClose(self.evaluate(diag_jacobian_shape_none[0]),
                        true_diag_jacobian,
                        atol=0.01, rtol=0.01)

if __name__ == '__main__':
  tf.test.main()
