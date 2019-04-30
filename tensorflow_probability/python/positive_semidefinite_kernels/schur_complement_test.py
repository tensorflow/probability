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
"""Tests for SchurComplement."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_probability import positive_semidefinite_kernels as tfpk
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class SchurComplementTest(tf.test.TestCase, parameterized.TestCase):

  def testMismatchedFloatTypesAreBad(self):
    base_kernel = tfpk.ExponentiatedQuadratic(
        np.float64(5.), np.float64(.2))

    # Should be OK
    tfpk.SchurComplement(
        base_kernel=base_kernel,  # float64
        fixed_inputs=np.random.uniform(-1., 1., [2, 1]))

    with self.assertRaises(TypeError):
      float32_inputs = np.random.uniform(
          -1., 1., [2, 1]).astype(np.float32)

      tfpk.SchurComplement(
          base_kernel=base_kernel,
          fixed_inputs=float32_inputs)

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testValuesAreCorrect(self, feature_ndims, dims):
    np.random.seed(42)
    num_obs = 5
    num_x = 3
    num_y = 3

    shape = [dims] * feature_ndims

    base_kernel = tfpk.ExponentiatedQuadratic(
        np.float64(5.), np.float64(.2), feature_ndims=feature_ndims)

    fixed_inputs = np.random.uniform(-1., 1., size=[num_obs] + shape)

    k = tfpk.SchurComplement(
        base_kernel=base_kernel,
        fixed_inputs=fixed_inputs)

    k_obs = self.evaluate(base_kernel.matrix(fixed_inputs, fixed_inputs))

    k_obs_chol_linop = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(k_obs))
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=[num_x] + shape)
      y = np.random.uniform(-1, 1, size=[num_y] + shape)

      k_x_y = self.evaluate(base_kernel.apply(x, y))
      k_x_obs = self.evaluate(base_kernel.matrix(x, fixed_inputs))
      k_obs_y = self.evaluate(base_kernel.matrix(y, fixed_inputs))

      k_x_obs = np.expand_dims(k_x_obs, -2)
      k_obs_y = np.expand_dims(k_obs_y, -1)

      k_obs_inv_k_obs_y = self.evaluate(
          k_obs_chol_linop.solve(
              k_obs_chol_linop.solve(k_obs_y),
              adjoint=True))

      cov_dec = np.einsum('ijk,ikl->ijl', k_x_obs, k_obs_inv_k_obs_y)
      cov_dec = cov_dec[..., 0, 0]  # np.squeeze didn't like list of axes
      expected = k_x_y - cov_dec
      self.assertAllClose(expected, self.evaluate(k.apply(x, y)))

  def testShapesAreCorrect(self):
    base_kernel = tfpk.ExponentiatedQuadratic(np.float64(1.), np.float64(1.))
    fixed_inputs = np.random.uniform(-1., 1., size=[2, 3])
    k = tfpk.SchurComplement(base_kernel, fixed_inputs)

    x = np.ones([4, 3], np.float64)
    y = np.ones([5, 3], np.float64)

    self.assertAllEqual([4], k.apply(x, x).shape)
    self.assertAllEqual([5], k.apply(y, y).shape)
    self.assertAllEqual([4, 5], k.matrix(x, y).shape)
    self.assertAllEqual(
        [2, 4, 5], k.matrix(tf.stack([x]*2), tf.stack([y]*2)).shape)

    base_kernel = tfpk.ExponentiatedQuadratic(
        amplitude=np.ones([2, 1, 1], np.float64),
        length_scale=np.ones([1, 3, 1], np.float64))
    # Batch these at the outermost shape position
    fixed_inputs = np.random.uniform(-1., 1., size=[7, 1, 1, 1, 2, 3])
    k = tfpk.SchurComplement(base_kernel, fixed_inputs)
    self.assertAllEqual(
        [7, 2, 3, 4],
        k.apply(x, x).shape)
    self.assertAllEqual(
        [7, 2, 3, 2, 4, 5],
        #|  `--'  |  `--'
        #|    |   |    `- matrix shape
        #|    |   `- from input batch shapes
        #|    `- from broadcasting kernel params
        #`- from batch of obs index points
        k.matrix(
            tf.stack([x]*2),  # shape [2, 4, 3]
            tf.stack([y]*2)   # shape [2, 5, 3]
        ).shape)

  def testEmptyFixedInputs(self):
    base_kernel = tfpk.ExponentiatedQuadratic(1., 1.)
    fixed_inputs = tf.ones([0, 2], np.float32)
    schur = tfpk.SchurComplement(base_kernel, fixed_inputs)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual(
        self.evaluate(base_kernel.matrix(x, y)),
        self.evaluate(schur.matrix(x, y)))

  def testNoneFixedInputs(self):
    base_kernel = tfpk.ExponentiatedQuadratic(1., 1.)
    schur = tfpk.SchurComplement(base_kernel, fixed_inputs=None)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual(
        self.evaluate(base_kernel.matrix(x, y)),
        self.evaluate(schur.matrix(x, y)))

  def testBaseKernelNoneDtype(self):
    # Test that we don't have problems when base_kernel has no explicit dtype
    # (ie, params are all None), but fixed_inputs has a different dtype than the
    # "common_dtype" default value of np.float32.
    fixed_inputs = np.arange(3, dtype=np.float64).reshape([3, 1])

    # Should raise when there's an explicit mismatch.
    with self.assertRaises(TypeError):
      schur_complement = tfpk.SchurComplement(
          tfpk.ExponentiatedQuadratic(np.float32(1)),
          fixed_inputs)

    # Should not throw an exception when the kernel doesn't get an explicit
    # dtype from its inputs.
    schur_complement = tfpk.SchurComplement(
        tfpk.ExponentiatedQuadratic(), fixed_inputs)
    schur_complement.matrix(fixed_inputs, fixed_inputs)

if __name__ == '__main__':
  tf.test.main()
