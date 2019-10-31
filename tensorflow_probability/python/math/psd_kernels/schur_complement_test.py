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


import functools
import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


# A shape broadcasting fn
def broadcast_shapes(*shapes):
  def _broadcast_ab(a, b):
    if a == b or a == 1: return b
    if b == 1: return a
    raise ValueError("Can't broadcast {} with {}".format(a, b))
  def _broadcast_2(s1, s2):
    init_s1 = list(s1)
    init_s2 = list(s2)
    if len(s1) > len(s2):
      return _broadcast_2(s2, s1)
    # Now len(s1) <= len(s2)
    s1 = [1] * (len(s2) - len(s1)) + list(s1)
    try:
      return [_broadcast_ab(a, b) for a, b in zip(s1, s2)]
    except ValueError:
      raise ValueError(
          "Couldn't broadcast shapes {} and {}".format(init_s1, init_s2))
  return functools.reduce(_broadcast_2, shapes)


@test_util.test_all_tf_execution_regimes
class SchurComplementTest(test_util.TestCase):

  def testMismatchedFloatTypesAreBad(self):
    base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        np.float64(5.), np.float64(.2))

    # Should be OK
    tfp.math.psd_kernels.SchurComplement(
        base_kernel=base_kernel,  # float64
        fixed_inputs=np.random.uniform(-1., 1., [2, 1]))

    with self.assertRaises(TypeError):
      float32_inputs = np.random.uniform(
          -1., 1., [2, 1]).astype(np.float32)

      tfp.math.psd_kernels.SchurComplement(
          base_kernel=base_kernel, fixed_inputs=float32_inputs)

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

    base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        np.float64(5.), np.float64(.2), feature_ndims=feature_ndims)

    fixed_inputs = np.random.uniform(-1., 1., size=[num_obs] + shape)

    k = tfp.math.psd_kernels.SchurComplement(
        base_kernel=base_kernel, fixed_inputs=fixed_inputs)

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

  def testApplyShapesAreCorrect(self):
    for example_ndims in range(0, 4):
      # An integer generator.
      ints = itertools.count(start=2, step=1)
      feature_shape = [next(ints), next(ints)]

      x_batch_shape = [next(ints)]
      z_batch_shape = [next(ints), 1]
      num_x = [next(ints) for _ in range(example_ndims)]
      num_z = [next(ints)]

      x_shape = x_batch_shape + num_x + feature_shape
      z_shape = z_batch_shape + num_z + feature_shape

      x = np.ones(x_shape, np.float64)
      z = np.random.uniform(-1., 1., size=z_shape)

      base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
          amplitude=np.ones([next(ints), 1, 1], np.float64),
          feature_ndims=len(feature_shape))

      k = tfp.math.psd_kernels.SchurComplement(base_kernel, fixed_inputs=z)

      expected = broadcast_shapes(
          base_kernel.batch_shape, x_batch_shape, z_batch_shape) + num_x
      actual = k.apply(x, x, example_ndims=example_ndims).shape

      self.assertAllEqual(expected, actual)

  def testTensorShapesAreCorrect(self):
    for x1_example_ndims in range(0, 3):
      for x2_example_ndims in range(0, 3):
        # An integer generator.
        ints = itertools.count(start=2, step=1)
        feature_shape = [next(ints), next(ints)]

        x_batch_shape = [next(ints)]
        y_batch_shape = [next(ints), 1]
        z_batch_shape = [next(ints), 1, 1]

        num_x = [next(ints) for _ in range(x1_example_ndims)]
        num_y = [next(ints) for _ in range(x2_example_ndims)]
        num_z = [next(ints)]

        x_shape = x_batch_shape + num_x + feature_shape
        y_shape = y_batch_shape + num_y + feature_shape
        z_shape = z_batch_shape + num_z + feature_shape

        x = np.ones(x_shape, np.float64)
        y = np.ones(y_shape, np.float64)
        z = np.random.uniform(-1., 1., size=z_shape)

        base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=np.ones([next(ints), 1, 1, 1], np.float64),
            feature_ndims=len(feature_shape))

        k = tfp.math.psd_kernels.SchurComplement(base_kernel, fixed_inputs=z)

        expected = broadcast_shapes(
            base_kernel.batch_shape,
            x_batch_shape,
            y_batch_shape,
            z_batch_shape) + num_x + num_y

        mat = k.tensor(x, y,
                       x1_example_ndims=x1_example_ndims,
                       x2_example_ndims=x2_example_ndims)
        actual = mat.shape
        self.assertAllEqual(expected, actual)

  def testEmptyFixedInputs(self):
    base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(1., 1.)
    fixed_inputs = tf.ones([0, 2], np.float32)
    schur = tfp.math.psd_kernels.SchurComplement(base_kernel, fixed_inputs)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual(
        self.evaluate(base_kernel.matrix(x, y)),
        self.evaluate(schur.matrix(x, y)))

    # Test batch shapes
    base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic([1., 2.])
    fixed_inputs = tf.ones([0, 2], np.float32)
    schur = tfp.math.psd_kernels.SchurComplement(base_kernel, fixed_inputs)
    self.assertAllEqual([2], schur.batch_shape)
    self.assertAllEqual([2], self.evaluate(schur.batch_shape_tensor()))

  def testNoneFixedInputs(self):
    base_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(1., 1.)
    schur = tfp.math.psd_kernels.SchurComplement(base_kernel, fixed_inputs=None)

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
      schur_complement = tfp.math.psd_kernels.SchurComplement(
          tfp.math.psd_kernels.ExponentiatedQuadratic(np.float32(1)),
          fixed_inputs)

    # Should not throw an exception when the kernel doesn't get an explicit
    # dtype from its inputs.
    schur_complement = tfp.math.psd_kernels.SchurComplement(
        tfp.math.psd_kernels.ExponentiatedQuadratic(), fixed_inputs)
    schur_complement.matrix(fixed_inputs, fixed_inputs)

if __name__ == '__main__':
  tf.test.main()
