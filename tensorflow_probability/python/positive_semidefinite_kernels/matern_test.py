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
"""Tests for Matern kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_probability import positive_semidefinite_kernels as psd_kernels


class MaternOneHalfTest(tf.test.TestCase, parameterized.TestCase):

  def _matern_one_half(self, amplitude, length_scale, x, y):
    norm = np.sqrt(np.sum((x - y)**2)) / length_scale
    return amplitude**2 * np.exp(-norm)

  def testMismatchedFloatTypesAreBad(self):
    with self.assertRaises(ValueError):
      psd_kernels.MaternOneHalf(np.float32(1.), np.float64(1.))

  def testBatchShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(np.float32)
    k = psd_kernels.MaternOneHalf(amplitude, length_scale)
    self.assertAllEqual(tf.TensorShape([3, 3, 2]), k.batch_shape)
    self.assertAllEqual([3, 3, 2], self.evaluate(k.batch_shape_tensor()))

  @parameterized.parameters({
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 3
  }, {
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 4
  }, {
      'feature_ndims': 2,
      'dtype': np.float32,
      'dims': 2
  }, {
      'feature_ndims': 2,
      'dtype': np.float64,
      'dims': 3
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 2
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 3
  })
  def testValuesAreCorrect(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    length_scale = np.array(.2, dtype=dtype)

    np.random.seed(42)
    k = psd_kernels.MaternOneHalf(amplitude, length_scale, feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(dtype)
      y = np.random.uniform(-1, 1, size=shape).astype(dtype)
      self.assertAllClose(
          self.evaluate(k.apply(x, y)),
          self._matern_one_half(amplitude, length_scale, x, y))

  def testShapesAreCorrect(self):
    k = psd_kernels.MaternOneHalf(amplitude=1., length_scale=1.)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual(k.matrix(x, y).shape, [4, 5])
    self.assertAllEqual(
        k.matrix(tf.stack([x] * 2), tf.stack([y] * 2)).shape, [2, 4, 5])

    k = psd_kernels.MaternOneHalf(
        amplitude=np.ones([2, 1, 1], np.float32),
        length_scale=np.ones([1, 3, 1], np.float32))
    self.assertAllEqual(
        k.matrix(
            tf.stack([x] * 2),  # shape [2, 4, 3]
            tf.stack([y] * 2)  # shape [2, 5, 3]
        ).shape,
        [2, 3, 2, 4, 5])
    #    `--'  |  `--'
    #      |   |    `- matrix shape
    #      |   `- from input batch shapes
    #      `- from broadcasting kernel params

  def testValidateArgs(self):
    k = psd_kernels.MaternOneHalf(
        amplitude=-1., length_scale=-1., validate_args=True)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(k.amplitude)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(k.length_scale)

    # But `None`'s are ok
    k = psd_kernels.MaternOneHalf(
        amplitude=None, length_scale=None, validate_args=True)
    self.evaluate(k.apply([1.], [1.]))


class MaternThreeHalvesTest(tf.test.TestCase, parameterized.TestCase):

  def _matern_three_halves(self, amplitude, length_scale, x, y):
    norm = np.sqrt(3. * np.sum((x - y)**2)) / length_scale
    return amplitude**2 * (1 + norm) * np.exp(-norm)

  def testMismatchedFloatTypesAreBad(self):
    with self.assertRaises(ValueError):
      psd_kernels.MaternThreeHalves(np.float32(1.), np.float64(1.))

  def testBatchShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(np.float32)
    k = psd_kernels.MaternOneHalf(amplitude, length_scale)
    self.assertAllEqual(tf.TensorShape([3, 3, 2]), k.batch_shape)
    self.assertAllEqual([3, 3, 2], self.evaluate(k.batch_shape_tensor()))

  @parameterized.parameters({
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 3
  }, {
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 4
  }, {
      'feature_ndims': 2,
      'dtype': np.float32,
      'dims': 2
  }, {
      'feature_ndims': 2,
      'dtype': np.float64,
      'dims': 3
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 2
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 3
  })
  def testValuesAreCorrect(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    length_scale = np.array(.2, dtype=dtype)

    np.random.seed(42)
    k = psd_kernels.MaternThreeHalves(amplitude, length_scale, feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(dtype)
      y = np.random.uniform(-1, 1, size=shape).astype(dtype)
      self.assertAllClose(
          self.evaluate(k.apply(x, y)),
          self._matern_three_halves(amplitude, length_scale, x, y))

  def testShapesAreCorrect(self):
    k = psd_kernels.MaternThreeHalves(amplitude=1., length_scale=1.)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual(k.matrix(x, y).shape, [4, 5])
    self.assertAllEqual(
        k.matrix(tf.stack([x] * 2), tf.stack([y] * 2)).shape, [2, 4, 5])

    k = psd_kernels.MaternThreeHalves(
        amplitude=np.ones([2, 1, 1], np.float32),
        length_scale=np.ones([1, 3, 1], np.float32))
    self.assertAllEqual(
        k.matrix(
            tf.stack([x] * 2),  # shape [2, 4, 3]
            tf.stack([y] * 2)  # shape [2, 5, 3]
        ).shape,
        [2, 3, 2, 4, 5])
    #    `--'  |  `--'
    #      |   |    `- matrix shape
    #      |   `- from input batch shapes
    #      `- from broadcasting kernel params

  def testValidateArgs(self):
    k = psd_kernels.MaternThreeHalves(
        amplitude=-1., length_scale=-1., validate_args=True)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(k.amplitude)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(k.length_scale)

    # But `None`'s are ok
    k = psd_kernels.MaternThreeHalves(
        amplitude=None, length_scale=None, validate_args=True)
    self.evaluate(k.apply([1.], [1.]))


class MaternFiveHalvesTest(tf.test.TestCase, parameterized.TestCase):

  def _matern_five_halves(self, amplitude, length_scale, x, y):
    norm = np.sqrt(5. * np.sum((x - y)**2)) / length_scale
    return amplitude**2 * (1 + norm + norm**2 / 3.) * np.exp(-norm)

  def testMismatchedFloatTypesAreBad(self):
    with self.assertRaises(ValueError):
      psd_kernels.MaternFiveHalves(np.float32(1.), np.float64(1.))

  def testBatchShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(np.float32)
    k = psd_kernels.MaternOneHalf(amplitude, length_scale)
    self.assertAllEqual(tf.TensorShape([3, 3, 2]), k.batch_shape)
    self.assertAllEqual([3, 3, 2], self.evaluate(k.batch_shape_tensor()))

  @parameterized.parameters({
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 3
  }, {
      'feature_ndims': 1,
      'dtype': np.float32,
      'dims': 4
  }, {
      'feature_ndims': 2,
      'dtype': np.float32,
      'dims': 2
  }, {
      'feature_ndims': 2,
      'dtype': np.float64,
      'dims': 3
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 2
  }, {
      'feature_ndims': 3,
      'dtype': np.float64,
      'dims': 3
  })
  def testValuesAreCorrect(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    length_scale = np.array(.2, dtype=dtype)

    np.random.seed(42)
    k = psd_kernels.MaternFiveHalves(amplitude, length_scale, feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-3, 3, size=shape).astype(dtype)
      y = np.random.uniform(-3, 3, size=shape).astype(dtype)
      self.assertAllClose(
          self.evaluate(k.apply(x, y)),
          self._matern_five_halves(amplitude, length_scale, x, y))

  def testShapesAreCorrect(self):
    k = psd_kernels.MaternFiveHalves(amplitude=1., length_scale=1.)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual(k.matrix(x, y).shape, [4, 5])
    self.assertAllEqual(
        k.matrix(tf.stack([x] * 2), tf.stack([y] * 2)).shape, [2, 4, 5])

    k = psd_kernels.MaternFiveHalves(
        amplitude=np.ones([2, 1, 1], np.float32),
        length_scale=np.ones([1, 3, 1], np.float32))
    self.assertAllEqual(
        k.matrix(
            tf.stack([x] * 2),  # shape [2, 4, 3]
            tf.stack([y] * 2)  # shape [2, 5, 3]
        ).shape,
        [2, 3, 2, 4, 5])
    #    `--'  |  `--'
    #      |   |    `- matrix shape
    #      |   `- from input batch shapes
    #      `- from broadcasting kernel params

  def testValidateArgs(self):
    k = psd_kernels.MaternFiveHalves(
        amplitude=-1., length_scale=-1., validate_args=True)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(k.amplitude)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(k.length_scale)

    # But `None`'s are ok
    k = psd_kernels.MaternFiveHalves(
        amplitude=None, length_scale=None, validate_args=True)
    self.evaluate(k.apply([1.], [1.]))


class MaternGradsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters({
      'matern_class': psd_kernels.MaternOneHalf
  }, {
      'matern_class': psd_kernels.MaternThreeHalves
  }, {
      'matern_class': psd_kernels.MaternFiveHalves
  })
  def testGradsAtIdenticalInputsAreZeroNotNaN(self, matern_class):
    k = matern_class()
    x = tf.constant(np.arange(3 * 5, dtype=np.float32).reshape(3, 5))

    kernel_values = k.apply(x, x)
    grads = [tf.gradients(kernel_values[i], x)[0] for i in range(3)]

    self.assertAllEqual(
        [self.evaluate(grad) for grad in grads],
        [np.zeros(grad.shape.as_list(), np.float32) for grad in grads])

if __name__ == '__main__':
  tf.test.main()
