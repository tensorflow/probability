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
"""Tests for ExpSinSquared."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_probability import positive_semidefinite_kernels as psd_kernels


class ExpSinSquaredTest(tf.test.TestCase, parameterized.TestCase):

  def testMismatchedFloatTypesAreBad(self):
    psd_kernels.ExpSinSquared(1, 1)  # Should be OK (float32 fallback).
    psd_kernels.ExpSinSquared(1, np.float64(1))  # Should be OK.
    with self.assertRaises(TypeError):
      psd_kernels.ExpSinSquared(1, np.float64(1), np.float32(1))  # Should fail.

  def _exp_sin_squared_kernel(self, amplitude, length_scale, period, x, y):
    norm = np.abs(x - y)
    log_kernel = np.sum(-2. * np.sin(np.pi / period * norm) ** 2)
    log_kernel /= length_scale ** 2
    return amplitude ** 2 * np.exp(log_kernel)

  @parameterized.parameters(
      {'feature_ndims': 1, 'dtype': np.float32, 'dims': 3},
      {'feature_ndims': 1, 'dtype': np.float32, 'dims': 4},
      {'feature_ndims': 2, 'dtype': np.float32, 'dims': 2},
      {'feature_ndims': 2, 'dtype': np.float64, 'dims': 3},
      {'feature_ndims': 3, 'dtype': np.float64, 'dims': 2},
      {'feature_ndims': 3, 'dtype': np.float64, 'dims': 3})
  def testValuesAreCorrect(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    period = np.array(1., dtype=dtype)
    length_scale = np.array(.2, dtype=dtype)

    np.random.seed(42)
    k = psd_kernels.ExpSinSquared(
        amplitude=amplitude,
        length_scale=length_scale,
        period=period,
        feature_ndims=feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-3., 3., size=shape).astype(dtype)
      y = np.random.uniform(-3., 3., size=shape).astype(dtype)
      self.assertAllClose(
          self.evaluate(k.apply(x, y)),
          self._exp_sin_squared_kernel(amplitude, length_scale, period, x, y),
          rtol=1e-4)

  def testNoneShapes(self):
    k = psd_kernels.ExpSinSquared(amplitude=np.reshape([1.] * 6, [3, 2]))
    self.assertEqual([3, 2], k.batch_shape.as_list())

  def testShapesAreCorrect(self):
    k = psd_kernels.ExpSinSquared(amplitude=1., length_scale=1., period=3.)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual(k.matrix(x, y).shape, [4, 5])
    self.assertAllEqual(
        k.matrix(tf.stack([x]*2), tf.stack([y]*2)).shape,
        [2, 4, 5])

    k = psd_kernels.ExpSinSquared(
        amplitude=np.ones([2, 1, 1], np.float32),
        length_scale=np.ones([1, 3, 1], np.float32),
        period=np.ones([2, 1, 1, 1], np.float32))
    self.assertAllEqual(
        k.matrix(
            tf.stack([x]*2),  # shape [2, 4, 3]
            tf.stack([y]*2)   # shape [2, 5, 3]
        ).shape, [2, 2, 3, 2, 4, 5])
    #             `-----'  |  `--'
    #              |       |    `- matrix shape
    #              |       `- from input batch shapes
    #              `- from broadcasting kernel params

  def testValidateArgs(self):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      k = psd_kernels.ExpSinSquared(
          amplitude=-1., length_scale=-1., period=-1., validate_args=True)
      self.evaluate(k.amplitude)

    if not tf.executing_eagerly():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(k.length_scale)

      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(k.period)

    # But `None`'s are ok
    k = psd_kernels.ExpSinSquared(
        amplitude=None, length_scale=None, period=None, validate_args=True)
    self.evaluate(k.apply([1.], [1.]))


if __name__ == '__main__':
  tf.test.main()
