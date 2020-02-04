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
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ExpSinSquaredTest(test_util.TestCase):

  def testMismatchedFloatTypesAreBad(self):
    tfp.math.psd_kernels.ExpSinSquared(1, 1)  # Should be OK (float32 fallback).
    tfp.math.psd_kernels.ExpSinSquared(1, np.float64(1))  # Should be OK.
    with self.assertRaises(TypeError):
      tfp.math.psd_kernels.ExpSinSquared(1, np.float64(1),
                                         np.float32(1))  # Should fail.

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
    k = tfp.math.psd_kernels.ExpSinSquared(
        amplitude=amplitude,
        length_scale=length_scale,
        period=period,
        feature_ndims=feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-3., 3., size=shape).astype(dtype)
      y = np.random.uniform(-3., 3., size=shape).astype(dtype)
      self.assertAllClose(
          self._exp_sin_squared_kernel(amplitude, length_scale, period, x, y),
          self.evaluate(k.apply(x, y)),
          rtol=1e-4)

  def testNoneShapes(self):
    k = tfp.math.psd_kernels.ExpSinSquared(
        amplitude=np.reshape([1.] * 6, [3, 2]))
    self.assertAllEqual((3, 2), k.batch_shape)

  def testShapesAreCorrect(self):
    k = tfp.math.psd_kernels.ExpSinSquared(
        amplitude=1., length_scale=1., period=3.)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual([4, 5], k.matrix(x, y).shape)
    self.assertAllEqual(
        [2, 4, 5],
        k.matrix(tf.stack([x]*2), tf.stack([y]*2)).shape)

    k = tfp.math.psd_kernels.ExpSinSquared(
        amplitude=np.ones([2, 1, 1], np.float32),
        length_scale=np.ones([1, 3, 1], np.float32),
        period=np.ones([2, 1, 1, 1], np.float32))
    self.assertAllEqual(
        [2, 2, 3, 2, 4, 5],
        #`-----'  |  `--'
        # |       |    `- matrix shape
        # |       `- from input batch shapes
        # `- from broadcasting kernel params
        k.matrix(
            tf.stack([x]*2),  # shape [2, 4, 3]
            tf.stack([y]*2)   # shape [2, 5, 3]
        ).shape)

  def testValidateArgs(self):
    with self.assertRaisesOpError('must be positive'):
      k = tfp.math.psd_kernels.ExpSinSquared(
          amplitude=-1.,
          length_scale=-1.,
          period=-1.,
          validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

    # But `None`'s are ok
    k = tfp.math.psd_kernels.ExpSinSquared(
        amplitude=None, length_scale=None, period=None, validate_args=True)
    self.evaluate(k.apply([1.], [1.]))

  @test_util.jax_disable_variable_test
  def testValidateVariableArgs(self):
    amplitude = tf.Variable(1.)
    length_scale = tf.Variable(1.)
    period = tf.Variable(1.)
    k = tfp.math.psd_kernels.ExpSinSquared(
        amplitude=amplitude,
        length_scale=length_scale,
        period=period,
        validate_args=True)
    self.evaluate([v.initializer for v in k.variables])
    self.evaluate(k.apply([1.], [1.]))

    with self.assertRaisesOpError('must be positive'):
      with tf.control_dependencies([period.assign(-7.3)]):
        self.evaluate(k.apply([1.], [1.]))


if __name__ == '__main__':
  tf.test.main()
