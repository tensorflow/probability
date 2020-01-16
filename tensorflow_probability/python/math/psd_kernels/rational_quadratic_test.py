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
"""Tests for RationalQuadratic kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class RationalQuadraticTest(test_util.TestCase):

  def _rational_quadratic(
      self, amplitude, length_scale, scale_mixture_rate, x, y):
    return (amplitude ** 2) * (1. + np.sum((x - y) ** 2) / (
        2 * scale_mixture_rate * length_scale ** 2)) ** (-scale_mixture_rate)

  def testMismatchedFloatTypesAreBad(self):
    with self.assertRaises(TypeError):
      tfp.math.psd_kernels.RationalQuadratic(np.float32(1.), np.float64(1.))

  def testBatchShape(self):
    amplitude = np.random.uniform(2, 3., size=[3, 1, 2]).astype(np.float32)
    length_scale = np.random.uniform(2, 3., size=[1, 3, 1]).astype(np.float32)
    scale_mixture_rate = np.random.uniform(
        2, 3., size=[3, 1, 1]).astype(np.float32)
    k = tfp.math.psd_kernels.RationalQuadratic(amplitude, length_scale,
                                               scale_mixture_rate)
    self.assertAllEqual(tf.TensorShape([3, 3, 2]), k.batch_shape)
    self.assertAllEqual([3, 3, 2], self.evaluate(k.batch_shape_tensor()))

  @parameterized.parameters(
      {'feature_ndims': 1, 'dtype': np.float32, 'dims': 3},
      {'feature_ndims': 1, 'dtype': np.float32, 'dims': 4},
      {'feature_ndims': 2, 'dtype': np.float32, 'dims': 2},
      {'feature_ndims': 2, 'dtype': np.float64, 'dims': 3},
      {'feature_ndims': 3, 'dtype': np.float64, 'dims': 2},
      {'feature_ndims': 3, 'dtype': np.float64, 'dims': 3})
  def testValuesAreCorrect(self, feature_ndims, dtype, dims):
    amplitude = np.array(5., dtype=dtype)
    length_scale = np.array(.2, dtype=dtype)
    scale_mixture_rate = np.array(3., dtype=dtype)

    np.random.seed(42)
    k = tfp.math.psd_kernels.RationalQuadratic(amplitude, length_scale,
                                               scale_mixture_rate,
                                               feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(dtype)
      y = np.random.uniform(-1, 1, size=shape).astype(dtype)
      self.assertAllClose(
          self._rational_quadratic(
              amplitude, length_scale, scale_mixture_rate, x, y),
          self.evaluate(k.apply(x, y)))

  def testNoneScaleMixture(self):
    amplitude = 5.
    length_scale = .2

    np.random.seed(42)

    k = tfp.math.psd_kernels.RationalQuadratic(
        amplitude=amplitude, length_scale=length_scale, scale_mixture_rate=None)
    x = np.random.uniform(-1, 1, size=[5]).astype(np.float32)
    y = np.random.uniform(-1, 1, size=[5]).astype(np.float32)
    self.assertAllClose(
        # Ensure that a None value for scale_mixture_rate has the same semantics
        # as scale_mixture_rate=1.
        self._rational_quadratic(
            amplitude=amplitude,
            length_scale=length_scale, scale_mixture_rate=1., x=x, y=y),
        self.evaluate(k.apply(x, y)))

  def testShapesAreCorrect(self):
    k = tfp.math.psd_kernels.RationalQuadratic(amplitude=1., length_scale=1.)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual([4, 5], k.matrix(x, y).shape)
    self.assertAllEqual(
        [2, 4, 5],
        k.matrix(tf.stack([x]*2), tf.stack([y]*2)).shape)

    k = tfp.math.psd_kernels.RationalQuadratic(
        amplitude=np.ones([2, 1, 1], np.float32),
        length_scale=np.ones([1, 3, 1], np.float32),
        scale_mixture_rate=np.ones([2, 1, 1, 1], np.float32))

    self.assertAllEqual(
        [2, 2, 3, 2, 4, 5],
        #`-----'  |  `--'
        #  |      |    `- matrix shape
        #  |      `- from input batch shapes
        #  `- from broadcasting kernel params
        k.matrix(
            tf.stack([x]*2),  # shape [2, 4, 3]
            tf.stack([y]*2)   # shape [2, 5, 3]
        ).shape)

  def testValidateArgs(self):
    with self.assertRaisesOpError('amplitude must be positive'):
      k = tfp.math.psd_kernels.RationalQuadratic(
          amplitude=-1.,
          length_scale=1.,
          scale_mixture_rate=1.,
          validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

    with self.assertRaisesOpError('length_scale must be positive'):
      k = tfp.math.psd_kernels.RationalQuadratic(
          amplitude=1.,
          length_scale=-1.,
          scale_mixture_rate=1.,
          validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

    with self.assertRaisesOpError('scale_mixture_rate must be positive'):
      k = tfp.math.psd_kernels.RationalQuadratic(
          amplitude=1.,
          length_scale=1.,
          scale_mixture_rate=-1.,
          validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

    # But `None`'s are ok
    k = tfp.math.psd_kernels.RationalQuadratic(
        amplitude=None,
        length_scale=None,
        scale_mixture_rate=None,
        validate_args=True)
    self.evaluate(k.apply([1.], [1.]))

if __name__ == '__main__':
  tf.test.main()
