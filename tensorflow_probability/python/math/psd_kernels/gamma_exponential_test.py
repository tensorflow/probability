# Copyright 2023 The TensorFlow Probability Authors.
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
"""Tests for GammaExponential."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.math.psd_kernels import gamma_exponential


@test_util.test_all_tf_execution_regimes
class GammaExponentialTest(test_util.TestCase):

  @test_util.disable_test_for_backend(
      disable_numpy=True, reason='DType mismatch not caught in numpy.')
  def testMismatchedFloatTypesAreBad(self):
    gamma_exponential.GammaExponential(
        1, 1)  # Should be OK (float32 fallback).
    gamma_exponential.GammaExponential(np.float32(1.),
                                       1.)  # Should be OK.
    with self.assertRaises(TypeError):
      gamma_exponential.GammaExponential(np.float32(1.), np.float64(1.))

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testValuesAreCorrect(self, feature_ndims, dims):
    amplitude = 5.
    length_scale = .2
    gamma = .85

    np.random.seed(42)
    k = gamma_exponential.GammaExponential(
        amplitude, length_scale=length_scale, feature_ndims=feature_ndims,
        power=gamma)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(np.float32)
      y = np.random.uniform(-1, 1, size=shape).astype(np.float32)
      self.assertAllClose(
          amplitude ** 2 * np.exp(
              -(np.sum((x - y)**2) /
                (np.float32(2.) * length_scale**2))**gamma),
          self.evaluate(k.apply(x, y)))

  def testNoneShapes(self):
    k = gamma_exponential.GammaExponential(
        amplitude=np.reshape(np.arange(12.), [2, 3, 2]))
    self.assertAllEqual((2, 3, 2), k.batch_shape)

  def testEqualsExponentiatedQuadratic(self):
    np.random.seed(42)
    ge = gamma_exponential.GammaExponential(
        amplitude=3., length_scale=0.5, power=1., feature_ndims=0)
    eq = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude=3., length_scale=0.5, feature_ndims=0)
    t1, t2 = np.random.rand(2, 10)
    self.assertAllClose(ge.apply(t1, t2), eq.apply(t1, t2))

  def testShapesAreCorrect(self):
    k = gamma_exponential.GammaExponential(
        amplitude=1., length_scale=1., power=2.)

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual([4, 5], k.matrix(x, y).shape)
    self.assertAllEqual(
        [2, 4, 5],
        k.matrix(tf.stack([x]*2), tf.stack([y]*2)).shape)

    k = gamma_exponential.GammaExponential(
        amplitude=np.ones([2, 1, 1, 1], np.float32),
        length_scale=np.ones([1, 3, 1, 1], np.float32),
        power=np.ones([1, 1, 7, 1], np.float32))
    self.assertAllEqual(
        [2, 3, 7, 2, 4, 5],
        #`--'  |  `--'
        #  |   |    `- matrix shape
        #  |   `- from input batch shapes
        #  `- from broadcasting kernel params
        k.matrix(
            tf.stack([x]*2),  # shape [2, 4, 3]
            tf.stack([y]*2)   # shape [2, 5, 3]
        ).shape)

  def testValidateArgs(self):
    with self.assertRaisesOpError('must be positive'):
      k = gamma_exponential.GammaExponential(
          -1., 1., power=0.5, validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

    with self.assertRaisesOpError('must be positive'):
      gamma_exponential.GammaExponential(
          2., -2., power=0.5, validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

    with self.assertRaisesOpError('must be positive'):
      gamma_exponential.GammaExponential(
          2., 2., power=-1., validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

    with self.assertRaisesOpError('must be <= 1'):
      gamma_exponential.GammaExponential(
          2., 2., power=1.1, validate_args=True)
      self.evaluate(k.apply([1.], [1.]))

    # But `None`'s are ok
    k = gamma_exponential.GammaExponential(
        None, None, power=None, validate_args=True)
    self.evaluate(k.apply([1.], [1.]))

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testInverseLengthScaleValuesAreCorrect(self, feature_ndims, dims):
    amplitude = 5.
    inverse_length_scale = np.float32([2., 3., 1., 0.5])
    power = np.float32([0.8, 0.2, 0.9, 0.7])

    np.random.seed(42)
    k = gamma_exponential.GammaExponential(
        amplitude,
        inverse_length_scale=inverse_length_scale,
        power=power,
        feature_ndims=feature_ndims)
    k_with_ls = gamma_exponential.GammaExponential(
        amplitude,
        length_scale=1. / inverse_length_scale,
        power=power,
        feature_ndims=feature_ndims)
    shape = [dims] * feature_ndims
    x = np.random.uniform(-1, 1, size=shape).astype(np.float32)
    y = np.random.uniform(-1, 1, size=shape).astype(np.float32)
    self.assertAllClose(
        self.evaluate(k.apply(x, y)),
        self.evaluate(k_with_ls.apply(x, y)))

  @test_util.jax_disable_variable_test
  def testValidateVariableArgs(self):
    amplitude = tf.Variable(1.)
    length_scale = tf.Variable(1.)
    power = tf.Variable(1.)
    k = gamma_exponential.GammaExponential(
        amplitude, length_scale, power=power, validate_args=True)
    self.evaluate([v.initializer for v in k.variables])

    with self.assertRaisesOpError('must be positive'):
      with tf.control_dependencies([amplitude.assign(-1.)]):
        self.evaluate(k.apply([1.], [1.]))

    with self.assertRaisesOpError('must be positive'):
      with tf.control_dependencies([amplitude.assign(2.),
                                    length_scale.assign(-1.)]):
        self.evaluate(k.apply([3.], [3.]))

    with self.assertRaisesOpError('must be positive'):
      with tf.control_dependencies([length_scale.assign(2.),
                                    power.assign(-1.)]):
        self.evaluate(k.apply([3.], [3.]))

    with self.assertRaisesOpError('must be <= 1'):
      with tf.control_dependencies([power.assign(1.1)]):
        self.evaluate(k.apply([3.], [3.]))

  def testAtMostOneLengthScale(self):
    with self.assertRaisesRegex(ValueError, 'Must specify at most one of'):
      gamma_exponential.GammaExponential(
          amplitude=1., length_scale=1., inverse_length_scale=2.)


if __name__ == '__main__':
  test_util.main()
