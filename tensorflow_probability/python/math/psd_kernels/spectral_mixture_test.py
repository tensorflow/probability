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
"""Tests for SpectralMixture."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class SpectralMixtureTest(test_util.TestCase):

  def _spectral_mixture(
      self, logits, locs, scales, x, y, feature_ndims):
    dims = tuple(range(-feature_ndims, 0, 1))
    summand = (np.pi * scales * (x - y)) ** 2
    summand = np.exp(-2 * np.sum(summand, axis=dims))

    cos_coeffs = np.cos(2 * np.pi * (x - y) * locs)
    cos_coeffs = np.reshape(cos_coeffs, [np.shape(cos_coeffs)[0], -1])
    cos_coeffs = np.prod(cos_coeffs, axis=-1)
    return np.sum(np.exp(logits) * summand * cos_coeffs)

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testValuesAreCorrect(self, feature_ndims, dims):
    num_components = 7
    seed_stream = test_util.test_seed_stream('params')
    logits = tf.random.normal([num_components], seed=seed_stream())
    locs = tf.random.normal(
        [num_components] + [dims] * feature_ndims, seed=seed_stream())
    scales = tf.random.uniform(
        [num_components] + [dims] * feature_ndims,
        minval=1., maxval=2., seed=seed_stream())
    logits, locs, scales = self.evaluate([logits, locs, scales])

    k = tfp.math.psd_kernels.SpectralMixture(
        logits,
        locs=locs,
        scales=scales,
        feature_ndims=feature_ndims)
    shape = [dims] * feature_ndims
    for _ in range(5):
      x = np.random.uniform(-1, 1, size=shape).astype(np.float32)
      y = np.random.uniform(-1, 1, size=shape).astype(np.float32)
      self.assertAllClose(
          self._spectral_mixture(logits, locs, scales, x, y, feature_ndims),
          self.evaluate(k.apply(x, y)))

    shape = [2] + [dims] * feature_ndims
    x = np.random.uniform(-1, 1, size=shape).astype(np.float32)
    kernel_matrix = [
        [self._spectral_mixture(
            logits, locs, scales, x[0], x[0], feature_ndims),
         self._spectral_mixture(
             logits, locs, scales, x[0], x[1], feature_ndims)],
        [self._spectral_mixture(
            logits, locs, scales, x[1], x[0], feature_ndims),
         self._spectral_mixture(
             logits, locs, scales, x[1], x[1], feature_ndims)]]
    self.assertAllClose(kernel_matrix, self.evaluate(k.matrix(x, x)), rtol=1e-4)

  def testShapesAreCorrect(self):
    k = tfp.math.psd_kernels.SpectralMixture(
        logits=np.ones([6, 7], np.float32),
        locs=np.ones([9, 1, 1, 7, 3], np.float32),
        scales=np.ones([2, 1, 7, 3], np.float32))

    batch_shape = [9, 2, 6]

    x = np.ones([4, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    self.assertAllEqual(batch_shape + [4, 5], k.matrix(x, y).shape)

    x = np.ones([2, 1, 3], np.float32)
    y = np.ones([1, 4, 3], np.float32)
    self.assertAllEqual(
        batch_shape + [2, 4], k.apply(x, y, example_ndims=2).shape)

  def testValidateArgs(self):
    with self.assertRaisesOpError('must be positive'):
      k = tfp.math.psd_kernels.SpectralMixture(
          logits=np.ones([7], np.float32),
          locs=np.ones([7, 1], np.float32),
          scales=-np.ones([7, 1], np.float32),
          validate_args=True)

      self.evaluate(k.apply([1.], [1.]))

  @test_util.jax_disable_variable_test
  def testValidateVariableArgs(self):
    logits = tf.Variable(np.ones([7], np.float32))
    locs = tf.Variable(np.ones([7, 1], np.float32))
    scales = tf.Variable(np.ones([7, 1], np.float32))
    k = tfp.math.psd_kernels.SpectralMixture(
        logits=logits,
        locs=locs,
        scales=scales,
        validate_args=True)
    self.evaluate([v.initializer for v in k.variables])

    with self.assertRaisesOpError('must be positive'):
      with tf.control_dependencies(
          [scales.assign(-np.ones([7, 1], np.float32))]):
        self.evaluate(k.apply([1.], [1.]))


if __name__ == '__main__':
  test_util.main()
