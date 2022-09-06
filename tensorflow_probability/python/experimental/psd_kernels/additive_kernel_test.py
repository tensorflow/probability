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
"""Tests for AdditiveKernel."""

import itertools

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.psd_kernels import additive_kernel
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic


@test_util.test_all_tf_execution_regimes
class AdditiveKernelTest(test_util.TestCase):

  def testBatchShape(self):
    amplitudes = np.ones((4, 1, 1, 3), np.float32)
    additive_inner_kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude=np.ones([2, 1, 1], np.float32),
        length_scale=np.ones([1, 3, 1], np.float32))
    kernel = additive_kernel.AdditiveKernel(
        kernel=additive_inner_kernel, amplitudes=amplitudes)
    self.assertAllEqual(tf.TensorShape([4, 2, 3]), kernel.batch_shape)
    self.assertAllEqual([4, 2, 3], self.evaluate(kernel.batch_shape_tensor()))

  def _compute_additive_kernel(
      self, amplitudes, length_scale, dim, x, y, method='apply'):
    expected = 0.
    for i in range(amplitudes.shape[-1]):
      # Get all (ordered) combinations of indices of length `i + 1`
      ind = itertools.combinations(range(dim), i + 1)

      # Sum over all combinations of indices of the given length.
      sum_i = 0.
      for ind_i in ind:
        # Multiply the kernel values at the given indices together.
        prod_k = 1.
        for d in ind_i:
          kernel = exponentiated_quadratic.ExponentiatedQuadratic(
              amplitude=1., length_scale=length_scale[..., d])
          if method == 'apply':
            prod_k *= kernel.apply(
                x[..., d, np.newaxis], y[..., d, np.newaxis])
          elif method == 'matrix':
            prod_k *= kernel.matrix(
                x[..., d, np.newaxis], y[..., d, np.newaxis])
        sum_i += prod_k
      expected += amplitudes[..., i]**2 * sum_i
    return expected

  @parameterized.parameters(
      {'x_batch': [1], 'y_batch': [1], 'dim': 2, 'amplitudes': [1., 2.]},
      {'x_batch': [5], 'y_batch': [1], 'dim': 4, 'amplitudes': [1., 2., 3.]},
      {'x_batch': [], 'y_batch': [], 'dim': 2, 'amplitudes': [3., 2.]},
      {'x_batch': [4, 1], 'y_batch': [3], 'dim': 2, 'amplitudes': [1.]},
      {'x_batch': [4, 1, 1], 'y_batch': [3, 1], 'dim': 2,
       'amplitudes': [[2., 3.], [1., 2.]]},
      {'x_batch': [5], 'y_batch': [1], 'dim': 17, 'amplitudes': [2., 2.]},
      {'x_batch': [10], 'y_batch': [2, 1], 'dim': 16,
       'amplitudes': [2., 1., 0.5]},
      )
  def testValuesAreCorrect(self, x_batch, y_batch, dim, amplitudes):

    amplitudes = np.array(amplitudes).astype(np.float32)

    length_scale = tf.random.stateless_uniform(
        [dim],
        seed=test_util.test_seed(sampler_type='stateless'),
        minval=0., maxval=2., dtype=tf.float32)
    inner_kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude=1., length_scale=length_scale)
    kernel = additive_kernel.AdditiveKernel(
        kernel=inner_kernel, amplitudes=amplitudes)

    x = tf.random.stateless_uniform(
        x_batch + [dim],
        seed=test_util.test_seed(sampler_type='stateless'),
        minval=-2., maxval=2., dtype=tf.float32)
    y = tf.random.stateless_uniform(
        y_batch + [dim], seed=test_util.test_seed(sampler_type='stateless'),
        minval=-2., maxval=2., dtype=tf.float32)

    actual = kernel.apply(x, y)

    expected = self._compute_additive_kernel(
        amplitudes, length_scale, dim, x, y, method='apply')

    self.assertAllClose(self.evaluate(actual), self.evaluate(expected))

  @parameterized.parameters(
      {'x_batch': [5], 'y_batch': [1], 'dim': 4, 'amplitudes': [1., 2., 3.]},
      {'x_batch': [], 'y_batch': [], 'dim': 2, 'amplitudes': [3., 2.]},
      {'x_batch': [4, 1], 'y_batch': [3], 'dim': 2, 'amplitudes': [1.]},
      {'x_batch': [5], 'y_batch': [1], 'dim': 17, 'amplitudes': [2., 2.]},
      {'x_batch': [10], 'y_batch': [2, 1], 'dim': 16,
       'amplitudes': [2., 1., 0.5]},
      )
  def testMatrixValuesAreCorrect(
      self, x_batch, y_batch, dim, amplitudes):

    amplitudes = np.array(amplitudes).astype(np.float32)

    length_scale = tf.random.stateless_uniform(
        [dim],
        seed=test_util.test_seed(sampler_type='stateless'),
        minval=0., maxval=2., dtype=tf.float32)
    inner_kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude=1., length_scale=length_scale)
    kernel = additive_kernel.AdditiveKernel(
        kernel=inner_kernel, amplitudes=amplitudes)

    x = tf.random.stateless_uniform(
        x_batch + [3, dim],
        seed=test_util.test_seed(sampler_type='stateless'),
        minval=-2., maxval=2., dtype=tf.float32)
    y = tf.random.stateless_uniform(
        y_batch + [5, dim], seed=test_util.test_seed(sampler_type='stateless'),
        minval=-2., maxval=2., dtype=tf.float32)

    actual = kernel.matrix(x, y)

    expected = self._compute_additive_kernel(
        amplitudes, length_scale, dim, x, y, method='matrix')

    self.assertAllClose(
        self.evaluate(actual), self.evaluate(expected), rtol=1e-5)

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      disable_jax=True,
      reason='GradientTape not available for JAX/NumPy backend.')
  @parameterized.parameters(
      {'input_shape': [3, 4], 'amplitudes': [1., 2., 3.]},
      {'input_shape': [10, 16], 'amplitudes': [2., 1., 0.5]})
  def test_gradient(self, input_shape, amplitudes):
    length_scale = tf.Variable(
        tf.random.stateless_uniform(
            [input_shape[-1]],
            seed=test_util.test_seed(sampler_type='stateless'),
            minval=0., maxval=2., dtype=tf.float32))
    inner_kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude=1., length_scale=length_scale)
    kernel = additive_kernel.AdditiveKernel(
        kernel=inner_kernel, amplitudes=amplitudes)

    x = tf.random.stateless_uniform(
        input_shape, seed=test_util.test_seed(sampler_type='stateless'),
        minval=-2., maxval=2., dtype=tf.float32)
    y = tf.random.stateless_uniform(
        input_shape, seed=test_util.test_seed(sampler_type='stateless'),
        minval=-2., maxval=2., dtype=tf.float32)

    with tf.GradientTape() as tape:
      tape.watch(x)
      k = kernel.apply(x, y)
    self.evaluate([v.initializer for v in kernel.trainable_variables])
    grads = tape.gradient(k, kernel.trainable_variables + (x,))
    self.assertAllNotNone(grads)
    self.assertLen(grads, 2)

if __name__ == '__main__':
  tf.test.main()
