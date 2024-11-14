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
"""Bijectors Tests."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.math.psd_kernels import feature_scaled


def _numpy_exp_quad(amplitude, length_scale, x, y, feature_ndims):
  dims = tuple(range(-feature_ndims, 0, 1))
  return amplitude ** 2 * np.exp(
      -0.5 * np.sum((x - y) ** 2, axis=dims) / length_scale ** 2)


def _numpy_exp_quad_matrix(amplitude, length_scale, x, feature_ndims):
  return _numpy_exp_quad(
      amplitude,
      length_scale,
      np.expand_dims(x, -feature_ndims - 2),
      np.expand_dims(x, -feature_ndims - 1),
      feature_ndims)


@test_util.test_all_tf_execution_regimes
class _FeatureScaledTest(test_util.TestCase):

  def testBatchShape(self):
    # Batch shape [10, 2]
    amplitude = np.random.uniform(
        low=1., high=10., size=[10, 2]).astype(self.dtype)
    inner_length_scale = self.dtype(1.)
    # Use 3 feature_ndims.
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, inner_length_scale, feature_ndims=3)
    scale_diag = tf.ones([20, 1, 2, 1, 1, 1], dtype=self.dtype)
    ard_kernel = feature_scaled.FeatureScaled(kernel, scale_diag=scale_diag)
    self.assertAllEqual([20, 10, 2], ard_kernel.batch_shape)
    self.assertAllEqual(
        [20, 10, 2], self.evaluate(ard_kernel.batch_shape_tensor()))

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testKernelParametersBroadcast(self, feature_ndims, dims):
    # Batch shape [10, 2]
    amplitude = np.random.uniform(
        low=1., high=10., size=[10, 2]).astype(self.dtype)
    inner_length_scale = self.dtype(1.)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale=inner_length_scale, feature_ndims=feature_ndims)
    input_shape = [dims] * feature_ndims

    # Batch shape [3, 1, 2].
    length_scale = np.random.uniform(
        2, 5, size=([3, 1, 2] + input_shape)).astype(self.dtype)

    ard_kernel = feature_scaled.FeatureScaled(kernel, scale_diag=length_scale)
    self.assertIs(dtype_util.as_numpy_dtype(ard_kernel.dtype), self.dtype)

    x = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)
    y = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)

    self.assertAllClose(
        _numpy_exp_quad(
            amplitude,
            inner_length_scale,
            x / length_scale,
            y / length_scale,
            feature_ndims=feature_ndims
        ),
        self.evaluate(ard_kernel.apply(x, y)))

    z = np.random.uniform(-1, 1, size=[10] + input_shape).astype(self.dtype)

    expanded_length_scale = np.expand_dims(length_scale, -(feature_ndims + 1))

    self.assertAllClose(
        _numpy_exp_quad_matrix(
            amplitude[..., None, None],
            inner_length_scale,
            z / expanded_length_scale,
            feature_ndims=feature_ndims
        ),
        self.evaluate(ard_kernel.matrix(z, z)))

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testKernelParametersInverseLengthScale(self, feature_ndims, dims):
    # Batch shape [10, 2]
    amplitude = np.random.uniform(
        low=1., high=10., size=[10, 2]).astype(self.dtype)

    inner_length_scale = self.dtype(1.)
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale=inner_length_scale, feature_ndims=feature_ndims)
    input_shape = [dims] * feature_ndims

    # Batch shape [3, 1, 2].
    inverse_length_scale = np.random.uniform(
        2, 5, size=([3, 1, 2] + input_shape)).astype(self.dtype)

    ard_kernel = feature_scaled.FeatureScaled(
        kernel, scale_diag=None, inverse_scale_diag=inverse_length_scale)

    ard_kernel_ls = feature_scaled.FeatureScaled(
        kernel, scale_diag=1. / inverse_length_scale)

    x = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)
    y = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)

    self.assertAllClose(
        self.evaluate(ard_kernel.apply(x, y)),
        self.evaluate(ard_kernel_ls.apply(x, y)))


class FeatureScaledFloat32Test(_FeatureScaledTest):
  dtype = np.float32


class FeatureScaledFloat64Test(_FeatureScaledTest):
  dtype = np.float64


del _FeatureScaledTest


if __name__ == '__main__':
  test_util.main()
