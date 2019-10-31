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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels.internal import util


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
class _FeatureTransformedTest(test_util.TestCase):

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testValuesAreCorrectIdentity(self, feature_ndims, dims):
    amplitude = self.dtype(5.)
    length_scale = self.dtype(0.2)
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale, feature_ndims)
    input_shape = [dims] * feature_ndims
    identity_transformed_kernel = tfp.math.psd_kernels.FeatureTransformed(
        kernel,
        transformation_fn=lambda x, feature_ndims, param_expansion_ndims: x)
    x = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)
    y = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)
    self.assertAllClose(
        _numpy_exp_quad(
            amplitude, length_scale, x, y, feature_ndims=feature_ndims),
        self.evaluate(identity_transformed_kernel.apply(x, y)))

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testValuesAreCorrectScalarTransform(self, feature_ndims, dims):
    amplitude = self.dtype(5.)
    length_scale = self.dtype(0.2)
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale, feature_ndims)
    input_shape = [dims] * feature_ndims

    bij = tfp.bijectors.AffineScalar(self.dtype(0.), self.dtype(2.))
    # Flat multiplication by 2.
    def scale_transform(x, feature_ndims, param_expansion_ndims):
      del feature_ndims, param_expansion_ndims
      return bij.forward(x)

    scale_transformed_kernel = tfp.math.psd_kernels.FeatureTransformed(
        kernel, transformation_fn=scale_transform)

    x = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)
    y = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)
    self.assertAllClose(
        _numpy_exp_quad
        (amplitude, length_scale, 2. * x, 2. * y, feature_ndims=feature_ndims),
        self.evaluate(scale_transformed_kernel.apply(x, y)))

  @parameterized.parameters(
      {'feature_ndims': 1, 'dims': 3},
      {'feature_ndims': 1, 'dims': 4},
      {'feature_ndims': 2, 'dims': 2},
      {'feature_ndims': 2, 'dims': 3},
      {'feature_ndims': 3, 'dims': 2},
      {'feature_ndims': 3, 'dims': 3})
  def testValuesAreCorrectVectorTransform(self, feature_ndims, dims):
    amplitude = self.dtype(5.)
    length_scale = self.dtype(0.2)
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale, feature_ndims)
    input_shape = [dims] * feature_ndims

    scale_diag = np.random.uniform(-1, 1, size=(dims,)).astype(self.dtype)
    bij = tfp.bijectors.Affine(scale_diag=scale_diag)

    # Scaling the last dimension.
    def vector_transform(x, feature_ndims, param_expansion_ndims):
      del feature_ndims, param_expansion_ndims
      return bij.forward(x)

    vector_transformed_kernel = tfp.math.psd_kernels.FeatureTransformed(
        kernel, transformation_fn=vector_transform)

    x = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)
    y = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)
    self.assertAllClose(
        _numpy_exp_quad(
            amplitude,
            length_scale,
            scale_diag * x,
            scale_diag * y,
            feature_ndims=feature_ndims),
        self.evaluate(vector_transformed_kernel.apply(x, y)))

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
    length_scale = np.random.uniform(
        low=1., high=10., size=[1, 2]).astype(self.dtype)
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale, feature_ndims)
    input_shape = [dims] * feature_ndims

    # Batch shape [3, 1, 2].
    scale_diag = np.random.uniform(
        -1, 1, size=(3, 1, 2, dims,)).astype(self.dtype)

    # Scaling the last dimension.
    def vector_transform(x, feature_ndims, param_expansion_ndims):
      diag = util.pad_shape_with_ones(
          scale_diag,
          param_expansion_ndims + feature_ndims - 1,
          start=-2)
      return diag * x

    vector_transformed_kernel = tfp.math.psd_kernels.FeatureTransformed(
        kernel, transformation_fn=vector_transform)

    x = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)
    y = np.random.uniform(-1, 1, size=input_shape).astype(self.dtype)

    # Pad for each feature dimension.
    expanded_scale_diag = scale_diag
    for _ in range(feature_ndims - 1):
      expanded_scale_diag = expanded_scale_diag[..., None, :]

    self.assertAllClose(
        _numpy_exp_quad(
            amplitude,
            length_scale,
            expanded_scale_diag * x,
            expanded_scale_diag * y,
            feature_ndims=feature_ndims
        ),
        self.evaluate(vector_transformed_kernel.apply(x, y)))

    z = np.random.uniform(-1, 1, size=[10] + input_shape).astype(self.dtype)

    self.assertAllClose(
        _numpy_exp_quad_matrix(
            # We need to take in to account the event dimension.
            amplitude[..., None, None],
            length_scale[..., None, None],
            # Extra dimension for the event dimension.
            expanded_scale_diag[..., None, :] * z,
            feature_ndims=feature_ndims
        ),
        self.evaluate(vector_transformed_kernel.matrix(z, z)))


class FeatureTransformedFloat32Test(_FeatureTransformedTest):
  dtype = np.float32


class FeatureTransformedFloat64Test(_FeatureTransformedTest):
  dtype = np.float64


del _FeatureTransformedTest


if __name__ == '__main__':
  tf.test.main()
