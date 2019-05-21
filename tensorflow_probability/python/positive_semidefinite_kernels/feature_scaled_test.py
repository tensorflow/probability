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
import tensorflow as tf
from tensorflow_probability.python import positive_semidefinite_kernels as tfpk
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


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


@test_util.run_all_in_graph_and_eager_modes
class _FeatureScaledTest(parameterized.TestCase):

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
    kernel = tfpk.ExponentiatedQuadratic(
        amplitude, inner_length_scale, feature_ndims)
    input_shape = [dims] * feature_ndims

    # Batch shape [3, 1, 2].
    length_scale = np.random.uniform(
        2, 5, size=([3, 1, 2] + input_shape)).astype(self.dtype)

    ard_kernel = tfpk.FeatureScaled(kernel, scale_diag=length_scale)

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


class FeatureScaledFloat32Test(_FeatureScaledTest, tf.test.TestCase):
  dtype = np.float32


class FeatureScaledFloat64Test(_FeatureScaledTest, tf.test.TestCase):
  dtype = np.float64


del _FeatureScaledTest


if __name__ == '__main__':
  tf.test.main()
