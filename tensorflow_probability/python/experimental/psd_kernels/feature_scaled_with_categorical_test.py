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
"""Tests for feature_scaled_with_categorical."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.psd_kernels import feature_scaled_with_categorical as fswc
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.math.psd_kernels import feature_scaled
from tensorflow_probability.python.math.psd_kernels import matern


@test_util.test_all_tf_execution_regimes
class FeatureScaledWithCategoricalTest(test_util.TestCase):

  def testBatchShape(self):
    base_kernel = matern.MaternFiveHalves()
    isd_continuous = np.ones([3, 1, 2], dtype=np.float32)
    isd_categorical = np.ones([1, 4, 1], dtype=np.float32)
    kernel = fswc.FeatureScaledWithCategorical(
        base_kernel,
        inverse_scale_diag=fswc.ContinuousAndCategoricalValues(
            isd_continuous, isd_categorical),
        validate_args=True)
    self.assertAllEqual(tf.TensorShape([3, 4]), kernel.batch_shape)
    self.assertAllEqual([3, 4], self.evaluate(kernel.batch_shape_tensor()))

  def testCategoricalDistance(self):
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    isd = np.ones([2], dtype=np.float32)
    kernel = fswc.FeatureScaledWithCategorical(
        base_kernel,
        inverse_scale_diag=fswc.ContinuousAndCategoricalValues(isd, isd),
        validate_args=True)

    # Sample some non-overlapping "categorical" data, represented by integers.
    x1_cat = np.random.randint(5, size=(3, 2))
    x2_cat = np.random.randint(6, 10, size=(2, 3, 2))

    x1_cont = np.ones(x1_cat.shape).astype(np.float32)
    x2_cont = np.ones(x2_cat.shape).astype(np.float32)

    x1 = fswc.ContinuousAndCategoricalValues(x1_cont, x1_cat)
    x2 = fswc.ContinuousAndCategoricalValues(x2_cont, x2_cat)

    kern_mat = self.evaluate(kernel.matrix(x1, x2))
    self.assertAllClose(kern_mat, np.exp(-1.) * np.ones_like(kern_mat))

    # Make some of the categorical features equal.
    x2_cat[0, 1, 0] = x1_cat[2, 0]
    x2_cat[1, 2, 1] = x1_cat[0, 1]
    x2_cat[1, 1, :] = x1_cat[1, :]
    x1a = fswc.ContinuousAndCategoricalValues(x1_cont, x1_cat)
    x2a = fswc.ContinuousAndCategoricalValues(x2_cont, x2_cat)

    # Assert that the pairs with equal categorical feature are as expected.
    kern_mat = self.evaluate(kernel.matrix(x1a, x2a))
    self.assertAllClose(kern_mat[0, 2, 1], np.exp(-0.5))
    self.assertAllClose(kern_mat[1, 0, 2], np.exp(-0.5))
    self.assertAllClose(kern_mat[1, 1, 1], 1.)

  @parameterized.parameters(
      {'dtype': np.float32, 'feature_ndims': 1},
      {'dtype': np.float64, 'feature_ndims': 2})
  def testValuesAreCorrectAgainstBinary(self, dtype, feature_ndims):
    cont_dim = 3
    cat_dim = 5
    base_kernel = matern.MaternFiveHalves(
        amplitude=np.ones([], dtype=dtype), feature_ndims=feature_ndims)
    inverse_scale_diag = np.random.uniform(
        size=[cont_dim + cat_dim]).astype(dtype)
    kernel = fswc.FeatureScaledWithCategorical(
        base_kernel,
        inverse_scale_diag=fswc.ContinuousAndCategoricalValues(
            *tf.split(inverse_scale_diag, [cont_dim, cat_dim], axis=-1)),
        validate_args=True)
    feature_scaled_kernel = feature_scaled.FeatureScaled(
        base_kernel,
        inverse_scale_diag=inverse_scale_diag,
        validate_args=True)

    x1_cat = np.random.randint(2, size=(3, 1, 4, cat_dim))
    x2_cat = np.random.randint(2, size=(3, 4, cat_dim))
    x1_cont = np.random.normal(size=(3, 1, 4, cont_dim)).astype(dtype)
    x2_cont = np.random.normal(size=(3, 4, cont_dim)).astype(dtype)

    x1 = fswc.ContinuousAndCategoricalValues(x1_cont, x1_cat)
    x2 = fswc.ContinuousAndCategoricalValues(x2_cont, x2_cat)

    x1_ = np.concatenate([x1_cont, x1_cat.astype(dtype)], axis=-1)
    x2_ = np.concatenate([x2_cont, x2_cat.astype(dtype)], axis=-1)

    # When there are only two categories, 0 and 1, the categorical distance is
    # the same as the Euclidean distance.
    expected_kern_mat = self.evaluate(feature_scaled_kernel.matrix(x1_, x2_))
    actual_kern_mat = self.evaluate(kernel.matrix(x1, x2))
    self.assertAllClose(expected_kern_mat, actual_kern_mat)
    self.assertDTypeEqual(actual_kern_mat, dtype)

    expected_kern_apply = self.evaluate(feature_scaled_kernel.apply(x1_, x2_))
    actual_kern_apply = self.evaluate(kernel.apply(x1, x2))
    self.assertAllClose(expected_kern_apply, actual_kern_apply)
    self.assertDTypeEqual(actual_kern_apply, dtype)

    expected_kern_tensor = self.evaluate(
        feature_scaled_kernel.tensor(
            x1_, x2_, x1_example_ndims=2, x2_example_ndims=1))
    actual_kern_tensor = self.evaluate(
        kernel.tensor(x1, x2, x1_example_ndims=2, x2_example_ndims=1))
    self.assertAllClose(expected_kern_tensor, actual_kern_tensor)
    self.assertDTypeEqual(actual_kern_tensor, dtype)

  @parameterized.parameters(
      {'continuous_feature_ndims': 1,
       'categorical_feature_ndims': 1,
       'continuous_dims': 3,
       'categorical_dims': 3},
      {'continuous_feature_ndims': 2,
       'categorical_feature_ndims': 2,
       'continuous_dims': 2,
       'categorical_dims': 3},
      {'continuous_feature_ndims': 1,
       'categorical_feature_ndims': 3,
       'continuous_dims': 3,
       'categorical_dims': 3},
      {'continuous_feature_ndims': 1,
       'categorical_feature_ndims': 2,
       'continuous_dims': 3,
       'categorical_dims': 4})
  def testBroadcastingParametersAndValuesMatchFeatureScaled(
      self, continuous_feature_ndims, categorical_feature_ndims,
      continuous_dims, categorical_dims):
    # Batch shape [10, 2]
    amplitude = np.random.uniform(low=1., high=10., size=[10, 2])
    kernel = exponentiated_quadratic.ExponentiatedQuadratic(
        amplitude, length_scale=1., feature_ndims=continuous_feature_ndims)
    continuous_input_shape = [continuous_dims] * continuous_feature_ndims
    categorical_input_shape = [categorical_dims] * categorical_feature_ndims

    # Batch shape [3, 1, 2].
    continuous_length_scale = np.random.uniform(
        2, 5, size=([3, 1, 2] + continuous_input_shape))
    categorical_length_scale = np.random.uniform(
        2, 5, size=([10, 2] + categorical_input_shape))

    ard_kernel = feature_scaled.FeatureScaled(
        kernel, scale_diag=continuous_length_scale)
    cat_kernel = fswc.FeatureScaledWithCategorical(
        kernel,
        scale_diag=fswc.ContinuousAndCategoricalValues(
            continuous_length_scale, categorical_length_scale),
        feature_ndims=fswc.ContinuousAndCategoricalValues(
            continuous_feature_ndims, categorical_feature_ndims))

    x = np.random.uniform(-1, 1, size=[1] + continuous_input_shape)
    y = np.random.uniform(-1, 1, size=[1] + continuous_input_shape)

    # Zero distance between categorical features.
    cat = np.ones([1] + categorical_input_shape, dtype=np.int32)
    x_ = fswc.ContinuousAndCategoricalValues(x, cat)
    y_ = fswc.ContinuousAndCategoricalValues(y, cat)

    self.assertAllClose(
        self.evaluate(ard_kernel.apply(x, y)),
        self.evaluate(cat_kernel.apply(x_, y_)))
    self.assertAllClose(
        self.evaluate(ard_kernel.matrix(x, y)),
        self.evaluate(cat_kernel.matrix(x_, y_)))

  def testValidateArgs(self):
    x = fswc.ContinuousAndCategoricalValues(
        np.random.normal(size=(5, 3)),
        np.random.randint(10, size=(5, 1)))
    scale_diag_parameter = fswc.ContinuousAndCategoricalValues(
        [-1., 3., 4.], [2.])
    with self.assertRaisesOpError('`inverse_scale_diag` must be non-negative'):
      k = fswc.FeatureScaledWithCategorical(
          exponentiated_quadratic.ExponentiatedQuadratic(),
          inverse_scale_diag=scale_diag_parameter,
          validate_args=True)
      self.evaluate(k.apply(x, x))

    with self.assertRaisesOpError('`scale_diag` must be positive'):
      k = fswc.FeatureScaledWithCategorical(
          exponentiated_quadratic.ExponentiatedQuadratic(),
          scale_diag=scale_diag_parameter,
          validate_args=True)
      self.evaluate(k.apply(x, x))

  def testEmptyInputs(self):
    dim = 4
    n_pts = 3
    isd_empty = np.ones([]).astype(np.float32)
    x_cont_empty = np.ones([3, 0]).astype(np.float32)
    x_cat_empty = np.ones([3, 0]).astype(np.int32)
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()

    isd_categorical = np.array([dim]).astype(np.float32)
    kernel_empty_cont = fswc.FeatureScaledWithCategorical(
        base_kernel,
        inverse_scale_diag=fswc.ContinuousAndCategoricalValues(
            isd_empty, isd_categorical),
        validate_args=True)
    x_cat = np.ones((n_pts, dim)).astype(np.int32)
    x_empty_cont = fswc.ContinuousAndCategoricalValues(x_cont_empty, x_cat)

    # Distances between points are 0 so kernel matrix containes ones.
    self.assertAllClose(kernel_empty_cont.matrix(x_empty_cont, x_empty_cont),
                        np.ones((n_pts, n_pts)))

    x_cont = np.random.normal(size=(4, n_pts, dim)).astype(np.float32)
    isd_continuous = np.ones([dim], dtype=np.float32)
    kernel_empty_cat = fswc.FeatureScaledWithCategorical(
        base_kernel,
        inverse_scale_diag=fswc.ContinuousAndCategoricalValues(
            isd_continuous, isd_empty),
        validate_args=True)
    x_empty_cat = fswc.ContinuousAndCategoricalValues(x_cont, x_cat_empty)

    # Without categorical data, the kernel matrix should be the same as the base
    # kernel matrix for continuous data.
    self.assertAllClose(kernel_empty_cat.matrix(x_empty_cat, x_empty_cat),
                        base_kernel.matrix(x_cont, x_cont))


if __name__ == '__main__':
  test_util.main()
