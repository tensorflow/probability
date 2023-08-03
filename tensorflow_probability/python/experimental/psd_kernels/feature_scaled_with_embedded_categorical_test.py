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
"""Tests for feature_scaled_with_embedded_categorical."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.psd_kernels import feature_scaled_with_categorical as fswc
from tensorflow_probability.python.experimental.psd_kernels import feature_scaled_with_embedded_categorical as fswec
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.math.psd_kernels import feature_scaled
from tensorflow_probability.python.math.psd_kernels import matern


def _naive_categorical_exponentiated_quadratic(x, y, inverse_length_scale):
  x_obs = x.shape[0]
  y_obs = y.shape[0]
  kernel_mat = np.zeros([x_obs, y_obs])
  for i in range(x_obs):
    for j in range(y_obs):
      dist_sq = 0.
      for k, ls in enumerate(inverse_length_scale):
        if x[i, k] != y[j, k]:
          dist_sq += ls[x[i, k]] ** 2 + ls[y[j, k]] ** 2
      kernel_mat[i, j] = np.exp(-dist_sq / 2.)
  return kernel_mat


def _make_one_hot(ints, num_categories):
  x = np.zeros((ints.shape[0], num_categories))
  x[np.arange(ints.shape[0]), ints] = 1
  return x.astype(np.float32)


@test_util.disable_test_for_backend(
    disable_numpy=True,
    reason='Numpy `gather_nd` does not support batch dims.'
)
@test_util.test_all_tf_execution_regimes
class FeatureScaledWithCategoricalTest(test_util.TestCase):

  def testBatchShape(self):
    base_kernel = matern.MaternFiveHalves()
    isd_continuous = np.ones([3, 1, 2], dtype=np.float32)
    isd_categorical = [
        tf.linalg.LinearOperatorDiag(np.ones([1, 4, 3], dtype=np.float32)),
        tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float32)]
    kernel = fswec.FeatureScaledWithEmbeddedCategorical(
        base_kernel,
        categorical_embedding_operators=isd_categorical,
        continuous_inverse_scale_diag=isd_continuous,
        validate_args=True)
    self.assertAllEqual(tf.TensorShape([3, 4]), kernel.batch_shape)
    self.assertAllEqual([3, 4], self.evaluate(kernel.batch_shape_tensor()))

  def testCorrectnessExponentiatedQuadratic(self):
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    sd_continuous = np.ones([1], dtype=np.float32)

    num_categories = [5, 3, 6, 4]
    x1_cat = np.stack(
        [np.random.randint(n, size=(10,)) for n in num_categories], axis=-1)
    x2_cat = np.stack(
        [np.random.randint(n, size=(8,)) for n in num_categories], axis=-1)

    x1_cont = np.ones([x1_cat.shape[0], 1]).astype(np.float32)
    x2_cont = np.ones([x2_cat.shape[0], 1]).astype(np.float32)
    x1 = fswc.ContinuousAndCategoricalValues(x1_cont, x1_cat)
    x2 = fswc.ContinuousAndCategoricalValues(x2_cont, x2_cat)

    ops = [
        tf.linalg.LinearOperatorDiag(
            np.random.uniform(
                size=[num_categories[0]], high=10.).astype(np.float32)),
        tf.linalg.LinearOperatorScaledIdentity(
            multiplier=np.float32(np.random.normal()),
            num_rows=num_categories[1]),
        tf.linalg.LinearOperatorIdentity(
            num_rows=num_categories[2], dtype=np.float32),
        tf.linalg.LinearOperatorDiag(
            np.random.uniform(
                size=[num_categories[3]], high=10.).astype(np.float32))
    ]
    diagonal_kernel = fswec.FeatureScaledWithEmbeddedCategorical(
        base_kernel,
        categorical_embedding_operators=ops,
        continuous_scale_diag=sd_continuous,
        validate_args=True)

    actual_kern_mat = self.evaluate(diagonal_kernel.matrix(x1, x2))
    expected_kern_mat = _naive_categorical_exponentiated_quadratic(
        x1_cat, x2_cat, [self.evaluate(op.diag_part()) for op in ops])
    self.assertAllClose(actual_kern_mat, expected_kern_mat)

  def testCorrectnessExponentiatedQuadraticFullMatrix(self):
    base_kernel = matern.MaternFiveHalves()
    sd_continuous = np.ones([1], dtype=np.float32)

    num_categories = [6, 4]
    x1_cat = np.stack(
        [np.random.randint(n, size=(2,)) for n in num_categories], axis=-1)
    x2_cat = np.stack(
        [np.random.randint(n, size=(5,)) for n in num_categories], axis=-1)

    x1_cont = np.ones(x1_cat.shape[:-1] + (1,)).astype(np.float32)
    x2_cont = np.ones(x2_cat.shape[:-1] + (1,)).astype(np.float32)
    x1 = fswc.ContinuousAndCategoricalValues(x1_cont, x1_cat)
    x2 = fswc.ContinuousAndCategoricalValues(x2_cont, x2_cat)
    ops = [
        tf.linalg.LinearOperatorFullMatrix(
            np.random.uniform(
                size=[num_categories[0], 3], high=10.
                ).astype(np.float32)),
        tf.linalg.LinearOperatorScaledIdentity(
            multiplier=np.float32(np.random.normal()),
            num_rows=num_categories[1]),
    ]
    embedding_kernel = fswec.FeatureScaledWithEmbeddedCategorical(
        base_kernel,
        categorical_embedding_operators=ops,
        continuous_scale_diag=sd_continuous,
        validate_args=True)

    actual_kern_mat = self.evaluate(embedding_kernel.matrix(x1, x2))

    x1_one_hot = [_make_one_hot(x1_cat[:, 0], num_categories[0]),
                  _make_one_hot(x1_cat[:, 1], num_categories[1])]
    x2_one_hot = [_make_one_hot(x2_cat[:, 0], num_categories[0]),
                  _make_one_hot(x2_cat[:, 1], num_categories[1])]
    x1_embeddings = tf.concat(
        [tf.matmul(x1_one_hot[0], ops[0].to_dense()),
         tf.matmul(x1_one_hot[1], ops[1].to_dense())],
        axis=-1)
    x2_embeddings = tf.concat(
        [tf.matmul(x2_one_hot[0], ops[0].to_dense()),
         tf.matmul(x2_one_hot[1], ops[1].to_dense())],
        axis=-1)
    expected_kern_mat = base_kernel.matrix(x1_embeddings, x2_embeddings)
    self.assertAllClose(
        actual_kern_mat, expected_kern_mat, rtol=1e-5, atol=1e-5)

  def testCategoricalDistance(self):
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    isd_continuous = np.ones([2], dtype=np.float32)
    num_categories = 10
    isd_categorical = [tf.linalg.LinearOperatorIdentity(
        num_rows=num_categories, dtype=tf.float32)] * 2
    kernel = fswec.FeatureScaledWithEmbeddedCategorical(
        base_kernel,
        categorical_embedding_operators=isd_categorical,
        continuous_inverse_scale_diag=isd_continuous,
        validate_args=True)

    # Sample some non-overlapping "categorical" data, represented by integers.
    x1_cat = np.random.randint(5, size=(3, 2))
    x2_cat = np.random.randint(6, 10, size=(2, 3, 2))

    x1_cont = np.ones(x1_cat.shape, dtype=np.float32)
    x2_cont = np.ones(x2_cat.shape, dtype=np.float32)

    x1 = fswc.ContinuousAndCategoricalValues(x1_cont, x1_cat)
    x2 = fswc.ContinuousAndCategoricalValues(x2_cont, x2_cat)

    kern_mat = self.evaluate(kernel.matrix(x1, x2))
    self.assertAllClose(kern_mat, np.exp(-2.) * np.ones_like(kern_mat))

    # Make some of the categorical features equal.
    x2_cat[0, 1, 0] = x1_cat[2, 0]
    x2_cat[1, 2, 1] = x1_cat[0, 1]
    x2_cat[1, 1, :] = x1_cat[1, :]
    x1a = fswc.ContinuousAndCategoricalValues(x1_cont, x1_cat)
    x2a = fswc.ContinuousAndCategoricalValues(x2_cont, x2_cat)

    # Assert that the pairs with equal categorical feature are as expected.
    kern_mat = self.evaluate(kernel.matrix(x1a, x2a))
    self.assertAllClose(kern_mat[0, 2, 1], np.exp(-1.))
    self.assertAllClose(kern_mat[1, 0, 2], np.exp(-1.))
    self.assertAllClose(kern_mat[1, 1, 1], 1.)

  @parameterized.parameters(
      {'dtype': np.float32, 'feature_ndims': 1},
      {'dtype': np.float64, 'feature_ndims': 1})
  def testValuesAreCorrectAgainstBinary(self, dtype, feature_ndims):
    cont_dim = 3
    cat_dim = 3
    base_kernel = matern.MaternFiveHalves(
        amplitude=np.ones([], dtype=dtype), feature_ndims=feature_ndims)
    cont_scale_diag = np.random.uniform(size=[cont_dim]).astype(dtype)
    cat_scale_diag = np.random.uniform(size=[cat_dim]).astype(dtype)
    cat_scale_diags = [
        tf.linalg.LinearOperatorDiag(np.array([0., d]).astype(dtype))
        for d in cat_scale_diag]
    kernel = fswec.FeatureScaledWithEmbeddedCategorical(
        base_kernel,
        categorical_embedding_operators=cat_scale_diags,
        continuous_inverse_scale_diag=cont_scale_diag,
        validate_args=True)
    feature_scaled_kernel = feature_scaled.FeatureScaled(
        base_kernel,
        inverse_scale_diag=np.concatenate(
            [cont_scale_diag, cat_scale_diag], axis=0),
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
       'categorical_feature_ndims': 0,
       'continuous_dims': 2,
       'categorical_dims': 3},
      {'continuous_feature_ndims': 1,
       'categorical_feature_ndims': 1,
       'continuous_dims': 3,
       'categorical_dims': 3},
      {'continuous_feature_ndims': 1,
       'categorical_feature_ndims': 0,
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
    categorical_length_scale = [
        tf.linalg.LinearOperatorDiag(np.random.uniform(2, 5, size=([10, 2, 1])))
    ] * categorical_dims ** categorical_feature_ndims

    ard_kernel = feature_scaled.FeatureScaled(
        kernel, scale_diag=continuous_length_scale)
    cat_kernel = fswec.FeatureScaledWithEmbeddedCategorical(
        kernel,
        categorical_embedding_operators=categorical_length_scale,
        continuous_scale_diag=continuous_length_scale,
        feature_ndims=fswc.ContinuousAndCategoricalValues(
            continuous_feature_ndims, categorical_feature_ndims))

    x = np.random.uniform(-1, 1, size=[1] + continuous_input_shape)
    y = np.random.uniform(-1, 1, size=[1] + continuous_input_shape)

    # Zero distance between categorical features.
    cat = np.zeros([1] + categorical_input_shape, dtype=np.int32)
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
        np.random.normal(size=(5, 3)), np.zeros(shape=(5, 0)))
    scale_diag_parameter = np.array([-1., 3., 4.])
    with self.assertRaisesOpError(
        '`continuous_inverse_scale_diag` must be non-negative'):
      k = fswec.FeatureScaledWithEmbeddedCategorical(
          exponentiated_quadratic.ExponentiatedQuadratic(),
          categorical_embedding_operators=[],
          continuous_inverse_scale_diag=scale_diag_parameter,
          validate_args=True)
      self.evaluate(k.apply(x, x))

    with self.assertRaisesOpError('`continuous_scale_diag` must be positive'):
      k = fswec.FeatureScaledWithEmbeddedCategorical(
          exponentiated_quadratic.ExponentiatedQuadratic(),
          categorical_embedding_operators=[],
          continuous_scale_diag=scale_diag_parameter,
          validate_args=True)
      self.evaluate(k.apply(x, x))

  def testEmptyInputs(self):
    dim = 4
    n_pts = 3
    x_cont_empty = np.ones([3, 0], dtype=np.float32)
    x_cat_empty = np.ones([3, 0], dtype=np.int32)
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()

    isd_categorical = [
        tf.linalg.LinearOperatorIdentity(5, dtype=np.float32)] * dim
    kernel_empty_cont = fswec.FeatureScaledWithEmbeddedCategorical(
        base_kernel,
        categorical_embedding_operators=isd_categorical,
        continuous_inverse_scale_diag=np.ones([], dtype=np.float32),
        validate_args=True)
    x_cat = np.ones((n_pts, dim)).astype(np.int32)
    x_empty_cont = fswc.ContinuousAndCategoricalValues(x_cont_empty, x_cat)

    # Distances between points are 0 so kernel matrix containes ones.
    self.assertAllClose(kernel_empty_cont.matrix(x_empty_cont, x_empty_cont),
                        np.ones((n_pts, n_pts)))

    x_cont = np.random.normal(size=(4, n_pts, dim)).astype(np.float32)
    isd_continuous = np.ones([dim], dtype=np.float32)
    kernel_empty_cat = fswec.FeatureScaledWithEmbeddedCategorical(
        base_kernel,
        continuous_inverse_scale_diag=isd_continuous,
        categorical_embedding_operators=[],
        validate_args=True)
    x_empty_cat = fswc.ContinuousAndCategoricalValues(x_cont, x_cat_empty)

    # Without categorical data, the kernel matrix should be the same as the base
    # kernel matrix for continuous data.
    self.assertAllClose(kernel_empty_cat.matrix(x_empty_cat, x_empty_cat),
                        base_kernel.matrix(x_cont, x_cont))

  def testGradient(self):
    num_categories = [5, 3]
    x1_cat = np.stack(
        [np.random.randint(n, size=(10,)) for n in num_categories], axis=-1)
    x2_cat = np.stack(
        [np.random.randint(n, size=(8,)) for n in num_categories], axis=-1)

    x1_cont = np.ones([x1_cat.shape[0], 2]).astype(np.float32)
    x2_cont = np.ones([x2_cat.shape[0], 2]).astype(np.float32)
    x1 = fswc.ContinuousAndCategoricalValues(x1_cont, x1_cat)
    x2 = fswc.ContinuousAndCategoricalValues(x2_cont, x2_cat)

    sd_continuous = np.random.uniform(size=[2], high=10.).astype(np.float32)
    sd_categorical = [np.random.uniform(size=[n], high=10.).astype(np.float32)
                      for n in num_categories]
    sd = (sd_continuous, sd_categorical)

    def _kernel_mat_first_entry(sd):
      sd_cont, cat_diags = sd
      base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()
      cat_ops = [tf.linalg.LinearOperatorDiag(d) for d in cat_diags]
      kernel = fswec.FeatureScaledWithEmbeddedCategorical(
          base_kernel,
          categorical_embedding_operators=cat_ops,
          continuous_scale_diag=sd_cont,
          validate_args=True)
      return kernel.matrix(x1, x2)[0, 0]

    y, grad = gradient.value_and_gradient(
        _kernel_mat_first_entry, sd, auto_unpack_single_arg=False)
    self.assertAllNotNone(tf.nest.flatten(grad))
    self.assertAllNotNan(y)
    self.assertAllAssertsNested(self.assertAllNotNan, grad)

if __name__ == '__main__':
  test_util.main()
