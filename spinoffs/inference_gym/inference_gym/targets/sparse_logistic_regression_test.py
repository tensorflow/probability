# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for inference_gym.targets.sparse_logistic_regression."""

import functools

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import sparse_logistic_regression


def _test_dataset(num_features, num_test_points=None):
  return dict(
      train_features=tf.zeros([10, num_features]),
      test_features=(tf.zeros([num_test_points, num_features])
                     if num_test_points else None),
      train_labels=tf.zeros([10], dtype=tf.int32),
      test_labels=(tf.zeros([num_test_points], dtype=tf.int32)
                   if num_test_points else None),
  )


class _SparseLogisticRegressionTest(test_util.InferenceGymTestCase):
  positive_constraint_fn = None

  @parameterized.named_parameters(
      ('NoTestF32', None, tf.float32),
      ('WithTestF32', 5, tf.float32),
      ('NoTestF64', None, tf.float64),
      ('WithTestF64', 5, tf.float64),
  )
  def testBasic(self, num_test_points, dtype):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.

    Args:
      num_test_points: Number of test points.
      dtype: Dtype to use for floating point computations.
    """
    num_features = 5
    model = sparse_logistic_regression.SparseLogisticRegression(
        positive_constraint_fn=self.positive_constraint_fn,
        dtype=dtype,
        **_test_dataset(num_features, num_test_points))
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'global_scale': [],
                'local_scales': [num_features + 1],
                'unscaled_weights': [num_features + 1],
            },
            test_nll=[],
            per_example_test_nll=[num_test_points],
        ),
        dtype=dtype,
    )

  @parameterized.named_parameters(
      ('NoTest', None),
      ('WithTest', 5),
  )
  def testDeferred(self, num_test_points):
    """Checks that the dataset is not prematurely materialized."""
    num_features = 5
    kwargs = _test_dataset(num_features, num_test_points)
    self.validate_deferred_materialization(
        functools.partial(
            sparse_logistic_regression.SparseLogisticRegression,
            positive_constraint_fn=self.positive_constraint_fn),
        **kwargs,
    )

  def testPartiallySpecifiedTestSet(self):
    """Check that partially specified test set raises an error."""
    num_features = 5
    num_test_points = 5
    dataset = _test_dataset(num_features, num_test_points)
    del dataset['test_features']
    with self.assertRaisesRegex(ValueError, 'both specified'):
      sparse_logistic_regression.SparseLogisticRegression(
          positive_constraint_fn=self.positive_constraint_fn, **dataset)

  @test_util.uses_tfds
  def testGermanCredit(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = (
        sparse_logistic_regression.GermanCreditNumericSparseLogisticRegression(
            positive_constraint_fn=self.positive_constraint_fn,))
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'global_scale': [],
                'local_scales': [25],
                'unscaled_weights': [25],
            },),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
    )

  @test_util.uses_tfds
  @test_util.numpy_disable_gradient_test
  def testGermanCreditHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    model = (
        sparse_logistic_regression.GermanCreditNumericSparseLogisticRegression(
            positive_constraint_fn=self.positive_constraint_fn,))

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=4000,
        num_leapfrog_steps=40,
        step_size=0.015,
    )


@test_util.multi_backend_test(globals(),
                              'targets.sparse_logistic_regression_test')
class SparseLogisticRegressionExpTest(_SparseLogisticRegressionTest):
  positive_constraint_fn = 'exp'


@test_util.multi_backend_test(globals(),
                              'targets.sparse_logistic_regression_test')
class SparseLogisticRegressionSoftplusTest(_SparseLogisticRegressionTest):
  positive_constraint_fn = 'softplus'


del _SparseLogisticRegressionTest

if __name__ == '__main__':
  tfp_test_util.main()
