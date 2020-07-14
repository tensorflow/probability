# Lint as: python2, python3
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
# See the License for the modelific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for inference_gym.targets.probit_regression."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.inference_gym.internal import test_util
from tensorflow_probability.python.experimental.inference_gym.targets import probit_regression
from tensorflow_probability.python.internal import test_util as tfp_test_util


def _test_dataset(num_features, num_test_points=None):
  return dict(
      train_features=tf.zeros([10, num_features]),
      test_features=(tf.zeros([num_test_points, num_features])
                     if num_test_points else None),
      train_labels=tf.zeros([10], dtype=tf.int32),
      test_labels=(tf.zeros([num_test_points], dtype=tf.int32)
                   if num_test_points else None),
  )


class ProbitRegressionTest(test_util.InferenceGymTestCase,
                           parameterized.TestCase):

  @parameterized.named_parameters(
      ('NoTest', None),
      ('WithTest', 5),
  )
  def testBasic(self, num_test_points):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.

    Args:
      num_test_points: Number of test points.
    """
    num_features = 5
    model = probit_regression.ProbitRegression(
        **_test_dataset(num_features, num_test_points))
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity=[num_features + 1],
            test_nll=[],
            per_example_test_nll=[num_test_points],
        ))

  def testPartiallySpecifiedTestSet(self):
    """Check that partially specified test set raises an error."""
    num_features = 5
    num_test_points = 5
    dataset = _test_dataset(num_features, num_test_points)
    del dataset['test_features']
    with self.assertRaisesRegex(ValueError, 'both specified'):
      probit_regression.ProbitRegression(**dataset)

  @test_util.uses_tfds
  def testGermanCredit(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = probit_regression.GermanCreditNumericProbitRegression()
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity=[25],),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
    )

  @test_util.uses_tfds
  @tfp_test_util.numpy_disable_gradient_test
  def testGermanCreditHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    model = probit_regression.GermanCreditNumericProbitRegression()

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=4000,
        num_leapfrog_steps=15,
        step_size=0.03,
    )


if __name__ == '__main__':
  tf.test.main()
