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
"""Tests for inference_gym.targets.item_response_theory."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import item_response_theory


def _test_dataset(num_test_pairs=None):
  return dict(
      train_student_ids=np.arange(20),
      train_question_ids=(np.arange(20) % 10),
      train_correct=np.arange(20) % 2,
      test_student_ids=(np.arange(num_test_pairs) if num_test_pairs else None),
      test_question_ids=(np.arange(num_test_pairs) %
                         10 if num_test_pairs else None),
      test_correct=(np.arange(num_test_pairs) % 2 if num_test_pairs else None),
  )


@test_util.multi_backend_test(globals(), 'targets.item_response_theory_test')
class ItemResponseTheoryTest(test_util.InferenceGymTestCase,
                             parameterized.TestCase):

  @parameterized.named_parameters(
      ('NoTest', None),
      ('WithTest', 5),
  )
  @test_util.numpy_disable_test_missing_functionality(
      'tf.gather_nd and batch_dims > 0')
  def testBasic(self, num_test_points):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.

    Args:
      num_test_points: Number of test points.
    """
    model = item_response_theory.ItemResponseTheory(
        **_test_dataset(num_test_points))
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'mean_student_ability': [],
                'centered_student_ability': [20],
                'question_difficulty': [10],
            },
            test_nll=[],
            per_example_test_nll=[num_test_points],
        ))

  def testPartiallySpecifiedTestSet(self):
    """Check that partially specified test set raises an error."""
    num_test_points = 5
    dataset = _test_dataset(num_test_points)
    del dataset['test_student_ids']
    with self.assertRaisesRegex(ValueError, 'all be specified'):
      item_response_theory.ItemResponseTheory(**dataset)

  @parameterized.named_parameters(
      ('NoTest', None),
      ('WithTest', 5),
  )
  @test_util.numpy_disable_test_missing_functionality(
      'tf.gather_nd and batch_dims > 0')
  def testCreateDataset(self, num_test_points):
    """Checks that creating a dataset works."""
    # Technically this is private functionality, but we don't have it tested
    # elsewhere.
    if not tf.executing_eagerly():
      self.skipTest('This is Eager only for now due to _sample_dataset being '
                    'Eager-only.')
    model = item_response_theory.ItemResponseTheory(
        **_test_dataset(num_test_points))
    model2 = item_response_theory.ItemResponseTheory(
        **model._sample_dataset(tfp_test_util.test_seed()))
    self.validate_log_prob_and_transforms(
        model2,
        sample_transformation_shapes=dict(
            identity={
                'mean_student_ability': [],
                'centered_student_ability': [20],
                'question_difficulty': [10],
            },
            test_nll=[],
            per_example_test_nll=[num_test_points],
        ))

  def testSyntheticItemResponseTheory(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = item_response_theory.SyntheticItemResponseTheory()
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'mean_student_ability': [],
                'centered_student_ability': [400],
                'question_difficulty': [100],
            },),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
    )

  @test_util.numpy_disable_gradient_test
  def testSyntheticItemResponseTheoryHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    model = item_response_theory.SyntheticItemResponseTheory()

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=4000,
        num_leapfrog_steps=10,
        step_size=0.025,
    )


if __name__ == '__main__':
  tfp_test_util.main()
