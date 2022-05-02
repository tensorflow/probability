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
"""Tests for inference_gym.targets.non_identifiable_quartic."""

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import non_identifiable_quartic


@test_util.multi_backend_test(globals(),
                              'targets.non_identifiable_quartic_test')
class NonIdentifiableQuarticMeasurementModelTest(test_util.InferenceGymTestCase
                                                ):

  def testBasic(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = non_identifiable_quartic.NonIdentifiableQuarticMeasurementModel(
        ndims=3)
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity=[3]),
        check_ground_truth_mean=True,
    )


if __name__ == '__main__':
  tfp_test_util.main()
