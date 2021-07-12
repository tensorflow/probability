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
"""Tests for inference_gym.targets.eight_schools."""

import tensorflow.compat.v2 as tf

from inference_gym.internal import test_util
from inference_gym.targets import eight_schools


@test_util.multi_backend_test(globals(), 'targets.eight_schools_test')
class EightSchoolsTest(test_util.InferenceGymTestCase):

  def testEightSchools(self):
    """Checks that unconstrained parameters yield finite joint densities."""
    model = eight_schools.EightSchools()
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity={
            'avg_effect': [],
            'log_stddev': [],
            'school_effects': [8],
        }),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True)

  @test_util.numpy_disable_gradient_test
  def testEightSchoolsHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    model = eight_schools.EightSchools()

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=4000,
        num_leapfrog_steps=10,
        step_size=0.4,
    )

if __name__ == '__main__':
  tf.test.main()
