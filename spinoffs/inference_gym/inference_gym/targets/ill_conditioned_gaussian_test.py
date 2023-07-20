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
"""Tests for inference_gym.targets.ill_conditioned_gaussian."""

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import ill_conditioned_gaussian


@test_util.multi_backend_test(globals(),
                              'targets.ill_conditioned_gaussian_test')
class IllConditionedGaussianTest(test_util.InferenceGymTestCase):

  def testBasic(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = ill_conditioned_gaussian.IllConditionedGaussian()
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity=[100],),
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
    )

  @test_util.numpy_disable_gradient_test
  def testHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    model = ill_conditioned_gaussian.IllConditionedGaussian(
        ndims=10,
        gamma_shape_parameter=2.0,
        seed=tfp_test_util.test_seed(sampler_type='integer'),
    )

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=2000,
        num_leapfrog_steps=3,
        step_size=0.5,
        seed=tfp_test_util.test_seed(),
    )

  def testMC(self):
    """Checks true samples from the model against the ground truth."""
    model = ill_conditioned_gaussian.IllConditionedGaussian(
        ndims=10,
        gamma_shape_parameter=2.0,
        seed=tfp_test_util.test_seed(sampler_type='integer'),
    )

    self.validate_ground_truth_using_monte_carlo(
        model,
        num_samples=int(1e6),
        seed=tfp_test_util.test_seed(),
    )


if __name__ == '__main__':
  tfp_test_util.main()
