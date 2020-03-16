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
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for inference_gym.targets.ill_conditioned_gaussian."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.inference_gym.internal import test_util
from tensorflow_probability.python.experimental.inference_gym.targets import ill_conditioned_gaussian
from tensorflow_probability.python.internal import test_util as tfp_test_util


class IllConditionedGaussianTest(test_util.InferenceGymTestCase):

  def testBasic(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = ill_conditioned_gaussian.IllConditionedGaussian()
    self.validate_log_prob_and_transforms(
        model, sample_transformation_shapes=dict(identity=[100],))

  @tfp_test_util.numpy_disable_gradient_test
  @tfp_test_util.jax_disable_test_missing_functionality('tfp.mcmc')
  def testHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    # Note the side-effect of setting the eager seed.
    seed = tfp_test_util.test_seed_stream()
    model = ill_conditioned_gaussian.IllConditionedGaussian(
        ndims=10, gamma_shape_parameter=2., seed=seed())

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=4000,
        num_leapfrog_steps=2,
        step_size=0.5,
        seed=seed(),
    )

  def testMC(self):
    """Checks true samples from the model against the ground truth."""
    # Note the side-effect of setting the eager seed.
    seed = tfp_test_util.test_seed_stream()
    model = ill_conditioned_gaussian.IllConditionedGaussian(
        ndims=10, gamma_shape_parameter=2., seed=seed())

    self.validate_ground_truth_using_monte_carlo(
        model,
        num_samples=int(1e6),
        seed=seed(),
    )


if __name__ == '__main__':
  tf.test.main()
