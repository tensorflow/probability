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
"""Tests for inference_gym.internal.test_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.experimental.inference_gym.internal import test_util
from tensorflow_probability.python.internal import test_util as tfp_test_util

gym = tfp.experimental.inference_gym
tfd = tfp.distributions
tfb = tfp.bijectors


class TestModel(gym.targets.BayesianModel):

  def __init__(self, bijector=None, ground_truth_mean=np.exp(1.5)):
    """Creates a test model.

    The defaults are 'correct', in that they correspond to the joint
    distribution defined herein. You can adjust them to make the model
    'incorrect' for testing purposes, e.g. an 'incorrect' `bijector` might not
    correctly constrain the values to the domain of the joint distribution. An
    'incorrect' `ground_truth_mean` is different from the true mean of the joint
    distribution.

    Args:
      bijector: Default event space bijector to use. Default: `tfb.Exp`.
      ground_truth_mean: What ground truth mean to use for the `'identity'`
        sample transformation.
    """
    self._distribution = tfd.LogNormal(1., 1.)

    if bijector is None:
      bijector = tfb.Exp()

    super(TestModel, self).__init__(
        default_event_space_bijector=bijector,
        event_shape=self._distribution.event_shape,
        dtype=self._distribution.dtype,
        name='test_model',
        pretty_name='TestModel',
        sample_transformations=dict(
            identity=gym.targets.BayesianModel.SampleTransformation(
                fn=lambda x: x,
                pretty_name='Identity',
                ground_truth_mean=ground_truth_mean,
            ),),
    )

  def _joint_distribution(self):
    return self._distribution

  def _evidence(self):
    return

  def _unnormalized_log_prob(self, x):
    return self.joint_distribution().log_prob(x)

  def sample(self, n, seed=None):
    return self.joint_distribution().sample(n, seed=seed)


class InferenceGymTestCaseTest(test_util.InferenceGymTestCase):

  def testWellFormedModel(self):
    """A well formed model won't raise an error."""
    model = TestModel()
    self.validate_log_prob_and_transforms(
        model, sample_transformation_shapes=dict(identity=[]))

  def testBadBijector(self):
    """Tests that an error is raised if bijector is incorrect."""
    model = TestModel(tfb.Identity())
    with self.assertRaisesRegexp(AssertionError, 'Arrays are not equal'):
      self.validate_log_prob_and_transforms(
          model, sample_transformation_shapes=dict(identity=[]))

  def testBadShape(self):
    """Tests that an error is raised if expectations shapes are wrong."""
    model = TestModel()
    with self.assertRaisesRegexp(AssertionError, 'Tuples differ'):
      self.validate_log_prob_and_transforms(
          model, sample_transformation_shapes=dict(identity=[13]))

  @tfp_test_util.numpy_disable_gradient_test
  @tfp_test_util.jax_disable_test_missing_functionality('tfp.mcmc')
  def testCorrectGroundTruthWithHMC(self):
    """Tests the ground truth with HMC for a well formed model."""
    model = TestModel()
    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=2000,
        num_leapfrog_steps=10,
        step_size=1.,
    )

  @tfp_test_util.numpy_disable_gradient_test
  @tfp_test_util.jax_disable_test_missing_functionality('tfp.mcmc')
  def testBadGroundTruthWithHMC(self):
    """Tests that an error is raised if the ground truth is wrong."""
    model = TestModel(ground_truth_mean=-10.)
    with self.assertRaisesRegexp(AssertionError, 'Not equal to tolerance'):
      self.validate_ground_truth_using_hmc(
          model,
          num_chains=4,
          num_steps=2000,
          num_leapfrog_steps=10,
          step_size=1.,
      )

  def testCorrectGroundTruthWithMC(self):
    """Tests the ground truth with MC for a well formed model."""
    model = TestModel()
    self.validate_ground_truth_using_monte_carlo(
        model,
        num_samples=4000,
    )

  def testBadGroundTruthWithMC(self):
    """Tests that an error is raised if the ground truth is wrong."""
    model = TestModel(ground_truth_mean=-10.)
    with self.assertRaisesRegexp(AssertionError, 'Not equal to tolerance'):
      self.validate_ground_truth_using_monte_carlo(
          model,
          num_samples=4000,
      )


if __name__ == '__main__':
  tf.test.main()
