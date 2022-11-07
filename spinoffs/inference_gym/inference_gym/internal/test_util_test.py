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

import atexit
import functools

from absl import logging
import jax
import numpy as np
import tensorflow.compat.v2 as tf
# Don't rewrite this one.
import tensorflow.compat.v2 as otf  # pylint: disable=reimported
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym import targets
from inference_gym.internal import test_util

tfd = tfp.distributions
tfb = tfp.bijectors

BACKEND = None  # Rewritten by backends/rewrite.py.


class TestModel(targets.Model):

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
            identity=targets.Model.SampleTransformation(
                fn=lambda x: x,
                pretty_name='Identity',
                ground_truth_mean=ground_truth_mean,
            ),),
    )

  def _unnormalized_log_prob(self, x):
    return self._distribution.log_prob(x)

  def sample(self, n, seed=None):
    return self._distribution.sample(n, seed=seed)


class TestModelWithDataset(targets.Model):

  def __init__(self, dataset, materialize_dataset=False):
    """Creates a model that uses a dataset.

    Args:
      dataset: A 1D Tensor.
      materialize_dataset: Python `bool`. If True, the dataset is erroneously
        materialized in the initializer.
    """
    self._distribution = tfd.Sample(tfd.Normal(0., 1.), dataset.shape[-1])
    if materialize_dataset:
      tf.convert_to_tensor(dataset)

    super(TestModelWithDataset, self).__init__(
        default_event_space_bijector=tfb.Identity(),
        event_shape=self._distribution.event_shape,
        dtype=self._distribution.dtype,
        name='test_model_width_dataset',
        pretty_name='TestModelWithDataset',
        sample_transformations=dict(
            identity=targets.Model.SampleTransformation(
                fn=lambda x: x,
                pretty_name='Identity',
                ground_truth_mean=0.,
            ),),
    )

  def _unnormalized_log_prob(self, x):
    return self._distribution.log_prob(x)


@test_util.multi_backend_test(globals(), 'internal.test_util_test')
class InferenceGymTestCaseTest(test_util.InferenceGymTestCase):

  # See _check_backends below.
  _BACKEND_TESTED = BACKEND

  def testActuallyRewritten(self):
    """Tests that this file is rewritten to actually use different backends."""
    a = tf.constant([0.]) + 1.
    expected_type = {
        'backend_tensorflow': otf.Tensor,
        'backend_numpy': np.ndarray,
        'backend_jax': jax.Array,
    }[BACKEND]
    self.assertIsInstance(a, expected_type)

  def testWellFormedModel(self):
    """A well formed model won't raise an error."""
    model = TestModel()
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity=[]),
        check_ground_truth_mean=True)

  def testBadBijector(self):
    """Tests that an error is raised if bijector is incorrect."""
    model = TestModel(tfb.Identity())
    with self.assertRaisesRegex(AssertionError, 'Arrays are not equal'):
      self.validate_log_prob_and_transforms(
          model, sample_transformation_shapes=dict(identity=[]))

  def testBadShape(self):
    """Tests that an error is raised if expectations shapes are wrong."""
    model = TestModel()
    with self.assertRaisesRegex(AssertionError,
                                r'Checking outputs(.|\n)*Tuples differ'):
      self.validate_log_prob_and_transforms(
          model, sample_transformation_shapes=dict(identity=[13]))

  def testBadGroundTruth(self):
    """Tests that an error is raised if ground truth shapes are wrong."""
    model = TestModel(ground_truth_mean=np.array([1, 2]))
    with self.assertRaisesRegex(
        AssertionError, r'Checking ground truth mean(.|\n)*Tuples differ'):
      self.validate_log_prob_and_transforms(
          model,
          sample_transformation_shapes=dict(identity=[]),
          check_ground_truth_mean=True)

  @test_util.numpy_disable_gradient_test
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

  @test_util.numpy_disable_gradient_test
  def testBadGroundTruthWithHMC(self):
    """Tests that an error is raised if the ground truth is wrong."""
    model = TestModel(ground_truth_mean=-1000.)
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

  def testCorrectMaterialization(self):
    """Tests the dataset materialization a well formed model."""
    dataset = np.zeros(5)
    self.validate_deferred_materialization(
        TestModelWithDataset, dataset=dataset)

  def testBadMaterialization(self):
    """Tests that an error is raised if dataset materialization is wrong."""
    dataset = np.zeros(5)
    with self.assertRaisesRegexp(AssertionError,
                                 'Erroneously materialized dataset'):
      self.validate_deferred_materialization(
          functools.partial(TestModelWithDataset, materialize_dataset=True),
          dataset=dataset)


def _check_backends():
  """Check that all the backends were tested.

  This works by collecting all the test cases in globals() and verifying that
  they were rewritten correctly. We assume that after that point, the test
  system works.
  """
  tested_backends = set()
  for test_case in globals().values():
    if hasattr(test_case, '_BACKEND_TESTED'):
      tested_backends.add(test_case._BACKEND_TESTED)

  expected_backends = set(
      ['backend_numpy', 'backend_jax', 'backend_tensorflow'])
  assert tested_backends == expected_backends, 'Missing backends: {}'.format(
      expected_backends - tested_backends)
  logging.info('Tested Inference Gym backends: %s.',
               ', '.join(expected_backends))


if BACKEND is None:
  # Only test this in the original module.
  atexit.register(_check_backends)

if __name__ == '__main__':
  tfp_test_util.main()
