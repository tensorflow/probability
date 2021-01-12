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
"""Tests for radon_contextual_effects."""

import functools

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import radon_contextual_effects


def _test_dataset(train_size, test_size=None, num_counties=3):
  return dict(
      train_log_uranium=np.random.rand(train_size).astype(np.float32),
      train_county=np.arange(train_size) % num_counties,
      train_floor=np.arange(train_size) % 2,
      train_floor_by_county=np.random.rand(train_size).astype(np.float32),
      train_log_radon=np.random.rand(train_size).astype(np.float32),
      num_counties=np.array(num_counties, dtype=np.int32),
      test_log_uranium=(np.random.rand(test_size).astype(np.float32)
                        if test_size else None),
      test_county=(np.arange(test_size) % num_counties if test_size else None),
      test_floor=(np.arange(test_size) % 2 if test_size else None),
      test_floor_by_county=(np.random.rand(test_size).astype(np.float32)
                            if test_size else None),
      test_log_radon=(np.random.rand(test_size).astype(np.float32)
                      if test_size else None),
      )


class _RadonContextualEffectsTest(test_util.InferenceGymTestCase):

  @parameterized.named_parameters(
      ('NoTest', None),
      ('WithTest', 5),
  )
  def testBasic(self, test_size):
    """Checks that unconstrained parameters yield finite joint densities."""
    num_counties = 3
    train_size = 20
    model = radon_contextual_effects.RadonContextualEffects(
        prior_scale=self.prior_scale,
        **_test_dataset(train_size, test_size, num_counties))
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'county_effect_mean': [],
                'county_effect_scale': [],
                'county_effect': [num_counties],
                'weight': [3],
                'log_radon_scale': []
            },
            test_nll=[],
            per_example_test_nll=[test_size]))

  @parameterized.named_parameters(
      ('NoTest', None),
      ('WithTest', 5),
  )
  def testDeferred(self, test_size):
    """Checks that the dataset is not prematurely materialized."""
    num_counties = 3
    train_size = 20
    kwargs = _test_dataset(train_size, test_size, num_counties)
    del kwargs['num_counties']
    self.validate_deferred_materialization(
        functools.partial(
            radon_contextual_effects.RadonContextualEffects,
            prior_scale=self.prior_scale,
            num_counties=num_counties), **kwargs)

  def testPartiallySpecifiedTestSet(self):
    """Check that partially specified test set raises an error."""
    num_counties = 3
    test_size = 5
    dataset = _test_dataset(
        train_size=20, test_size=test_size, num_counties=num_counties)
    del dataset['test_county']
    with self.assertRaisesRegex(ValueError, 'all be specified'):
      radon_contextual_effects.RadonContextualEffects(
          prior_scale=self.prior_scale, **dataset)

  @parameterized.named_parameters(
      ('NoTest', None),
      ('WithTest', 5),
  )
  def testCreateDataset(self, test_size):
    """Checks that creating a dataset works."""
    train_size = 30
    num_counties = 3
    model = radon_contextual_effects.RadonContextualEffects(
        prior_scale=self.prior_scale,
        **_test_dataset(train_size, test_size, num_counties))
    model2 = radon_contextual_effects.RadonContextualEffects(
        prior_scale=self.prior_scale,
        **model._sample_dataset(tfp_test_util.test_seed()))
    self.validate_log_prob_and_transforms(
        model2,
        sample_transformation_shapes=dict(
            identity={
                'county_effect_mean': [],
                'county_effect_scale': [],
                'county_effect': [num_counties],
                'weight': [3],
                'log_radon_scale': []
            },
            test_nll=[],
            per_example_test_nll=[test_size]))

  @test_util.uses_tfds
  def testRadon(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    num_counties = 85
    model = self.build_model()
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'county_effect_mean': [],
                'county_effect_scale': [],
                'county_effect': [num_counties],
                'weight': [3],
                'log_radon_scale': []
            },
            test_nll=[],
            per_example_test_nll=[]),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
    )

  @test_util.uses_tfds
  @tfp_test_util.numpy_disable_gradient_test
  def testRadonHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    model = self.build_model()
    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=4000,
        num_leapfrog_steps=15,
        step_size=0.03,
        dtype=tf.float64,
        standard_deviation_fudge_atol=1e-4)


@test_util.multi_backend_test(globals(),
                              'targets.radon_contextual_effects_test')
class RadonContextualEffectsTest(_RadonContextualEffectsTest):
  prior_scale = 'uniform'
  build_model = radon_contextual_effects.RadonContextualEffectsMinnesota

  def testInvalidPriorScaleRaises(self):
    with self.assertRaisesRegex(ValueError, 'not a valid value'):
      radon_contextual_effects.RadonContextualEffects(
          prior_scale='invalid_input',
          **_test_dataset(train_size=20))


@test_util.multi_backend_test(globals(),
                              'targets.radon_contextual_effects_test')
class RadonContextualEffectsHalfNormalTest(_RadonContextualEffectsTest):
  prior_scale = 'halfnormal'
  build_model = radon_contextual_effects.RadonContextualEffectsHalfNormalMinnesota


del _RadonContextualEffectsTest


if __name__ == '__main__':
  tf.test.main()
