# Lint as: python3
# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for inference_gym.targets.plasma_spectroscopy."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import plasma_spectroscopy

BACKEND = None  # Rewritten by the rewrite.py.


def _test_dataset():
  return dict(
      measurements=np.zeros((20, 30)),
      wavelengths=np.linspace(1., 2., 20),
      center_wavelength=1.5,
  )


@test_util.multi_backend_test(globals(), 'targets.plasma_spectroscopy_test')
class PlasmaSpectroscopyTest(test_util.InferenceGymTestCase):

  @parameterized.named_parameters(
      ('Smooth', True),
      ('NotSmooth', False),
  )
  def testBasic(self, use_bump_function):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.

    Args:
      use_bump_function: Whether to use the bump function.
    """
    model = plasma_spectroscopy.PlasmaSpectroscopy(
        **_test_dataset(), use_bump_function=use_bump_function)
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity=model.dtype._replace(
                amplitude=[16],
                temperature=[16],
                velocity=[16],
                shift=[],
            )))

  def testCreateDataset(self):
    """Checks that creating a dataset works."""
    # Technically this is private functionality, but we don't have it tested
    # elsewhere.
    model = plasma_spectroscopy.PlasmaSpectroscopy(**_test_dataset())
    model2 = plasma_spectroscopy.PlasmaSpectroscopy(
        **model._sample_dataset(tfp_test_util.test_seed())[1])
    self.validate_log_prob_and_transforms(
        model2,
        sample_transformation_shapes=dict(
            identity=model.dtype._replace(
                amplitude=[16],
                temperature=[16],
                velocity=[16],
                shift=[],
            )))

  def testSyntheticPlasmaSpectroscopy(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = plasma_spectroscopy.SyntheticPlasmaSpectroscopy()
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity=model.dtype._replace(
                amplitude=[16],
                temperature=[16],
                velocity=[16],
                shift=[],
            )))

  def testSyntheticPlasmaSpectroscopyWithBump(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = plasma_spectroscopy.SyntheticPlasmaSpectroscopyWithBump()
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity=model.dtype._replace(
                amplitude=[16],
                temperature=[16],
                velocity=[16],
                shift=[],
            )))


if __name__ == '__main__':
  tf.test.main()
