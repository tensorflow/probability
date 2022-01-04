# Lint as: python3
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
"""Tests for inference_gym.targets.log_gaussian_cox_process."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import log_gaussian_cox_process

BACKEND = None  # Rewritten by the rewrite.py.


def _test_dataset():
  return dict(
      train_locations=np.arange(20).reshape((10, 2)),
      train_extents=1. + (np.arange(10) % 10),
      train_counts=50. + np.arange(10),
  )


@test_util.multi_backend_test(globals(),
                              'targets.log_gaussian_cox_process_test')
class LogGaussianCoxProcessTest(test_util.InferenceGymTestCase):

  def testBasic(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = log_gaussian_cox_process.LogGaussianCoxProcess(**_test_dataset())
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'amplitude': [],
                'length_scale': [],
                'log_intensity': [10],
            },))

  def testCreateDataset(self):
    """Checks that creating a dataset works."""
    # Technically this is private functionality, but we don't have it tested
    # elsewhere.
    model = log_gaussian_cox_process.LogGaussianCoxProcess(**_test_dataset())
    model2 = log_gaussian_cox_process.LogGaussianCoxProcess(
        **model._sample_dataset(tfp_test_util.test_seed()))
    self.validate_log_prob_and_transforms(
        model2,
        sample_transformation_shapes=dict(
            identity={
                'amplitude': [],
                'length_scale': [],
                'log_intensity': [10],
            },))

  def testSyntheticLogGaussianCoxProcess(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = log_gaussian_cox_process.SyntheticLogGaussianCoxProcess()
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'amplitude': [],
                'length_scale': [],
                'log_intensity': [100],
            },),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
    )

  @test_util.numpy_disable_gradient_test
  def testSyntheticLogGaussianCoxProcessHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    model = log_gaussian_cox_process.SyntheticLogGaussianCoxProcess()

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=3000,
        num_leapfrog_steps=5,
        step_size=0.001,
        dtype=tf.float64,
        use_xla=BACKEND == 'backend_jax',  # TF XLA is very slow on this problem
    )


if __name__ == '__main__':
  tfp_test_util.main()
