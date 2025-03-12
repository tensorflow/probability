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
"""Tests for inference_gym.targets.brownian_motion."""

import functools

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import brownian_motion

_fake_observed_data = np.array([
    0.21592641, 0.118771404, -0.07945447, 0.037677474, -0.27885845, -0.1484156,
    -0.3250906, -0.22957903, -0.44110894, -0.09830782, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -0.8786016,
    -0.83736074, -0.7384849, -0.8939254, -0.7774566, -0.70238715, -0.87771565,
    -0.51853573, -0.6948214, -0.6202789
]).astype(dtype=np.float32)

_small_observed_data = np.array(
    [0.21592641, 0.118771404, np.nan, 0.037677474,
     -0.27885845]).astype(dtype=np.float32)


def _test_dataset():
  return dict(
      observed_locs=_fake_observed_data,
      innovation_noise_scale=.1,
      observation_noise_scale=.15)


@test_util.multi_backend_test(globals(), 'targets.brownian_motion_test')
class BrownianMotionTest(test_util.InferenceGymTestCase):

  @parameterized.parameters(tf.float32, tf.float64)
  def testBrownianMotion(self, dtype):
    """Checks that unconstrained parameters yield finite joint densities."""
    model = brownian_motion.BrownianMotion(
        **_test_dataset(), dtype=dtype
    )
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity=[30]),
        dtype=dtype)

  @parameterized.parameters(tf.float32, tf.float64)
  def testBrownianMotionMissingMiddleObservations(self, dtype):
    """Checks that unconstrained parameters yield finite joint densities."""
    model = brownian_motion.BrownianMotionMissingMiddleObservations(
        dtype=dtype
    )
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity=[30]),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
        dtype=dtype)

  @parameterized.parameters(tf.float32, tf.float64)
  def testBrownianMotionUnknownScalesMissingMiddleObservations(
      self, dtype
  ):
    """Checks that unconstrained parameters yield finite joint densities."""
    model = (
        brownian_motion.BrownianMotionUnknownScalesMissingMiddleObservations(
            dtype=dtype
        )
    )
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={'innovation_noise_scale': [],
                      'observation_noise_scale': [],
                      'locs': [30]}),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
        dtype=dtype)

  def testDeferred(self):
    """Checks that the dataset is not prematurely materialized."""
    kwargs = dict(observed_locs=_small_observed_data)
    func = functools.partial(
        brownian_motion.BrownianMotion,
        observation_noise_scale=0.15,
        innovation_noise_scale=0.1)
    self.validate_deferred_materialization(func, **kwargs)

  @test_util.numpy_disable_gradient_test
  def testBrownianMotionMissingMiddleObservationsHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    model = brownian_motion.BrownianMotionMissingMiddleObservations()

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=4000,
        num_leapfrog_steps=18,
        step_size=0.02,
    )

  @test_util.numpy_disable_gradient_test
  def testBrownianMotionUnknownScalesMissingMiddleObservationsHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    self.skipTest('b/171518508')
    model = (
        brownian_motion.BrownianMotionUnknownScalesMissingMiddleObservations())

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=4000,
        num_leapfrog_steps=15,
        step_size=0.03,
    )

  def testBrownianMotionMarkovChainLogprobMatchesOriginal(self):
    model = (
        brownian_motion.BrownianMotionMissingMiddleObservations(
            use_markov_chain=False))
    markov_chain_model = (
        brownian_motion.BrownianMotionMissingMiddleObservations(
            use_markov_chain=True))

    x = self.evaluate(model.prior_distribution().sample(
        400, seed=tfp_test_util.test_seed()))
    self.assertAllClose(model.unnormalized_log_prob(x),
                        markov_chain_model.unnormalized_log_prob(
                            tf.stack(x, axis=-1)),
                        atol=1e-2)

  def testBrownianMotionUnknownScalesMarkovChainLogprobMatchesOriginal(self):
    model = (
        brownian_motion.BrownianMotionUnknownScalesMissingMiddleObservations(
            use_markov_chain=False))
    markov_chain_model = (
        brownian_motion.BrownianMotionUnknownScalesMissingMiddleObservations(
            use_markov_chain=True))

    x = self.evaluate(model.prior_distribution().sample(
        400, seed=tfp_test_util.test_seed()))
    self.assertAllClose(model.unnormalized_log_prob(x),
                        markov_chain_model.unnormalized_log_prob(
                            type(markov_chain_model.dtype)(
                                x[0], x[1], tf.stack(x[2:], axis=-1))),
                        atol=1e-2)

if __name__ == '__main__':
  tfp_test_util.main()
