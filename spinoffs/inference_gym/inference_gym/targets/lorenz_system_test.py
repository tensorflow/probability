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
"""Tests for inference_gym.targets.lorenz_system."""

import functools

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import lorenz_system

_full_observed_data = np.array([
    1.1774399, 2.3005185, 1.3215489, 0.48964, -2.93706, -1.2511115, -0.23023647,
    -0.07261133, -1.4139587, -2.9960678, -2.0675511, -3.706842, -5.97136,
    -6.109783, -5.214157, -6.611416, -7.7250977, -10.704396, -14.218651,
    -18.137701, -17.255384, -19.98641, -22.580467, -22.586863, -21.432135,
    -16.92844, -9.660896, -2.9241529, 3.199096, 7.6405044
]).astype(dtype=np.float32)

_partially_observed_data = np.array([
    1.1774399, 2.3005185, 1.3215489, 0.48964, -2.93706, -1.2511115, -0.23023647,
    -0.07261133, -1.4139587, -2.9960678, np.nan, np.nan, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, -17.255384, -19.98641, -22.580467,
    -22.586863, -21.432135, -16.92844, -9.660896, -2.9241529, 3.199096,
    7.6405044
]).astype(dtype=np.float32)


def _make_dataset(data):
  return dict(
      observed_values=data,
      observation_mask=~np.isnan(data),
      observation_index=0,
      innovation_scale=np.array(.1, np.float32),
      observation_scale=np.array(1., np.float32),
      step_size=np.array(0.02, np.float32))


_test_dataset = functools.partial(_make_dataset, _full_observed_data)


@test_util.multi_backend_test(globals(), 'targets.lorenz_system_test')
class LorenzSystemTest(test_util.InferenceGymTestCase):

  @parameterized.parameters(tf.float32, tf.float64)
  def testLorenzSystem(self, dtype):
    """Checks that unconstrained parameters yield finite joint densities."""
    model = lorenz_system.LorenzSystem(
        **_test_dataset(), use_markov_chain=True, dtype=dtype)
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity=[30, 3]),
        dtype=dtype)

  def testDeferred(self):
    """Checks that the dataset is not prematurely materialized."""
    dataset = _test_dataset()
    func = functools.partial(lorenz_system.LorenzSystem,
                             step_size=dataset.pop('step_size'),
                             innovation_scale=dataset.pop('innovation_scale'),
                             observation_scale=dataset.pop('observation_scale'),
                             observation_index=dataset.pop('observation_index'))
    self.validate_deferred_materialization(func, **dataset)

  @parameterized.parameters(tf.float32, tf.float64)
  def testConvectionLorenzBridge(self, dtype):
    """Checks that unconstrained parameters yield finite joint densities."""
    model = lorenz_system.ConvectionLorenzBridge(
        use_markov_chain=True, dtype=dtype)
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(identity=[30, 3]),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
        dtype=dtype)

  @test_util.numpy_disable_gradient_test
  def testConvectionLorenzBridgeHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    model = lorenz_system.ConvectionLorenzBridge(use_markov_chain=True)

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=2000,
        num_leapfrog_steps=240,
        step_size=0.03,
    )

  def testMarkovChainLogprobMatchesOriginal(self):
    model = lorenz_system.ConvectionLorenzBridge(use_markov_chain=False)
    markov_chain_model = lorenz_system.ConvectionLorenzBridge(
        use_markov_chain=True)

    x = self.evaluate(model.prior_distribution().sample(
        20, seed=tfp_test_util.test_seed()))
    self.assertAllClose(model.unnormalized_log_prob(x),
                        markov_chain_model.unnormalized_log_prob(
                            tf.stack(x, axis=-2)),
                        atol=1e-2)


@test_util.multi_backend_test(globals(), 'targets.lorenz_system_test')
class LorenzSystemUnknownScalesTest(test_util.InferenceGymTestCase):

  @parameterized.parameters(tf.float32, tf.float64)
  def testLorenzSystemUnknownScales(self, dtype):
    """Checks that unconstrained parameters yield finite joint densities."""
    dataset = _test_dataset()
    del dataset['innovation_scale']
    del dataset['observation_scale']
    model = lorenz_system.LorenzSystemUnknownScales(use_markov_chain=True,
                                                    dtype=dtype,
                                                    **dataset)
    self.validate_log_prob_and_transforms(
        model, sample_transformation_shapes=dict(
            identity={'innovation_scale': [],
                      'observation_scale': [],
                      'latents': [30, 3]}),
        dtype=dtype)

  def testDeferred(self):
    """Checks that the dataset is not prematurely materialized."""
    dataset = _test_dataset()
    del dataset['innovation_scale']
    del dataset['observation_scale']
    func = functools.partial(
        lorenz_system.LorenzSystemUnknownScales,
        step_size=dataset.pop('step_size'),
        observation_index=dataset.pop('observation_index'))
    self.validate_deferred_materialization(func, **dataset)

  @parameterized.parameters(tf.float32, tf.float64)
  def testConvectionLorenzBridge(self, dtype):
    """Checks that unconstrained parameters yield finite joint densities."""
    model = lorenz_system.ConvectionLorenzBridgeUnknownScales(
        use_markov_chain=True, dtype=dtype)
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={'innovation_scale': [],
                      'observation_scale': [],
                      'latents': [30, 3]}),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True,
        dtype=dtype)

  @test_util.numpy_disable_gradient_test
  def testConvectionLorenzBridgeHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    self.skipTest('b/171518508')
    model = lorenz_system.ConvectionLorenzBridgeUnknownScales(
        use_markov_chain=True)

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=2,
        num_steps=2000,
        num_leapfrog_steps=240,
        step_size=0.03,
    )

  def testMarkovChainLogprobMatchesOriginal(self):
    model = lorenz_system.ConvectionLorenzBridgeUnknownScales(
        use_markov_chain=False)
    markov_chain_model = lorenz_system.ConvectionLorenzBridgeUnknownScales(
        use_markov_chain=True)

    x = self.evaluate(model.prior_distribution().sample(
        20, seed=tfp_test_util.test_seed()))
    self.assertAllClose(model.unnormalized_log_prob(x),
                        markov_chain_model.unnormalized_log_prob(
                            type(markov_chain_model.dtype)(
                                x[0], x[1], tf.stack(x[2:], axis=-2))),
                        atol=1e-2)


if __name__ == '__main__':
  tfp_test_util.main()
