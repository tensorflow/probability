# Copyright 2018 The TensorFlow Probability Authors.
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
"""Smooth Seasonal Model Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import SmoothSeasonal
from tensorflow_probability.python.sts import SmoothSeasonalStateSpaceModel


class _SmoothSeasonalStateSpaceModelTest(object):

  def test_accepts_tensor_valued_period_and_frequency_multipliers(self):

    period = tf.constant(100.)
    frequency_multipliers = tf.constant([1., 3.])
    drift_scale = tf.constant([2.])

    component = SmoothSeasonal(
        period=period, frequency_multipliers=frequency_multipliers)

    ssm = component.make_state_space_model(
        num_timesteps=3, param_vals=[drift_scale])

    self.assertAllEqual(component.latent_size, 4)
    self.assertAllEqual(ssm.latent_size, 4)

  def test_basic_statistics_no_latent_variance_one_frequency(self):
    # fix the latent variables at the value 1 so the results are deterministic
    num_timesteps = 10
    period = 42
    frequency_multipliers = [3]
    drift_scale = 0.

    initial_state_loc = self._build_placeholder(np.ones([2]))
    initial_state_scale = tf.zeros_like(initial_state_loc)

    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=initial_state_loc, scale_diag=initial_state_scale)

    ssm = SmoothSeasonalStateSpaceModel(
        num_timesteps=num_timesteps,
        period=period,
        frequency_multipliers=frequency_multipliers,
        drift_scale=drift_scale,
        initial_state_prior=initial_state_prior)

    two_pi = 6.283185307179586
    sine_terms = tf.sin(two_pi * 3 * tf.range(
        0, num_timesteps, dtype=tf.float32) / 42)
    cosine_terms = tf.cos(two_pi * 3 * tf.range(
        0, num_timesteps, dtype=tf.float32) / 42)
    predicted_time_series_ = self.evaluate(
        (sine_terms + cosine_terms)[..., tf.newaxis])

    self.assertAllClose(self.evaluate(ssm.mean()), predicted_time_series_)
    self.assertAllClose(*self.evaluate((ssm.stddev(),
                                        tf.zeros_like(predicted_time_series_))))

  def test_matrices_from_component(self):
    num_timesteps = 4
    drift_scale = 1.23
    period = 12
    frequency_multipliers = [1, 3]

    component = SmoothSeasonal(
        period=period, frequency_multipliers=frequency_multipliers)

    ssm = component.make_state_space_model(num_timesteps, [drift_scale])

    frequency_0 = 2 * np.pi * frequency_multipliers[0] / period
    frequency_1 = 2 * np.pi * frequency_multipliers[1] / period

    first_frequency_transition = tf.linalg.LinearOperatorFullMatrix([
        [tf.cos(frequency_0), tf.sin(frequency_0)],
        [-tf.sin(frequency_0), tf.cos(frequency_0)]])

    second_frequency_transition = tf.linalg.LinearOperatorFullMatrix([
        [tf.cos(frequency_1), tf.sin(frequency_1)],
        [-tf.sin(frequency_1), tf.cos(frequency_1)]])

    latents_transition = self.evaluate(tf.linalg.LinearOperatorBlockDiag([
        first_frequency_transition, second_frequency_transition]).to_dense())

    for t in range(num_timesteps):
      observation_matrix = self.evaluate(
          ssm.get_observation_matrix_for_timestep(t).to_dense())

      self.assertAllClose([[1.0, 0.0, 1.0, 0.0]], observation_matrix)

      observation_noise_mean = self.evaluate(
          ssm.get_observation_noise_for_timestep(t).mean())
      observation_noise_covariance = self.evaluate(
          ssm.get_observation_noise_for_timestep(t).covariance())

      self.assertAllClose([0.0], observation_noise_mean)
      self.assertAllClose([[0.0]], observation_noise_covariance)

      transition_matrix = self.evaluate(
          ssm.get_transition_matrix_for_timestep(t).to_dense())

      self.assertAllClose(latents_transition, transition_matrix)

      transition_noise_mean = self.evaluate(
          ssm.get_transition_noise_for_timestep(t).mean())
      transition_noise_covariance = self.evaluate(
          ssm.get_transition_noise_for_timestep(t).covariance())

      self.assertAllClose(np.zeros([4]),
                          transition_noise_mean)
      self.assertAllClose(np.square(drift_scale) * np.eye(4),
                          transition_noise_covariance)

  def _build_placeholder(self, ndarray):
    ndarray = np.asarray(ndarray).astype(self.dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.test_all_tf_execution_regimes
class SmoothSeasonalStateSpaceModelTestStaticShape32(
    test_util.TestCase, _SmoothSeasonalStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class SmoothSeasonalStateSpaceModelTestDynamicShape32(
    test_util.TestCase, _SmoothSeasonalStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class SmoothSeasonalStateSpaceModelTestStaticShape64(
    test_util.TestCase, _SmoothSeasonalStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == "__main__":
  tf.test.main()
