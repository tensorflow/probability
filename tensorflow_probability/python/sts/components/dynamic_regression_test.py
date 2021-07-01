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
"""Dynamic Linear Regression State Space Model Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import DynamicLinearRegression
from tensorflow_probability.python.sts import DynamicLinearRegressionStateSpaceModel


class _DynamicLinearRegressionStateSpaceModelTest(object):

  def test_basic_statistics_no_latent_variance(self):
    batch_shape = [4, 3]
    num_timesteps = 10
    num_features = 2
    drift_scale = 0.

    design_matrix = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_timesteps, num_features])))

    initial_state_loc = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_features])))
    initial_state_scale = tf.zeros_like(initial_state_loc)
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=initial_state_loc, scale_diag=initial_state_scale)

    ssm = DynamicLinearRegressionStateSpaceModel(
        num_timesteps=num_timesteps,
        design_matrix=design_matrix,
        drift_scale=drift_scale,
        initial_state_prior=initial_state_prior)

    predicted_time_series = tf.linalg.matmul(
        design_matrix, initial_state_loc[..., tf.newaxis])

    self.assertAllEqual(self.evaluate(ssm.mean()), predicted_time_series)
    self.assertAllEqual(*self.evaluate((ssm.stddev(),
                                        tf.zeros_like(predicted_time_series))))

  def test_initial_state_broadcasts_over_batch(self):
    batch_shape = [4, 3]
    num_timesteps = 10
    num_features = 2
    drift_scale = 1.

    initial_state_loc = self._build_placeholder([0.1, -0.2])
    initial_state_scale = self._build_placeholder([0.42, 0.314])

    design_matrix = self._build_placeholder(
        np.random.randn(*(batch_shape + [num_timesteps, num_features])))

    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=initial_state_loc, scale_diag=initial_state_scale)

    ssm = DynamicLinearRegressionStateSpaceModel(
        num_timesteps=num_timesteps,
        design_matrix=design_matrix,
        drift_scale=drift_scale,
        initial_state_prior=initial_state_prior)

    sample = ssm.sample()

    ll, means, _, _, _, _, _ = ssm.forward_filter(sample)

    self.assertAllEqual(batch_shape + [num_timesteps, 1],
                        self.evaluate(sample).shape)

    self.assertAllEqual(batch_shape + [num_timesteps],
                        self.evaluate(ll).shape)

    self.assertAllEqual(batch_shape + [num_timesteps, num_features],
                        self.evaluate(means).shape)

  def test_matrices_from_component(self):
    num_timesteps = 4
    num_features = 3
    drift_scale = 1.23

    design_matrix = self._build_placeholder(
        np.random.randn(num_timesteps, num_features))

    component = DynamicLinearRegression(design_matrix=design_matrix)

    ssm = component.make_state_space_model(num_timesteps, [drift_scale])

    for t in range(num_timesteps):
      observation_matrix = self.evaluate(
          ssm.get_observation_matrix_for_timestep(t).to_dense())

      self.assertAllClose(self.evaluate(design_matrix[tf.newaxis, t]),
                          observation_matrix)

      observation_noise_mean = self.evaluate(
          ssm.get_observation_noise_for_timestep(t).mean())
      observation_noise_covariance = self.evaluate(
          ssm.get_observation_noise_for_timestep(t).covariance())

      self.assertAllClose([0.0], observation_noise_mean)
      self.assertAllClose([[0.0]], observation_noise_covariance)

      transition_matrix = self.evaluate(
          ssm.get_transition_matrix_for_timestep(t).to_dense())

      self.assertAllClose(np.eye(num_features), transition_matrix)

      transition_noise_mean = self.evaluate(
          ssm.get_transition_noise_for_timestep(t).mean())
      transition_noise_covariance = self.evaluate(
          ssm.get_transition_noise_for_timestep(t).covariance())

      self.assertAllClose(np.zeros([num_features]),
                          transition_noise_mean)
      self.assertAllClose(np.square(drift_scale) * np.eye(num_features),
                          transition_noise_covariance)

  def _build_placeholder(self, ndarray):
    ndarray = np.asarray(ndarray).astype(self.dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.test_all_tf_execution_regimes
class DynamicRegressionStateSpaceModelTestStaticShape32(
    test_util.TestCase, _DynamicLinearRegressionStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class DynamicRegressionStateSpaceModelTestDynamicShape32(
    test_util.TestCase, _DynamicLinearRegressionStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class DynamicRegressionStateSpaceModelTestStaticShape64(
    test_util.TestCase, _DynamicLinearRegressionStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == "__main__":
  tf.test.main()
