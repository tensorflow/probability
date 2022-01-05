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
"""ARIMA tests."""

import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import AutoregressiveMovingAverageStateSpaceModel
from tensorflow_probability.python.sts import IntegratedStateSpaceModel

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_graph_and_eager_modes
class IntegratedStateSpaceModelTest(test_util.TestCase):

  def test_sum_of_white_noise_is_random_walk(self):
    num_timesteps = 20
    level_scale = 0.6
    noise_scale = 0.3
    random_walk_ssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=[[1.]],
        transition_noise=tfd.MultivariateNormalDiag(
            loc=[0.], scale_diag=[level_scale]),
        observation_matrix=[[1.]],
        observation_noise=tfd.MultivariateNormalDiag(
            loc=[0.], scale_diag=[noise_scale]),
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=[0.], scale_diag=[level_scale]))

    white_noise_ssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=[[0.]],
        transition_noise=tfd.MultivariateNormalDiag(
            loc=[0.], scale_diag=[level_scale]),
        observation_matrix=[[1.]],
        observation_noise=tfd.MultivariateNormalDiag(
            loc=[0.], scale_diag=[noise_scale]),
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=[0.], scale_diag=[level_scale]))
    cumsum_white_noise_ssm = IntegratedStateSpaceModel(white_noise_ssm)
    x, lp = cumsum_white_noise_ssm.experimental_sample_and_log_prob(
        [3], seed=test_util.test_seed())
    self.assertAllClose(lp, random_walk_ssm.log_prob(x), atol=1e-5)

  def test_noiseless_is_consistent_with_cumsum_bijector(self):
    num_timesteps = 10
    ssm = AutoregressiveMovingAverageStateSpaceModel(
        num_timesteps=num_timesteps,
        ar_coefficients=[0.7, -0.2, 0.1],
        ma_coefficients=[0.6],
        level_scale=0.6,
        level_drift=-0.3,
        observation_noise_scale=0.,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=tf.zeros([3]), scale_diag=tf.ones([3])))
    cumsum_ssm = IntegratedStateSpaceModel(ssm)
    x, lp = cumsum_ssm.experimental_sample_and_log_prob(
        [2], seed=test_util.test_seed())

    flatten_event = tfb.Reshape([num_timesteps],
                                event_shape_in=[num_timesteps, 1])
    cumsum_dist = tfb.Chain([tfb.Invert(flatten_event),
                             tfb.Cumsum(),
                             flatten_event])(ssm)
    self.assertAllClose(lp, cumsum_dist.log_prob(x), atol=1e-5)


if __name__ == '__main__':
  test_util.main()
