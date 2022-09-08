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
"""Local Linear Trend State Space Model Tests."""

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts.components.local_linear_trend import LocalLinearTrendStateSpaceModel


tfl = tf.linalg


class _LocalLinearTrendStateSpaceModelTest(object):

  def test_logprob(self):

    y = self._build_placeholder([1.0, 2.5, 4.3, 6.1, 7.8])

    ssm = LocalLinearTrendStateSpaceModel(
        num_timesteps=5,
        level_scale=0.5,
        slope_scale=0.5,
        initial_state_prior=mvn_diag.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([1., 1.])))

    lp = ssm.log_prob(y[..., np.newaxis])
    expected_lp = -5.801624298095703
    self.assertAllClose(self.evaluate(lp), expected_lp)

  def test_stats(self):

    # Build a model with expected initial loc 0 and slope 1.
    level_scale = self._build_placeholder(1.0)
    slope_scale = self._build_placeholder(1.0)
    initial_state_prior = mvn_diag.MultivariateNormalDiag(
        loc=self._build_placeholder([0, 1.]),
        scale_diag=self._build_placeholder([1., 1.]))

    ssm = LocalLinearTrendStateSpaceModel(
        num_timesteps=10,
        level_scale=level_scale,
        slope_scale=slope_scale,
        initial_state_prior=initial_state_prior)

    # In expectation, the process grows linearly.
    mean = self.evaluate(ssm.mean())
    self.assertAllClose(mean, np.arange(0, 10)[:, np.newaxis])

    # slope variance at time T is linear: T * slope_scale
    expected_variance = [1, 3, 8, 18, 35, 61, 98, 148, 213, 295]
    variance = self.evaluate(ssm.variance())
    self.assertAllClose(variance, np.array(expected_variance)[:, np.newaxis])

  def test_batch_shape(self):
    batch_shape = [4, 2]
    partial_batch_shape = [2]

    level_scale = self._build_placeholder(
        np.exp(np.random.randn(*partial_batch_shape)))
    slope_scale = self._build_placeholder(np.exp(np.random.randn(*batch_shape)))
    initial_state_prior = mvn_diag.MultivariateNormalDiag(
        scale_diag=self._build_placeholder([1., 1.]))

    ssm = LocalLinearTrendStateSpaceModel(
        num_timesteps=10,
        level_scale=level_scale,
        slope_scale=slope_scale,
        initial_state_prior=initial_state_prior)
    self.assertAllEqual(self.evaluate(ssm.batch_shape_tensor()), batch_shape)

    seed = test_util.test_seed(sampler_type='stateless')
    y = ssm.sample(seed=seed)
    self.assertAllEqual(self.evaluate(tf.shape(y))[:-2], batch_shape)

  def test_joint_sample(self):
    seed = test_util.test_seed(sampler_type='stateless')
    batch_shape = [4, 3]

    level_scale = self._build_placeholder(2 * np.ones(batch_shape))
    slope_scale = self._build_placeholder(0.2 * np.ones(batch_shape))
    observation_noise_scale = self._build_placeholder(1.)
    initial_state_prior = mvn_diag.MultivariateNormalDiag(
        loc=self._build_placeholder([-3, 0.5]),
        scale_diag=self._build_placeholder([0.1, 0.2]))

    ssm = LocalLinearTrendStateSpaceModel(
        num_timesteps=10,
        level_scale=level_scale,
        slope_scale=slope_scale,
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=initial_state_prior)

    num_samples = 10000
    sampled_latents, sampled_obs = ssm._joint_sample_n(n=num_samples,
                                                       seed=seed)
    latent_mean, obs_mean = ssm._joint_mean()
    latent_cov, obs_cov = ssm._joint_covariances()
    (sampled_latents_, sampled_obs_,
     latent_mean_, obs_mean_,
     latent_level_std_,
     level_slope_std_,
     obs_std_) = self.evaluate(
         (sampled_latents, sampled_obs,
          latent_mean, obs_mean,
          tf.sqrt(latent_cov[..., 0, 0]),
          tf.sqrt(latent_cov[..., 1, 1]),
          tf.sqrt(obs_cov[..., 0])))
    latent_std_ = np.stack([latent_level_std_, level_slope_std_], axis=-1)

    # Instead of directly comparing means and stddevs, we normalize by stddev
    # to make the stderr constant.
    self.assertAllClose(np.mean(sampled_latents_, axis=0) / latent_std_,
                        latent_mean_ / latent_std_,
                        atol=4. / np.sqrt(num_samples))
    self.assertAllClose(np.mean(sampled_obs_, axis=0) / obs_std_,
                        obs_mean_ / obs_std_,
                        atol=4. / np.sqrt(num_samples))
    self.assertAllClose(np.std(sampled_latents_, axis=0) / latent_std_,
                        np.ones(latent_std_.shape, dtype=latent_std_.dtype),
                        atol=4. / np.sqrt(num_samples))
    self.assertAllClose(np.std(sampled_obs_, axis=0) / obs_std_,
                        np.ones(obs_std_.shape, dtype=obs_std_.dtype),
                        atol=4. / np.sqrt(num_samples))

  def _build_placeholder(self, ndarray):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype`, and shape specified
      statically only if `self.use_static_shape` is `True`.
    """

    ndarray = np.asarray(ndarray).astype(self.dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.test_all_tf_execution_regimes
class LocalLinearTrendStateSpaceModelTestStaticShape32(
    test_util.TestCase, _LocalLinearTrendStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class LocalLinearTrendStateSpaceModelTestDynamicShape32(
    test_util.TestCase, _LocalLinearTrendStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class LocalLinearTrendStateSpaceModelTestStaticShape64(
    test_util.TestCase, _LocalLinearTrendStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == '__main__':
  test_util.main()
