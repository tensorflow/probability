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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import LocalLevelStateSpaceModel


@test_util.test_all_tf_execution_regimes
class _LocalLevelStateSpaceModelTest(object):

  def test_logprob(self):
    y = self._build_placeholder([1.0, 1.3, 1.9, 2.9, 2.1])

    ssm = LocalLevelStateSpaceModel(
        num_timesteps=5,
        level_scale=0.5,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([1.])))

    lp = ssm.log_prob(y[..., np.newaxis])
    expected_lp = -6.5021
    self.assertAllClose(self.evaluate(lp), expected_lp)

  def test_stats(self):
    # Build a model with expected initial scale 0.
    level_scale = self._build_placeholder(1.0)
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=self._build_placeholder([0.]),
        scale_diag=self._build_placeholder([1.]))

    ssm = LocalLevelStateSpaceModel(
        num_timesteps=10,
        level_scale=level_scale,
        initial_state_prior=initial_state_prior)

    # In expectation, the process is constant.
    mean = self.evaluate(ssm.mean())
    self.assertAllClose(mean, np.zeros(10)[:, np.newaxis])

    # variance of level[T] is T * level_scale
    expected_variance = np.arange(1, 11)[:, np.newaxis]
    variance = self.evaluate(ssm.variance())
    self.assertAllClose(variance, expected_variance)

  def test_batch_shape(self):
    batch_shape = [4, 2]

    level_scale = self._build_placeholder(
        np.exp(np.random.randn(*batch_shape)))
    initial_state_prior = tfd.MultivariateNormalDiag(
        scale_diag=self._build_placeholder([1.]))

    ssm = LocalLevelStateSpaceModel(
        num_timesteps=10,
        level_scale=level_scale,
        initial_state_prior=initial_state_prior)
    self.assertAllEqual(self.evaluate(ssm.batch_shape_tensor()), batch_shape)

    y = ssm.sample()
    self.assertAllEqual(self.evaluate(tf.shape(y))[:-2], batch_shape)

  def test_joint_sample(self):
    strm = test_util.test_seed_stream()
    batch_shape = [4, 2]

    level_scale = self._build_placeholder(2 * np.ones(batch_shape))
    observation_noise_scale = self._build_placeholder(1.)
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=self._build_placeholder([-3]),
        scale_diag=self._build_placeholder([1.]))

    ssm = LocalLevelStateSpaceModel(
        num_timesteps=10,
        level_scale=level_scale,
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=initial_state_prior)

    num_samples = 10000
    sampled_latents, sampled_obs = ssm._joint_sample_n(n=num_samples,
                                                       seed=strm())
    latent_mean, obs_mean = ssm._joint_mean()
    latent_cov, obs_cov = ssm._joint_covariances()
    (sampled_latents_, sampled_obs_,
     latent_mean_, obs_mean_,
     latent_std_, obs_std_) = self.evaluate((sampled_latents, sampled_obs,
                                             latent_mean, obs_mean,
                                             tf.sqrt(latent_cov[..., 0]),
                                             tf.sqrt(obs_cov[..., 0])))

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
class LocalLevelStateSpaceModelTestStaticShape32(
    test_util.TestCase, _LocalLevelStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class LocalLevelStateSpaceModelTestDynamicShape32(
    test_util.TestCase, _LocalLevelStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class LocalLevelStateSpaceModelTestStaticShape64(
    test_util.TestCase, _LocalLevelStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == "__main__":
  tf.test.main()
