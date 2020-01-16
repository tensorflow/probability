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
"""Semilocal Linear Trend Model Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import LocalLinearTrendStateSpaceModel
from tensorflow_probability.python.sts import SemiLocalLinearTrendStateSpaceModel


@test_util.test_all_tf_execution_regimes
class _SemiLocalLinearTrendStateSpaceModelTest(object):

  def test_logprob(self):

    num_timesteps = 5
    y = self._build_placeholder([1.0, 2.5, 4.3, 6.1, 7.8])

    ssm = SemiLocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        level_scale=self._build_placeholder(0.5),
        slope_scale=self._build_placeholder(0.5),
        slope_mean=self._build_placeholder(0.2),
        autoregressive_coef=self._build_placeholder(0.3),
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([1., 1.])))

    lp = ssm.log_prob(y[:, np.newaxis])
    expected_lp = -9.846248626708984
    self.assertAllClose(self.evaluate(lp), expected_lp)

  def test_matches_locallineartrend(self):
    """SemiLocalLinearTrend with trivial AR process is a LocalLinearTrend."""

    level_scale = self._build_placeholder(0.5)
    slope_scale = self._build_placeholder(0.5)
    initial_level = self._build_placeholder(3.)
    initial_slope = self._build_placeholder(-2.)
    num_timesteps = 5
    y = self._build_placeholder([1.0, 2.5, 4.3, 6.1, 7.8])

    semilocal_ssm = SemiLocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        level_scale=level_scale,
        slope_scale=slope_scale,
        slope_mean=self._build_placeholder(0.),
        autoregressive_coef=self._build_placeholder(1.),
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=[initial_level, initial_slope],
            scale_diag=self._build_placeholder([1., 1.])))

    local_ssm = LocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        level_scale=level_scale,
        slope_scale=slope_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=[initial_level, initial_slope],
            scale_diag=self._build_placeholder([1., 1.])))

    semilocal_lp = semilocal_ssm.log_prob(y[:, tf.newaxis])
    local_lp = local_ssm.log_prob(y[:, tf.newaxis])
    self.assertAllClose(self.evaluate(semilocal_lp), self.evaluate(local_lp))

    semilocal_mean = semilocal_ssm.mean()
    local_mean = local_ssm.mean()
    self.assertAllClose(
        self.evaluate(semilocal_mean), self.evaluate(local_mean))

    semilocal_variance = semilocal_ssm.variance()
    local_variance = local_ssm.variance()
    self.assertAllClose(
        self.evaluate(semilocal_variance), self.evaluate(local_variance))

  def test_slope_mean_and_variance(self):
    """Check that slope follows `slope_mean` and has stationary variance."""

    level_scale = 0.1
    slope_scale = 0.2
    initial_level = 3.
    initial_slope = 0.
    slope_mean = -2.
    initial_level = 3.
    autoregressive_coef = 0.9
    num_timesteps = 50

    # Stationary distribution of an AR1 process, from
    # (https://en.wikipedia.org/wiki/Autoregressive_model#Example:_An_AR(1)_process)  # pylint: disable=line-too-long
    stationary_slope_variance = slope_scale**2 / (1. - autoregressive_coef**2)

    # Initialize the slope prior at the stationary variance.
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=self._build_placeholder([initial_level, initial_slope]),
        scale_diag=self._build_placeholder(
            [1., np.sqrt(stationary_slope_variance)]))

    semilocal_ssm = SemiLocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        level_scale=self._build_placeholder(level_scale),
        slope_scale=self._build_placeholder(slope_scale),
        slope_mean=self._build_placeholder(slope_mean),
        autoregressive_coef=self._build_placeholder(autoregressive_coef),
        initial_state_prior=initial_state_prior)

    # The slope of the mean should converge to `slope_mean` (as opposed to
    # staying fixed at `initial_slope` as in a LocalLinearTrend).
    mean_ = self.evaluate(semilocal_ssm.mean()[..., 0])
    final_slope = mean_[num_timesteps-1] - mean_[num_timesteps-2]
    self.assertAllClose(final_slope, slope_mean, atol=0.05)

    # The variance in latent `slope` should converge to the stationary
    # distribution of an AR1 process:
    latent_covs, _ = semilocal_ssm._joint_covariances()
    actual_slope_variances = tf.linalg.diag_part(latent_covs)[:, 1]
    converged_slope_variance = actual_slope_variances[-1]
    self.assertAllClose(self.evaluate(converged_slope_variance),
                        stationary_slope_variance, atol=1e-4)

  def test_batch_shape(self):
    batch_shape = [4, 2]

    level_scale = self._build_placeholder(
        np.exp(np.random.randn(*(batch_shape))))
    slope_scale = self._build_placeholder(np.exp(np.random.randn(*batch_shape)))
    autoregressive_coef = self._build_placeholder(np.random.randn(*batch_shape))
    slope_mean = self._build_placeholder(np.random.randn(*batch_shape))

    ssm = SemiLocalLinearTrendStateSpaceModel(
        num_timesteps=10,
        level_scale=level_scale,
        slope_scale=slope_scale,
        autoregressive_coef=autoregressive_coef,
        slope_mean=slope_mean,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([1., 1.])))

    if self.use_static_shape:
      model_batch_shape = ssm.batch_shape.as_list()
    else:
      model_batch_shape = self.evaluate(ssm.batch_shape_tensor())
    self.assertAllEqual(model_batch_shape, batch_shape)

    y = ssm.sample()
    if self.use_static_shape:
      y_batch_shape = y.shape.as_list()[:-2]
    else:
      y_batch_shape = self.evaluate(tf.shape(y))[:-2]
    self.assertAllEqual(y_batch_shape, batch_shape)

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
class SemiLocalLinearTrendStateSpaceModelTestStaticShape32(
    test_util.TestCase, _SemiLocalLinearTrendStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class SemiLocalLinearTrendStateSpaceModelTestDynamicShape32(
    test_util.TestCase, _SemiLocalLinearTrendStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class SemiLocalLinearTrendStateSpaceModelTestStaticShape64(
    test_util.TestCase, _SemiLocalLinearTrendStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True

if __name__ == "__main__":
  tf.test.main()
