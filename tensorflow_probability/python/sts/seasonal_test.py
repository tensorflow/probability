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
"""Seasonal Model Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import ConstrainedSeasonalStateSpaceModel
from tensorflow_probability.python.sts import SeasonalStateSpaceModel


tfl = tf.linalg


class _SeasonalStateSpaceModelTest(test_util.TestCase):

  def test_day_of_week_example(self):

    # Test that the Seasonal SSM is equivalent to individually modeling
    # a random walk on each season's slice of timesteps.

    drift_scale = 0.6
    observation_noise_scale = 0.1

    day_of_week = SeasonalStateSpaceModel(
        num_timesteps=28,
        num_seasons=7,
        drift_scale=self._build_placeholder(drift_scale),
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(np.ones([7]))),
        num_steps_per_season=1)

    random_walk_model = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=4,
        transition_matrix=self._build_placeholder([[1.]]),
        transition_noise=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([drift_scale])),
        observation_matrix=self._build_placeholder([[1.]]),
        observation_noise=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([observation_noise_scale])),
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([1.])))

    sampled_time_series = day_of_week.sample()
    (sampled_time_series_, total_lp_,
     prior_mean_, prior_variance_) = self.evaluate([
         sampled_time_series,
         day_of_week.log_prob(sampled_time_series),
         day_of_week.mean(),
         day_of_week.variance()])

    expected_daily_means_, expected_daily_variances_ = self.evaluate([
        random_walk_model.mean(),
        random_walk_model.variance()])

    # For the (noncontiguous) indices corresponding to each season, assert
    # that the model's mean, variance, and log_prob match a random-walk model.
    daily_lps = []
    for day_idx in range(7):
      self.assertAllClose(prior_mean_[day_idx::7], expected_daily_means_)
      self.assertAllClose(prior_variance_[day_idx::7],
                          expected_daily_variances_)

      daily_lps.append(self.evaluate(random_walk_model.log_prob(
          sampled_time_series_[day_idx::7])))

    self.assertAlmostEqual(total_lp_, sum(daily_lps), places=3)

  def test_month_of_year_example(self):

    num_days_per_month = np.array(
        [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31])

    # put wildly different near-deterministic priors on the effect for each
    # month, so we can easily distinguish the months and diagnose off-by-one
    # errors.
    monthly_effect_prior_means = np.linspace(-1000, 1000, 12)
    monthly_effect_prior_scales = 0.1 * np.ones(12)
    drift_scale = 0.3
    observation_noise_scale = 0.1
    num_timesteps = 365 * 2  # 2 years.
    initial_step = 22

    month_of_year = SeasonalStateSpaceModel(
        num_timesteps=num_timesteps,
        num_seasons=12,
        num_steps_per_season=num_days_per_month,
        drift_scale=self._build_placeholder(drift_scale),
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=self._build_placeholder(monthly_effect_prior_means),
            scale_diag=self._build_placeholder(monthly_effect_prior_scales)),
        initial_step=initial_step)

    sampled_series_, prior_mean_, prior_variance_ = self.evaluate(
        (month_of_year.sample()[..., 0],
         month_of_year.mean()[..., 0],
         month_of_year.variance()[..., 0]))

    # For each month, ensure the mean (and samples) of each day matches the
    # expected effect for that month, and that the variances all match
    # including expected drift from year to year
    current_idx = -initial_step  # start at beginning of year
    current_drift_variance = 0.
    for _ in range(3):  # loop over three years to include entire 2-year period
      for (num_days_in_month, prior_effect_mean, prior_effect_scale) in zip(
          num_days_per_month,
          monthly_effect_prior_means,
          monthly_effect_prior_scales):

        month_indices = range(current_idx, current_idx + num_days_in_month)
        current_idx += num_days_in_month

        # Trim any indices outside of the observed window.
        month_indices = [idx for idx in month_indices
                         if 0 <= idx < num_timesteps]

        if month_indices:
          ones = np.ones(len(month_indices))
          self.assertAllClose(sampled_series_[month_indices],
                              prior_effect_mean * ones, atol=20.)
          self.assertAllClose(prior_mean_[month_indices],
                              prior_effect_mean * ones, atol=1e-4)
          self.assertAllClose(prior_variance_[month_indices],
                              (prior_effect_scale**2 +
                               current_drift_variance +
                               observation_noise_scale**2) * ones,
                              atol=1e-4)

      # At the end of each year, allow the effects to drift.
      current_drift_variance += drift_scale**2

  def test_month_of_year_with_leap_day_example(self):

    num_days_per_month = np.array(
        [[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
         [31, 29, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],  # year with leap day
         [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
         [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31]])

    # put wildly different near-deterministic priors on the effect for each
    # month, so we can easily distinguish the months and diagnose off-by-one
    # errors.
    monthly_effect_prior_means = np.linspace(-1000, 1000, 12)
    monthly_effect_prior_scales = 0.1 * np.ones(12)
    drift_scale = 0.3
    observation_noise_scale = 0.1
    num_timesteps = 365 * 6 + 2   # 1.5 cycles of 4 years with leap days.
    initial_step = 22

    month_of_year = SeasonalStateSpaceModel(
        num_timesteps=num_timesteps,
        num_seasons=12,
        num_steps_per_season=num_days_per_month,
        drift_scale=self._build_placeholder(drift_scale),
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=self._build_placeholder(monthly_effect_prior_means),
            scale_diag=self._build_placeholder(monthly_effect_prior_scales)),
        initial_step=initial_step)

    sampled_series_, prior_mean_, prior_variance_ = self.evaluate(
        (month_of_year.sample()[..., 0],
         month_of_year.mean()[..., 0],
         month_of_year.variance()[..., 0]))

    # For each month, ensure the mean (and samples) of each day matches the
    # expected effect for that month, and that the variances all match
    # including expected drift from year to year
    current_idx = -initial_step  # start at beginning of year
    current_drift_variance = 0.
    for i in range(6):  # loop over the 1.5 cycles of the 4 years
      for (num_days_in_month, prior_effect_mean, prior_effect_scale) in zip(
          num_days_per_month[i % 4],
          monthly_effect_prior_means,
          monthly_effect_prior_scales):

        month_indices = range(current_idx, current_idx + num_days_in_month)
        current_idx += num_days_in_month

        # Trim any indices outside of the observed window.
        month_indices = [idx for idx in month_indices
                         if 0 <= idx < num_timesteps]

        if month_indices:
          ones = np.ones(len(month_indices))
          self.assertAllClose(sampled_series_[month_indices],
                              prior_effect_mean * ones, atol=20.)
          self.assertAllClose(prior_mean_[month_indices],
                              prior_effect_mean * ones, atol=1e-4)
          self.assertAllClose(prior_variance_[month_indices],
                              (prior_effect_scale**2 +
                               current_drift_variance +
                               observation_noise_scale**2) * ones,
                              atol=1e-4)

      # At the end of each year, allow the effects to drift.
      current_drift_variance += drift_scale**2

  def test_batch_shape(self):
    batch_shape = [3, 2]
    partial_batch_shape = [2]

    num_seasons = 24
    initial_state_prior = tfd.MultivariateNormalDiag(
        scale_diag=self._build_placeholder(
            np.exp(np.random.randn(*(partial_batch_shape + [num_seasons])))))
    drift_scale = self._build_placeholder(
        np.exp(np.random.randn(*batch_shape)))
    observation_noise_scale = self._build_placeholder(
        np.exp(np.random.randn(*partial_batch_shape)))

    ssm = SeasonalStateSpaceModel(
        num_timesteps=9,
        num_seasons=24,
        num_steps_per_season=2,
        drift_scale=drift_scale,
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=initial_state_prior)

    # First check that the model's batch shape is the broadcast batch shape
    # of parameters, as expected.

    self.assertAllEqual(self.evaluate(ssm.batch_shape_tensor()), batch_shape)
    y_ = self.evaluate(ssm.sample())
    self.assertAllEqual(y_.shape[:-2], batch_shape)

    # Next check that the broadcasting works as expected, and the batch log_prob
    # actually matches the log probs of independent models.
    individual_ssms = [SeasonalStateSpaceModel(
        num_timesteps=9,
        num_seasons=num_seasons,
        num_steps_per_season=2,
        drift_scale=drift_scale[i, j, ...],
        observation_noise_scale=observation_noise_scale[j, ...],
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=initial_state_prior.scale.diag[j, ...]))
                       for i in range(batch_shape[0])
                       for j in range(batch_shape[1])]

    batch_lps_ = self.evaluate(ssm.log_prob(y_)).flatten()

    individual_ys = [y_[i, j, ...]
                     for i in range(batch_shape[0])
                     for j in range(batch_shape[1])]
    individual_lps_ = self.evaluate([
        individual_ssm.log_prob(individual_y)
        for (individual_ssm, individual_y)
        in zip(individual_ssms, individual_ys)])

    self.assertAllClose(individual_lps_, batch_lps_)

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
class SeasonalStateSpaceModelTestStaticShape32(_SeasonalStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class SeasonalStateSpaceModelTestDynamicShape32(_SeasonalStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class SeasonalStateSpaceModelTestStaticShape64(_SeasonalStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True


class _ConstrainedSeasonalStateSpaceModelTest(test_util.TestCase):

  # TODO(b/128635942): write additional tests for ConstrainedSeasonalSSM

  def test_batch_shape(self):
    batch_shape = [3, 2]
    partial_batch_shape = [2]

    num_seasons = 24
    initial_state_prior = tfd.MultivariateNormalDiag(
        scale_diag=self._build_placeholder(
            np.exp(np.random.randn(
                *(partial_batch_shape + [num_seasons - 1])))))
    drift_scale = self._build_placeholder(
        np.exp(np.random.randn(*batch_shape)))
    observation_noise_scale = self._build_placeholder(
        np.exp(np.random.randn(*partial_batch_shape)))

    ssm = ConstrainedSeasonalStateSpaceModel(
        num_timesteps=9,
        num_seasons=24,
        num_steps_per_season=2,
        drift_scale=drift_scale,
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=initial_state_prior)

    # First check that the model's batch shape is the broadcast batch shape
    # of parameters, as expected.

    self.assertAllEqual(self.evaluate(ssm.batch_shape_tensor()), batch_shape)
    y_ = self.evaluate(ssm.sample())
    self.assertAllEqual(y_.shape[:-2], batch_shape)

    # Next check that the broadcasting works as expected, and the batch log_prob
    # actually matches the log probs of independent models.
    individual_ssms = []
    for i in range(batch_shape[0]):
      for j in range(batch_shape[1]):
        individual_ssms.append(ConstrainedSeasonalStateSpaceModel(
            num_timesteps=9,
            num_seasons=num_seasons,
            num_steps_per_season=2,
            drift_scale=drift_scale[i, j, ...],
            observation_noise_scale=observation_noise_scale[j, ...],
            initial_state_prior=tfd.MultivariateNormalDiag(
                scale_diag=initial_state_prior.scale.diag[j, ...])))

    batch_lps_ = self.evaluate(ssm.log_prob(y_)).flatten()

    individual_ys = [y_[i, j, ...]  # pylint: disable=g-complex-comprehension
                     for i in range(batch_shape[0])
                     for j in range(batch_shape[1])]
    individual_lps_ = self.evaluate([
        individual_ssm.log_prob(individual_y)
        for (individual_ssm, individual_y)
        in zip(individual_ssms, individual_ys)])

    self.assertAllClose(individual_lps_, batch_lps_)

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
class ConstrainedSeasonalStateSpaceModelTestStaticShape32(
    _ConstrainedSeasonalStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_all_tf_execution_regimes
class ConstrainedSeasonalStateSpaceModelTestDynamicShape32(
    _ConstrainedSeasonalStateSpaceModelTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_all_tf_execution_regimes
class ConstrainedSeasonalStateSpaceModelTestStaticShape64(
    _ConstrainedSeasonalStateSpaceModelTest):
  dtype = np.float64
  use_static_shape = True

# Don't run tests for the abstract base classes.
del _SeasonalStateSpaceModelTest
del _ConstrainedSeasonalStateSpaceModelTest

if __name__ == "__main__":
  tf.test.main()
