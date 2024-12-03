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

import math

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import square
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import linear_gaussian_ssm
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import mixture_same_family
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts.components.seasonal import ConstrainedSeasonalStateSpaceModel
from tensorflow_probability.python.sts.components.seasonal import Seasonal
from tensorflow_probability.python.sts.components.seasonal import SeasonalStateSpaceModel


NUMPY_MODE = False

tfl = tf.linalg


class _SeasonalStateSpaceModelTest(test_util.TestCase):

  def test_day_of_week_example(self):

    # Test that the Seasonal SSM is equivalent to individually modeling
    # a random walk on each season's slice of timesteps.

    seed = test_util.test_seed(sampler_type='stateless')
    drift_scale = 0.6
    observation_noise_scale = 0.1

    day_of_week = SeasonalStateSpaceModel(
        num_timesteps=28,
        num_seasons=7,
        drift_scale=self._build_placeholder(drift_scale),
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=mvn_diag.MultivariateNormalDiag(
            scale_diag=self._build_placeholder(np.ones([7]))),
        num_steps_per_season=1)

    random_walk_model = linear_gaussian_ssm.LinearGaussianStateSpaceModel(
        num_timesteps=4,
        transition_matrix=self._build_placeholder([[1.]]),
        transition_noise=mvn_diag.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([drift_scale])),
        observation_matrix=self._build_placeholder([[1.]]),
        observation_noise=mvn_diag.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([observation_noise_scale])),
        initial_state_prior=mvn_diag.MultivariateNormalDiag(
            scale_diag=self._build_placeholder([1.])))

    sampled_time_series = day_of_week.sample(seed=seed)
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

    seed = test_util.test_seed(sampler_type='stateless')
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
        initial_state_prior=mvn_diag.MultivariateNormalDiag(
            loc=self._build_placeholder(monthly_effect_prior_means),
            scale_diag=self._build_placeholder(monthly_effect_prior_scales)),
        initial_step=initial_step)

    sampled_series_, prior_mean_, prior_variance_ = self.evaluate(
        (month_of_year.sample(seed=seed)[..., 0],
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

    seed = test_util.test_seed(sampler_type='stateless')
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
        initial_state_prior=mvn_diag.MultivariateNormalDiag(
            loc=self._build_placeholder(monthly_effect_prior_means),
            scale_diag=self._build_placeholder(monthly_effect_prior_scales)),
        initial_step=initial_step)

    sampled_series_, prior_mean_, prior_variance_ = self.evaluate(
        (month_of_year.sample(seed=seed)[..., 0],
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
    seed = test_util.test_seed(sampler_type='stateless')

    num_seasons = 24
    initial_state_prior = mvn_diag.MultivariateNormalDiag(
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
    y_ = self.evaluate(ssm.sample(seed=seed))
    self.assertAllEqual(y_.shape[:-2], batch_shape)

    # Next check that the broadcasting works as expected, and the batch log_prob
    # actually matches the log probs of independent models.
    individual_ssms = [
        SeasonalStateSpaceModel(  # pylint:disable=g-complex-comprehension
            num_timesteps=9,
            num_seasons=num_seasons,
            num_steps_per_season=2,
            drift_scale=drift_scale[i, j, ...],
            observation_noise_scale=observation_noise_scale[j, ...],
            initial_state_prior=mvn_diag.MultivariateNormalDiag(
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
    seed = test_util.test_seed(sampler_type='stateless')

    num_seasons = 24
    initial_state_prior = mvn_diag.MultivariateNormalDiag(
        scale_diag=self._build_placeholder(
            np.exp(np.random.randn(*(partial_batch_shape +
                                     [num_seasons - 1])))))
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
    y_ = self.evaluate(ssm.sample(seed=seed))
    self.assertAllEqual(y_.shape[:-2], batch_shape)

    # Next check that the broadcasting works as expected, and the batch log_prob
    # actually matches the log probs of independent models.
    individual_ssms = []
    for i in range(batch_shape[0]):
      for j in range(batch_shape[1]):
        individual_ssms.append(
            ConstrainedSeasonalStateSpaceModel(
                num_timesteps=9,
                num_seasons=num_seasons,
                num_steps_per_season=2,
                drift_scale=drift_scale[i, j, ...],
                observation_noise_scale=observation_noise_scale[j, ...],
                initial_state_prior=mvn_diag.MultivariateNormalDiag(
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


class SeasonalComponentResampleDriftScaleInputs(test_util.TestCase):

  def testDriftRequired(self):
    seasonal_component = Seasonal(num_seasons=3, allow_drift=False)
    with self.assertRaisesRegex(NotImplementedError,
                                'implemented only .* with drift'):
      seasonal_component.experimental_resample_drift_scale(
          latents=tf.constant(
              value=[], shape=[0, seasonal_component.num_seasons - 1]),
          seed=test_util.test_seed(sampler_type='stateless'))

  def testDriftScalePriorRequiresTransformedDistribution(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        'experimental_resample_drift_scale requires drift scale prior'):
      seasonal_component = Seasonal(
          num_seasons=3,
          drift_scale_prior=inverse_gamma.InverseGamma(
              tf.constant(16.), tf.constant(4.)))
      seasonal_component.experimental_resample_drift_scale(
          latents=tf.constant(
              value=[], shape=[0, seasonal_component.num_seasons - 1]),
          seed=test_util.test_seed(sampler_type='stateless'))

  def testDriftScalePriorRequiresInvertedBijector(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        'experimental_resample_drift_scale requires drift scale prior'):
      seasonal_component = Seasonal(
          num_seasons=3,
          drift_scale_prior=transformed_distribution.TransformedDistribution(
              bijector=square.Square(),
              distribution=inverse_gamma.InverseGamma(
                  tf.constant(16.), tf.constant(4.))))
      seasonal_component.experimental_resample_drift_scale(
          latents=tf.constant(
              value=[], shape=[0, seasonal_component.num_seasons - 1]),
          seed=test_util.test_seed(sampler_type='stateless'))

  def testDriftScalePriorRequiresInvertedSquaredBijector(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        'experimental_resample_drift_scale requires drift scale prior'):
      seasonal_component = Seasonal(
          num_seasons=3,
          drift_scale_prior=transformed_distribution.TransformedDistribution(
              bijector=invert.Invert(identity.Identity()),
              distribution=inverse_gamma.InverseGamma(
                  tf.constant(16.), tf.constant(4.))))
      seasonal_component.experimental_resample_drift_scale(
          latents=tf.constant(
              value=[], shape=[0, seasonal_component.num_seasons - 1]),
          seed=test_util.test_seed(sampler_type='stateless'))

  def testDriftScalePriorRequiresTransformedInverseGamma(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        'experimental_resample_drift_scale requires drift scale prior'):
      seasonal_component = Seasonal(
          num_seasons=3,
          drift_scale_prior=transformed_distribution.TransformedDistribution(
              bijector=invert.Invert(square.Square()),
              distribution=lognormal.LogNormal(loc=3., scale=4.)))
      seasonal_component.experimental_resample_drift_scale(
          latents=tf.constant(
              value=[], shape=[0, seasonal_component.num_seasons - 1]),
          seed=test_util.test_seed(sampler_type='stateless'))

  def testProperInputsThrowsNoExceptionNonEmptyTimeseries(self):
    seasonal_component = Seasonal(
        num_seasons=3,
        drift_scale_prior=transformed_distribution.TransformedDistribution(
            bijector=invert.Invert(square.Square()),
            distribution=inverse_gamma.InverseGamma(
                tf.constant(16.), tf.constant(4.))))
    seasonal_component.experimental_resample_drift_scale(
        latents=tf.constant(
            value=2., shape=[1, seasonal_component.num_seasons - 1]),
        seed=test_util.test_seed(sampler_type='stateless'))

  def testProperInputsThrowsNoExceptionEmptyTimeseries(self):
    seasonal_component = Seasonal(
        num_seasons=3,
        drift_scale_prior=transformed_distribution.TransformedDistribution(
            bijector=invert.Invert(square.Square()),
            distribution=inverse_gamma.InverseGamma(
                tf.constant(16.), tf.constant(4.))))
    seasonal_component.experimental_resample_drift_scale(
        latents=tf.zeros(shape=[
            # Verify that a timeseries of size 0 still succeeds. Some
            # implementations (e.g. using `tf.range`) could fail with 0.
            0,
            seasonal_component.num_seasons - 1
        ]),
        seed=test_util.test_seed(sampler_type='stateless'))


@tf.function()
def _compute_summary_for_normal_samples_with_inverse_gamma_scale_prior(
    variance_inverse_gamma_prior, batch_shape, num_samples,
    num_observations_per_sample, sampled_fixed_scale, seed):
  """Computes the mean and standard deviation from num_samples.

  Given an inverse gamma prior on the variance of Normal distributions scale,
  this returns summary statistics of what is expected given `num_samples`
  where each sample has `num_observations_per_sample` from
  N(0, sampled_fixed_scale).

  Args:
    variance_inverse_gamma_prior: Prior on the variance.
    batch_shape: Shape of batches.
    num_samples: Number of samples.
    num_observations_per_sample: How many observations is in each sample.
    sampled_fixed_scale: The scale to draw observations with.
    seed: The seed to use.

  Returns:
    Tuple of (mean, standard error) of the variance.
  """
  sample_square_sum = tf.math.reduce_sum(
      tf.square(
          normal.Normal(0., sampled_fixed_scale).sample(
              ps.concat(
                  [batch_shape, (num_samples, num_observations_per_sample)],
                  axis=0),
              seed=seed)),
      axis=-1)

  posterior_concentration = variance_inverse_gamma_prior.concentration + (
      num_observations_per_sample / 2.)
  posterior_scales = variance_inverse_gamma_prior.concentration + (
      sample_square_sum / 2.)
  d = inverse_gamma.InverseGamma(posterior_concentration, posterior_scales)

  mix = mixture_same_family.MixtureSameFamily(
      categorical.Categorical(logits=tf.zeros(d.batch_shape)), d)
  return mix.mean(), tf.math.sqrt(mix.variance() /
                                  tf.cast(num_samples, dtype=tf.float32))


ENABLE_DRIFT_SCALE_TESTS = True
# For reasons currently unclear, drift scale tests fail on external numpy.
ENABLE_DRIFT_SCALE_TESTS = not NUMPY_MODE  # EnableOnExport


@test_util.test_graph_and_eager_modes
class SeasonalComponentResampleDriftScaleNumerical(test_util.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'Unconstrained',
          'constrain_mean_effect_to_zero': False,
      }, {
          'testcase_name': 'Constrained',
          'constrain_mean_effect_to_zero': True,
      }, {
          'testcase_name':
              'UnconstrainedSmallNumberOfObservationsShouldRespectPrior',
          'constrain_mean_effect_to_zero':
              False,
          'num_observations_of_transition_noise':
              100
      }, {
          'testcase_name':
              'ConstrainedSmallNumberOfObservationsShouldRespectPrior',
          'constrain_mean_effect_to_zero':
              True,
          'num_observations_of_transition_noise':
              100
      }, {
          'testcase_name': 'UnconstrainedWithStepsPerSeason',
          'constrain_mean_effect_to_zero': False,
          'num_steps_per_season': [2, 4, 3],
      }, {
          'testcase_name': 'ConstrainedWithStepsPerSeason',
          'constrain_mean_effect_to_zero': True,
          'num_steps_per_season': [2, 4, 3],
      }, {
          'testcase_name': 'UnconstrainedWithBatchShape',
          'constrain_mean_effect_to_zero': False,
          # TODO(kloveless): Add tests with different batch value, not just
          # a batch shape.
          'batch_shape': (3,),
      }, {
          'testcase_name': 'ConstrainedWithBatchShape',
          'constrain_mean_effect_to_zero': True,
          'batch_shape': (3,),
      })
  def testBasicSeasonality(self,
                           constrain_mean_effect_to_zero,
                           batch_shape=(),
                           num_steps_per_season=None,
                           num_observations_of_transition_noise=1000):
    if not ENABLE_DRIFT_SCALE_TESTS:
      print('Aborting drift scale test.')
      return

    # Use enough samples so we can get a reasonable estimate of the
    # distribution.
    num_samples = 1000
    distribution_batch_shape = batch_shape + (num_samples,)

    if num_steps_per_season is None:
      num_steps_per_season = [1, 1, 1]
    inverse_gamma_prior = inverse_gamma.InverseGamma(
        tf.constant(16., shape=distribution_batch_shape),
        tf.constant(4., shape=distribution_batch_shape))
    seasonal_component = Seasonal(
        num_seasons=3,
        constrain_mean_effect_to_zero=constrain_mean_effect_to_zero,
        num_steps_per_season=num_steps_per_season,
        drift_scale_prior=transformed_distribution.TransformedDistribution(
            bijector=invert.Invert(square.Square()),
            distribution=inverse_gamma_prior))
    # Scale the number of time steps to be a constant number of season
    # transitions. This means all variations should converge similarly.
    num_timesteps = num_observations_of_transition_noise * math.ceil(
        sum(num_steps_per_season) / len(num_steps_per_season))
    initial_state_size = (
        seasonal_component.num_seasons -
        1 if constrain_mean_effect_to_zero else seasonal_component.num_seasons)
    drift_scale = 20.
    ssm = seasonal_component.make_state_space_model(
        num_timesteps=num_timesteps,
        initial_state_prior=mvn_diag.MultivariateNormalDiag(scale_diag=[1.] *
                                                            initial_state_size),
        param_vals={},
        drift_scale=drift_scale,
        # This needs to be non-zero for steps_per_season to be any value other
        # than 1 (otherwise NaN is returned after cholesky decomposition
        # failure). When there is not a season transition, LGSSM will encounter
        # variances and covariances of 0 - and even though there are no
        # differences to explain for our data (since it is generated from the
        # same LGSSM), these 0's are not supported.
        observation_noise_scale=1e-4,
    )
    seed = test_util.test_seed(sampler_type='stateless')

    @tf.function
    def sample_drift():
      sample = ssm.sample(sample_shape=(num_samples,), seed=seed)
      latents = ssm.posterior_sample(x=sample, seed=seed)
      return seasonal_component.experimental_resample_drift_scale(
          latents, seed=seed)

    drift_scale_samples = sample_drift()
    drift_variance_sample_mean = tf.math.reduce_mean(
        tf.math.square(drift_scale_samples))

    # Verify that the samples of the drift scale are close to the actual drift
    # scale. This computes, given a certain number of samples, how close
    # random samples would be (a very limited duplication of what is being
    # tested).
    expected_scale_variance_mean, expected_variance_standard_error = _compute_summary_for_normal_samples_with_inverse_gamma_scale_prior(
        inverse_gamma_prior,
        batch_shape=batch_shape,
        num_samples=num_samples,
        num_observations_per_sample=num_observations_of_transition_noise,
        sampled_fixed_scale=drift_scale,
        seed=seed,
    )

    z_scores = ((expected_scale_variance_mean - drift_variance_sample_mean) /
                expected_variance_standard_error)
    self.assertAllLess(z_scores, 4.2)


if __name__ == '__main__':
  test_util.main()
