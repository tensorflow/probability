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
"""Durbin-Koopman tests."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import square
from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions.linear_gaussian_ssm import linear_gaussian_update
from tensorflow_probability.python.experimental.distributions import mvn_precision_factor_linop as mvnpflo
from tensorflow_probability.python.experimental.sts_gibbs import gibbs_sampler
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.stats import sample_stats
from tensorflow_probability.python.sts.components import local_level
from tensorflow_probability.python.sts.components import local_linear_trend
from tensorflow_probability.python.sts.components import regression
from tensorflow_probability.python.sts.components import seasonal
from tensorflow_probability.python.sts.components import semilocal_linear_trend
from tensorflow_probability.python.sts.components import sum as sum_lib
from tensorflow_probability.python.sts.forecast import forecast
from tensorflow_probability.python.sts.internal import missing_values_util
from tensorflow_probability.python.sts.internal import util as sts_util


tfl = tf.linalg

JAX_MODE = False


@test_util.test_graph_and_eager_modes
class GibbsSamplerTests(test_util.TestCase):

  def _build_test_model(self,
                        num_timesteps=5,
                        num_features=2,
                        batch_shape=(),
                        missing_prob=0,
                        true_noise_scale=0.1,
                        true_level_scale=0.04,
                        true_slope_scale=0.02,
                        prior_class=inverse_gamma.InverseGamma,
                        weights=None,
                        weights_prior_scale=10.,
                        sparse_weights_nonzero_prob=None,
                        time_series_shift=0.,
                        dtype=tf.float32,
                        design_matrix=False,
                        seed=None,
                        seasonal_components=None,
                        tiled_residual_offset=None):
    if seasonal_components is None:
      seasonal_components = []
    if seed is None:
      seed = test_util.test_seed(sampler_type='stateless')
    (design_seed,
     weights_seed,
     noise_seed,
     level_seed,
     slope_seed,
     is_missing_seed) = samplers.split_seed(seed, 6, salt='_build_test_model')

    if weights is None:
      weights = samplers.normal(
          list(batch_shape) + [num_features], dtype=dtype, seed=weights_seed)
    if design_matrix is None:
      reg = tf.zeros(num_timesteps, dtype)
    else:
      if isinstance(design_matrix, bool) and not design_matrix:
        design_matrix = samplers.normal([num_timesteps, num_features],
                                        dtype=dtype,
                                        seed=design_seed)
      reg = tf.linalg.matvec(design_matrix, weights)
    noise = samplers.normal(
        list(batch_shape) + [num_timesteps],
        dtype=dtype, seed=noise_seed) * true_noise_scale

    level_residuals = samplers.normal(
        list(batch_shape) + [num_timesteps],
        dtype=dtype, seed=level_seed) * true_level_scale
    if true_slope_scale is not None:
      slope = tf.cumsum(samplers.normal(
          list(batch_shape) + [num_timesteps],
          dtype=dtype, seed=slope_seed) * true_slope_scale, axis=-1)
      level_residuals += slope
    level = tf.cumsum(level_residuals, axis=-1)

    residual_offset = 0.
    if tiled_residual_offset:
      residual_offset = tf.convert_to_tensor([
          tiled_residual_offset[x % len(tiled_residual_offset)]
          for x in range(num_timesteps)
      ])

    time_series = (reg + noise + level + time_series_shift + residual_offset)
    is_missing = samplers.uniform(
        list(batch_shape) + [num_timesteps],
        dtype=dtype, seed=is_missing_seed) < missing_prob

    observation_noise_variance_prior = prior_class(
        concentration=tf.cast(0.01, dtype),
        scale=tf.cast(0.01 * 0.01, dtype))
    observation_noise_variance_prior.upper_bound = tf.constant(
        100.0, dtype=dtype)

    observed_time_series = missing_values_util.MaskedTimeSeries(
        time_series[..., tf.newaxis], is_missing)
    if time_series_shift != 0.:
      observed_mean, observed_stddev, observed_initial = (
          sts_util.empirical_statistics(observed_time_series))
      initial_level_prior = normal.Normal(
          loc=observed_mean + observed_initial,
          scale=tf.abs(observed_initial) + observed_stddev)
    else:
      initial_level_prior = None

    model = gibbs_sampler.build_model_for_gibbs_fitting(
        observed_time_series=observed_time_series,
        design_matrix=design_matrix,
        weights_prior=(None if weights_prior_scale is None else normal.Normal(
            loc=tf.cast(0., dtype), scale=tf.cast(weights_prior_scale, dtype))),
        level_variance_prior=prior_class(
            concentration=tf.cast(0.01, dtype),
            scale=tf.cast(0.01 * 0.01, dtype)),
        slope_variance_prior=None if true_slope_scale is None else prior_class(
            concentration=tf.cast(0.01, dtype),
            scale=tf.cast(0.01 * 0.01, dtype)),
        initial_level_prior=initial_level_prior,
        observation_noise_variance_prior=observation_noise_variance_prior,
        sparse_weights_nonzero_prob=sparse_weights_nonzero_prob,
        seasonal_components=seasonal_components)
    return model, time_series, is_missing

  @parameterized.named_parameters(
      {
          'testcase_name': 'LocalLinearTrend',
          'use_slope': True,
          'num_chains': (),
          'time_series_shift': 0.
      },
      {
          'testcase_name': 'LocalLinearTrend_4chains',
          'use_slope': True,
          'num_chains': 4,
          'time_series_shift': 0.
      },
      {
          'testcase_name': 'LocalLevel',
          'use_slope': False,
          'num_chains': (),
          'time_series_shift': 0.
      },
      {
          'testcase_name': 'LocalLevel_OneSeasonality',
          'use_slope': False,
          'num_chains': 4,  # Use chains to speed up tests.
          'time_series_shift': 0.,
          'seasonal_components': [7],
          # Seasonal effects do not support forecasting, thus forecast
          # via masked values.
          'use_num_forecast_steps_for_forecast': False,
          'tiled_residual_offset': [4., 2., 0., 3., 6., -12., -4.],
          # Sets a fairly tight stddev bound to ensure the seasonal effect
          # is being learned and it is not just being pushed to
          # observation noise. For instance, incorrectly not accounting
          # for seasonal effect when sampling observation noise results
          # in a value of ~0.3 - measured empirically by inserting a defect.
          'forecast_stddev_atol': 0.05
      },
      {
          'testcase_name': 'LocalLevel_TwoSeasonality',
          'use_slope': False,
          'num_chains': 4,  # Use chains to speed up tests.
          'time_series_shift': 0.,
          'seasonal_components': [7, 18],
          # Seasonal effects do not support forecasting, thus forecast
          # via masked values.
          'use_num_forecast_steps_for_forecast': False,
          'tiled_residual_offset': [4., 2., 0., 3., 6., -12., -4.],
      },
      {
          'testcase_name': 'LocalLevel_ZeroStepPrediction',
          'use_slope': False,
          'num_chains': (),
          'time_series_shift': 0.,
          'use_zero_step_prediction': True,
      },
      {
          'testcase_name': 'LocalLevel_4chains',
          'use_slope': False,
          'num_chains': 4,
          'time_series_shift': 0.
      },
      {
          'testcase_name': 'UnscaledTimeSeries_LocalLinear',
          'use_slope': False,
          'num_chains': (),
          'time_series_shift': 100.
      },
      {
          'testcase_name': 'UnscaledTimeSeries_LocalLinearTrend',
          'use_slope': True,
          'num_chains': (),
          'time_series_shift': 100.
      })
  def test_forecasts_match_reference(self,
                                     use_slope,
                                     num_chains,
                                     time_series_shift,
                                     use_zero_step_prediction=False,
                                     seasonal_components=None,
                                     use_num_forecast_steps_for_forecast=True,
                                     tiled_residual_offset=None,
                                     forecast_stddev_atol=None):
    if seasonal_components is None:
      seasonal_components = []
    dtype = tf.float32
    # pylint:disable=g-complex-comprehension
    seasonal_components = [
        seasonal.Seasonal(
            name=f'season{season_index}',
            num_seasons=num_seasons,
            drift_scale_prior=transformed_distribution.TransformedDistribution(
                bijector=invert.Invert(square.Square()),
                distribution=inverse_gamma.InverseGamma(
                    tf.constant(0.01, dtype=dtype),
                    tf.constant(0.0001, dtype=dtype))),
            initial_effect_prior=normal.Normal(
                tf.constant(0., dtype=dtype), tf.constant(1., dtype=dtype)))
        for season_index, num_seasons in enumerate(seasonal_components)
    ]
    # pylint:enable=g-complex-comprehension
    seed = test_util.test_seed()
    # Have enough steps to fit seasonality to.
    num_observed_steps = 100
    num_forecast_steps = 4
    num_results = 1000

    # Dividing the number of results with number of chains so we sample the same
    # total number of MCMC samples.
    if not tf.nest.is_nested(num_chains):
      num_results = num_results // num_chains

    batch_shape = [3]
    model, observed_time_series, is_missing = self._build_test_model(
        num_timesteps=num_observed_steps + num_forecast_steps,
        true_slope_scale=0.5 if use_slope else None,
        batch_shape=batch_shape,
        time_series_shift=time_series_shift,
        seasonal_components=seasonal_components,
        tiled_residual_offset=tiled_residual_offset)

    if use_num_forecast_steps_for_forecast:
      time_series_to_fit = missing_values_util.MaskedTimeSeries(
          observed_time_series[..., :num_observed_steps, tf.newaxis],
          is_missing[..., :num_observed_steps])
    else:
      # Use missing values for the last N points instead of forecasting.
      time_series_to_fit = missing_values_util.MaskedTimeSeries(
          observed_time_series[..., tf.newaxis],
          tf.concat([
              is_missing[..., :num_observed_steps],
              tf.broadcast_to(False, is_missing.shape[:-1] +
                              (num_forecast_steps,))
          ],
                    axis=-1))

    @tf.function(autograph=False)
    def do_sampling():
      return gibbs_sampler.fit_with_gibbs_sampling(
          model,
          time_series_to_fit,
          num_chains=num_chains,
          num_results=num_results,
          num_warmup_steps=100,
          seed=seed)

    samples = self.evaluate(do_sampling())

    def reshape_chain_and_sample(x):
      if np.ndim(x) > 2:
        return np.reshape(x, [x.shape[0] * x.shape[1], *x.shape[2:]])
      return x

    if not tf.nest.is_nested(num_chains):
      samples = tf.nest.map_structure(reshape_chain_and_sample, samples)

    predictive_dist = gibbs_sampler.one_step_predictive(
        model,
        samples,
        num_forecast_steps=(
            # If we are not using forecast steps for forecast, then the
            # predictions are already included through the masked inputs.
            num_forecast_steps if use_num_forecast_steps_for_forecast else 0),
        thin_every=1,
        use_zero_step_prediction=use_zero_step_prediction)
    predictive_mean, predictive_stddev = self.evaluate((
        predictive_dist.mean(), predictive_dist.stddev()))
    self.assertAllEqual(predictive_mean.shape,
                        batch_shape + [num_observed_steps + num_forecast_steps])
    self.assertAllEqual(predictive_stddev.shape,
                        batch_shape + [num_observed_steps + num_forecast_steps])

    # big tolerance, but makes sure the predictive mean initializes near
    # the initial time series value
    self.assertAllClose(tf.reduce_mean(predictive_mean[:, 0]),
                        observed_time_series[0, 0],
                        atol=10.)
    parameter_samples = gibbs_sampler.model_parameter_samples_from_gibbs_samples(
        model, samples)

    # Note that although we expect the Gibbs-sampled forecasts to match a
    # reference implementation, we *don't* expect the one-step predictions to
    # match `tfp.sts.one_step_predictive`, because that makes predictions using
    # a filtered posterior (i.e., given only previous observations) whereas the
    # Gibbs-sampled latent `level`s will incorporate some information from
    # future observations.
    reference_forecast_dist = forecast(
        model,
        observed_time_series=observed_time_series[..., :num_observed_steps],
        parameter_samples=parameter_samples,
        num_steps_forecast=num_forecast_steps)

    reference_forecast_mean, reference_forecast_stddev = self.evaluate(
        (reference_forecast_dist.mean()[..., 0],
         reference_forecast_dist.stddev()[..., 0]))

    self.assertAllClose(
        predictive_mean[..., -num_forecast_steps:],
        reference_forecast_mean,
        atol=1.0 if use_slope else 0.3)
    if forecast_stddev_atol is None:
      forecast_stddev_atol = 2.0 if use_slope else 1.00
    self.assertAllClose(
        predictive_stddev[..., -num_forecast_steps:],
        reference_forecast_stddev,
        atol=forecast_stddev_atol)

  @parameterized.named_parameters(
      {'testcase_name': 'float32_xla',
       'dtype': tf.float32,
       'use_xla': True,
       'use_spike_and_slab': False},
      {'testcase_name': 'float64',
       'dtype': tf.float64,
       'use_xla': False,
       'use_spike_and_slab': False},
      {'testcase_name': 'float64_xla_sparse',
       'dtype': tf.float32,
       'use_xla': True,
       'use_spike_and_slab': True})
  def test_end_to_end_prediction_works_and_is_deterministic(
      self, dtype, use_xla, use_spike_and_slab):
    if not tf.executing_eagerly():
      return
    seed = test_util.test_seed(sampler_type='stateless')
    model, observed_time_series, is_missing = self._build_test_model(
        num_timesteps=5,
        batch_shape=[3],
        prior_class=gibbs_sampler.XLACompilableInverseGamma,
        sparse_weights_nonzero_prob=0.5 if use_spike_and_slab else None,
        dtype=dtype,
        seasonal_components=[
            seasonal.Seasonal(
                num_seasons=6,
                drift_scale_prior=transformed_distribution
                .TransformedDistribution(
                    bijector=invert.Invert(square.Square()),
                    distribution=inverse_gamma.InverseGamma(
                        tf.constant(16., dtype=dtype),
                        tf.constant(4., dtype=dtype))),
                initial_effect_prior=normal.Normal(
                    tf.constant(0., dtype=dtype), tf.constant(1., dtype=dtype)))
        ])

    @tf.function(jit_compile=use_xla)
    def do_sampling(observed_time_series, is_missing):
      return gibbs_sampler.fit_with_gibbs_sampling(
          model,
          missing_values_util.MaskedTimeSeries(observed_time_series,
                                               is_missing),
          num_results=4,
          num_warmup_steps=1,
          seed=seed)
    samples = do_sampling(observed_time_series[..., tf.newaxis], is_missing)
    predictive_dist = gibbs_sampler.one_step_predictive(
        model, samples, thin_every=1)

    # Test that the seeded calculation gives the same result on multiple runs.
    samples2 = do_sampling(observed_time_series[..., tf.newaxis], is_missing)
    predictive_dist2 = gibbs_sampler.one_step_predictive(
        model, samples2, thin_every=1)

    (predictive_mean_, predictive_stddev_,
     predictive_mean2_, predictive_stddev2_) = self.evaluate((
         predictive_dist.mean(), predictive_dist.stddev(),
         predictive_dist2.mean(), predictive_dist2.stddev()))
    self.assertAllEqual(predictive_mean_, predictive_mean2_)
    self.assertAllEqual(predictive_stddev_, predictive_stddev2_)

  def test_no_covariates_is_similar_to_zero_design_matrix(self):
    if not tf.executing_eagerly():
      return
    seed = test_util.test_seed(sampler_type='stateless')
    build_model_seed, sample_seed = samplers.split_seed(seed)
    dtype = tf.float32
    num_timesteps = 5
    num_features = 2
    seed = test_util.test_seed(sampler_type='stateless')
    model, observed_time_series, is_missing = self._build_test_model(
        num_timesteps=num_timesteps,
        num_features=num_features,
        batch_shape=[3],
        prior_class=gibbs_sampler.XLACompilableInverseGamma,
        time_series_shift=10.,
        dtype=dtype,
        design_matrix=None,
        weights_prior_scale=None,
        seed=build_model_seed)

    @tf.function(jit_compile=True)
    def do_sampling(observed_time_series, is_missing):
      return gibbs_sampler.fit_with_gibbs_sampling(
          model,
          missing_values_util.MaskedTimeSeries(observed_time_series,
                                               is_missing),
          num_results=30,
          num_warmup_steps=10,
          seed=sample_seed)

    samples = do_sampling(observed_time_series[..., tf.newaxis], is_missing)

    dummy_model, observed_time_series, is_missing = self._build_test_model(
        num_timesteps=num_timesteps,
        num_features=num_features,
        batch_shape=[3],
        prior_class=gibbs_sampler.XLACompilableInverseGamma,
        dtype=dtype,
        time_series_shift=10.,
        design_matrix=tf.zeros([num_timesteps, num_features]),
        weights_prior_scale=None,
        sparse_weights_nonzero_prob=0.5,
        seed=build_model_seed)  # reuse seed!

    @tf.function(jit_compile=True)
    def do_sampling_again(observed_time_series, is_missing):
      return gibbs_sampler.fit_with_gibbs_sampling(
          dummy_model,
          missing_values_util.MaskedTimeSeries(observed_time_series,
                                               is_missing),
          num_results=30,
          num_warmup_steps=10,
          seed=sample_seed)

    new_samples = do_sampling_again(observed_time_series[..., tf.newaxis],
                                    is_missing)
    for key in ('observation_noise_scale', 'level_scale', 'level',
                'slope_scale', 'slope'):
      first_mean = tf.reduce_mean(getattr(samples, key), axis=0)
      second_mean = tf.reduce_mean(getattr(new_samples, key), axis=0)
      self.assertAllClose(first_mean, second_mean, atol=0.15,
                          msg=f'{key} mean differ')

      first_std = tf.math.reduce_std(getattr(samples, key), axis=0)
      second_std = tf.math.reduce_std(getattr(new_samples, key), axis=0)
      self.assertAllClose(first_std, second_std, atol=0.2,
                          msg=f'{key} stddev differ')

  def test_invalid_model_spec_raises_error(self):
    observed_time_series = tf.ones([2])
    design_matrix = tf.eye(2)
    with self.assertRaisesRegex(
        ValueError, 'Weights prior must be a normal distribution'):
      gibbs_sampler.build_model_for_gibbs_fitting(
          observed_time_series,
          design_matrix=design_matrix,
          weights_prior=student_t.StudentT(df=10, loc=0., scale=1.),
          level_variance_prior=inverse_gamma.InverseGamma(0.01, 0.01),
          observation_noise_variance_prior=inverse_gamma.InverseGamma(
              0.01, 0.01))

    with self.assertRaisesRegex(
        ValueError, 'Level variance prior must be an inverse gamma'):
      gibbs_sampler.build_model_for_gibbs_fitting(
          observed_time_series,
          design_matrix=design_matrix,
          weights_prior=normal.Normal(loc=0., scale=1.),
          level_variance_prior=lognormal.LogNormal(0., 3.),
          observation_noise_variance_prior=inverse_gamma.InverseGamma(
              0.01, 0.01))

    with self.assertRaisesRegex(
        ValueError, 'noise variance prior must be an inverse gamma'):
      gibbs_sampler.build_model_for_gibbs_fitting(
          observed_time_series,
          design_matrix=design_matrix,
          weights_prior=normal.Normal(loc=0., scale=1.),
          level_variance_prior=inverse_gamma.InverseGamma(0.01, 0.01),
          observation_noise_variance_prior=lognormal.LogNormal(0., 3.))

  def test_model_with_linop_precision_works(self):
    observed_time_series = tf.ones([2])
    design_matrix = tf.eye(2)
    sampler = gibbs_sampler.build_model_for_gibbs_fitting(
        observed_time_series,
        design_matrix=design_matrix,
        weights_prior=mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
            precision_factor=tf.linalg.LinearOperatorDiag(tf.ones(2))),
        level_variance_prior=inverse_gamma.InverseGamma(0.01, 0.01),
        observation_noise_variance_prior=inverse_gamma.InverseGamma(0.01, 0.01))
    self.assertIsNotNone(sampler)

  def test_invalid_options_with_none_design_matrix_raises_error(self):
    observed_time_series = tf.ones([2])
    with self.assertRaisesRegex(
        ValueError,
        'Design matrix is None thus sparse_weights_nonzero_prob should '
        'not be defined'):
      gibbs_sampler.build_model_for_gibbs_fitting(
          observed_time_series,
          design_matrix=None,
          weights_prior=None,
          sparse_weights_nonzero_prob=0.4,
          level_variance_prior=inverse_gamma.InverseGamma(0.01, 0.01),
          observation_noise_variance_prior=inverse_gamma.InverseGamma(
              0.01, 0.01))

    with self.assertRaisesRegex(
        ValueError,
        'Design matrix is None thus weights_prior should not be defined'):
      gibbs_sampler.build_model_for_gibbs_fitting(
          observed_time_series,
          design_matrix=None,
          weights_prior=normal.Normal(loc=0., scale=1.),
          level_variance_prior=inverse_gamma.InverseGamma(0.01, 0.01),
          observation_noise_variance_prior=inverse_gamma.InverseGamma(
              0.01, 0.01))

  def test_invalid_model_raises_error(self):
    observed_time_series = tf.convert_to_tensor([1., 0., -1., 2.])
    bad_model = sum_lib.Sum([
        regression.LinearRegression(design_matrix=tf.ones([4, 2])),
        local_level.LocalLevel(observed_time_series=observed_time_series),
    ],
                            observed_time_series=observed_time_series)

    with self.assertRaisesRegex(ValueError, 'does not support Gibbs sampling'):
      gibbs_sampler.fit_with_gibbs_sampling(
          bad_model, observed_time_series, seed=test_util.test_seed())

    bad_model.supports_gibbs_sampling = True
    bad_model_with_correct_params = sum_lib.Sum([
        # An unsupported model component.
        semilocal_linear_trend.SemiLocalLinearTrend(
            observed_time_series=observed_time_series),
        local_level.LocalLevel(observed_time_series=observed_time_series),
        regression.LinearRegression(design_matrix=tf.ones([5, 2]))
    ])
    bad_model_with_correct_params.supports_gibbs_sampling = True

    with self.assertRaisesRegex(
        NotImplementedError,
        'Found unsupported model component for Gibbs Sampling'):
      gibbs_sampler.fit_with_gibbs_sampling(bad_model_with_correct_params,
                                            observed_time_series,
                                            seed=test_util.test_seed())

  @parameterized.named_parameters(
      {
          'testcase_name': 'LocalLinearTrend',
          'use_slope': True
      },
      {
          'testcase_name': 'LocalLevel',
          'use_slope': False
      },
      {
          'testcase_name':
              'LocalLevelWithSingleConstrainedSeasonality',
          'use_slope':
              False,
          # Include a true seasonal effect so there is something interesting
          # for the seasonal component to try to infer.
          'tiled_residual_offset': [4., 2., 0., 3., 6., -12., -4.],
          'seasonal_components_and_drift_scales_factory':
              lambda: [(  # pylint: disable=g-long-lambda
                  seasonal.Seasonal(
                      constrain_mean_effect_to_zero=True,
                      num_seasons=7,
                      drift_scale_prior=transformed_distribution.
                      TransformedDistribution(
                          bijector=invert.Invert(square.Square()),
                          distribution=inverse_gamma.InverseGamma(16., 4.)),
                      initial_effect_prior=normal.Normal(0., 1.)), 0.2)]
      },
      {
          'testcase_name':
              'LocalLevelWithSingleUnconstrainedSeasonality',
          'use_slope':
              False,
          # Include a true seasonal effect so there is something interesting
          # for the seasonal component to try to infer.
          'tiled_residual_offset': [4., 2., 0., 3., 6., -12., -4.],
          'seasonal_components_and_drift_scales_factory':
              lambda: [(  # pylint: disable=g-long-lambda
                  seasonal.Seasonal(
                      constrain_mean_effect_to_zero=False,
                      num_seasons=7,
                      drift_scale_prior=transformed_distribution.
                      TransformedDistribution(
                          bijector=invert.Invert(square.Square()),
                          distribution=inverse_gamma.InverseGamma(16., 4.)),
                      initial_effect_prior=normal.Normal(0., 1.)),
                  # drift_scale
                  0.2)]
      },
      {
          'testcase_name':
              'LocalLevelWithDoubleSeasonality',
          'use_slope':
              False,
          # Include a true seasonal effect so there is something interesting
          # for the seasonal component to try to infer.
          'tiled_residual_offset': [4., 2., 0., 3., 6., -12.],
          'seasonal_components_and_drift_scales_factory':
              lambda: [  # pylint: disable=g-long-lambda
                  (
                      seasonal.Seasonal(
                          num_seasons=2,
                          drift_scale_prior=transformed_distribution.
                          TransformedDistribution(
                              bijector=invert.Invert(square.Square()),
                              distribution=inverse_gamma.InverseGamma(16., 4.)),
                          initial_effect_prior=normal.Normal(0., 1.)),
                      # drift_scale
                      0.2),
                  (
                      seasonal.Seasonal(
                          num_seasons=3,
                          drift_scale_prior=transformed_distribution.
                          TransformedDistribution(
                              bijector=invert.Invert(square.Square()),
                              distribution=inverse_gamma.InverseGamma(16., 4.)),
                          initial_effect_prior=normal.Normal(0., 1.)),
                      # drift_scale
                      0.5)
              ]
      })
  def test_sampled_latents_have_correct_marginals(
      self,
      use_slope,
      tiled_residual_offset=None,
      seasonal_components_and_drift_scales_factory=None,
  ):
    if seasonal_components_and_drift_scales_factory is None:
      seasonal_components_and_drift_scales = []
    else:
      # Use a factory since distributions in JAX can not be initialized
      # before main runs.
      seasonal_components_and_drift_scales = (
          seasonal_components_and_drift_scales_factory())

    seed = test_util.test_seed(sampler_type='stateless')
    residuals_seed, is_missing_seed, level_seed = samplers.split_seed(
        seed, 3, 'test_sampled_level_has_correct_marginals')

    # Have enough timesteps that seasonal effects can be observed.
    num_timesteps = 100
    batch_shape = (3, 1)

    residual_offset = 0.
    if tiled_residual_offset:
      residual_offset = tf.convert_to_tensor([
          tiled_residual_offset[x % len(tiled_residual_offset)]
          for x in range(num_timesteps)
      ])

    observed_residuals = samplers.normal(
        batch_shape + (num_timesteps,), seed=residuals_seed) + (
            residual_offset)
    is_missing = samplers.uniform(
        batch_shape + (num_timesteps,), seed=is_missing_seed) > 0.8
    level_scale = 1.5 * tf.ones(batch_shape)
    observation_noise_scale = 0.2 * tf.ones(batch_shape)

    if use_slope:
      initial_state_prior = mvn_diag.MultivariateNormalDiag(
          loc=[-30., 2.], scale_diag=[1., 0.2])
      slope_scale = 0.5 * tf.ones(batch_shape)
      level_ssm = local_linear_trend.LocalLinearTrendStateSpaceModel(
          num_timesteps=num_timesteps,
          initial_state_prior=initial_state_prior,
          observation_noise_scale=observation_noise_scale,
          level_scale=level_scale,
          slope_scale=slope_scale)
    else:
      initial_state_prior = mvn_diag.MultivariateNormalDiag(
          loc=[-30.], scale_diag=[100.])
      slope_scale = None
      level_ssm = local_level.LocalLevelStateSpaceModel(
          num_timesteps=num_timesteps,
          initial_state_prior=initial_state_prior,
          observation_noise_scale=observation_noise_scale,
          level_scale=level_scale)

    seasonal_ssms = []
    for seasonal_component, drift_scale in seasonal_components_and_drift_scales:
      seasonal_ssms.append(
          seasonal_component.make_state_space_model(
              num_timesteps=num_timesteps,
              initial_state_prior=seasonal_component.initial_state_prior,
              param_vals={},
              drift_scale=drift_scale,
              # Set this to exactly 0, since we want the observation noise
              # of the entire additive space to match observation_noise_scale,
              # but that is already specified on the local ssm.
              observation_noise_scale=0.,
          ))

    ssm = sum_lib.AdditiveStateSpaceModel([level_ssm] + seasonal_ssms)

    posterior_means, posterior_covs = ssm.posterior_marginals(
        observed_residuals[..., tf.newaxis], mask=is_missing)
    latents_samples = gibbs_sampler._resample_latents(
        observed_residuals=observed_residuals,
        level_scale=level_scale,
        slope_scale=slope_scale,
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=initial_state_prior,
        is_missing=is_missing,
        sample_shape=10000,
        seed=level_seed,
        seasonal_components=[
            x[0] for x in seasonal_components_and_drift_scales
        ],
        seasonal_drift_scales=tf.convert_to_tensor(
            value=[x[1] for x in seasonal_components_and_drift_scales],
            name='drift_scale'))

    (posterior_means_, posterior_covs_, latents_means_,
     latents_covs_) = self.evaluate((posterior_means, posterior_covs,
                                     tf.reduce_mean(latents_samples, axis=0),
                                     sample_stats.covariance(
                                         latents_samples,
                                         sample_axis=0,
                                         event_axis=-1)))
    # TODO(axch, cgs): Can we use assertAllMeansClose here?  The
    # latents_samples are presumably not IID across axis=0, so the
    # statistical assumptions are not satisfied.
    self.assertAllClose(latents_means_,
                        posterior_means_, atol=0.1)
    self.assertAllClose(latents_covs_,
                        posterior_covs_, atol=0.1)

  def test_sampled_scale_follows_correct_distribution(self):
    strm = test_util.test_seed_stream()
    prior = inverse_gamma.InverseGamma(concentration=0.1, scale=0.1)

    num_timesteps = 100
    observed_samples = tf.random.normal([2, num_timesteps], seed=strm()) * 3.
    is_missing = tf.random.uniform([2, num_timesteps], seed=strm()) > 0.9

    # Check that posterior variance samples have the moments of the correct
    # InverseGamma distribution.
    posterior_scale_samples = tf.vectorized_map(
        lambda seed: gibbs_sampler._resample_scale(  # pylint: disable=g-long-lambda
            prior=prior,
            observed_residuals=observed_samples,
            is_missing=is_missing,
            seed=seed),
        samplers.split_seed(strm(), tf.constant(10000)))

    concentration = prior.concentration + tf.reduce_sum(
        1 - tf.cast(is_missing, tf.float32), axis=-1)/2.
    scale = prior.scale + tf.reduce_sum(
        (observed_samples * tf.cast(~is_missing, tf.float32))**2, axis=-1)/2.
    posterior_scale_samples_, concentration_, scale_ = self.evaluate(
        (posterior_scale_samples, concentration, scale))
    self.assertAllClose(np.mean(posterior_scale_samples_**2, axis=0),
                        scale_ / (concentration_ - 1), atol=0.05)
    self.assertAllClose(
        np.std(posterior_scale_samples_**2, axis=0),
        scale_ / ((concentration_ - 1) * np.sqrt(concentration_ - 2)),
        atol=0.05)

  def test_sampled_weights_follow_correct_distribution(self):
    seed = test_util.test_seed(sampler_type='stateless')
    design_seed, true_weights_seed, sampled_weights_seed = samplers.split_seed(
        seed, 3, 'test_sampled_weights_follow_correct_distribution')
    num_timesteps = 10
    num_features = 2
    batch_shape = [3, 1]
    design_matrix = self.evaluate(samplers.normal(
        batch_shape + [num_timesteps, num_features], seed=design_seed))
    true_weights = self.evaluate(samplers.normal(
        batch_shape + [num_features, 1], seed=true_weights_seed) * 10.0)
    targets = np.matmul(design_matrix, true_weights)
    is_missing = np.array([False, False, False, True, True,
                           False, False, True, False, False])
    prior_scale = tf.convert_to_tensor(5.)
    likelihood_scale = tf.convert_to_tensor(0.1)

    # Analytically compute the true posterior distribution on weights.
    valid_design_matrix = design_matrix[..., ~is_missing, :]
    valid_targets = targets[..., ~is_missing, :]
    num_valid_observations = tf.shape(valid_design_matrix)[-2]
    weights_posterior_mean, weights_posterior_cov, _ = linear_gaussian_update(
        prior_mean=tf.zeros([num_features, 1]),
        prior_cov=tf.eye(num_features) * prior_scale**2,
        observation_matrix=tfl.LinearOperatorFullMatrix(valid_design_matrix),
        observation_noise=mvn_diag.MultivariateNormalDiag(
            loc=tf.zeros([num_valid_observations]),
            scale_diag=likelihood_scale * tf.ones([num_valid_observations])),
        x_observed=valid_targets)

    # Check that the empirical moments of sampled weights match the true values.
    sampled_weights = tf.vectorized_map(
        lambda seed: gibbs_sampler._resample_weights(  # pylint: disable=g-long-lambda
            design_matrix=tf.where(is_missing[..., tf.newaxis],
                                   tf.zeros_like(design_matrix), design_matrix),
            target_residuals=targets[..., 0],
            observation_noise_scale=likelihood_scale,
            weights_prior_scale=tf.linalg.LinearOperatorScaledIdentity(
                num_features, prior_scale),
            seed=seed),
        samplers.split_seed(sampled_weights_seed, tf.constant(10000)))
    sampled_weights_mean = tf.reduce_mean(sampled_weights, axis=0)
    centered_weights = sampled_weights - weights_posterior_mean[..., 0]
    sampled_weights_cov = tf.reduce_mean(centered_weights[..., :, tf.newaxis] *
                                         centered_weights[..., tf.newaxis, :],
                                         axis=0)

    (sampled_weights_mean_, weights_posterior_mean_,
     sampled_weights_cov_, weights_posterior_cov_) = self.evaluate((
         sampled_weights_mean, weights_posterior_mean[..., 0],
         sampled_weights_cov, weights_posterior_cov))
    self.assertAllClose(sampled_weights_mean_, weights_posterior_mean_,
                        atol=0.01, rtol=0.05)
    self.assertAllClose(sampled_weights_cov_, weights_posterior_cov_,
                        atol=0.01, rtol=0.05)

  def test_sparse_weights_nonzero_prob_of_one_works(self):
    true_weights = tf.constant([0., 0., 2., 0., -2.])
    model, observed_time_series, _ = self._build_test_model(
        num_timesteps=20,
        num_features=5,
        missing_prob=0.,
        true_noise_scale=0.1,
        weights=true_weights,
        weights_prior_scale=None,  # Default g-prior.
        sparse_weights_nonzero_prob=1.)

    @tf.function(autograph=False)
    def do_sampling():
      return gibbs_sampler.fit_with_gibbs_sampling(
          model,
          observed_time_series,
          num_results=100,
          num_warmup_steps=100,
          seed=test_util.test_seed(sampler_type='stateless'))

    samples = self.evaluate(do_sampling())
    mean_weights = tf.reduce_mean(samples.weights, axis=-2)
    nonzero_probs = tf.reduce_mean(
        tf.cast(tf.not_equal(samples.weights, 0.), tf.float32),
        axis=-2)
    # Increasing `num_timesteps` relative to `num_features` would give more
    # precise weight estimates, at the cost of longer test runtime.
    # TODO(axch, cgs): Can we use assertAllMeansClose here too?  The
    # samples are presumably not IID across axis=0, so the
    # statistical assumptions are not satisfied.
    self.assertAllClose(mean_weights, true_weights, atol=0.3)
    self.assertAllClose(nonzero_probs, [1., 1., 1., 1., 1.])

  @parameterized.named_parameters(
      {
          'testcase_name': 'Rank1Updates',
          'use_dynamic_cholesky': False,
      }, {
          'testcase_name': 'DynamicCholesky',
          'use_dynamic_cholesky': True,
      })
  def test_sparse_regression_recovers_plausible_weights(self,
                                                        use_dynamic_cholesky):
    true_weights = tf.constant([0., 0., 2., 0., -2.])
    model, observed_time_series, _ = self._build_test_model(
        num_timesteps=20,
        num_features=5,
        missing_prob=0.,
        true_noise_scale=0.1,
        weights=true_weights,
        weights_prior_scale=None,  # Default g-prior.
        sparse_weights_nonzero_prob=0.4)

    @tf.function(autograph=False)
    def do_sampling():
      return gibbs_sampler.fit_with_gibbs_sampling(
          model,
          observed_time_series,
          num_results=100,
          num_warmup_steps=100,
          seed=test_util.test_seed(sampler_type='stateless'),
          experimental_use_dynamic_cholesky=use_dynamic_cholesky)

    if JAX_MODE and use_dynamic_cholesky:
      with self.assertRaises(ValueError):
        self.evaluate(do_sampling())
      return
    else:
      samples = self.evaluate(do_sampling())
    mean_weights = tf.reduce_mean(samples.weights, axis=-2)
    nonzero_probs = tf.reduce_mean(
        tf.cast(tf.not_equal(samples.weights, 0.), tf.float32),
        axis=-2)
    # Increasing `num_timesteps` relative to `num_features` would give more
    # precise weight estimates, at the cost of longer test runtime.
    # TODO(axch, cgs): Can we use assertAllMeansClose here too?  The
    # samples are presumably not IID across axis=0, so the
    # statistical assumptions are not satisfied.
    self.assertAllClose(mean_weights, true_weights, atol=0.3)
    self.assertAllClose(nonzero_probs, [0., 0., 1., 0., 1.], atol=0.2)

  def test_regression_does_not_explain_seasonal_variation(self):
    """Tests that seasonality is used, not regression, when it is best.

    This creates a situation where a regressor is added that explains
    the seasonality for only a portion of the timeseries, but seasonality
    explains it for whole time.
    """
    num_timesteps = 200
    tiled_seasonal_effect_one = [-2., -4., 2., 8., 5.]
    tiled_seasonal_effect_two = [4., 6., -3., 0.]
    seasonal_effect = [
        tiled_seasonal_effect_one[x % len(tiled_seasonal_effect_one)] +
        tiled_seasonal_effect_two[x % len(tiled_seasonal_effect_two)]
        for x in range(num_timesteps)
    ]
    # Regression explains just a portion of the effect.
    regression_value = [
        seasonal_effect[x] if x < 25 else 0. for x in range(num_timesteps)
    ]
    model, observed_time_series, _ = self._build_test_model(
        num_timesteps=num_timesteps,
        num_features=1,
        missing_prob=0.,
        true_noise_scale=0.1,
        design_matrix=tf.convert_to_tensor(regression_value)[..., tf.newaxis],
        # Do not generate the observed series using the regression, just
        # use the tiled_residual_offset since the goal is to see what happens
        # when the regression explains a portion of the seasonal effect.
        weights=tf.constant([0.]),
        weights_prior_scale=None,  # Default g-prior.
        sparse_weights_nonzero_prob=0.01,
        tiled_residual_offset=seasonal_effect,
        true_slope_scale=None,  # Simplify the test data to have no trend.
        seasonal_components=[
            seasonal.Seasonal(
                name='season1',
                num_seasons=len(tiled_seasonal_effect_one),
                drift_scale_prior=transformed_distribution
                .TransformedDistribution(
                    bijector=invert.Invert(square.Square()),
                    distribution=inverse_gamma.InverseGamma(0.01, 0.0001)),
                initial_effect_prior=normal.Normal(0., 1.)),
            seasonal.Seasonal(
                name='season2',
                num_seasons=len(tiled_seasonal_effect_two),
                drift_scale_prior=transformed_distribution
                .TransformedDistribution(
                    bijector=invert.Invert(square.Square()),
                    distribution=inverse_gamma.InverseGamma(0.01, 0.0001)),
                initial_effect_prior=normal.Normal(0., 1.))
        ])

    @tf.function(autograph=False)
    def do_sampling():
      return gibbs_sampler.fit_with_gibbs_sampling(
          model,
          observed_time_series,
          num_results=100,
          num_warmup_steps=100,
          seed=test_util.test_seed(sampler_type='stateless'))

    samples = do_sampling()
    nonzero_probs = tf.reduce_mean(
        tf.cast(tf.not_equal(samples.weights, 0.), tf.float32), axis=-2)
    # Verify that essentially none of the sample are non-zero, since we
    # explained the timeseries with a seasonal effect. If the regression
    # has a defect introduced to not take the seasonal effect into account,
    # this becomes near 1. Similarly, if either of the seasonal components
    # are removed, it becomes near 1 - proving that multiple-seasonal components
    # is respected by regression.
    self.assertAllLess(nonzero_probs, 0.05)

  @parameterized.named_parameters(
      {
          'testcase_name':
              'DriftRequired',
          'seasonal_component_factory':
              lambda: seasonal.Seasonal(num_seasons=3, allow_drift=False),  # pylint: disable=g-long-lambda
          'assertion_regex':
              'Only seasonality with drift'
      },
      {
          'testcase_name':
              'PriorRequiresTransformedDistribution',
          'seasonal_component_factory':
              lambda: seasonal.Seasonal(  # pylint: disable=g-long-lambda
                  num_seasons=3,
                  drift_scale_prior=inverse_gamma.InverseGamma(
                      tf.constant(16.), tf.constant(4.))),
          'assertion_regex':
              'components drift scale must be'
      },
      {
          'testcase_name':
              'PriorRequiresInvertedBijector',
          'seasonal_component_factory':
              lambda: seasonal.Seasonal(  # pylint: disable=g-long-lambda
                  num_seasons=3,
                  drift_scale_prior=transformed_distribution.
                  TransformedDistribution(
                      bijector=square.Square(),
                      distribution=inverse_gamma.InverseGamma(
                          tf.constant(16.), tf.constant(4.)))),
          'assertion_regex':
              'components drift scale must be'
      },
      {
          'testcase_name':
              'PriorRequiresInvertedSquaredBijector',
          'seasonal_component_factory':
              lambda: seasonal.Seasonal(  # pylint: disable=g-long-lambda
                  num_seasons=3,
                  drift_scale_prior=transformed_distribution.
                  TransformedDistribution(
                      bijector=invert.Invert(identity.Identity()),
                      distribution=inverse_gamma.InverseGamma(
                          tf.constant(16.), tf.constant(4.)))),
          'assertion_regex':
              'components drift scale must be'
      },
      {
          'testcase_name':
              'PriorRequiresTransformedInverseGamma',
          'seasonal_component_factory':
              lambda: seasonal.Seasonal(  # pylint: disable=g-long-lambda
                  num_seasons=3,
                  drift_scale_prior=transformed_distribution.
                  TransformedDistribution(
                      bijector=invert.Invert(square.Square()),
                      distribution=lognormal.LogNormal(loc=3., scale=4.))),
          'assertion_regex':
              'components drift scale must be'
      })
  def test_seasonal_component_assumptions_enforced(self,
                                                   seasonal_component_factory,
                                                   assertion_regex):
    with self.assertRaisesRegex(NotImplementedError, assertion_regex):
      self._build_test_model(
          # Use a factory since distributions in JAX can not be initialized
          # before main runs.
          seasonal_components=[seasonal_component_factory()])

  def test_get_seasonal_latents_shape_no_chains_constrained_and_unconstrained(
      self):
    model = sum_lib.Sum([
        local_level.LocalLevel(),
        seasonal.Seasonal(
            name='s1', num_seasons=6, constrain_mean_effect_to_zero=True),
        seasonal.Seasonal(
            name='s2', num_seasons=10, constrain_mean_effect_to_zero=False)
    ])
    model.supports_gibbs_sampling = True  # Work-around to simplify testing.
    self.assertAllEqual(
        [
            3,
            4,
            120,
            # 5 from the constrained, 10 from the unconstrained.
            15
        ],
        gibbs_sampler.get_seasonal_latents_shape(
            timeseries=tf.ones(shape=[3, 4, 120]), model=model))

  def test_get_seasonal_latents_shape_chains_constrained_and_unconstrained(
      self):
    model = sum_lib.Sum([
        local_level.LocalLevel(),
        seasonal.Seasonal(
            name='s1', num_seasons=6, constrain_mean_effect_to_zero=True),
        seasonal.Seasonal(
            name='s2', num_seasons=10, constrain_mean_effect_to_zero=False)
    ])
    model.supports_gibbs_sampling = True  # Work-around to simplify testing.
    self.assertAllEqual(
        [
            7,
            3,
            4,
            120,
            # 5 from the constrained, 10 from the unconstrained.
            15
        ],
        gibbs_sampler.get_seasonal_latents_shape(
            num_chains=[7], timeseries=tf.ones(shape=[3, 4, 120]), model=model))

  def test_get_seasonal_latents_shape_no_seasonal_components(self):
    model = sum_lib.Sum([
        local_level.LocalLevel(),
    ])
    model.supports_gibbs_sampling = True  # Work-around to simplify testing.
    self.assertAllEqual(
        [
            3,
            4,
            120,
            # No components
            0
        ],
        gibbs_sampler.get_seasonal_latents_shape(
            timeseries=tf.ones(shape=[3, 4, 120]), model=model))


if __name__ == '__main__':
  test_util.main()
