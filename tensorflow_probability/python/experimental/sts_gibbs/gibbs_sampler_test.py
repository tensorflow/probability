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
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions.linear_gaussian_ssm import linear_gaussian_update
from tensorflow_probability.python.experimental.sts_gibbs import gibbs_sampler
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts.internal import util as sts_util


tfd = tfp.distributions
tfl = tf.linalg


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
                        prior_class=tfd.InverseGamma,
                        weights=None,
                        weights_prior_scale=10.,
                        sparse_weights_nonzero_prob=None,
                        time_series_shift=0.,
                        dtype=tf.float32,
                        design_matrix_is_none=False):
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
    if design_matrix_is_none:
      design_matrix = None
      regression = tf.zeros(num_timesteps, dtype)
    else:
      design_matrix = samplers.normal([num_timesteps, num_features],
                                      dtype=dtype,
                                      seed=design_seed)
      regression = tf.linalg.matvec(design_matrix, weights)
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
    time_series = (regression + noise + level + time_series_shift)
    is_missing = samplers.uniform(
        list(batch_shape) + [num_timesteps],
        dtype=dtype, seed=is_missing_seed) < missing_prob

    observation_noise_variance_prior = prior_class(
        concentration=tf.cast(0.01, dtype),
        scale=tf.cast(0.01 * 0.01, dtype))
    observation_noise_variance_prior.upper_bound = 100.0

    observed_time_series = tfp.sts.MaskedTimeSeries(
        time_series[..., tf.newaxis], is_missing)
    if time_series_shift != 0.:
      observed_mean, observed_stddev, observed_initial = (
          sts_util.empirical_statistics(observed_time_series))
      initial_level_prior = tfd.Normal(
          loc=observed_mean + observed_initial,
          scale=tf.abs(observed_initial) + observed_stddev)
    else:
      initial_level_prior = None

    model = gibbs_sampler.build_model_for_gibbs_fitting(
        observed_time_series=observed_time_series,
        design_matrix=design_matrix,
        weights_prior=(
            None if weights_prior_scale is None
            else tfd.Normal(loc=tf.cast(0., dtype),
                            scale=tf.cast(weights_prior_scale, dtype))),
        level_variance_prior=prior_class(
            concentration=tf.cast(0.01, dtype),
            scale=tf.cast(0.01 * 0.01, dtype)),
        slope_variance_prior=None if true_slope_scale is None else prior_class(
            concentration=tf.cast(0.01, dtype),
            scale=tf.cast(0.01 * 0.01, dtype)),
        initial_level_prior=initial_level_prior,
        observation_noise_variance_prior=observation_noise_variance_prior,
        sparse_weights_nonzero_prob=sparse_weights_nonzero_prob)
    return model, time_series, is_missing

  @parameterized.named_parameters(
      {
          'testcase_name': 'LocalLinearTrend',
          'use_slope': True,
          'num_chains': (),
          'time_series_shift': 0.
      }, {
          'testcase_name': 'LocalLinearTrend_4chains',
          'use_slope': True,
          'num_chains': 4,
          'time_series_shift': 0.
      }, {
          'testcase_name': 'LocalLevel',
          'use_slope': False,
          'num_chains': (),
          'time_series_shift': 0.
      }, {
          'testcase_name': 'LocalLevel_4chains',
          'use_slope': False,
          'num_chains': 4,
          'time_series_shift': 0.
      }, {
          'testcase_name': 'UnscaledTimeSeries_LocalLinear',
          'use_slope': False,
          'num_chains': (),
          'time_series_shift': 100.
      }, {
          'testcase_name': 'UnscaledTimeSeries_LocalLinearTrend',
          'use_slope': True,
          'num_chains': (),
          'time_series_shift': 100.
      })
  def test_forecasts_match_reference(
      self, use_slope, num_chains, time_series_shift):
    seed = test_util.test_seed()
    num_observed_steps = 5
    num_forecast_steps = 4
    num_results = 10000

    # Dividing the number of results with number of chains so we sample the same
    # total number of MCMC samples.
    if not tf.nest.is_nested(num_chains):
      num_results = num_results // num_chains

    model, observed_time_series, is_missing = self._build_test_model(
        num_timesteps=num_observed_steps + num_forecast_steps,
        true_slope_scale=0.5 if use_slope else None,
        batch_shape=[3],
        time_series_shift=time_series_shift)

    @tf.function(autograph=False)
    def do_sampling():
      return gibbs_sampler.fit_with_gibbs_sampling(
          model,
          tfp.sts.MaskedTimeSeries(
              observed_time_series[..., :num_observed_steps, tf.newaxis],
              is_missing[..., :num_observed_steps]),
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
        model, samples, num_forecast_steps=num_forecast_steps,
        thin_every=1)
    predictive_mean, predictive_stddev = self.evaluate((
        predictive_dist.mean(), predictive_dist.stddev()))
    self.assertAllEqual(predictive_mean.shape,
                        [3, num_observed_steps + num_forecast_steps])
    self.assertAllEqual(predictive_stddev.shape,
                        [3, num_observed_steps + num_forecast_steps])

    # big tolerance, but makes sure the predictive mean initializes near
    # the initial time series value
    self.assertAllClose(tf.reduce_mean(predictive_mean[:, 0]),
                        observed_time_series[0, 0],
                        atol=10.)

    if use_slope:
      parameter_samples = (samples.observation_noise_scale,
                           samples.level_scale,
                           samples.slope_scale,
                           samples.weights)
    else:
      parameter_samples = (samples.observation_noise_scale,
                           samples.level_scale,
                           samples.weights)

    # Note that although we expect the Gibbs-sampled forecasts to match a
    # reference implementation, we *don't* expect the one-step predictions to
    # match `tfp.sts.one_step_predictive`, because that makes predictions using
    # a filtered posterior (i.e., given only previous observations) whereas the
    # Gibbs-sampled latent `level`s will incorporate some information from
    # future observations.
    reference_forecast_dist = tfp.sts.forecast(
        model,
        observed_time_series=observed_time_series[..., :num_observed_steps],
        parameter_samples=parameter_samples,
        num_steps_forecast=num_forecast_steps)

    reference_forecast_mean, reference_forecast_stddev = self.evaluate((
        reference_forecast_dist.mean()[..., 0],
        reference_forecast_dist.stddev()[..., 0]))

    self.assertAllClose(predictive_mean[..., -num_forecast_steps:],
                        reference_forecast_mean,
                        atol=1.0 if use_slope else 0.3)
    self.assertAllClose(predictive_stddev[..., -num_forecast_steps:],
                        reference_forecast_stddev,
                        atol=2.0 if use_slope else 1.0)

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
        dtype=dtype)

    @tf.function(jit_compile=use_xla)
    def do_sampling(observed_time_series, is_missing):
      return gibbs_sampler.fit_with_gibbs_sampling(
          model, tfp.sts.MaskedTimeSeries(
              observed_time_series, is_missing),
          num_results=4, num_warmup_steps=1, seed=seed)
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

  def test_no_covariates_support(self):
    if not tf.executing_eagerly():
      return
    seed = test_util.test_seed(sampler_type='stateless')
    dtype = tf.float32
    model, observed_time_series, is_missing = self._build_test_model(
        num_timesteps=5,
        batch_shape=[3],
        prior_class=gibbs_sampler.XLACompilableInverseGamma,
        dtype=dtype,
        design_matrix_is_none=True,
        weights_prior_scale=None)

    @tf.function(jit_compile=True)
    def do_sampling(observed_time_series, is_missing):
      return gibbs_sampler.fit_with_gibbs_sampling(
          model,
          tfp.sts.MaskedTimeSeries(observed_time_series, is_missing),
          num_results=4,
          num_warmup_steps=1,
          seed=seed)

    # This simply ensures we can get samples without throwing an error.
    # TODO(kloveless): Add tests that compare the results with either another
    # method of inference, or to a model with covariates, but all covariates
    # are zero.
    samples = do_sampling(observed_time_series[..., tf.newaxis], is_missing)
    gibbs_sampler.one_step_predictive(model, samples, thin_every=1)

  def test_invalid_model_spec_raises_error(self):
    observed_time_series = tf.ones([2])
    design_matrix = tf.eye(2)
    with self.assertRaisesRegexp(
        ValueError, 'Weights prior must be a normal distribution'):
      gibbs_sampler.build_model_for_gibbs_fitting(
          observed_time_series, design_matrix=design_matrix,
          weights_prior=tfd.StudentT(df=10, loc=0., scale=1.),
          level_variance_prior=tfd.InverseGamma(0.01, 0.01),
          observation_noise_variance_prior=tfd.InverseGamma(0.01, 0.01))

    with self.assertRaisesRegexp(
        ValueError, 'Level variance prior must be an inverse gamma'):
      gibbs_sampler.build_model_for_gibbs_fitting(
          observed_time_series, design_matrix=design_matrix,
          weights_prior=tfd.Normal(loc=0., scale=1.),
          level_variance_prior=tfd.LogNormal(0., 3.),
          observation_noise_variance_prior=tfd.InverseGamma(0.01, 0.01))

    with self.assertRaisesRegexp(
        ValueError, 'noise variance prior must be an inverse gamma'):
      gibbs_sampler.build_model_for_gibbs_fitting(
          observed_time_series, design_matrix=design_matrix,
          weights_prior=tfd.Normal(loc=0., scale=1.),
          level_variance_prior=tfd.InverseGamma(0.01, 0.01),
          observation_noise_variance_prior=tfd.LogNormal(0., 3.))

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
          level_variance_prior=tfd.InverseGamma(0.01, 0.01),
          observation_noise_variance_prior=tfd.InverseGamma(0.01, 0.01))

    with self.assertRaisesRegex(
        ValueError,
        'Design matrix is None thus weights_prior should not be defined'):
      gibbs_sampler.build_model_for_gibbs_fitting(
          observed_time_series,
          design_matrix=None,
          weights_prior=tfd.Normal(loc=0., scale=1.),
          level_variance_prior=tfd.InverseGamma(0.01, 0.01),
          observation_noise_variance_prior=tfd.InverseGamma(0.01, 0.01))

  def test_invalid_model_raises_error(self):
    observed_time_series = tf.convert_to_tensor([1., 0., -1., 2.])
    bad_model = tfp.sts.Sum(
        [tfp.sts.LinearRegression(design_matrix=tf.ones([4, 2])),
         tfp.sts.LocalLevel(observed_time_series=observed_time_series),],
        observed_time_series=observed_time_series)

    with self.assertRaisesRegexp(ValueError, 'does not support Gibbs sampling'):
      gibbs_sampler.fit_with_gibbs_sampling(
          bad_model, observed_time_series, seed=test_util.test_seed())

    bad_model.supports_gibbs_sampling = True
    with self.assertRaisesRegexp(
        ValueError, 'Expected the first model component to be an instance of'):
      gibbs_sampler.fit_with_gibbs_sampling(
          bad_model, observed_time_series, seed=test_util.test_seed())

    bad_model_with_correct_params = tfp.sts.Sum([
        # A seasonal model with no drift has no parameters, so adding it
        # won't break the check for correct params.
        tfp.sts.Seasonal(num_seasons=2,
                         allow_drift=False,
                         observed_time_series=observed_time_series),
        tfp.sts.LocalLevel(observed_time_series=observed_time_series),
        tfp.sts.LinearRegression(design_matrix=tf.ones([5, 2]))])
    bad_model_with_correct_params.supports_gibbs_sampling = True

    with self.assertRaisesRegexp(ValueError,
                                 'Expected the first model component to be an '
                                 'instance of `tfp.sts.LocalLevel`'):
      gibbs_sampler.fit_with_gibbs_sampling(bad_model_with_correct_params,
                                            observed_time_series,
                                            seed=test_util.test_seed())

  @parameterized.named_parameters(
      {'testcase_name': 'LocalLinearTrend', 'use_slope': True},
      {'testcase_name': 'LocalLevel', 'use_slope': False})
  def test_sampled_latents_have_correct_marginals(self, use_slope):
    seed = test_util.test_seed(sampler_type='stateless')
    residuals_seed, is_missing_seed, level_seed = samplers.split_seed(
        seed, 3, 'test_sampled_level_has_correct_marginals')

    num_timesteps = 10

    observed_residuals = samplers.normal(
        [3, 1, num_timesteps], seed=residuals_seed)
    is_missing = samplers.uniform(
        [3, 1, num_timesteps], seed=is_missing_seed) > 0.8
    level_scale = 1.5 * tf.ones([3, 1])
    observation_noise_scale = 0.2 * tf.ones([3, 1])

    if use_slope:
      initial_state_prior = tfd.MultivariateNormalDiag(loc=[-30., 2.],
                                                       scale_diag=[1., 0.2])
      slope_scale = 0.5 * tf.ones([3, 1])
      ssm = tfp.sts.LocalLinearTrendStateSpaceModel(
          num_timesteps=num_timesteps,
          initial_state_prior=initial_state_prior,
          observation_noise_scale=observation_noise_scale,
          level_scale=level_scale,
          slope_scale=slope_scale)
    else:
      initial_state_prior = tfd.MultivariateNormalDiag(loc=[-30.],
                                                       scale_diag=[100.])
      slope_scale = None
      ssm = tfp.sts.LocalLevelStateSpaceModel(
          num_timesteps=num_timesteps,
          initial_state_prior=initial_state_prior,
          observation_noise_scale=observation_noise_scale,
          level_scale=level_scale)

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
        seed=level_seed)

    (posterior_means_,
     posterior_covs_,
     latents_means_,
     latents_covs_) = self.evaluate((
         posterior_means,
         posterior_covs,
         tf.reduce_mean(latents_samples, axis=0),
         tfp.stats.covariance(latents_samples,
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
    prior = tfd.InverseGamma(concentration=0.1, scale=0.1)

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
        tfp.random.split_seed(strm(), tf.constant(10000)))

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
        observation_noise=tfd.MultivariateNormalDiag(
            loc=tf.zeros([num_valid_observations]),
            scale_diag=likelihood_scale * tf.ones([num_valid_observations])),
        x_observed=valid_targets)

    # Check that the empirical moments of sampled weights match the true values.
    sampled_weights = tf.vectorized_map(
        lambda seed: gibbs_sampler._resample_weights(  # pylint: disable=g-long-lambda
            design_matrix=tf.where(is_missing[..., tf.newaxis],
                                   tf.zeros_like(design_matrix),
                                   design_matrix),
            target_residuals=targets[..., 0],
            observation_noise_scale=likelihood_scale,
            weights_prior_scale=tf.linalg.LinearOperatorScaledIdentity(
                num_features, prior_scale),
            seed=seed),
        tfp.random.split_seed(sampled_weights_seed, tf.constant(10000)))
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

  def test_sparse_regression_recovers_plausible_weights(self):
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
    self.assertAllClose(nonzero_probs, [0., 0., 1., 0., 1.], atol=0.2)


if __name__ == '__main__':
  test_util.main()
