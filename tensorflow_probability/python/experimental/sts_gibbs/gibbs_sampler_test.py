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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd


from tensorflow_probability.python.distributions.linear_gaussian_ssm import linear_gaussian_update
from tensorflow_probability.python.experimental.sts_gibbs import gibbs_sampler
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

from tensorflow.python.ops import parallel_for  # pylint: disable=g-direct-tensorflow-import


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
                        dtype=tf.float32):
    seed = test_util.test_seed(sampler_type='stateless')
    (design_seed,
     weights_seed,
     noise_seed,
     level_seed,
     is_missing_seed) = samplers.split_seed(seed, 5, salt='_build_test_model')

    design_matrix = samplers.normal(
        [num_timesteps, num_features], dtype=dtype, seed=design_seed)
    weights = samplers.normal(
        list(batch_shape) + [num_features], dtype=dtype, seed=weights_seed)
    regression = tf.linalg.matvec(design_matrix, weights)
    noise = samplers.normal(
        list(batch_shape) + [num_timesteps],
        dtype=dtype, seed=noise_seed) * true_noise_scale
    level = tf.cumsum(samplers.normal(
        list(batch_shape) + [num_timesteps],
        dtype=dtype, seed=level_seed) * true_level_scale, axis=-1)
    time_series = (regression + noise + level)
    is_missing = samplers.uniform(
        list(batch_shape) + [num_timesteps],
        dtype=dtype, seed=is_missing_seed) < missing_prob

    model = gibbs_sampler.build_model_for_gibbs_fitting(
        observed_time_series=tfp.sts.MaskedTimeSeries(
            time_series[..., tf.newaxis], is_missing),
        design_matrix=design_matrix,
        weights_prior=tfd.Normal(loc=tf.cast(0., dtype),
                                 scale=tf.cast(10.0, dtype)),
        level_variance_prior=tfd.InverseGamma(
            concentration=tf.cast(0.01, dtype),
            scale=tf.cast(0.01 * 0.01, dtype)),
        observation_noise_variance_prior=tfd.InverseGamma(
            concentration=tf.cast(0.01, dtype),
            scale=tf.cast(0.01 * 0.01, dtype)))
    return model, time_series, is_missing

  def test_forecasts_are_sane(self):
    seed = test_util.test_seed()
    num_observed_steps = 5
    num_forecast_steps = 3
    model, observed_time_series, is_missing = self._build_test_model(
        num_timesteps=num_observed_steps + num_forecast_steps,
        batch_shape=[3])

    samples = gibbs_sampler.fit_with_gibbs_sampling(
        model, tfp.sts.MaskedTimeSeries(
            observed_time_series[..., :num_observed_steps, tf.newaxis],
            is_missing[..., :num_observed_steps]),
        num_results=5, num_warmup_steps=10,
        seed=seed, compile_steps_with_xla=False)
    predictive_dist = gibbs_sampler.one_step_predictive(
        model, samples, num_forecast_steps=num_forecast_steps,
        thin_every=1)
    predictive_mean, predictive_stddev = self.evaluate((
        predictive_dist.mean(), predictive_dist.stddev()))

    self.assertAllEqual(predictive_mean.shape,
                        [3, num_observed_steps + num_forecast_steps])
    self.assertAllEqual(predictive_stddev.shape,
                        [3, num_observed_steps + num_forecast_steps])

    # Uncertainty should increase over the forecast period.
    self.assertTrue(
        np.all(predictive_stddev[..., num_observed_steps + 1:] >
               predictive_stddev[..., num_observed_steps:-1]))

  @parameterized.named_parameters(
      {'testcase_name': 'float32_xla', 'dtype': tf.float32, 'use_xla': True},
      {'testcase_name': 'float16', 'dtype': tf.float16, 'use_xla': False})
  def test_end_to_end_prediction_works_and_is_deterministic(
      self, dtype, use_xla):
    if not tf.executing_eagerly():
      return
    seed = test_util.test_seed()
    model, observed_time_series, is_missing = self._build_test_model(
        num_timesteps=5, batch_shape=[3])

    samples = gibbs_sampler.fit_with_gibbs_sampling(
        model, tfp.sts.MaskedTimeSeries(
            observed_time_series[..., tf.newaxis], is_missing),
        num_results=4, num_warmup_steps=1, seed=seed,
        compile_steps_with_xla=use_xla)
    predictive_dist = gibbs_sampler.one_step_predictive(
        model, samples, thin_every=1)

    # Test that the seeded calculation gives the same result on multiple runs.
    samples2 = gibbs_sampler.fit_with_gibbs_sampling(
        model, tfp.sts.MaskedTimeSeries(observed_time_series, is_missing),
        num_results=4, num_warmup_steps=1, seed=seed,
        compile_steps_with_xla=use_xla)
    predictive_dist2 = gibbs_sampler.one_step_predictive(
        model, samples2, thin_every=1)

    (predictive_mean_, predictive_stddev_,
     predictive_mean2_, predictive_stddev2_) = self.evaluate((
         predictive_dist.mean(), predictive_dist.stddev(),
         predictive_dist2.mean(), predictive_dist2.stddev()))
    self.assertAllEqual(predictive_mean_, predictive_mean2_)
    self.assertAllEqual(predictive_stddev_, predictive_stddev2_)

  def test_invalid_model_spec_raises_error(self):
    observed_time_series = tf.ones([2])
    design_matrix = tf.eye(2)
    with self.assertRaisesRegexp(ValueError,
                                 'Weights prior must be a univariate normal'):
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

  def test_invalid_model_raises_error(self):
    observed_time_series = tf.convert_to_tensor([1., 0., -1., 2.])
    bad_model = tfp.sts.Sum(
        [tfp.sts.LinearRegression(design_matrix=tf.ones([4, 2])),
         tfp.sts.LocalLevel(observed_time_series=observed_time_series),],
        observed_time_series=observed_time_series)

    with self.assertRaisesRegexp(ValueError, 'does not support Gibbs sampling'):
      gibbs_sampler.fit_with_gibbs_sampling(bad_model, observed_time_series)

    bad_model.supports_gibbs_sampling = True
    with self.assertRaisesRegexp(
        ValueError, 'parameters .* do not match the expected sampler state'):
      gibbs_sampler.fit_with_gibbs_sampling(bad_model, observed_time_series)

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
                                            observed_time_series)

  def test_sampled_level_has_correct_marginals(self):
    seed = test_util.test_seed(sampler_type='stateless')
    residuals_seed, is_missing_seed, level_seed = samplers.split_seed(
        seed, 3, 'test_sampled_level_has_correct_marginals')

    num_timesteps = 10
    initial_state_prior = tfd.MultivariateNormalDiag(loc=[-30.],
                                                     scale_diag=[100.])
    observed_residuals = samplers.normal(
        [3, 1, num_timesteps], seed=residuals_seed)
    is_missing = samplers.uniform(
        [3, 1, num_timesteps], seed=is_missing_seed) > 0.8
    level_scale = 1.5 * tf.ones([3, 1])
    observation_noise_scale = 0.2 * tf.ones([3, 1])

    ssm = tfp.sts.LocalLevelStateSpaceModel(
        num_timesteps=num_timesteps,
        initial_state_prior=initial_state_prior,
        observation_noise_scale=observation_noise_scale,
        level_scale=level_scale)

    resample_level = gibbs_sampler._build_resample_level_fn(
        initial_state_prior, is_missing=is_missing)
    posterior_means, posterior_covs = ssm.posterior_marginals(
        observed_residuals[..., tf.newaxis], mask=is_missing)
    level_samples = resample_level(
        observed_residuals=observed_residuals,
        level_scale=level_scale,
        observation_noise_scale=observation_noise_scale,
        sample_shape=10000,
        seed=level_seed)

    posterior_means_, posterior_covs_, level_samples_ = self.evaluate((
        posterior_means, posterior_covs, level_samples))
    self.assertAllClose(np.mean(level_samples_, axis=0),
                        posterior_means_[..., 0], atol=0.1)
    self.assertAllClose(np.std(level_samples_, axis=0),
                        np.sqrt(posterior_covs_[..., 0, 0]), atol=0.1)

  def test_sampled_scale_follows_correct_distribution(self):
    strm = test_util.test_seed_stream()
    prior_concentration = 0.1
    prior_scale = 0.1

    num_timesteps = 100
    observed_samples = tf.random.normal([2, num_timesteps], seed=strm()) * 3.
    is_missing = tf.random.uniform([2, num_timesteps], seed=strm()) > 0.9

    # Check that posterior variance samples have the moments of the correct
    # InverseGamma distribution.
    posterior_scale_samples = parallel_for.pfor(
        lambda i: gibbs_sampler._resample_scale(  # pylint: disable=g-long-lambda
            prior_concentration=prior_concentration,
            prior_scale=prior_scale,
            observed_residuals=observed_samples,
            is_missing=is_missing,
            seed=strm()), 10000)

    concentration = prior_concentration + tf.reduce_sum(
        1 - tf.cast(is_missing, tf.float32), axis=-1)/2.
    scale = prior_scale + tf.reduce_sum(
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
    design_matrix = samplers.normal(
        batch_shape + [num_timesteps, num_features], seed=design_seed)
    true_weights = samplers.normal(
        batch_shape + [num_features, 1], seed=true_weights_seed) * 10.0
    targets = tf.matmul(design_matrix, true_weights)
    is_missing = tf.convert_to_tensor([False, False, False, True, True,
                                       False, False, True, False, False],
                                      dtype=tf.bool)
    prior_scale = tf.convert_to_tensor(5.)
    likelihood_scale = tf.convert_to_tensor(0.1)

    # Analytically compute the true posterior distribution on weights.
    valid_design_matrix = tf.boolean_mask(design_matrix, ~is_missing, axis=-2)
    valid_targets = tf.boolean_mask(targets, ~is_missing, axis=-2)
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
    sampled_weights = parallel_for.pfor(
        lambda i: gibbs_sampler._resample_weights(  # pylint: disable=g-long-lambda
            design_matrix=design_matrix,
            target_residuals=targets[..., 0],
            observation_noise_scale=likelihood_scale,
            weights_prior_scale=prior_scale,
            is_missing=is_missing,
            seed=sampled_weights_seed),
        10000)
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

if __name__ == '__main__':
  tf.test.main()
