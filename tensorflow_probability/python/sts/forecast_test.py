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
"""Tests for STS forecasting methods."""

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


class _ForecastTest(object):

  def _build_model(self, observed_time_series,
                   prior_batch_shape=(),
                   initial_effect_prior_scale=1.,
                   constant_offset=None):
    seasonal = tfp.sts.Seasonal(
        num_seasons=4,
        observed_time_series=observed_time_series,
        initial_effect_prior=tfd.Normal(
            loc=self._build_tensor(np.zeros(prior_batch_shape)),
            scale=self._build_tensor(initial_effect_prior_scale)),
        constrain_mean_effect_to_zero=False,  # Simplifies analysis.
        name='seasonal')
    return tfp.sts.Sum(components=[seasonal],
                       constant_offset=constant_offset,
                       observed_time_series=observed_time_series)

  def test_one_step_predictive_correctness(self):
    observed_time_series_ = np.array([1., -1., -3., 4., 0.5, 2., 1., 3.])
    observed_time_series = self._build_tensor(observed_time_series_)
    model = self._build_model(
        observed_time_series,
        constant_offset=0.)  # Simplifies analytic calculations.

    drift_scale = 0.1
    observation_noise_scale = 0.01
    params = {'seasonal/_drift_scale': self._build_tensor([drift_scale]),
              'observation_noise_scale': self._build_tensor(
                  [observation_noise_scale])}

    onestep_dist = tfp.sts.one_step_predictive(model, observed_time_series,
                                               parameter_samples=params)
    onestep_mean_, onestep_scale_ = self.evaluate(
        (onestep_dist.mean(), onestep_dist.stddev()))

    # Since Seasonal is just a set of interleaved random walks, it's
    # straightforward to compute the forecast analytically.
    # For the first (num_seasons - 1) steps, the one-step-ahead
    # forecast mean/scale are just the prior Normal(0., 1.). After that,
    # the predicted `n`th step depends on the posterior from the
    # `n - num_seasons` step.
    num_seasons = 4
    effect_posterior_precision = 1. + 1/observation_noise_scale**2
    effect_posterior_means = (
        (observed_time_series_[:num_seasons] / observation_noise_scale**2)
        / effect_posterior_precision)
    effect_posterior_variance = 1/effect_posterior_precision
    observation_predictive_variance = (
        effect_posterior_variance + drift_scale**2 + observation_noise_scale**2)

    expected_onestep_mean = np.concatenate([np.zeros(4),
                                            effect_posterior_means])
    expected_onestep_scale = np.concatenate([
        [np.sqrt(1.**2 + observation_noise_scale**2)] * 4,
        [np.sqrt(observation_predictive_variance)] * 4])
    self.assertAllClose(onestep_mean_, expected_onestep_mean)
    self.assertAllClose(onestep_scale_, expected_onestep_scale)

  def test_one_step_predictive_with_batch_shape(self):
    num_param_samples = 5
    num_timesteps = 4
    batch_shape = [3, 2]
    observed_time_series = self._build_tensor(np.random.randn(
        *(batch_shape + [num_timesteps])))
    model = self._build_model(observed_time_series,
                              prior_batch_shape=batch_shape[1:])
    prior_samples = [param.prior.sample(num_param_samples)
                     for param in model.parameters]

    onestep_dist = tfp.sts.one_step_predictive(model, observed_time_series,
                                               parameter_samples=prior_samples)

    self.evaluate(tf1.global_variables_initializer())
    if self.use_static_shape:
      self.assertAllEqual(onestep_dist.batch_shape.as_list(), batch_shape)
    else:
      self.assertAllEqual(self.evaluate(onestep_dist.batch_shape_tensor()),
                          batch_shape)
    onestep_mean_ = self.evaluate(onestep_dist.mean())
    self.assertAllEqual(onestep_mean_.shape, batch_shape + [num_timesteps])

  def test_forecast_correctness(self):
    observed_time_series_ = np.array([1., -1., -3., 4.])
    observed_time_series = self._build_tensor(observed_time_series_)
    model = self._build_model(
        observed_time_series,
        constant_offset=0.)  # Simplifies analytic calculations.

    drift_scale = 0.1
    observation_noise_scale = 0.01
    params = {'seasonal/_drift_scale': self._build_tensor([drift_scale]),
              'observation_noise_scale': self._build_tensor(
                  [observation_noise_scale])}

    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                     parameter_samples=params,
                                     num_steps_forecast=8,
                                     include_observation_noise=True)
    forecast_mean = forecast_dist.mean()[..., 0]
    forecast_scale = forecast_dist.stddev()[..., 0]
    forecast_mean_, forecast_scale_ = self.evaluate(
        (forecast_mean, forecast_scale))

    # Since Seasonal is just a set of interleaved random walks, it's
    # straightforward to compute the forecast analytically.
    effect_posterior_precision = 1. + 1/observation_noise_scale**2
    effect_posterior_means = (
        (observed_time_series_ / observation_noise_scale**2)
        / effect_posterior_precision)
    effect_posterior_variance = 1/effect_posterior_precision
    observation_predictive_variance = (
        effect_posterior_variance + drift_scale**2 + observation_noise_scale**2)

    expected_forecast_mean = np.concatenate([effect_posterior_means,
                                             effect_posterior_means])
    expected_forecast_scale = np.concatenate([
        [np.sqrt(observation_predictive_variance)] * 4,
        [np.sqrt(observation_predictive_variance + drift_scale**2)] * 4])
    self.assertAllClose(forecast_mean_, expected_forecast_mean)
    self.assertAllClose(forecast_scale_, expected_forecast_scale)

    # Also test forecasting the noise-free function.
    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                     parameter_samples=params,
                                     num_steps_forecast=8,
                                     include_observation_noise=False)
    forecast_mean = forecast_dist.mean()[..., 0]
    forecast_scale = forecast_dist.stddev()[..., 0]
    forecast_mean_, forecast_scale_ = self.evaluate(
        (forecast_mean, forecast_scale))

    noiseless_predictive_variance = (effect_posterior_variance + drift_scale**2)
    expected_forecast_scale = np.concatenate([
        [np.sqrt(noiseless_predictive_variance)] * 4,
        [np.sqrt(noiseless_predictive_variance + drift_scale**2)] * 4])
    self.assertAllClose(forecast_mean_, expected_forecast_mean)
    self.assertAllClose(forecast_scale_, expected_forecast_scale)

  def test_forecast_from_hmc(self):
    # test that we can directly plug in the output of an HMC chain as
    # the input to `forecast`, as done in the example, with no `sess.run` call.
    num_results = 5
    num_timesteps = 4
    num_steps_forecast = 3
    batch_shape = [1, 2]
    observed_time_series = self._build_tensor(np.random.randn(
        *(batch_shape + [num_timesteps])))
    model = self._build_model(observed_time_series)
    samples, _ = tfp.sts.fit_with_hmc(
        model, observed_time_series,
        num_results=num_results,
        num_warmup_steps=2,
        num_variational_steps=2)

    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                     parameter_samples=samples,
                                     num_steps_forecast=num_steps_forecast)

    forecast_mean = forecast_dist.mean()[..., 0]
    forecast_scale = forecast_dist.stddev()[..., 0]

    sample_shape = [10]
    forecast_samples = forecast_dist.sample(sample_shape)[..., 0]

    self.evaluate(tf1.global_variables_initializer())
    forecast_mean_, forecast_scale_, forecast_samples_ = self.evaluate(
        (forecast_mean, forecast_scale, forecast_samples))
    self.assertAllEqual(forecast_mean_.shape,
                        batch_shape + [num_steps_forecast])
    self.assertAllEqual(forecast_scale_.shape,
                        batch_shape + [num_steps_forecast])
    self.assertAllEqual(forecast_samples_.shape,
                        sample_shape + batch_shape + [num_steps_forecast])

  def test_forecast_with_batch_shape(self):
    num_param_samples = 5
    num_timesteps = 4
    num_steps_forecast = 6
    batch_shape = [3, 2]
    observed_time_series = self._build_tensor(np.random.randn(
        *(batch_shape + [num_timesteps])))

    # By not passing a constant offset, we test that the default behavior
    # (setting constant offset to the observed mean) works when the observed
    # time series has batch shape.
    model = self._build_model(observed_time_series,
                              prior_batch_shape=batch_shape[1:])
    prior_samples = [param.prior.sample(num_param_samples)
                     for param in model.parameters]

    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                     parameter_samples=prior_samples,
                                     num_steps_forecast=num_steps_forecast)

    self.evaluate(tf1.global_variables_initializer())
    if self.use_static_shape:
      self.assertAllEqual(forecast_dist.batch_shape.as_list(), batch_shape)
    else:
      self.assertAllEqual(self.evaluate(forecast_dist.batch_shape_tensor()),
                          batch_shape)
    forecast_mean = forecast_dist.mean()[..., 0]
    forecast_mean_ = self.evaluate(forecast_mean)
    self.assertAllEqual(forecast_mean_.shape,
                        batch_shape + [num_steps_forecast])

  def test_methods_handle_masked_inputs(self):
    num_param_samples = 5
    num_timesteps = 4
    num_steps_forecast = 2

    # Build a time series with `NaN`s that will propagate if not properly
    # masked.
    observed_time_series_ = np.random.randn(num_timesteps)
    is_missing_ = np.random.randn(num_timesteps) > 0
    observed_time_series_[is_missing_] = np.nan
    observed_time_series = tfp.sts.MaskedTimeSeries(
        self._build_tensor(observed_time_series_),
        is_missing=self._build_tensor(is_missing_, dtype=np.bool))

    model = self._build_model(observed_time_series)
    prior_samples = [param.prior.sample(num_param_samples)
                     for param in model.parameters]

    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                     parameter_samples=prior_samples,
                                     num_steps_forecast=num_steps_forecast)

    forecast_mean_, forecast_stddev_ = self.evaluate((
        forecast_dist.mean(),
        forecast_dist.stddev()))
    self.assertTrue(np.all(np.isfinite(forecast_mean_)))
    self.assertTrue(np.all(np.isfinite(forecast_stddev_)))

    onestep_dist = tfp.sts.one_step_predictive(
        model, observed_time_series,
        parameter_samples=prior_samples)
    onestep_mean_, onestep_stddev_ = self.evaluate((
        onestep_dist.mean(),
        onestep_dist.stddev()))
    self.assertTrue(np.all(np.isfinite(onestep_mean_)))
    self.assertTrue(np.all(np.isfinite(onestep_stddev_)))

  def test_impute_missing(self):
    time_series_with_nans = self._build_tensor(
        [-1., 1., np.nan, 2.4, np.nan, np.nan, 2.])
    observed_time_series = tfp.sts.MaskedTimeSeries(
        time_series=time_series_with_nans,
        is_missing=tf.math.is_nan(time_series_with_nans))

    # Build model with a near-uniform prior on the initial effect. In principle
    # we should use a large scale like 1e8 here, but we use 1e2 because
    # increasing the scale triggers numerical issues with Kalman smoothing
    # described in b/138414045.
    model = self._build_model(observed_time_series,
                              initial_effect_prior_scale=1e2)

    # Impute values using manually-set parameters, which will allow us to
    # compute the expected results analytically.
    drift_scale = 1.0
    noise_scale = 0.1
    parameter_samples = {'observation_noise_scale': [noise_scale],
                         'seasonal/_drift_scale': [drift_scale]}
    imputed_series_dist = tfp.sts.impute_missing_values(
        model, observed_time_series, parameter_samples)
    imputed_noisy_series_dist = tfp.sts.impute_missing_values(
        model, observed_time_series, parameter_samples,
        include_observation_noise=True)

    # Compare imputed mean to expected mean.
    mean_, stddev_ = self.evaluate([imputed_series_dist.mean(),
                                    imputed_series_dist.stddev()])
    noisy_mean_, noisy_stddev_ = self.evaluate([
        imputed_noisy_series_dist.mean(),
        imputed_noisy_series_dist.stddev()])
    self.assertAllClose(mean_, [-1., 1., 2., 2.4, -1., 1., 2.], atol=1e-2)
    self.assertAllClose(mean_, noisy_mean_, atol=1e-2)

    # Compare imputed stddevs to expected stddevs.
    drift_plus_noise_scale = np.sqrt(noise_scale**2 + drift_scale**2)
    expected_stddev = np.array([noise_scale,
                                noise_scale,
                                drift_plus_noise_scale,
                                noise_scale,
                                drift_plus_noise_scale,
                                drift_plus_noise_scale,
                                noise_scale])
    self.assertAllClose(stddev_, expected_stddev, atol=1e-2)
    self.assertAllClose(noisy_stddev_,
                        np.sqrt(stddev_**2 + noise_scale**2), atol=1e-2)

  def _build_tensor(self, ndarray, dtype=None):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.
      dtype: optional `dtype`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype` (if not specified), and
      shape specified statically only if `self.use_static_shape` is `True`.
    """

    ndarray = np.asarray(ndarray).astype(self.dtype if dtype is None else dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.test_all_tf_execution_regimes
class ForecastTestStatic32(test_util.TestCase, _ForecastTest):
  dtype = np.float32
  use_static_shape = True


# Run in graph mode only to reduce test weight.
class ForecastTestDynamic32(test_util.TestCase, _ForecastTest):
  dtype = np.float32
  use_static_shape = False


# Run in graph mode only to reduce test weight.
class ForecastTestStatic64(test_util.TestCase, _ForecastTest):
  dtype = np.float64
  use_static_shape = True

if __name__ == '__main__':
  tf.test.main()
