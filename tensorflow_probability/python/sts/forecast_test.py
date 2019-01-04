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
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

tfl = tf.linalg
tfd = tfp.distributions


class _ForecastTest(object):

  def _build_model(self, observed_time_series):
    seasonal = tfp.sts.Seasonal(
        num_seasons=4,
        observed_time_series=observed_time_series,
        initial_effect_prior=tfd.Normal(loc=self._build_tensor(0.),
                                        scale=self._build_tensor(1.)),
        name='seasonal')
    return tfp.sts.Sum(components=[seasonal],
                       observed_time_series=observed_time_series)

  def test_one_step_predictive_correctness(self):
    observed_time_series_ = np.array([1., -1., -3., 4., 0.5, 2., 1., 3.])
    observed_time_series = self._build_tensor(observed_time_series_)
    model = self._build_model(observed_time_series)

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
    model = self._build_model(observed_time_series)
    prior_samples = [param.prior.sample(num_param_samples)
                     for param in model.parameters]

    onestep_dist = tfp.sts.one_step_predictive(model, observed_time_series,
                                               parameter_samples=prior_samples)

    self.evaluate(tf.global_variables_initializer())
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
    model = self._build_model(observed_time_series)

    drift_scale = 0.1
    observation_noise_scale = 0.01
    params = {'seasonal/_drift_scale': self._build_tensor([drift_scale]),
              'observation_noise_scale': self._build_tensor(
                  [observation_noise_scale])}

    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                     parameter_samples=params,
                                     num_steps_forecast=8)
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

    self.evaluate(tf.global_variables_initializer())
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
    model = self._build_model(observed_time_series)
    prior_samples = [param.prior.sample(num_param_samples)
                     for param in model.parameters]

    forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                     parameter_samples=prior_samples,
                                     num_steps_forecast=num_steps_forecast)

    self.evaluate(tf.global_variables_initializer())
    if self.use_static_shape:
      self.assertAllEqual(forecast_dist.batch_shape.as_list(), batch_shape)
    else:
      self.assertAllEqual(self.evaluate(forecast_dist.batch_shape_tensor()),
                          batch_shape)
    forecast_mean = forecast_dist.mean()[..., 0]
    forecast_mean_ = self.evaluate(forecast_mean)
    self.assertAllEqual(forecast_mean_.shape,
                        batch_shape + [num_steps_forecast])

  def _build_tensor(self, ndarray):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype`, and shape specified
      statically only if `self.use_static_shape` is `True`.
    """

    ndarray = np.asarray(ndarray).astype(self.dtype)
    return tf.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.run_all_in_graph_and_eager_modes
class ForecastTestStatic32(tf.test.TestCase, _ForecastTest):
  dtype = np.float32
  use_static_shape = True


# Run in graph mode only to reduce test weight.
class ForecastTestDynamic32(tf.test.TestCase, _ForecastTest):
  dtype = np.float32
  use_static_shape = False


# Run in graph mode only to reduce test weight.
class ForecastTestStatic64(tf.test.TestCase, _ForecastTest):
  dtype = np.float64
  use_static_shape = True

if __name__ == '__main__':
  test.main()
