# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for automatically building StructuralTimeSeries models."""

# Dependency imports

import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import default_model

tfb = tfp.bijectors
tfd = tfp.distributions


class DefaultModelTests(test_util.TestCase):

  def _build_test_series(self, shape, freq, start='2020-01-01 00:00:00'):
    values = self.evaluate(tf.random.stateless_normal(
        shape, seed=test_util.test_seed(sampler_type='stateless')))
    index = pd.date_range('2020-01-01 00:00:00',
                          periods=shape[0],
                          freq=freq)
    if len(shape) > 1:
      num_columns = shape[1]
      return pd.DataFrame(values,
                          columns=['series{}'.format(i)
                                   for i in range(num_columns)],
                          index=index)
    else:
      return pd.Series(values, index=index)

  def test_has_expected_seasonality(self):
    model = default_model.build_default_model(
        self._build_test_series(shape=[168 * 2], freq=pd.DateOffset(hours=1)))

    self.assertIsInstance(model, tfp.sts.Sum)
    self.assertLen(model.components, 3)
    self.assertIsInstance(model.components[0], tfp.sts.LocalLinearTrend)
    self.assertIsInstance(model.components[1], tfp.sts.Seasonal)
    self.assertContainsSubsequence(model.components[1].name, 'HOUR_OF_DAY')
    self.assertIsInstance(model.components[2], tfp.sts.Seasonal)
    self.assertContainsSubsequence(model.components[2].name, 'DAY_OF_WEEK')

  def test_explicit_base_component(self):
    series = self._build_test_series(shape=[48], freq=pd.DateOffset(hours=1))
    model = default_model.build_default_model(
        series,
        base_component=tfp.sts.SemiLocalLinearTrend(
            level_scale_prior=tfd.Exponential(5.),
            slope_scale_prior=tfd.Exponential(0.1),
            slope_mean_prior=tfd.Normal(0., 100.),
            constrain_ar_coef_positive=True,
            constrain_ar_coef_stationary=True,
            observed_time_series=series))
    self.assertLen(model.components, 2)

    param_by_name = lambda n: [p for p in model.parameters if n in p.name][0]
    self.assertAllClose(param_by_name('level_scale').prior.rate, 5.)
    self.assertAllClose(param_by_name('slope_mean').prior.scale, 100.)
    self.assertAllClose(param_by_name('slope_scale').prior.rate, 0.1)

  def test_creates_batch_model_from_multiple_series(self):
    model = default_model.build_default_model(
        self._build_test_series(shape=[48, 3], freq=pd.DateOffset(hours=1)))
    self.assertAllEqual(model.batch_shape, [3])

  @test_util.jax_disable_variable_test
  @test_util.numpy_disable_variable_test
  def test_docstring_fitting_example(self):
    # Construct a series of eleven data points, covering a period of two weeks
    # with three missing days.
    series = pd.Series(
        [100., 27., 92., 66., 51., 126., 113., 95., 48., 20., 59.,],
        index=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04',
                              '2020-01-05', '2020-01-06', '2020-01-07',
                              '2020-01-10', '2020-01-11', '2020-01-12',
                              '2020-01-13', '2020-01-14']))
    series = tfp.sts.regularize_series(series)
    self.assertLen(series, 14)

    # Default model captures day-of-week effects with a LocalLinearTrend
    # baseline.
    model = default_model.build_default_model(series)
    self.assertLen(model.components, 2)

    # Fit the model using variational inference.
    surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(model)
    _ = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_log_prob(series),
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(0.1),
        num_steps=1000,
        convergence_criterion=(tfp.optimizer.convergence_criteria.
                               SuccessiveGradientsAreUncorrelated(
                                   window_size=15, min_num_steps=50)),
        jit_compile=True)

    # Forecast the next week.
    parameter_samples = surrogate_posterior.sample(50)
    forecast_dist = tfp.sts.forecast(model,
                                     observed_time_series=series,
                                     parameter_samples=parameter_samples,
                                     num_steps_forecast=7)
    # Strip trailing unit dimension from LinearGaussianStateSpaceModel events.
    self.evaluate(
        [v.initializer for v in surrogate_posterior.trainable_variables])
    forecast_mean, forecast_stddev = self.evaluate(
        (forecast_dist.mean()[..., 0], forecast_dist.stddev()[..., 0]))

    pd.DataFrame(
        {'mean': forecast_mean,
         'lower_bound': forecast_mean - 2. * forecast_stddev,
         'upper_bound': forecast_mean + 2. * forecast_stddev},
        index=pd.date_range(start=series.index[-1] + series.index.freq,
                            periods=7,
                            freq=series.index.freq))


if __name__ == '__main__':
  test_util.main()
