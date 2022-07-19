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
"""Tests for anomaly detection."""

from absl.testing import parameterized

import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts import anomaly_detection

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_graph_and_eager_modes()
class AnomalyDetectionTests(test_util.TestCase):

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

  @parameterized.named_parameters(
      ('_no_forecast', [168 * 2], False),
      ('_no_forecast_gibbs_predictive', [168 * 2], True),
      ('_batch', [24, 3], False),
      ('_batch_gibbs_predictive', [24, 3], True))
  def test_returns_expected_shapes(self, shape, use_gibbs_predictive_dist):
    series = self._build_test_series(shape=shape, freq=pd.DateOffset(hours=1))
    predictions = anomaly_detection.detect_anomalies(
        series,
        anomaly_threshold=0.01,
        use_gibbs_predictive_dist=use_gibbs_predictive_dist,
        num_warmup_steps=5,
        num_samples=5,
        jit_compile=False,
        seed=test_util.test_seed(sampler_type='stateless'))

    results_length = shape[0]
    results_shape = shape[1:] + [results_length]
    self.assertAllEqual(predictions.observed_time_series.shape, results_shape)
    self.assertAllEqual(predictions.mean.shape, results_shape)
    self.assertAllEqual(predictions.lower_limit.shape, results_shape)
    self.assertAllEqual(predictions.upper_limit.shape, results_shape)
    self.assertAllEqual(predictions.tail_probabilities.shape, results_shape)
    self.assertAllEqual(predictions.is_anomaly.shape, results_shape)
    self.assertLen(predictions.times, results_length)

  def test_runs_with_xla(self):
    series = self._build_test_series(shape=[72, 3], freq=pd.DateOffset(hours=1))
    anomaly_detection.detect_anomalies(
        series, anomaly_threshold=0.01, use_gibbs_predictive_dist=False,
        seed=test_util.test_seed(sampler_type='stateless'),
        num_warmup_steps=5,
        num_samples=5,
        jit_compile=True)

  def test_plot_predictions_runs(self):
    series = self._build_test_series(shape=[28], freq=pd.DateOffset(days=1))
    predictions = anomaly_detection.detect_anomalies(
        series, anomaly_threshold=0.01, use_gibbs_predictive_dist=False,
        seed=test_util.test_seed(sampler_type='stateless'),
        num_warmup_steps=5,
        num_samples=5)
    predictions = tf.nest.map_structure(
        lambda x: self.evaluate(x) if tf.is_tensor(x) else x, predictions)
    anomaly_detection.plot_predictions(predictions)

    batch_predictions = tf.nest.map_structure(
        lambda x: np.stack([x, x], axis=0) if isinstance(x, np.ndarray) else x,
        predictions)
    with self.assertRaisesRegex(ValueError, 'must be one-dimensional'):
      anomaly_detection.plot_predictions(batch_predictions)

  def test_adapts_to_series_scale(self):
    # Create a batch of two series with very different means and stddevs.
    freq = pd.DateOffset(hours=1)
    df = self._build_test_series(shape=[72, 2], freq=freq)
    mean0, mean1 = -5e6, 1e3
    stddev0, stddev1 = 1e3, 1e-2
    df.series0 = df.series0 * stddev0 + mean0
    df.series1 = df.series1 * stddev1 + mean1
    # Ask the model to impute values before and after the observations.
    df.reindex(pd.date_range(start=df.index[0] - 5 * freq,
                             end=df.index[-1] + 5 * freq,
                             freq=freq))
    predictions = anomaly_detection.detect_anomalies(
        df,
        # Predictive bounds will contain 68% of mass, roughly +- 1 stddev.
        anomaly_threshold=0.32,
        seed=test_util.test_seed(sampler_type='stateless'),
        num_samples=100,
        num_warmup_steps=50,
        jit_compile=False)
    # The forecasts should have roughly the same mean and scale as the original
    # series.
    ones = tf.ones(predictions.mean[0, :].shape, dtype=predictions.mean.dtype)
    self.assertAllClose(predictions.mean[0, :],
                        ones * np.nanmean(df.series0),
                        atol=stddev0 * 4)
    self.assertAllClose(predictions.mean[1, :],
                        ones * np.nanmean(df.series1),
                        atol=stddev1 * 4)

    self.assertAllClose(predictions.lower_limit[0, :],
                        predictions.mean[0, :] - stddev0,
                        rtol=1e-3)
    self.assertAllClose(predictions.lower_limit[1, :],
                        predictions.mean[1, :] - stddev1,
                        rtol=1e-3)
    self.assertAllClose(predictions.upper_limit[0, :],
                        predictions.mean[0, :] + stddev0,
                        rtol=1e-3)
    self.assertAllClose(predictions.upper_limit[1, :],
                        predictions.mean[1, :] + stddev1,
                        rtol=1e-3)

  def test_constant_series(self):
    values = np.concatenate([np.nan * np.ones([5]),
                             np.ones([10]),
                             np.nan * np.ones([5])], axis=0)
    series = pd.Series(values,
                       index=pd.date_range('2020-01-01', periods=len(values),
                                           freq=pd.DateOffset(days=1)))
    predictions = anomaly_detection.detect_anomalies(
        series,
        seed=test_util.test_seed(sampler_type='stateless'),
        num_samples=50,
        num_warmup_steps=10,
        jit_compile=False)
    self.assertAllClose(predictions.mean,
                        tf.ones_like(predictions.mean),
                        atol=0.1)

  @parameterized.named_parameters(('', False),
                                  ('_gibbs_predictive', True))
  def test_predictions_align_with_series(self, use_gibbs_predictive_dist):
    np.random.seed(0)
    # Simulate data with very clear daily and hourly effects, so that an
    # off-by-one error will almost certainly lead to out-of-bounds predictions.
    daily_effects = [100., 0., 20., -50., -100., -20., 70.]
    hourly_effects = [
        20., 0., 10., -10., 0., -20., -10., -30., -15., -5., -10., 0.] * 2
    effects = [daily_effects[(t // 24) % 7] + hourly_effects[t % 24]
               for t in range(24 * 7 * 2)]
    series = pd.Series(effects + np.random.randn(len(effects)),
                       index=pd.date_range('2020-01-01',
                                           periods=len(effects),
                                           freq=pd.DateOffset(hours=1)))
    predictions = anomaly_detection.detect_anomalies(
        series,
        seed=test_util.test_seed(sampler_type='stateless'),
        num_samples=100,
        num_warmup_steps=50,
        use_gibbs_predictive_dist=use_gibbs_predictive_dist,
        jit_compile=False)
    # An off-by-one error in the predictive distribution would generate
    # anomalies at most steps.
    num_anomalies = tf.reduce_sum(tf.cast(predictions.is_anomaly, tf.int32))
    self.assertLessEqual(self.evaluate(num_anomalies), 5)


if __name__ == '__main__':
  test_util.main()
