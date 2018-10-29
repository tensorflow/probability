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
"""Tests for STS fitting methods."""

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

tfl = tf.linalg
tfd = tfp.distributions


class VariationalInferenceTests(tf.test.TestCase):

  def _build_model(self, observed_time_series):
    day_of_week = tfp.sts.Seasonal(
        num_seasons=7,
        observed_time_series=observed_time_series,
        name='day_of_week')
    local_linear_trend = tfp.sts.LocalLinearTrend(
        observed_time_series=observed_time_series,
        name='local_linear_trend')
    return tfp.sts.Sum(components=[day_of_week, local_linear_trend],
                       observed_time_series=observed_time_series)

  def test_multiple_inits_example(self):
    batch_shape = [2, 3]
    num_timesteps = 5
    num_inits = 10
    observed_time_series = np.random.randn(
        *(batch_shape + [num_timesteps])).astype(np.float32)

    model = self._build_model(observed_time_series)

    (variational_loss,
     _) = tfp.sts.build_factored_variational_loss(
         model=model,
         observed_time_series=observed_time_series,
         init_batch_shape=num_inits)

    train_op = tf.train.AdamOptimizer(0.1).minimize(variational_loss)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      for _ in range(5):  # don't actually run to completion
        _, _ = sess.run((train_op, variational_loss))

      # Draw multiple samples to reduce Monte Carlo error in the optimized
      # variational bounds.
      avg_loss = np.mean(
          [sess.run(variational_loss) for _ in range(25)], axis=0)
      self.assertAllEqual(avg_loss.shape, [num_inits] + batch_shape)


@test_util.run_all_in_graph_and_eager_modes
class HMCTests(tf.test.TestCase):

  def _build_model(self, observed_time_series):
    day_of_week = tfp.sts.Seasonal(
        num_seasons=7,
        observed_time_series=observed_time_series,
        name='day_of_week')
    local_linear_trend = tfp.sts.LocalLinearTrend(
        observed_time_series=observed_time_series,
        name='local_linear_trend')
    return tfp.sts.Sum(components=[day_of_week, local_linear_trend],
                       observed_time_series=observed_time_series)

  def test_basic_hmc_example(self):
    batch_shape = [2, 3]
    num_timesteps = 5
    observed_time_series = np.random.randn(
        *(batch_shape + [num_timesteps])).astype(np.float32)
    model = self._build_model(observed_time_series)
    samples, kernel_results = tfp.sts.fit_with_hmc(
        model,
        observed_time_series,
        num_results=10,
        num_warmup_steps=5,
        num_variational_steps=5)

    self.evaluate(tf.global_variables_initializer())
    samples_, kernel_results_ = self.evaluate((samples, kernel_results))

    acceptance_rate = np.mean(
        kernel_results_.inner_results.is_accepted, axis=0)

    posterior_means = {
        param.name: np.mean(param_draws, axis=0)
        for (param, param_draws) in zip(model.parameters, samples_)}

    # Perfunctory checks to ensure the code executed and we got results
    # of the expected shape.
    self.assertAllEqual(acceptance_rate.shape, batch_shape)
    for parameter in model.parameters:
      self.assertAllEqual(posterior_means[parameter.name].shape,
                          parameter.prior.batch_shape.as_list() +
                          parameter.prior.event_shape.as_list())

  def test_multiple_chains_example(self):
    batch_shape = [2, 3]
    num_timesteps = 5
    num_results = 10
    num_chains = 4
    observed_time_series = np.random.randn(
        *(batch_shape + [num_timesteps])).astype(np.float32)
    model = self._build_model(observed_time_series)
    samples, kernel_results = tfp.sts.fit_with_hmc(
        model,
        observed_time_series,
        num_results=num_results,
        chain_batch_shape=num_chains,
        num_warmup_steps=5,
        num_variational_steps=5)

    self.evaluate(tf.global_variables_initializer())
    samples_, kernel_results_ = self.evaluate((samples, kernel_results))

    acceptance_rate = np.mean(
        kernel_results_.inner_results.is_accepted, axis=0)

    # Combining the samples from multiple chains into a single dimension allows
    # us to easily pass sampled parameters to downstream forecasting methods.
    combined_samples_ = [np.reshape(param_draws,
                                    [-1] + list(param_draws.shape[2:]))
                         for param_draws in samples_]

    self.assertAllEqual(acceptance_rate.shape, [num_chains] + batch_shape)
    for parameter, samples_ in zip(model.parameters, combined_samples_):
      self.assertAllEqual(samples_.shape,
                          [num_results * num_chains] +
                          parameter.prior.batch_shape.as_list() +
                          parameter.prior.event_shape.as_list())

  def test_chain_batch_shape(self):
    batch_shape = [2, 3]
    num_results = 1
    num_timesteps = 5
    observed_time_series = np.random.randn(
        *(batch_shape + [num_timesteps])).astype(np.float32)
    model = self._build_model(observed_time_series)

    for i, (shape_in, expected_batch_shape_out) in enumerate(
        [([], []), (3, [3]), ([3], [3]), ([5, 2], [5, 2])]):
      # Use variable scope to prevents name conflicts from multiple fits
      # in the same graph.
      with tf.variable_scope('case_{}'.format(i), reuse=False):
        samples, _ = tfp.sts.fit_with_hmc(
            model,
            observed_time_series,
            num_results=num_results,
            chain_batch_shape=shape_in,
            num_warmup_steps=1,
            num_variational_steps=1)
      for parameter, parameter_samples in zip(model.parameters, samples):
        self.assertAllEqual(parameter_samples.shape.as_list(),
                            [num_results] +
                            expected_batch_shape_out +
                            parameter.prior.batch_shape.as_list() +
                            parameter.prior.event_shape.as_list())

if __name__ == '__main__':
  test.main()
