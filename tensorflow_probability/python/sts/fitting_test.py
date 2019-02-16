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
from absl.testing import parameterized

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
tfl = tf.linalg


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

    def build_variational_loss():
      (variational_loss, _) = tfp.sts.build_factored_variational_loss(
          model=model,
          observed_time_series=observed_time_series,
          init_batch_shape=num_inits)
      return variational_loss

    # We provide graph- and eager-mode optimization for TF 2.0 compatibility.
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
    if tf.executing_eagerly():
      for _ in range(5):  # don't actually run to completion
        optimizer.minimize(build_variational_loss)
      # Draw multiple samples to reduce Monte Carlo error in the optimized
      # variational bounds.
      avg_loss = np.mean(
          [self.evaluate(build_variational_loss()) for _ in range(25)], axis=0)
    else:
      variational_loss = build_variational_loss()
      train_op = optimizer.minimize(variational_loss)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      for _ in range(5):  # don't actually run to completion
        _ = self.evaluate(train_op)
      # Draw multiple samples to reduce Monte Carlo error in the optimized
      # variational bounds.
      avg_loss = np.mean(
          [self.evaluate(variational_loss) for _ in range(25)], axis=0)
    self.assertAllEqual(avg_loss.shape, [num_inits] + batch_shape)


class _HMCTests(object):

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
    observed_time_series = self._build_tensor(np.random.randn(
        *(batch_shape + [num_timesteps])))
    model = self._build_model(observed_time_series)
    samples, kernel_results = tfp.sts.fit_with_hmc(
        model,
        observed_time_series,
        num_results=4,
        num_warmup_steps=2,
        num_variational_steps=2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
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
                          self._batch_shape_as_list(parameter.prior) +
                          self._event_shape_as_list(parameter.prior))

  def test_multiple_chains_example(self):
    batch_shape = [2, 3]
    num_timesteps = 5
    num_results = 6
    num_chains = 4
    observed_time_series = self._build_tensor(np.random.randn(
        *(batch_shape + [num_timesteps])))
    model = self._build_model(observed_time_series)
    samples, kernel_results = tfp.sts.fit_with_hmc(
        model,
        observed_time_series,
        num_results=num_results,
        chain_batch_shape=num_chains,
        num_warmup_steps=2,
        num_variational_steps=2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
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
                          self._batch_shape_as_list(parameter.prior) +
                          self._event_shape_as_list(parameter.prior))

  def _shape_as_list(self, tensor):
    if self.use_static_shape:
      return tensor.shape.as_list()
    else:
      return list(self.evaluate(tf.shape(input=tensor)))

  def _batch_shape_as_list(self, distribution):
    if self.use_static_shape:
      return distribution.batch_shape.as_list()
    else:
      return list(self.evaluate(distribution.batch_shape_tensor()))

  def _event_shape_as_list(self, distribution):
    if self.use_static_shape:
      return distribution.event_shape.as_list()
    else:
      return list(self.evaluate(distribution.event_shape_tensor()))

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
    return tf.compat.v1.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)


@test_util.run_all_in_graph_and_eager_modes
class HMCTestsStatic32(tf.test.TestCase, parameterized.TestCase, _HMCTests):
  dtype = np.float32
  use_static_shape = True

  # Parameterized tests appear to require that their direct containing class
  # inherits from `parameterized.TestCase`, so we have to put this test here
  # rather than the base class. As a bonus, running this test only in the
  # Static32 case reduces overall test weight.
  @parameterized.parameters(([], []),
                            (3, [3]),
                            ([3], [3]),
                            ([5, 2], [5, 2]))
  def test_chain_batch_shape(self, shape_in, expected_batch_shape_out):
    batch_shape = [2, 3]
    num_results = 1
    num_timesteps = 5
    observed_time_series = self._build_tensor(np.random.randn(
        *(batch_shape + [num_timesteps])))
    model = self._build_model(observed_time_series)
    samples, _ = tfp.sts.fit_with_hmc(
        model,
        observed_time_series,
        num_results=num_results,
        chain_batch_shape=shape_in,
        num_warmup_steps=1,
        num_variational_steps=1)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    for parameter, parameter_samples in zip(model.parameters, samples):
      self.assertAllEqual(self._shape_as_list(parameter_samples),
                          [num_results] +
                          expected_batch_shape_out +
                          self._batch_shape_as_list(parameter.prior) +
                          self._event_shape_as_list(parameter.prior))


# This test runs in graph mode only to reduce test weight.
class HMCTestsDynamic32(tf.test.TestCase, _HMCTests):
  dtype = np.float32
  use_static_shape = False


# This test runs in graph mode only to reduce test weight.
class HMCTestsStatic64(tf.test.TestCase, _HMCTests):
  dtype = np.float64
  use_static_shape = True

if __name__ == '__main__':
  tf.test.main()
