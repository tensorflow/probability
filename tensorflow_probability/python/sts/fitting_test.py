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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


class _VariationalInferenceTests(object):

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

  def test_basic_variational_fitting(self):
    batch_shape = [2, 3]
    num_timesteps = 5
    num_inits = 10
    observed_time_series = self._build_tensor(np.random.randn(
        *(batch_shape + [num_timesteps])))

    model = self._build_model(observed_time_series)

    variational_posterior = tfp.sts.build_factored_surrogate_posterior(
        model, batch_shape=num_inits)
    loss_curve = tfp.vi.fit_surrogate_posterior(
        model.joint_log_prob(observed_time_series),
        surrogate_posterior=variational_posterior,
        sample_size=3,
        num_steps=10,
        optimizer=tf1.train.AdamOptimizer(learning_rate=0.1))
    self.evaluate(tf1.global_variables_initializer())
    with tf.control_dependencies([loss_curve]):
      posterior_samples = variational_posterior.sample(10)
    loss_curve_, _ = self.evaluate((loss_curve, posterior_samples))
    self.assertLess(np.mean(loss_curve_[-1]), np.mean(loss_curve_[0]))

  def test_custom_eager_optimization_loop(self):
    if not tf.executing_eagerly():
      return

    batch_shape = [2, 3]
    num_timesteps = 5
    observed_time_series = self._build_tensor(np.random.randn(
        *(batch_shape + [num_timesteps])))

    model = self._build_model(observed_time_series)

    surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(
        model=model)
    self.assertLen(surrogate_posterior.trainable_variables,
                   len(model.parameters) * 2)  # Loc and scale for each param.

    @tf.function(autograph=False)  # Ensure the loss is computed efficiently
    def loss_fn(sample_size=3):
      return tfp.vi.monte_carlo_variational_loss(
          model.joint_log_prob(observed_time_series),
          surrogate_posterior,
          sample_size=sample_size)

    initial_loss = self.evaluate(loss_fn(sample_size=10))

    # TODO(b/137299119) Replace with TF2 optimizer.
    optimizer = tf1.train.AdamOptimizer(learning_rate=0.1)
    for _ in range(10):
      with tf.GradientTape() as tape:
        loss = loss_fn()
      grads = tape.gradient(loss, surrogate_posterior.trainable_variables)
      self.evaluate(optimizer.apply_gradients(
          zip(grads, surrogate_posterior.trainable_variables)))

    final_loss = self.evaluate(loss_fn(sample_size=10))
    self.assertAllEqual(final_loss.shape, batch_shape)
    self.assertLess(np.mean(final_loss), np.mean(initial_loss))

  def test_init_is_valid_for_large_observations(self):
    num_timesteps = 20
    observed_time_series = self._build_tensor(
        -1e8 + 1e6 * np.random.randn(num_timesteps))
    model = self._build_model(observed_time_series)
    surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(
        model=model)
    variational_loss = tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=model.joint_log_prob(observed_time_series),
        surrogate_posterior=surrogate_posterior)
    self.evaluate(tf1.global_variables_initializer())
    loss_ = self.evaluate(variational_loss)
    self.assertTrue(np.isfinite(loss_))

    # When this test was written, the variational loss with default
    # initialization and seed was 431.5 nats. Larger finite initial losses are
    # not 'incorrect' as such, but if your change makes the next line fail,
    # you have probably done something questionable.
    self.assertLessEqual(loss_, 10000)

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
class VariationalInferenceTestsStatic64(test_util.TestCase,
                                        _VariationalInferenceTests):
  dtype = np.float64
  use_static_shape = True


# This test runs in graph mode only to reduce test weight.
class VariationalInferenceTestsDynamic32(test_util.TestCase,
                                         _VariationalInferenceTests):
  dtype = np.float32
  use_static_shape = False


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

    self.evaluate(tf1.global_variables_initializer())
    samples_, kernel_results_ = self.evaluate((samples, kernel_results))

    acceptance_rate = np.mean(
        kernel_results_.inner_results.inner_results.is_accepted, axis=0)

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

    # Use an observation mask to additionally test that masks are
    # threaded through the HMC (and VI) APIs.
    observed_time_series_ = np.random.randn(
        *(batch_shape + [num_timesteps]))
    observed_time_series = tfp.sts.MaskedTimeSeries(
        self._build_tensor(observed_time_series_),
        is_missing=self._build_tensor([False, True, False, False, True],
                                      dtype=np.bool))

    model = self._build_model(observed_time_series)
    samples, kernel_results = tfp.sts.fit_with_hmc(
        model,
        observed_time_series,
        num_results=num_results,
        chain_batch_shape=num_chains,
        num_warmup_steps=2,
        num_variational_steps=2)

    self.evaluate(tf1.global_variables_initializer())
    samples_, kernel_results_ = self.evaluate((samples, kernel_results))

    acceptance_rate = np.mean(
        kernel_results_.inner_results.inner_results.is_accepted, axis=0)

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
      return list(self.evaluate(tf.shape(tensor)))

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
class HMCTestsStatic32(test_util.TestCase, _HMCTests):
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

    self.evaluate(tf1.global_variables_initializer())
    for parameter, parameter_samples in zip(model.parameters, samples):
      self.assertAllEqual(self._shape_as_list(parameter_samples),
                          [num_results] +
                          expected_batch_shape_out +
                          self._batch_shape_as_list(parameter.prior) +
                          self._event_shape_as_list(parameter.prior))


# This test runs in graph mode only to reduce test weight.
class HMCTestsDynamic32(test_util.TestCase, _HMCTests):
  dtype = np.float32
  use_static_shape = False


# This test runs in graph mode only to reduce test weight.
class HMCTestsStatic64(test_util.TestCase, _HMCTests):
  dtype = np.float64
  use_static_shape = True

if __name__ == '__main__':
  tf.test.main()
