# Lint as: python3
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
"""Tests for vectorized_stochastic_volatility."""

import functools

# Dependency imports

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from inference_gym.internal import test_util
from inference_gym.targets import vectorized_stochastic_volatility
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions

BACKEND = None  # Rewritten by backends/rewrite.py.


def _test_dataset():
  return tf.convert_to_tensor([5., -2.1, 8., 4., 1.1])


@test_util.multi_backend_test(globals(),
                              'targets.vectorized_stochastic_volatility_test')
class VectorizedStochasticVolatilityTest(test_util.InferenceGymTestCase,
                                         parameterized.TestCase):

  @parameterized.named_parameters(
      ('Centered', True, False),
      ('Noncentered', False, False),
      ('NoncenteredFFT', False, True),
  )
  def testBasic(self, centered, use_fft):
    """Checks that you get finite values given unconstrained samples.

    We check `log_prob` as well as the values of the expectations.

    Args:
      centered: Whether or not to use the centered parameterization.
      use_fft: Whether or not to use FFT-based convolution to implement the
        centering transformation.
    """
    model = vectorized_stochastic_volatility.VectorizedStochasticVolatility(
        centered_returns=_test_dataset(), centered=centered, use_fft=use_fft)
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'persistence_of_volatility': [],
                'mean_log_volatility': [],
                'white_noise_shock_scale': [],
                'log_volatility': [5]
            }))

  def testDeferred(self):
    """Checks that the dataset is not prematurely materialized."""
    self.validate_deferred_materialization(
        vectorized_stochastic_volatility.VectorizedStochasticVolatility,
        centered_returns=_test_dataset())

  @test_util.numpy_disable_gradient_test
  def testParameterizationsConsistent(self):
    """Run HMC for both parameterizations, and compare posterior means."""
    self.skipTest('Broken by omnistaging b/168705919')
    centered_returns = _test_dataset()
    centered_model = (
        vectorized_stochastic_volatility.VectorizedStochasticVolatility(
            centered_returns, centered=True))
    non_centered_model = (
        vectorized_stochastic_volatility.VectorizedStochasticVolatility(
            centered_returns, centered=False, use_fft=False))
    non_centered_fft_model = (
        vectorized_stochastic_volatility.VectorizedStochasticVolatility(
            centered_returns, centered=False, use_fft=True))

    logging.info('Centered:')
    centered_results = self.evaluate(
        test_util.run_hmc_on_model(
            centered_model,
            num_chains=4,
            num_steps=1000,
            num_leapfrog_steps=10,
            step_size=0.1,
            # TF XLA is very slow on this problem
            use_xla=BACKEND == 'backend_jax',
            target_accept_prob=0.7))
    logging.info('Acceptance rate: %s', centered_results.accept_rate)
    logging.info('ESS: %s', centered_results.ess)
    logging.info('r_hat: %s', centered_results.r_hat)

    logging.info('Non-centered (without FFT):')
    non_centered_results = self.evaluate(
        test_util.run_hmc_on_model(
            non_centered_model,
            num_chains=4,
            num_steps=1000,
            num_leapfrog_steps=10,
            step_size=0.1,
            # TF XLA is very slow on this problem
            use_xla=BACKEND == 'backend_jax',
            target_accept_prob=0.7))
    logging.info('Acceptance rate: %s', non_centered_results.accept_rate)
    logging.info('ESS: %s', non_centered_results.ess)
    logging.info('r_hat: %s', non_centered_results.r_hat)

    logging.info('Non-centered (with FFT):')
    non_centered_fft_results = self.evaluate(
        test_util.run_hmc_on_model(
            non_centered_fft_model,
            num_chains=4,
            num_steps=1000,
            num_leapfrog_steps=10,
            step_size=0.1,
            # TF XLA is very slow on this problem
            use_xla=BACKEND == 'backend_jax',
            target_accept_prob=0.7))
    logging.info('Acceptance rate: %s', non_centered_fft_results.accept_rate)
    logging.info('ESS: %s', non_centered_fft_results.ess)
    logging.info('r_hat: %s', non_centered_fft_results.r_hat)

    centered_params = self.evaluate(
        tf.nest.map_structure(
            tf.identity, centered_model.sample_transformations['identity'](
                centered_results.chain)))
    non_centered_params = self.evaluate(
        tf.nest.map_structure(
            tf.identity, non_centered_model.sample_transformations['identity'](
                non_centered_results.chain)))
    non_centered_fft_params = self.evaluate(
        tf.nest.map_structure(
            tf.identity,
            non_centered_fft_model.sample_transformations['identity'](
                non_centered_fft_results.chain)))

    def get_mean_and_var(chain):

      def one_part(chain):
        mean = chain.mean((0, 1))
        var = chain.var((0, 1))
        ess = 1. / (1. / self.evaluate(
            tfp.mcmc.effective_sample_size(
                chain, filter_beyond_positive_pairs=True))).mean(0)
        return mean, var / ess

      mean_var = tf.nest.map_structure(one_part, chain)
      return (nest.map_structure_up_to(chain, lambda x: x[0], mean_var),
              nest.map_structure_up_to(chain, lambda x: x[1], mean_var))

    centered_mean, centered_var = get_mean_and_var(centered_params)
    non_centered_mean, non_centered_var = get_mean_and_var(non_centered_params)
    non_centered_fft_mean, non_centered_fft_var = get_mean_and_var(
        non_centered_fft_params)

    def get_atol(var1, var2):
      # TODO(b/144290399): Use the full atol vector.
      max_var_per_rv = tf.nest.map_structure(
          lambda v1, v2: (3. * np.sqrt(v1 + v2)).max(), var1, var2)
      return functools.reduce(max, tf.nest.flatten(max_var_per_rv))

    self.assertAllCloseNested(
        centered_mean,
        non_centered_mean,
        atol=get_atol(centered_var, non_centered_var),
    )
    self.assertAllCloseNested(
        centered_mean,
        non_centered_fft_mean,
        atol=get_atol(centered_var, non_centered_fft_var),
    )

  def testFFTConv(self):
    np.random.seed(10003)
    x = np.random.randn(100).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    x_conv_y_np = np.convolve(x, y)
    x_conv_y_fft = vectorized_stochastic_volatility._fft_conv(x, y)
    self.assertAllClose(x_conv_y_np, x_conv_y_fft, rtol=1e-3)

  def testFFTConvCenter(self):
    np.random.seed(10003)
    persistence = 0.95
    noncentered = np.random.randn(100).astype(np.float32)
    centered = np.zeros_like(noncentered)
    centered[0] = noncentered[0] / np.sqrt(1 - persistence**2)
    for i in range(1, len(centered)):
      centered[i] = noncentered[i] + persistence * centered[i - 1]
    self.assertAllClose(
        centered,
        vectorized_stochastic_volatility._fft_conv_center(
            noncentered, persistence),
        rtol=1e-3)

  def testSP500Small(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    if BACKEND == 'backend_jax':
      self.skipTest('Too slow.')
    model = (
        vectorized_stochastic_volatility
        .VectorizedStochasticVolatilitySP500Small())
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'persistence_of_volatility': [],
                'mean_log_volatility': [],
                'white_noise_shock_scale': [],
                'log_volatility': [100]
            }),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True)

  @test_util.numpy_disable_gradient_test
  def testSP500SmallHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    self.skipTest('Broken by omnistaging b/168705919')
    model = (
        vectorized_stochastic_volatility
        .VectorizedStochasticVolatilitySP500Small())

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=500,
        num_leapfrog_steps=15,
        step_size=1.,
        use_xla=BACKEND == 'backend_jax',  # TF XLA is very slow on this problem
    )

  def testSP500(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = (
        vectorized_stochastic_volatility.VectorizedStochasticVolatilitySP500())
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'persistence_of_volatility': [],
                'mean_log_volatility': [],
                'white_noise_shock_scale': [],
                'log_volatility': [2516]
            }),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True)

  @test_util.numpy_disable_gradient_test
  def testSP500HMC(self):
    """Checks approximate samples from the model against the ground truth."""
    self.skipTest('b/169073800')
    model = (
        vectorized_stochastic_volatility.VectorizedStochasticVolatilitySP500())

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=2000,
        num_leapfrog_steps=25,
        step_size=1.,
        use_xla=BACKEND == 'backend_jax',  # TF XLA is very slow on this problem
    )

  def testLogSP500Small(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    if BACKEND == 'backend_jax':
      self.skipTest('Too slow.')
    model = (
        vectorized_stochastic_volatility
        .VectorizedStochasticVolatilityLogSP500Small())
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'persistence_of_volatility': [],
                'mean_log_volatility': [],
                'white_noise_shock_scale': [],
                'log_volatility': [100]
            }),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True)

  @test_util.numpy_disable_gradient_test
  def testLogSP500SmallHMC(self):
    """Checks approximate samples from the model against the ground truth."""
    self.skipTest('Broken by omnistaging b/168705919')
    model = (
        vectorized_stochastic_volatility
        .VectorizedStochasticVolatilityLogSP500Small())

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=500,
        num_leapfrog_steps=15,
        step_size=1.,
        use_xla=BACKEND == 'backend_jax',  # TF XLA is very slow on this problem
    )

  def testLogSP500(self):
    """Checks that you get finite values given unconstrained samples.

    We check `unnormalized_log_prob` as well as the values of the sample
    transformations.
    """
    model = (
        vectorized_stochastic_volatility.VectorizedStochasticVolatilityLogSP500(
        ))
    self.validate_log_prob_and_transforms(
        model,
        sample_transformation_shapes=dict(
            identity={
                'persistence_of_volatility': [],
                'mean_log_volatility': [],
                'white_noise_shock_scale': [],
                'log_volatility': [2516]
            }),
        check_ground_truth_mean_standard_error=True,
        check_ground_truth_mean=True,
        check_ground_truth_standard_deviation=True)

  @test_util.numpy_disable_gradient_test
  def testLogSP500HMC(self):
    """Checks approximate samples from the model against the ground truth."""
    self.skipTest('b/169073800')
    model = (
        vectorized_stochastic_volatility.VectorizedStochasticVolatilityLogSP500(
        ))

    self.validate_ground_truth_using_hmc(
        model,
        num_chains=4,
        num_steps=2000,
        num_leapfrog_steps=25,
        step_size=1.,
        use_xla=BACKEND == 'backend_jax',  # TF XLA is very slow on this problem
    )


if __name__ == '__main__':
  tfp_test_util.main()
