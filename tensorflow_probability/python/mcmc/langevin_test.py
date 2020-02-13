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
"""Tests for MetropolisAdjustedLangevinAlgorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_graph_and_eager_modes
class LangevinTest(test_util.TestCase):

  def testLangevin1DNormal(self):
    """Sampling from the Standard Normal Distribution."""
    dtype = np.float32

    target = tfd.Normal(loc=dtype(0), scale=dtype(1))
    samples, _ = tfp.mcmc.sample_chain(
        num_results=1000,
        current_state=dtype(1),
        kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target.log_prob,
            step_size=0.75,
            seed=test_util.test_seed()),
        num_burnin_steps=500,
        parallel_iterations=1)  # For determinism.

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_std = tf.sqrt(
        tf.reduce_mean(
            tf.math.squared_difference(samples, sample_mean),
            axis=0))

    sample_mean_, sample_std_ = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(sample_mean_, 0., atol=0.1, rtol=0.1)
    self.assertAllClose(sample_std_, 1., atol=0.1, rtol=0.1)

  def testLangevin3DNormal(self):
    """Sampling from a 3-D Multivariate Normal distribution."""
    dtype = np.float32
    true_mean = dtype([1, 2, 7])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
    num_results = 500
    num_chains = 500

    # Target distribution is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a list of tensors `x` and `y`.
    # Then the target log-density is defined as follows:
    def target_log_prob(x, y):
      # Stack the input tensors together
      z = tf.concat([x, y], axis=-1)
      return target.log_prob(z)

    # Initial state of the chain
    init_state = [np.ones([num_chains, 2], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]

    # Run MALA with normal proposal for `num_results` iterations for
    # `num_chains` independent chains:
    states, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target_log_prob,
            step_size=.1,
            seed=test_util.test_seed()),
        num_burnin_steps=200,
        num_steps_between_results=1,
        parallel_iterations=1)

    states = tf.concat(states, axis=-1)
    sample_mean = tf.reduce_mean(states, axis=[0, 1])
    x = (states - sample_mean)[..., tf.newaxis]
    sample_cov = tf.reduce_mean(
        tf.matmul(x, tf.transpose(a=x, perm=[0, 1, 3, 2])),
        axis=[0, 1])

    sample_mean_, sample_cov_ = self.evaluate([sample_mean, sample_cov])

    self.assertAllClose(np.squeeze(sample_mean_), true_mean, atol=0.1, rtol=0.1)
    self.assertAllClose(np.squeeze(sample_cov_), true_cov, atol=0.1, rtol=0.1)

  def testLangevin3DNormalDynamicVolatility(self):
    """Sampling from a 3-D Multivariate Normal distribution."""
    dtype = np.float32
    true_mean = dtype([1, 2, 7])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
    num_results = 500
    num_chains = 500

    # Targeg distribution is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a list of 1-d tensors `x` and `y`.
    # Then the target log-density is defined as follows:
    def target_log_prob(x, y):
      # Stack the input tensors together
      z = tf.concat([x, y], axis=-1)
      return target.log_prob(z)

    # Here we define the volatility function to be non-caonstant
    def volatility_fn(x, y):
      # Stack the input tensors together
      return [1. / (0.5 + 0.1 * tf.abs(x + y)),
              1. / (0.5 + 0.1 * tf.abs(y))]

    # Initial state of the chain
    init_state = [np.ones([num_chains, 2], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]

    # Run Random Walk Metropolis with normal proposal for `num_results`
    # iterations for `num_chains` independent chains:
    states, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target_log_prob,
            volatility_fn=volatility_fn,
            step_size=.1,
            seed=test_util.test_seed()),
        num_burnin_steps=200,
        num_steps_between_results=1,
        parallel_iterations=1)

    states = tf.concat(states, axis=-1)
    sample_mean = tf.reduce_mean(states, axis=[0, 1])
    x = (states - sample_mean)[..., tf.newaxis]
    sample_cov = tf.reduce_mean(tf.matmul(x, x, transpose_b=True), axis=[0, 1])

    sample_mean_, sample_cov_ = self.evaluate([sample_mean, sample_cov])

    self.assertAllClose(np.squeeze(sample_mean_), true_mean, atol=0.1, rtol=0.1)
    self.assertAllClose(np.squeeze(sample_cov_), true_cov, atol=0.1, rtol=0.1)

  def testLangevinCorrectVolatilityGradient(self):
    """Check that the gradient of the volatility is computed correctly."""
    # Consider the example target distribution as in `testLangevin3DNormal`
    dtype = np.float32
    true_mean = dtype([1, 2, 7])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
    num_chains = 100

    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    def target_log_prob(x, y):
      # Stack the input tensors together
      z = tf.concat([x, y], axis=-1)
      return target.log_prob(z)

    def volatility_fn(x, y):
      # Stack the input tensors together
      return [1. / (0.5 + 0.1 * tf.abs(x + y)),
              1. / (0.5 + 0.1 * tf.abs(y))]

    # Initial state of the chain
    init_state = [np.ones([num_chains, 2], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]

    strm = test_util.test_seed_stream()
    # Define MALA with constant volatility
    langevin_unit = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=target_log_prob,
        step_size=0.1,
        seed=strm())
    # Define MALA with volatility being `volatility_fn`
    langevin_general = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=target_log_prob,
        step_size=0.1,
        volatility_fn=volatility_fn,
        seed=strm())

    # Initialize the samplers
    kernel_unit_volatility = langevin_unit.bootstrap_results(init_state)
    kernel_general = langevin_general.bootstrap_results(init_state)

    # For `langevin_general` volatility gradient should be zero.
    grad_1, grad_2 = kernel_unit_volatility.accepted_results.grads_volatility
    self.assertAllEqual(self.evaluate(grad_1),
                        np.zeros(shape=init_state[0].shape, dtype=dtype))
    self.assertAllEqual(self.evaluate(grad_2),
                        np.zeros(shape=init_state[1].shape, dtype=dtype))

    # For `langevin_unit` volatility gradient should be around -0.926 for
    # each direction.
    grad_1, grad_2 = kernel_general.accepted_results.grads_volatility
    self.assertAllClose(self.evaluate(grad_1),
                        -0.583 * np.ones(shape=init_state[0].shape,
                                         dtype=dtype),
                        atol=0.01, rtol=0.01)
    self.assertAllClose(self.evaluate(grad_2),
                        -0.926 * np.ones(shape=init_state[1].shape,
                                         dtype=dtype),
                        atol=0.01, rtol=0.01)

  def testMALAIsCalibrated(self):
    mala = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,
        step_size=0.1,
    )
    self.assertTrue(mala.is_calibrated)

  def testUncalibratedLangevinIsNotCalibrated(self):
    uncal_langevin = tfp.mcmc.UncalibratedLangevin(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,
        step_size=0.1,
    )
    self.assertFalse(uncal_langevin.is_calibrated)


if __name__ == '__main__':
  tf.test.main()
