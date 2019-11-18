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
"""Tests for RandomWalkMetropolisNormal and RandomWalkMetropolisUniform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class RWMTest(test_util.TestCase):

  def testRWM1DUniform(self):
    """Sampling from the Standard Normal Distribution."""
    dtype = np.float32

    target = tfd.Normal(loc=dtype(0), scale=dtype(1))

    samples, _ = tfp.mcmc.sample_chain(
        num_results=2000,
        current_state=dtype(1),
        kernel=tfp.mcmc.RandomWalkMetropolis(
            target.log_prob,
            new_state_fn=tfp.mcmc.random_walk_uniform_fn(scale=dtype(2.)),
            seed=421),
        num_burnin_steps=500,
        parallel_iterations=1)  # For determinism.

    sample_mean = tf.math.reduce_mean(samples, axis=0)
    sample_std = tf.math.sqrt(
        tf.math.reduce_mean(
            tf.math.squared_difference(samples, sample_mean),
            axis=0))
    [sample_mean_, sample_std_] = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(sample_mean_, 0., atol=0.1, rtol=0.1)
    self.assertAllClose(sample_std_, 1., atol=0.1, rtol=0.1)

  def testRWM1DNormal(self):
    """Sampling from the Standard Normal Distribution with adaptation."""
    dtype = np.float32

    target = tfd.Normal(loc=dtype(0), scale=dtype(1))
    samples, _ = tfp.mcmc.sample_chain(
        num_results=1000,
        current_state=dtype(1),
        kernel=tfp.mcmc.RandomWalkMetropolis(
            target.log_prob,
            seed=42),
        num_burnin_steps=500,
        parallel_iterations=1)  # For determinism.

    sample_mean = tf.math.reduce_mean(samples, axis=0)
    sample_std = tf.math.sqrt(
        tf.math.reduce_mean(
            tf.math.squared_difference(samples, sample_mean),
            axis=0))

    [sample_mean_, sample_std_] = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(sample_mean_, 0., atol=0.2, rtol=0.2)
    self.assertAllClose(sample_std_, 1., atol=0.1, rtol=0.1)

  def testRWM1DCauchy(self):
    """Sampling from the Standard Normal Distribution using Cauchy proposal."""
    dtype = np.float32
    num_burnin_steps = 1000
    num_chain_results = 500

    target = tfd.Normal(loc=dtype(0), scale=dtype(1))

    def cauchy_new_state_fn(scale, dtype):
      cauchy = tfd.Cauchy(loc=dtype(0), scale=dtype(scale))
      def _fn(state_parts, seed):
        seed_stream = tfp.util.SeedStream(
            seed, salt='RandomWalkCauchyIncrement')
        next_state_parts = [
            state + cauchy.sample(
                sample_shape=state.shape, seed=seed_stream())
            for state in state_parts]
        return next_state_parts
      return _fn

    samples, _ = tfp.mcmc.sample_chain(
        num_results=num_chain_results,
        num_burnin_steps=num_burnin_steps,
        current_state=dtype(1),
        kernel=tfp.mcmc.RandomWalkMetropolis(
            target.log_prob,
            new_state_fn=cauchy_new_state_fn(scale=0.5, dtype=dtype),
            seed=42),
        parallel_iterations=1)  # For determinism.

    sample_mean = tf.math.reduce_mean(samples, axis=0)
    sample_std = tf.math.sqrt(
        tf.math.reduce_mean(
            tf.math.squared_difference(samples, sample_mean),
            axis=0))
    [sample_mean_, sample_std_] = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(sample_mean_, 0., atol=0.2, rtol=0.2)
    self.assertAllClose(sample_std_, 1., atol=0.1, rtol=0.1)

  def testRWM2DNormal(self):
    """Sampling from a 2-D Multivariate Normal distribution."""
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    num_results = 500
    num_chains = 100
    # Target distribution is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a list of 1-d tensors `x` and `y`.
    # Then the target log-density is defined as follows:
    def target_log_prob(x, y):
      # Stack the input tensors together
      z = tf.stack([x, y], axis=-1) - true_mean
      return target.log_prob(tf.squeeze(z))

    # Initial state of the chain
    init_state = [np.ones([num_chains, 1], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]

    # Run Random Walk Metropolis with normal proposal for `num_results`
    # iterations for `num_chains` independent chains:
    states, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob,
            seed=54),
        num_burnin_steps=200,
        num_steps_between_results=1,
        parallel_iterations=1)

    states = tf.stack(states, axis=-1)
    sample_mean = tf.math.reduce_mean(states, axis=[0, 1])
    x = states - sample_mean
    sample_cov = tf.math.reduce_mean(
        tf.linalg.matmul(x, x, transpose_a=True), axis=[0, 1])
    [sample_mean_, sample_cov_] = self.evaluate([
        sample_mean, sample_cov])

    self.assertAllClose(np.squeeze(sample_mean_), true_mean, atol=0.1, rtol=0.1)
    self.assertAllClose(np.squeeze(sample_cov_), true_cov, atol=0.1, rtol=0.1)

  def testRWMIsCalibrated(self):
    rwm = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,
    )
    self.assertTrue(rwm.is_calibrated)

  def testUncalibratedRWIsNotCalibrated(self):
    uncal_rw = tfp.mcmc.UncalibratedRandomWalk(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,
    )
    self.assertFalse(uncal_rw.is_calibrated)


if __name__ == '__main__':
  tf.test.main()
