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

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc import langevin
from tensorflow_probability.python.mcmc import sample


JAX_MODE = False


@test_util.test_graph_and_eager_modes
class LangevinTest(test_util.TestCase):

  def testLangevin1DNormal(self):
    """Sampling from the Standard Normal Distribution."""
    dtype = np.float32
    nchains = 32

    target = normal.Normal(loc=dtype(0), scale=dtype(1))
    samples = sample.sample_chain(
        num_results=500,
        current_state=np.ones([nchains], dtype=dtype),
        kernel=langevin.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target.log_prob,
            step_size=0.75,
            volatility_fn=lambda *args: .5),
        num_burnin_steps=200,
        trace_fn=None,
        seed=test_util.test_seed())

    sample_mean = tf.reduce_mean(samples, axis=(0, 1))
    sample_std = tf.math.reduce_std(samples, axis=(0, 1))

    sample_mean_, sample_std_ = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(sample_mean_, 0., atol=0.12)
    self.assertAllClose(sample_std_, 1., atol=0.1)

  def testLangevin3DNormal(self):
    """Sampling from a 3-D Multivariate Normal distribution."""
    dtype = np.float32
    true_mean = dtype([1, 2, 7])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
    num_results = 500
    num_chains = 500

    # Target distribution is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)
    target = mvn_tril.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

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
    states = sample.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=langevin.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target_log_prob, step_size=.1),
        num_burnin_steps=200,
        num_steps_between_results=1,
        trace_fn=None,
        seed=test_util.test_seed())

    states = tf.concat(states, axis=-1)
    sample_mean = tf.reduce_mean(states, axis=[0, 1])
    x = (states - sample_mean)[..., tf.newaxis]
    sample_cov = tf.reduce_mean(
        tf.matmul(x, tf.transpose(a=x, perm=[0, 1, 3, 2])),
        axis=[0, 1])

    sample_mean_, sample_cov_ = self.evaluate([sample_mean, sample_cov])

    self.assertAllClose(true_mean, np.squeeze(sample_mean_), atol=0.1, rtol=0.1)
    self.assertAllClose(true_cov, np.squeeze(sample_cov_), atol=0.1, rtol=0.1)

  def testLangevin3DNormalDynamicVolatility(self):
    """Sampling from a 3-D Multivariate Normal distribution."""
    dtype = np.float32
    true_mean = dtype([1, 2, 7])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
    num_results = 500
    num_chains = 500

    # Targeg distribution is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)
    target = mvn_tril.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

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
    states = sample.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=langevin.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target_log_prob,
            volatility_fn=volatility_fn,
            step_size=.1),
        num_burnin_steps=200,
        num_steps_between_results=1,
        trace_fn=None,
        seed=test_util.test_seed())

    states = tf.concat(states, axis=-1)
    sample_mean = tf.reduce_mean(states, axis=[0, 1])
    x = (states - sample_mean)[..., tf.newaxis]
    sample_cov = tf.reduce_mean(tf.matmul(x, x, transpose_b=True), axis=[0, 1])

    sample_mean_, sample_cov_ = self.evaluate([sample_mean, sample_cov])

    self.assertAllClose(true_mean, np.squeeze(sample_mean_), atol=0.1, rtol=0.1)
    self.assertAllClose(true_cov, np.squeeze(sample_cov_), atol=0.1, rtol=0.1)

  def testLangevinCorrectVolatilityGradient(self):
    """Check that the gradient of the volatility is computed correctly."""
    # Consider the example target distribution as in `testLangevin3DNormal`
    dtype = np.float32
    true_mean = dtype([1, 2, 7])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
    num_chains = 100

    chol = tf.linalg.cholesky(true_cov)
    target = mvn_tril.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

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

    # Define MALA with constant volatility
    langevin_unit = langevin.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=target_log_prob, step_size=0.1)
    # Define MALA with volatility being `volatility_fn`
    langevin_general = langevin.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=target_log_prob,
        step_size=0.1,
        volatility_fn=volatility_fn)

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
    mala = langevin.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,
        step_size=0.1,
    )
    self.assertTrue(mala.is_calibrated)

  def testUncalibratedLangevinIsNotCalibrated(self):
    uncal_langevin = langevin.UncalibratedLangevin(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,
        step_size=0.1,
    )
    self.assertFalse(uncal_langevin.is_calibrated)


@test_util.test_all_tf_execution_regimes
class DistributedLangevinTest(distribute_test_lib.DistributedTest):

  def test_langevin_kernel_tracks_axis_names(self):
    kernel = langevin.MetropolisAdjustedLangevinAlgorithm(
        normal.Normal(0, 1).log_prob, step_size=1.9)
    self.assertIsNone(kernel.experimental_shard_axis_names)
    kernel = langevin.MetropolisAdjustedLangevinAlgorithm(
        normal.Normal(0, 1).log_prob,
        step_size=1.9,
        experimental_shard_axis_names=['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])
    kernel = langevin.MetropolisAdjustedLangevinAlgorithm(
        normal.Normal(0, 1).log_prob,
        step_size=1.9).experimental_with_shard_axes(['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])

  def test_computes_same_log_acceptance_correction_with_sharded_state(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) + distribute_lib.psum(
          normal.Normal(distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b),
          'foo'))

    kernel = langevin.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob, step_size=1.9)
    sharded_kernel = kernel.experimental_with_shard_axes([None, ['foo']])

    def run(seed):
      state = [tf.convert_to_tensor(0.), tf.convert_to_tensor(0.)]
      kr = sharded_kernel.bootstrap_results(state)
      _, kr = sharded_kernel.one_step(state, kr, seed=seed)
      return kr.proposed_results.log_acceptance_correction

    log_acceptance_correction = self.evaluate(self.per_replica_to_tensor(
        self.strategy_run(run, args=(samplers.zeros_seed(),),
                          in_axes=None, axis_name='foo'), 0))

    for i in range(distribute_test_lib.NUM_DEVICES):
      self.assertAllClose(log_acceptance_correction[i],
                          log_acceptance_correction[0])

  def test_unsharded_state_remains_synchronized_across_devices(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) + distribute_lib.psum(
          normal.Normal(distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b),
          'foo'))

    kernel = langevin.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob, step_size=1e-1)
    sharded_kernel = kernel.experimental_with_shard_axes([None, ['foo']])

    def run(seed):
      state = [tf.convert_to_tensor(-10.),
               tf.convert_to_tensor(-10.)]
      kr = sharded_kernel.bootstrap_results(state)
      state, _ = sharded_kernel.one_step(state, kr, seed=seed)
      return state

    state = self.evaluate(self.per_replica_to_tensor(
        self.strategy_run(run, args=(samplers.zeros_seed(),),
                          in_axes=None, axis_name='foo'), 0))

    for i in range(distribute_test_lib.NUM_DEVICES):
      self.assertAllClose(state[0][i],
                          state[0][0])


if __name__ == '__main__':
  test_util.main()
