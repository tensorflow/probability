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
"""Tests for RandomWalkMetropolis."""

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import cauchy
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc import random_walk_metropolis
from tensorflow_probability.python.mcmc import sample

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class RWMTest(test_util.TestCase):

  def testRWM1DUniform(self):
    """Sampling from the Standard Normal Distribution."""
    dtype = np.float32

    target = normal.Normal(loc=dtype(0), scale=dtype(1))

    samples = sample.sample_chain(
        num_results=2000,
        current_state=dtype(1),
        kernel=random_walk_metropolis.RandomWalkMetropolis(
            target.log_prob,
            new_state_fn=random_walk_metropolis.random_walk_uniform_fn(
                scale=dtype(2.))),
        num_burnin_steps=500,
        trace_fn=None,
        seed=test_util.test_seed())

    sample_mean = tf.math.reduce_mean(samples, axis=0)
    sample_std = tf.math.reduce_std(samples, axis=0)
    [sample_mean_, sample_std_] = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(0., sample_mean_, atol=0.17, rtol=0.)
    self.assertAllClose(1., sample_std_, atol=0.2, rtol=0.)

  def testRWM1DNormal(self):
    """Sampling from the Standard Normal Distribution with adaptation."""
    dtype = np.float32

    target = normal.Normal(loc=dtype(0), scale=dtype(1))
    samples = sample.sample_chain(
        num_results=500,
        current_state=dtype([1] * 8),  # 8 parallel chains
        kernel=random_walk_metropolis.RandomWalkMetropolis(target.log_prob),
        num_burnin_steps=500,
        trace_fn=None,
        seed=test_util.test_seed())

    sample_mean = tf.math.reduce_mean(samples, axis=(0, 1))
    sample_std = tf.math.reduce_std(samples, axis=(0, 1))

    [sample_mean_, sample_std_] = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(0., sample_mean_, atol=0.2, rtol=0.)
    self.assertAllClose(1., sample_std_, atol=0.2, rtol=0.)

  def testRWM1DNormalWithDynamicScaleForNextState(self):
    """Sampling from the Standard Normal Distribution with adaptation."""
    if tf.executing_eagerly() or JAX_MODE:
      raise self.skipTest(
          'Dynamic scale makes no sense in Eager or JAX modes.')

    scale_ph = tf1.placeholder_with_default(1.0, shape=None)

    def target_log_prob(x):
      return -tf.reduce_sum(x**2) / 2

    samples = sample.sample_chain(
        num_results=500,
        current_state=np.float32([0.] * 4),  # 4 parallel chains
        kernel=random_walk_metropolis.RandomWalkMetropolis(
            target_log_prob,
            new_state_fn=random_walk_metropolis.random_walk_normal_fn(scale_ph),
        ),
        num_burnin_steps=500,
        trace_fn=None,
        seed=test_util.test_seed(),
    )

    sample_mean = tf.math.reduce_mean(samples, axis=(0, 1))
    sample_std = tf.math.reduce_std(samples, axis=(0, 1))

    sample_mean_, sample_std_ = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(0., sample_mean_, atol=0.2, rtol=0.)
    self.assertAllClose(1., sample_std_, atol=0.2, rtol=0.)

  def testRWM1DCauchy(self):
    """Sampling from the Standard Normal Distribution using Cauchy proposal."""
    dtype = np.float32
    num_burnin_steps = 750
    num_chain_results = 400

    target = normal.Normal(loc=dtype(0), scale=dtype(1))

    def cauchy_new_state_fn(scale, dtype):
      dist = cauchy.Cauchy(loc=dtype(0), scale=dtype(scale))
      def _fn(state_parts, seed):
        seeds = samplers.split_seed(seed, n=len(state_parts), salt='rwmcauchy')
        next_state_parts = [
            state + dist.sample(state.shape, seed=part_seed)
            for state, part_seed in zip(state_parts, seeds)
        ]
        return next_state_parts
      return _fn

    samples = sample.sample_chain(
        num_results=num_chain_results,
        num_burnin_steps=num_burnin_steps,
        current_state=dtype([1] * 8),  # 8 parallel chains
        kernel=random_walk_metropolis.RandomWalkMetropolis(
            target.log_prob,
            new_state_fn=cauchy_new_state_fn(scale=0.5, dtype=dtype)),
        trace_fn=None,
        seed=test_util.test_seed())

    sample_mean = tf.math.reduce_mean(samples, axis=(0, 1))
    sample_std = tf.math.reduce_std(samples, axis=(0, 1))
    [sample_mean_, sample_std_] = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(0., sample_mean_, atol=0.2, rtol=0.)
    self.assertAllClose(1., sample_std_, atol=0.2, rtol=0.)

  def testRWM2DNormal(self):
    """Sampling from a 2-D Multivariate Normal distribution."""
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    num_results = 500
    num_chains = 100
    # Target distribution is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)
    target = mvn_tril.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a list of 1-d tensors `x` and `y`.
    # Then the target log-density is defined as follows:
    def target_log_prob(x, y):
      # Stack the input tensors together
      z = tf.stack([x, y], axis=-1) - true_mean
      return target.log_prob(z)

    # Initial state of the chain
    init_state = [np.ones([num_chains, 1], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]

    # Run Random Walk Metropolis with normal proposal for `num_results`
    # iterations for `num_chains` independent chains:
    states = sample.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=random_walk_metropolis.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob),
        num_burnin_steps=200,
        num_steps_between_results=1,
        trace_fn=None,
        seed=test_util.test_seed())

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
    rwm = random_walk_metropolis.RandomWalkMetropolis(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,)
    self.assertTrue(rwm.is_calibrated)

  def testUncalibratedRWIsNotCalibrated(self):
    uncal_rw = random_walk_metropolis.UncalibratedRandomWalk(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,)
    self.assertFalse(uncal_rw.is_calibrated)


@test_util.test_all_tf_execution_regimes
class DistributedRWMTest(distribute_test_lib.DistributedTest):

  def test_rwm_kernel_tracks_axis_names(self):
    kernel = random_walk_metropolis.RandomWalkMetropolis(
        normal.Normal(0, 1).log_prob)
    self.assertIsNone(kernel.experimental_shard_axis_names)
    kernel = random_walk_metropolis.RandomWalkMetropolis(
        normal.Normal(0, 1).log_prob, experimental_shard_axis_names=['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])
    kernel = random_walk_metropolis.RandomWalkMetropolis(
        normal.Normal(0, 1).log_prob,).experimental_with_shard_axes(['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])

  @test_util.numpy_disable_test_missing_functionality('No SPMD support.')
  def test_computes_same_log_acceptance_correction_with_sharded_state(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) + distribute_lib.psum(
          normal.Normal(distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b),
          'foo'))

    kernel = random_walk_metropolis.RandomWalkMetropolis(target_log_prob)
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

  @test_util.numpy_disable_test_missing_functionality('No SPMD support.')
  def test_unsharded_state_remains_synchronized_across_devices(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) + distribute_lib.psum(
          normal.Normal(distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b),
          'foo'))

    kernel = random_walk_metropolis.RandomWalkMetropolis(target_log_prob)
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
