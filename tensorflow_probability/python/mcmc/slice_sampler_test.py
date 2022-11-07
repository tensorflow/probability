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
"""Tests for slice_sampler_utils.py and slice_sampler_kernel.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc import random_walk_metropolis
from tensorflow_probability.python.mcmc import sample
from tensorflow_probability.python.mcmc import slice_sampler_kernel
from tensorflow_probability.python.stats import sample_stats


JAX_MODE = False


def _get_mode_dependent_settings():
  if tf.executing_eagerly() and not JAX_MODE:
    num_results = 100
    tolerance = .2
  else:
    num_results = 400
    tolerance = .1
  return num_results, tolerance


@test_util.test_graph_and_eager_modes
class SliceSamplerTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_slow_asserts', asserts=True),
      dict(testcase_name='_fast_execute_only', asserts=False),
  )
  def testOneDimNormal(self, asserts):
    """Sampling from the Standard Normal Distribution."""
    dtype = np.float32
    num_results, tolerance = _get_mode_dependent_settings()
    # We're not using multiple chains in this test, so dial up samples.
    num_results = int(num_results * (5 if asserts else .2))
    target = normal.Normal(loc=dtype(0), scale=dtype(1))

    kernel = slice_sampler_kernel.SliceSampler(
        target.log_prob, step_size=1.0, max_doublings=5)
    if asserts:
      kernel.one_step = tf.function(kernel.one_step, autograph=False)
    samples = sample.sample_chain(
        num_results=num_results,
        current_state=dtype(1),
        kernel=kernel,
        num_burnin_steps=100,
        trace_fn=None,
        seed=test_util.test_seed_stream())

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_std = sample_stats.stddev(samples)

    if not asserts:
      return
    self.assertAllClose(0., sample_mean, atol=tolerance, rtol=tolerance)
    self.assertAllClose(1., sample_std, atol=tolerance, rtol=tolerance)

  @parameterized.named_parameters(
      dict(testcase_name='StaticShape', static_shape=True, static_rank=True),
      dict(testcase_name='DynamicShape', static_shape=False, static_rank=True),
      dict(testcase_name='DynamicRank', static_shape=False, static_rank=False),
  )
  def testTwoDimNormal(self, static_shape, static_rank):
    """Sampling from a 2-D Multivariate Normal distribution."""

    # Disabling this test in eager mode as it is very slow.
    # Other tests check eager behavior, so we can safely disable this test.
    if tf.executing_eagerly() and not JAX_MODE:
      return

    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    num_results, tolerance = _get_mode_dependent_settings()
    num_chains = 16
    # Target distribution is defined through the Cholesky decomposition.
    chol = tf.linalg.cholesky(true_cov)
    target = mvn_tril.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Initial state of the chain
    if static_shape and static_rank:
      shape = [num_chains, 2]
    elif static_rank:
      shape = [None, 2]
    else:
      shape = None

    init_state = tf1.placeholder_with_default(
        np.ones([num_chains, 2], dtype=dtype), shape=shape)

    # Run Slice Samper for `num_results` iterations for `num_chains`
    # independent chains:
    states = sample.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=slice_sampler_kernel.SliceSampler(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            step_size=1.0,
            max_doublings=5),
        num_burnin_steps=1000,
        trace_fn=None,
        seed=test_util.test_seed_stream())

    states = tf.reshape(states, [-1, 2])
    sample_mean = tf.reduce_mean(states, axis=0)
    sample_cov = sample_stats.covariance(states)

    self.assertAllClose(true_mean, sample_mean, atol=tolerance, rtol=tolerance)
    self.assertAllClose(true_cov, sample_cov, atol=tolerance, rtol=tolerance)

  def testFourDimNormal(self):
    """Sampling from a 4-D Multivariate Normal distribution."""

    dtype = np.float32
    true_mean = dtype([0, 4, -8, 2])
    true_cov = np.eye(4, dtype=dtype)
    num_results, tolerance = _get_mode_dependent_settings()
    num_chains = 16
    target = mvn_tril.MultivariateNormalTriL(loc=true_mean, scale_tril=true_cov)

    # Initial state of the chain
    init_state = np.ones([num_chains, 4], dtype=dtype)

    # Run Slice Samper for `num_results` iterations for `num_chains`
    # independent chains:
    states = sample.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=slice_sampler_kernel.SliceSampler(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            step_size=1.0,
            max_doublings=5),
        num_burnin_steps=100,
        trace_fn=None,
        seed=test_util.test_seed_stream())

    result = tf.reshape(states, [-1, 4])
    sample_mean = tf.reduce_mean(result, axis=0)
    sample_cov = sample_stats.covariance(result)

    self.assertAllClose(true_mean, sample_mean, atol=tolerance, rtol=tolerance)
    self.assertAllClose(true_cov, sample_cov, atol=tolerance, rtol=tolerance)

  def testTwoStateParts(self):
    dtype = np.float32
    num_results, tolerance = _get_mode_dependent_settings()

    true_loc1 = 1.
    true_scale1 = 1.
    true_loc2 = -1.
    true_scale2 = 2.
    target = jds.JointDistributionSequential([
        normal.Normal(true_loc1, true_scale1),
        normal.Normal(true_loc2, true_scale2),
    ])

    num_chains = 16

    init_state = [np.ones([num_chains], dtype=dtype),
                  np.ones([num_chains], dtype=dtype)]

    target_fn = tf.function(
        lambda *states: target.log_prob(states), autograph=False)
    [states1, states2] = sample.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=slice_sampler_kernel.SliceSampler(
            target_log_prob_fn=target_fn, step_size=1.0, max_doublings=5),
        num_burnin_steps=100,
        trace_fn=None,
        seed=test_util.test_seed_stream())

    states1 = tf.reshape(states1, [-1])
    states2 = tf.reshape(states2, [-1])
    sample_mean1 = tf.reduce_mean(states1, axis=0)
    sample_stddev1 = sample_stats.stddev(states1)
    sample_mean2 = tf.reduce_mean(states2, axis=0)
    sample_stddev2 = sample_stats.stddev(states2)

    self.assertAllClose(true_loc1, sample_mean1, atol=tolerance, rtol=tolerance)
    self.assertAllClose(
        true_scale1, sample_stddev1, atol=tolerance, rtol=tolerance)
    self.assertAllClose(true_loc2, sample_mean2, atol=tolerance, rtol=tolerance)
    self.assertAllClose(
        true_scale2, sample_stddev2, atol=tolerance, rtol=tolerance)


@test_util.test_all_tf_execution_regimes
class DistributedRWMTest(distribute_test_lib.DistributedTest):

  def test_slice_sampler_kernel_tracks_axis_names(self):
    kernel = slice_sampler_kernel.SliceSampler(
        normal.Normal(0, 1).log_prob, 1e-1, 5)
    self.assertIsNone(kernel.experimental_shard_axis_names)
    kernel = slice_sampler_kernel.SliceSampler(
        normal.Normal(0, 1).log_prob,
        1e-1,
        5,
        experimental_shard_axis_names=['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])
    kernel = random_walk_metropolis.RandomWalkMetropolis(
        normal.Normal(0, 1).log_prob, 1e-1,
        5).experimental_with_shard_axes(['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])

  @test_util.numpy_disable_test_missing_functionality('No SPMD support.')
  def test_computes_same_direction_for_unsharded_state(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) + distribute_lib.psum(
          normal.Normal(distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b),
          'foo'))

    kernel = slice_sampler_kernel.SliceSampler(target_log_prob, 1e-1, 5)
    sharded_kernel = kernel.experimental_with_shard_axes([None, ['foo']])

    def run(seed):
      state = [tf.convert_to_tensor(0.), tf.convert_to_tensor(0.)]
      kr = sharded_kernel.bootstrap_results(state)
      _, kr = sharded_kernel.one_step(state, kr, seed=seed)
      return kr.direction

    direction = self.evaluate(self.per_replica_to_tensor(
        self.strategy_run(run, args=(samplers.zeros_seed(),),
                          in_axes=None, axis_name='foo'), 0))

    for i in range(distribute_test_lib.NUM_DEVICES):
      self.assertAllClose(direction[0][i],
                          direction[0][0])

  @test_util.numpy_disable_test_missing_functionality('No SPMD support.')
  def test_unsharded_state_remains_synchronized_across_devices(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) + distribute_lib.psum(
          normal.Normal(distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b),
          'foo'))

    kernel = slice_sampler_kernel.SliceSampler(target_log_prob, 1e-1, 5)
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
