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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_graph_and_eager_modes
class SliceSamplerTest(test_util.TestCase):

  def _get_mode_dependent_settings(self):
    if tf.executing_eagerly():
      num_results = 100
      tolerance = .2
    else:
      num_results = 400
      tolerance = .1
    return num_results, tolerance

  def testOneDimNormal(self):
    """Sampling from the Standard Normal Distribution."""
    dtype = np.float32
    num_results, tolerance = self._get_mode_dependent_settings()
    target = tfd.Normal(loc=dtype(0), scale=dtype(1))

    samples, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=dtype(1),
        kernel=tfp.mcmc.SliceSampler(
            target.log_prob,
            step_size=1.0,
            max_doublings=5,
            seed=test_util.test_seed_stream()),
        num_burnin_steps=100,
        parallel_iterations=1)  # For determinism.

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_std = tfp.stats.stddev(samples)

    self.assertAllClose(0., sample_mean, atol=tolerance, rtol=tolerance)
    self.assertAllClose(1., sample_std, atol=tolerance, rtol=tolerance)

  @parameterized.named_parameters(
      dict(testcase_name='StaticShape', static_shape=True, static_rank=True),
      dict(testcase_name='DynamicShape', static_shape=False, static_rank=True),
      dict(testcase_name='DynamicRank', static_shape=False, static_rank=False),
  )
  def testTwoDimNormal(self, static_shape, static_rank):
    """Sampling from a 2-D Multivariate Normal distribution."""
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    num_results, tolerance = self._get_mode_dependent_settings()
    num_chains = 10
    # Target distribution is defined through the Cholesky decomposition.
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

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
    states, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.SliceSampler(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            step_size=1.0,
            max_doublings=5,
            seed=test_util.test_seed_stream()),
        num_burnin_steps=100,
        parallel_iterations=1)

    states = tf.reshape(states, [-1, 2])
    sample_mean = tf.reduce_mean(states, axis=0)
    sample_cov = tfp.stats.covariance(states)

    self.assertAllClose(true_mean, sample_mean, atol=tolerance, rtol=tolerance)
    self.assertAllClose(true_cov, sample_cov, atol=tolerance, rtol=tolerance)

  def testFourDimNormal(self):
    """Sampling from a 4-D Multivariate Normal distribution."""

    dtype = np.float32
    true_mean = dtype([0, 4, -8, 2])
    true_cov = np.eye(4, dtype=dtype)
    num_results, tolerance = self._get_mode_dependent_settings()
    num_chains = 10
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=true_cov)

    # Initial state of the chain
    init_state = np.ones([num_chains, 4], dtype=dtype)

    # Run Slice Samper for `num_results` iterations for `num_chains`
    # independent chains:
    states, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.SliceSampler(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            step_size=1.0,
            max_doublings=5,
            seed=test_util.test_seed_stream()),
        num_burnin_steps=100,
        parallel_iterations=1)

    result = tf.reshape(states, [-1, 4])
    sample_mean = tf.reduce_mean(result, axis=0)
    sample_cov = tfp.stats.covariance(result)

    self.assertAllClose(true_mean, sample_mean, atol=tolerance, rtol=tolerance)
    self.assertAllClose(true_cov, sample_cov, atol=tolerance, rtol=tolerance)

  def testTwoStateParts(self):
    dtype = np.float32
    num_results, tolerance = self._get_mode_dependent_settings()

    true_loc1 = 1.
    true_scale1 = 1.
    true_loc2 = -1.
    true_scale2 = 2.
    target = tfd.JointDistributionSequential([
        tfd.Normal(true_loc1, true_scale1),
        tfd.Normal(true_loc2, true_scale2),
    ])

    num_chains = 10

    init_state = [np.ones([num_chains], dtype=dtype),
                  np.ones([num_chains], dtype=dtype)]

    target_fn = tf.function(
        lambda *states: target.log_prob(states), autograph=False)
    [states1, states2], _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.SliceSampler(
            target_log_prob_fn=target_fn,
            step_size=1.0,
            max_doublings=5,
            seed=test_util.test_seed_stream()),
        num_burnin_steps=100,
        parallel_iterations=1)

    states1 = tf.reshape(states1, [-1])
    states2 = tf.reshape(states2, [-1])
    sample_mean1 = tf.reduce_mean(states1, axis=0)
    sample_stddev1 = tfp.stats.stddev(states1)
    sample_mean2 = tf.reduce_mean(states2, axis=0)
    sample_stddev2 = tfp.stats.stddev(states2)

    self.assertAllClose(true_loc1, sample_mean1, atol=tolerance, rtol=tolerance)
    self.assertAllClose(
        true_scale1, sample_stddev1, atol=tolerance, rtol=tolerance)
    self.assertAllClose(true_loc2, sample_mean2, atol=tolerance, rtol=tolerance)
    self.assertAllClose(
        true_scale2, sample_stddev2, atol=tolerance, rtol=tolerance)


if __name__ == '__main__':
  tf.test.main()
