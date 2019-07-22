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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class SliceSamplerTest(tf.test.TestCase):

  def testOneDimNormal(self):
    """Sampling from the Standard Normal Distribution."""
    dtype = np.float32

    target = tfd.Normal(loc=dtype(0), scale=dtype(1))

    samples, _ = tfp.mcmc.sample_chain(
        num_results=500,
        current_state=dtype(1),
        kernel=tfp.mcmc.SliceSampler(
            target.log_prob,
            step_size=1.0,
            max_doublings=5,
            seed=1234),
        num_burnin_steps=500,
        parallel_iterations=1)  # For determinism.

    sample_mean = tf.reduce_mean(input_tensor=samples, axis=0)
    sample_std = tf.sqrt(
        tf.reduce_mean(
            input_tensor=tf.math.squared_difference(samples, sample_mean),
            axis=0))
    [sample_mean, sample_std] = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(0., b=sample_mean, atol=0.1, rtol=0.1)
    self.assertAllClose(1., b=sample_std, atol=0.1, rtol=0.1)

  def testTwoDimNormal(self):
    """Sampling from a 2-D Multivariate Normal distribution."""
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    num_results = 200
    num_chains = 75
    # Target distribution is defined through the Cholesky decomposition.
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a list of 1-d tensors `x` and `y`.
    # Then the target log-density is defined as follows:
    def target_log_prob(x, y):
      # Stack the input tensors together
      z = tf.stack([x, y], axis=-1) - true_mean
      return target.log_prob(z)

    # Initial state of the chain
    init_state = [np.ones([num_chains, 1], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]

    # Run Slice Samper for `num_results` iterations for `num_chains`
    # independent chains:
    [x, y], _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.SliceSampler(
            target_log_prob_fn=target_log_prob,
            step_size=1.0,
            max_doublings=5,
            seed=47),
        num_burnin_steps=200,
        num_steps_between_results=1,
        parallel_iterations=1)

    states = tf.stack([x, y], axis=-1)
    sample_mean = tf.reduce_mean(input_tensor=states, axis=[0, 1])
    z = states - sample_mean
    sample_cov = tf.reduce_mean(
        input_tensor=tf.matmul(z, z, transpose_a=True), axis=[0, 1])
    [sample_mean, sample_cov] = self.evaluate([
        sample_mean, sample_cov])

    self.assertAllClose(true_mean, b=np.squeeze(sample_mean),
                        atol=0.1, rtol=0.1)
    self.assertAllClose(true_cov, b=np.squeeze(sample_cov), atol=0.1, rtol=0.1)

  def testTwoDimNormalDynamicShape(self):
    """Checks that dynamic batch shapes for the initial state are supported."""
    if tf.executing_eagerly(): return

    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    num_results = 200
    num_chains = 75
    # Target distribution is defined through the Cholesky decomposition.
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a list of 1-d tensors `x` and `y`.
    # Then the target log-density is defined as follows:
    def target_log_prob(x, y):
      # Stack the input tensors together
      z = tf.stack([x, y], axis=-1) - true_mean
      return target.log_prob(z)

    # Initial state of the chain
    init_state = [np.ones([num_chains, 1], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]
    placeholder_init_state = [
        tf.compat.v1.placeholder_with_default(init_state[0], shape=[None, 1]),
        tf.compat.v1.placeholder_with_default(init_state[1], shape=[None, 1])
    ]
    # Run Slice Samper for `num_results` iterations for `num_chains`
    # independent chains:
    [x, y], _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=placeholder_init_state,
        kernel=tfp.mcmc.SliceSampler(
            target_log_prob_fn=target_log_prob,
            step_size=1.0,
            max_doublings=5,
            seed=47),
        num_burnin_steps=200,
        num_steps_between_results=1,
        parallel_iterations=1)

    states = tf.stack([x, y], axis=-1)
    sample_mean = tf.reduce_mean(input_tensor=states, axis=[0, 1])
    z = states - sample_mean
    sample_cov = tf.reduce_mean(
        input_tensor=tf.matmul(z, z, transpose_a=True), axis=[0, 1])
    [sample_mean, sample_cov] = self.evaluate([sample_mean, sample_cov])

    self.assertAllClose(true_mean, b=np.squeeze(sample_mean),
                        atol=0.1, rtol=0.1)
    self.assertAllClose(true_cov, b=np.squeeze(sample_cov), atol=0.1, rtol=0.1)

  def testTwoDimNormalDynamicRank(self):
    """Checks that fully dynamic shape for the initial state is supported."""
    if tf.executing_eagerly(): return
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    num_results = 200
    num_chains = 75
    # Target distribution is defined through the Cholesky decomposition.
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a list of 1-d tensors `x` and `y`.
    # Then the target log-density is defined as follows:
    def target_log_prob(x, y):
      # Stack the input tensors together
      z = tf.stack([x, y], axis=-1) - true_mean
      return target.log_prob(z)

    # Initial state of the chain
    init_state = [np.ones([num_chains, 1], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]
    placeholder_init_state = [
        tf.compat.v1.placeholder_with_default(init_state[0], shape=None),
        tf.compat.v1.placeholder_with_default(init_state[1], shape=None)
    ]
    # Run Slice Samper for `num_results` iterations for `num_chains`
    # independent chains:
    [x, y], _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=placeholder_init_state,
        kernel=tfp.mcmc.SliceSampler(
            target_log_prob_fn=target_log_prob,
            step_size=1.0,
            max_doublings=5,
            seed=47),
        num_burnin_steps=200,
        num_steps_between_results=1,
        parallel_iterations=1)

    states = tf.stack([x, y], axis=-1)
    sample_mean = tf.reduce_mean(input_tensor=states, axis=[0, 1])
    z = states - sample_mean
    sample_cov = tf.reduce_mean(
        input_tensor=tf.matmul(z, z, transpose_a=True), axis=[0, 1])
    [sample_mean, sample_cov] = self.evaluate([sample_mean, sample_cov])

    self.assertAllClose(true_mean, b=np.squeeze(sample_mean),
                        atol=0.1, rtol=0.1)
    self.assertAllClose(true_cov, b=np.squeeze(sample_cov), atol=0.1, rtol=0.1)

  def testFourDimNormal(self):
    """Sampling from a 4-D Multivariate Normal distribution."""

    dtype = np.float32
    true_mean = dtype([0, 4, -8, 2])
    true_cov = dtype([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    num_results = 25
    num_chains = 500
    # Target distribution is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Initial state of the chain
    init_state = [np.ones([num_chains, 4], dtype=dtype)]

    # Run Slice Samper for `num_results` iterations for `num_chains`
    # independent chains:
    states, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.SliceSampler(
            target_log_prob_fn=target.log_prob,
            step_size=1.0,
            max_doublings=5,
            seed=47),
        num_burnin_steps=300,
        num_steps_between_results=1,
        parallel_iterations=1)
    result = states[0]
    sample_mean = tf.reduce_mean(input_tensor=result, axis=[0, 1])
    deviation = tf.reshape(result - sample_mean, shape=[-1, 4])
    sample_cov = tf.matmul(deviation, b=deviation, transpose_a=True)
    sample_cov /= tf.cast(tf.shape(input=deviation)[0], dtype=tf.float32)
    sample_mean_err = sample_mean - true_mean
    sample_cov_err = sample_cov - true_cov

    [sample_mean_err, sample_cov_err] = self.evaluate([sample_mean_err,
                                                       sample_cov_err])

    self.assertAllClose(np.zeros_like(sample_mean_err), b=sample_mean_err,
                        atol=0.1, rtol=0.1)
    self.assertAllClose(np.zeros_like(sample_cov_err), b=sample_cov_err,
                        atol=0.1, rtol=0.1)

if __name__ == '__main__':
  tf.test.main()
