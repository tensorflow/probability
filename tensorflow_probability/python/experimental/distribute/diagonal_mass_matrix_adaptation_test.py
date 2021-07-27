# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for tensorflow_probability.python.experimental.distribute.joint_distribution."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import distribute_test_lib as test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

tfd = tfp.distributions
tfp_dist = tfp.experimental.distribute


class EchoKernel(tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo):

  def __init__(self, *args, **kwargs):
    super().__init__(
        target_log_prob_fn=lambda x: -x * x,
        step_size=0.1,
        num_leapfrog_steps=2,
    )

  def one_step(self, current_state, previous_kernel_results, seed=None):
    _, nkr = super().one_step(current_state, previous_kernel_results, seed=seed)
    return current_state, nkr


@test_util.test_all_tf_execution_regimes
class DiagonalAdaptationTest(test_lib.DistributedTest):

  def test_diagonal_mass_matrix_no_distribute(self):
    """Nothing distributed. Make sure EchoKernel works."""
    kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
        EchoKernel(),
        tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=10., mean=tf.zeros(3), variance=tf.ones(3)))
    state = tf.zeros(3)
    pkr = kernel.bootstrap_results(state)
    draws = np.random.randn(10, 3).astype(np.float32)

    def body(pkr_seed, draw):
      pkr, seed = pkr_seed
      seed, kernel_seed = samplers.split_seed(seed)
      _, pkr = kernel.one_step(draw, pkr, seed=kernel_seed)
      return (pkr, seed)

    (pkr, _), _ = mcmc_util.trace_scan(body,
                                       (pkr, samplers.sanitize_seed(self.key)),
                                       draws, lambda _: ())

    running_variance = pkr.running_variance[0]
    emp_mean = draws.sum(axis=0) / 20.
    emp_squared_residuals = (np.sum((draws - emp_mean) ** 2, axis=0) +
                             10 * emp_mean ** 2 +
                             10)
    self.assertAllClose(emp_mean, running_variance.mean)
    self.assertAllClose(emp_squared_residuals,
                        running_variance.sum_squared_residuals)

  def test_diagonal_mass_matrix_independent(self):
    @tf.function(autograph=False)
    def run(seed):
      dist_seed, *seeds = samplers.split_seed(seed, 11)
      dist = tfp_dist.Sharded(
          tfd.Independent(tfd.Normal(tf.zeros(3), tf.ones(3)), 1),
          shard_axis_name=self.axis_name)
      state = dist.sample(seed=dist_seed)
      kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
          EchoKernel(),
          tfp.experimental.stats.RunningVariance.from_stats(
              num_samples=10., mean=tf.zeros(3), variance=tf.ones(3)))
      pkr = kernel.bootstrap_results(state)

      def body(draw_pkr, seed):
        _, pkr = draw_pkr
        draw_seed, step_seed = samplers.split_seed(seed)
        draw = dist.sample(seed=draw_seed)
        _, pkr = kernel.one_step(draw, pkr, seed=step_seed)
        return draw, pkr

      (_, pkr), draws = mcmc_util.trace_scan(body,
                                             (tf.zeros(dist.event_shape), pkr),
                                             seeds, lambda v: v[0])

      return draws, pkr

    draws, pkr = self.strategy_run(run, (self.key,), in_axes=None)
    running_variance = self.per_replica_to_composite_tensor(
        pkr.running_variance[0])
    draws = self.per_replica_to_tensor(draws, axis=1)
    mean, sum_squared_residuals, draws = self.evaluate(
        (running_variance.mean, running_variance.sum_squared_residuals, draws))
    emp_mean = tf.reduce_sum(draws, axis=0) / 20.
    emp_squared_residuals = (
        tf.reduce_sum((draws - emp_mean)**2, axis=0) + 10 * emp_mean**2 + 10)
    self.assertAllClose(emp_mean, mean)
    self.assertAllClose(emp_squared_residuals, sum_squared_residuals)

  def test_diagonal_mass_matrix_sample(self):
    @tf.function(autograph=False)
    def run(seed):
      dist_seed, *seeds = samplers.split_seed(seed, 11)
      dist = tfp_dist.Sharded(
          tfd.Sample(tfd.Normal(0., 1.), 3),
          shard_axis_name=self.axis_name)
      state = dist.sample(seed=dist_seed)
      kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
          EchoKernel(),
          tfp.experimental.stats.RunningVariance.from_stats(
              num_samples=10., mean=tf.zeros(3), variance=tf.ones(3)))
      pkr = kernel.bootstrap_results(state)
      def body(draw_pkr, seed):
        _, pkr = draw_pkr
        draw_seed, step_seed = samplers.split_seed(seed)
        draw = dist.sample(seed=draw_seed)
        _, pkr = kernel.one_step(draw, pkr, seed=step_seed)
        return draw, pkr

      (_, pkr), draws = mcmc_util.trace_scan(body,
                                             (tf.zeros(dist.event_shape), pkr),
                                             seeds, lambda v: v[0])
      return draws, pkr

    draws, pkr = self.strategy_run(run, (self.key,), in_axes=None)
    running_variance = self.per_replica_to_composite_tensor(
        pkr.running_variance[0])
    draws = self.per_replica_to_tensor(draws, axis=1)
    mean, sum_squared_residuals, draws = self.evaluate(
        (running_variance.mean, running_variance.sum_squared_residuals, draws))
    emp_mean = tf.reduce_sum(draws, axis=0) / 20.
    emp_squared_residuals = tf.reduce_sum(
        (draws - emp_mean[None, ...])**2, axis=0) + 10 * emp_mean**2 + 10
    self.assertAllClose(emp_mean, mean)
    self.assertAllClose(emp_squared_residuals, sum_squared_residuals)


if __name__ == '__main__':
  tf.test.main()
