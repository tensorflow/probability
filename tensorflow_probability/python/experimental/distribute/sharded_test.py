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
# limitations under the License.
# ============================================================================
"""Tests for tensorflow_probability.python.experimental.distribute.sharded."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.distribute import distribute_test_lib as test_lib
from tensorflow_probability.python.experimental.distribute import sharded
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfde = tfp.experimental.distributions
tfp_dist = tfp.experimental.distribute


@test_util.test_all_tf_execution_regimes
class ShardedDistributionTest(test_lib.DistributedTest):

  def test_sharded_sample_samples_differently_across_shards(self):

    @tf.function(autograph=False)
    def run(key):
      return sharded.ShardedSample(
          tfd.Normal(0., 1.),
          test_lib.NUM_DEVICES,
          shard_axis_name=self.axis_name).sample(seed=key)

    sample = self.evaluate(
        self.per_replica_to_tensor(
            self.strategy_run(run, (self.key,), in_axes=None)))
    for i in range(4):
      for j in range(4):
        if i == j:
          continue
        self.assertNotEqual(sample[i], sample[j])

  def test_sharded_independent_samples_differently_across_shards(self):

    @tf.function(autograph=False)
    def run(key):
      return sharded.ShardedIndependent(
          tfd.Normal(tf.zeros(1), tf.ones(1)),
          1,
          shard_axis_name=self.axis_name).sample(seed=key)

    sample = self.evaluate(
        self.per_replica_to_tensor(
            self.strategy_run(run, (self.key,), in_axes=None)))
    for i in range(4):
      for j in range(4):
        if i == j:
          continue
        self.assertNotEqual(sample[i], sample[j])

  def test_kahan_custom_grad(self):
    def model_fn():
      root = tfp.experimental.distribute.JointDistributionCoroutine.Root
      _ = yield root(tfp.experimental.distribute.ShardedIndependent(
          tfd.Normal(0, tf.ones([7])),
          reinterpreted_batch_ndims=1,
          experimental_use_kahan_sum=True,
          shard_axis_name=self.axis_name))
    model = tfp.experimental.distribute.JointDistributionCoroutine(
        model_fn, shard_axis_name=self.axis_name)

    samps = self.strategy_run(lambda seed: model.sample(seed=seed),
                              (test_util.test_seed(sampler_type='stateless'),),
                              in_axes=None)

    @tf.function(jit_compile=True)
    def lp_grad(x):
      return tfp.math.value_and_gradient(model.log_prob, x)

    self.evaluate(
        self.per_replica_to_tensor(self.strategy_run(lp_grad, (samps,))))

  def test_log_prob_ratio_sample(self):
    dist = tfd.Sample(tfd.Normal(0., 1.), 8)

    x0 = 0.
    x1 = 1.

    oracle_value = tfde.log_prob_ratio(dist, x0, dist, x1)

    dist_sharded = tfp_dist.ShardedSample(
        tfd.Normal(0., 1.), 8, shard_axis_name=self.axis_name)
    @tf.function
    def test():
      return (tfde.log_prob_ratio(dist_sharded, x0, dist_sharded, x1),
              dist_sharded.log_prob(x0) - dist_sharded.log_prob(x1))

    lp1, lp2 = self.strategy_run(test, in_axes=None)
    sharded_ratio = self.per_replica_to_tensor(lp1)
    sharded_subtraction = self.per_replica_to_tensor(lp2)

    self.assertAllClose(oracle_value + tf.zeros_like(sharded_ratio),
                        sharded_ratio)
    self.assertAllClose(oracle_value + tf.zeros_like(sharded_subtraction),
                        sharded_subtraction)

  def test_log_prob_ratio_independent(self):
    dist = tfd.Independent(tfd.Normal(tf.zeros([2 * test_lib.NUM_DEVICES]), 1.),
                           reinterpreted_batch_ndims=1)

    x0 = 0.
    x1 = 1.

    oracle_value = tfde.log_prob_ratio(dist, x0, dist, x1)

    dist_sharded = tfp_dist.ShardedIndependent(
        tfd.Normal(tf.zeros([2]), 1.),
        reinterpreted_batch_ndims=1,
        shard_axis_name=self.axis_name)
    @tf.function
    def test():
      return (tfde.log_prob_ratio(dist_sharded, x0, dist_sharded, x1),
              dist_sharded.log_prob(x0) - dist_sharded.log_prob(x1))

    lp1, lp2 = self.strategy_run(test, in_axes=None)
    sharded_ratio = self.per_replica_to_tensor(lp1)
    sharded_subtraction = self.per_replica_to_tensor(lp2)

    self.assertAllClose(oracle_value + tf.zeros_like(sharded_ratio),
                        sharded_ratio)
    self.assertAllClose(oracle_value + tf.zeros_like(sharded_subtraction),
                        sharded_subtraction)


if __name__ == '__main__':
  tf.test.main()
