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
from tensorflow_probability.python.experimental.distribute import sharded
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions

NUM_DEVICES = 4


def per_replica_to_tensor(value):
  return tf.nest.map_structure(
      lambda per_replica: tf.stack(per_replica.values, axis=0), value)


class ShardedDistributionTest(test_util.TestCase):

  def setUp(self):
    super(ShardedDistributionTest, self).setUp()
    self.strategy = tf.distribute.MirroredStrategy(
        devices=tf.config.list_logical_devices())

  def test_sharded_sample_samples_differently_across_shards(self):

    @tf.function(autograph=False)
    def run(key):
      return sharded.ShardedSample(tfd.Normal(0., 1.),
                                   NUM_DEVICES).sample(seed=key)

    sample = self.evaluate(
        per_replica_to_tensor(self.strategy.run(run, (tf.zeros(2, tf.int32),))))
    for i in range(4):
      for j in range(4):
        if i == j:
          continue
        self.assertNotEqual(sample[i], sample[j])

  def test_sharded_independent_samples_differently_across_shards(self):

    @tf.function(autograph=False)
    def run(key):
      return sharded.ShardedIndependent(
          tfd.Normal(tf.zeros(1), tf.ones(1)), 1).sample(seed=key)

    sample = self.evaluate(
        per_replica_to_tensor(self.strategy.run(run, (tf.zeros(2, tf.int32),))))
    for i in range(4):
      for j in range(4):
        if i == j:
          continue
        self.assertNotEqual(sample[i], sample[j])


if __name__ == "__main__":
  tf.enable_v2_behavior()
  physical_devices = tf.config.experimental.list_physical_devices()

  num_logical_devices = 4
  tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0],
      [tf.config.experimental.VirtualDeviceConfiguration()] * NUM_DEVICES)
  tf.test.main()
