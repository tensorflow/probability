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
"""Tests for tensorflow_probability.python.experimental.distribute.joint_distribution."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.distribute import joint_distribution as jd
from tensorflow_probability.python.experimental.distribute import sharded
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions

NUM_DEVICES = 4


def per_replica_to_tensor(value):
  return tf.nest.map_structure(
      lambda per_replica: tf.stack(per_replica.values, axis=0), value)


def model_coroutine():
  w = yield tfd.JointDistributionCoroutine.Root(tfd.Normal(0., 1.))
  x = yield sharded.ShardedSample(tfd.Normal(w, 1.), NUM_DEVICES)
  yield sharded.ShardedIndependent(tfd.Normal(x, 1.), 1)


distributions = (
    ('coroutine', lambda: jd.JointDistributionCoroutine(model_coroutine)),
    ('sequential', lambda: jd.JointDistributionSequential([  # pylint: disable=g-long-lambda
        tfd.Normal(0., 1.),
        lambda w: sharded.ShardedSample(tfd.Normal(w, 1.), NUM_DEVICES),
        lambda x: sharded.ShardedIndependent(tfd.Normal(x, 1.), 1),
    ])),
    ('named', lambda: jd.JointDistributionNamed(  # pylint: disable=g-long-lambda
        dict(
            w=tfd.Normal(0., 1.),
            x=lambda w: sharded.ShardedSample(tfd.Normal(w, 1.), NUM_DEVICES),
            data=lambda x: sharded.ShardedIndependent(tfd.Normal(x, 1.), 1),
        ))),
)


@test_util.test_all_tf_execution_regimes
class JointDistributionTest(test_util.TestCase):

  def setUp(self):
    super(JointDistributionTest, self).setUp()
    self.strategy = tf.distribute.MirroredStrategy(
        devices=tf.config.list_logical_devices())

  def shard_values(self, values):

    def value_fn(ctx):
      return values[ctx.replica_id_in_sync_group]

    return self.strategy.experimental_distribute_values_from_function(value_fn)

  def test_get_sharded_distribution_coroutine(self):
    dist = distributions[0][1]()
    self.assertTupleEqual(dist.get_sharded_distributions(),
                          (False, True, True))

  def test_get_sharded_distribution_sequential(self):
    dist = distributions[1][1]()
    self.assertListEqual(dist.get_sharded_distributions(),
                         [False, True, True])

  def test_get_sharded_distribution_named(self):
    dist = distributions[2][1]()
    self.assertDictEqual(dist.get_sharded_distributions(),
                         dict(w=False, x=True, data=True))

  @parameterized.named_parameters(*distributions)
  def test_jd(self, dist_fn):
    dist = dist_fn()

    @tf.function(autograph=False)
    def run(key):
      sample = dist.sample(seed=key)
      # The identity is to prevent reparameterization gradients from kicking in.
      log_prob, (log_prob_grads,) = tfp.math.value_and_gradient(
          dist.log_prob, (tf.nest.map_structure(tf.identity, sample),))
      return sample, log_prob, log_prob_grads

    sample, log_prob, log_prob_grads = self.strategy.run(
        run, (tf.ones(2, tf.int32),))
    sample, log_prob, log_prob_grads = per_replica_to_tensor(
        (sample, log_prob, log_prob_grads))

    def true_log_prob_fn(w, x, data):
      return (tfd.Normal(0., 1.).log_prob(w) +
              tfd.Sample(tfd.Normal(w, 1.), (4, 1)).log_prob(x) +
              tfd.Independent(tfd.Normal(x, 1.), 2).log_prob(data))

    if isinstance(dist, jd.JointDistributionNamed):
      # N.B. the global RV 'w' gets replicated, so we grab any single replica's
      # result.
      w, x, data = sample['w'][0], sample['x'], sample['data']
      log_prob_grads = (log_prob_grads['w'][0], log_prob_grads['x'],
                        log_prob_grads['data'])
    else:
      w, x, data = sample[0][0], sample[1], sample[2]
      log_prob_grads = (log_prob_grads[0][0], log_prob_grads[1],
                        log_prob_grads[2])

    true_log_prob, true_log_prob_grads = tfp.math.value_and_gradient(
        true_log_prob_fn, (w, x, data))

    self.assertAllClose(
        self.evaluate(log_prob), self.evaluate(tf.ones(4) * true_log_prob))
    self.assertAllCloseNested(
        self.evaluate(log_prob_grads), self.evaluate(true_log_prob_grads))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  physical_devices = tf.config.experimental.list_physical_devices()

  num_logical_devices = 4
  tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0],
      [tf.config.experimental.VirtualDeviceConfiguration()] * NUM_DEVICES)
  tf.test.main()
