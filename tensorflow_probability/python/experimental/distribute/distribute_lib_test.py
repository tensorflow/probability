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
"""Tests for tensorflow_probability.python.experimental.distribute.distribute_lib."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.distribute import distribute_lib
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions

NUM_DEVICES = 4


def per_replica_to_tensor(value):
  return tf.nest.map_structure(
      lambda per_replica: tf.stack(per_replica.values, axis=0), value)


@test_util.test_all_tf_execution_regimes
class LogProbPartsTest(test_util.TestCase):

  def setUp(self):
    super(LogProbPartsTest, self).setUp()
    self.strategy = tf.distribute.MirroredStrategy(
        devices=tf.config.list_logical_devices())

  def shard_values(self, values):

    def value_fn(ctx):
      return values[ctx.replica_id_in_sync_group]

    return self.strategy.experimental_distribute_values_from_function(value_fn)

  def test_can_shard_values_across_logical_devices(self):

    @tf.function(autograph=False)
    def value_fn(ctx):
      return tf.cast(ctx.replica_id_in_sync_group, tf.float32)

    def add_one(x):
      return x + 1.

    values = self.strategy.experimental_distribute_values_from_function(
        value_fn)
    out_values = self.evaluate(
        per_replica_to_tensor(self.strategy.run(add_one, (values,))))
    self.assertAllEqual(out_values, [1., 2., 3., 4.])

  def test_correct_log_prob_for_global_variable_no_strategy(self):
    data = tf.ones(4)

    def log_prob_parts(value):
      x, data = value
      return [
          tfd.Normal(0., 1.).log_prob(x),
          tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
      ]

    sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
        log_prob_parts, [False, True])
    self.assertAllEqualNested(
        self.evaluate(sharded_log_prob_parts([tf.constant(0.), data])),
        self.evaluate([
            tfd.Normal(0., 1.).log_prob(0.),
            tf.reduce_sum(tfd.Normal(0., 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_local_variable_no_strategy(self):

    data = tf.ones(4)

    def log_prob_parts(value):
      x, data = value
      return [
          tf.reduce_sum(tfd.Normal(0., 1.).log_prob(x)),
          tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
      ]

    sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
        log_prob_parts, [True, True])
    self.assertAllEqualNested(
        self.evaluate(sharded_log_prob_parts([tf.ones(4), data])),
        self.evaluate([
            tf.reduce_sum(tfd.Normal(0., 1.).log_prob(tf.ones(4))),
            tf.reduce_sum(tfd.Normal(tf.ones(4), 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_global_variable(self):

    @tf.function(autograph=False)
    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            tfd.Normal(0., 1.).log_prob(x),
            tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
        ]

      sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
          log_prob_parts, [False, True])

      return sharded_log_prob_parts([x, data])

    x = tf.constant(0.)
    data = tf.ones(4)
    sharded_data = self.shard_values(data)
    out_parts = per_replica_to_tensor(self.strategy.run(run, (x, sharded_data)))

    self.assertAllEqualNested(
        self.evaluate(out_parts),
        self.evaluate([
            tf.ones(4) * tfd.Normal(0., 1.).log_prob(0.),
            tf.ones(4) * tf.reduce_sum(tfd.Normal(0., 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_local_variable(self):

    @tf.function(autograph=False)
    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            tfd.Normal(0., 1.).log_prob(x),
            tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
        ]

      sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
          log_prob_parts, [True, True])

      return sharded_log_prob_parts([x, data])

    x = tf.zeros(4)
    sharded_x = self.shard_values(x)
    data = tf.ones(4)
    sharded_data = self.shard_values(data)
    out_parts = per_replica_to_tensor(
        self.strategy.run(run, (sharded_x, sharded_data)))

    self.assertAllEqualNested(
        self.evaluate(out_parts),
        self.evaluate([
            tf.ones(4) * tf.reduce_sum(tfd.Normal(0., 1.).log_prob(x)),
            tf.ones(4) * tf.reduce_sum(tfd.Normal(0., 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_global_and_local_variable(self):

    @tf.function(autograph=False)
    def run(w, x, data):

      def log_prob_parts(values):
        w, x, data = values
        return [
            tfd.Normal(0., 1.).log_prob(w),
            tfd.Normal(w, 1.).log_prob(x),
            tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
        ]

      sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
          log_prob_parts, [False, True, True])

      return sharded_log_prob_parts([w, x, data])

    w = tf.constant(1.)
    x = 2 * tf.ones(4)
    sharded_x = self.shard_values(x)
    data = 3 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_parts = per_replica_to_tensor(
        self.strategy.run(run, (w, sharded_x, sharded_data)))

    self.assertAllEqualNested(
        self.evaluate(out_parts),
        self.evaluate([
            tf.ones(4) * tfd.Normal(0., 1.).log_prob(w),
            tf.ones(4) * tf.reduce_sum(tfd.Normal(w, 1.).log_prob(x)),
            tf.ones(4) * tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
        ]))

  def test_correct_gradient_for_global_variable(self):

    @tf.function(autograph=False)
    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            tfd.Normal(0., 1.).log_prob(x),
            tfd.Normal(x, 1.).log_prob(data)
        ]

      def log_prob(x):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [False, True])
        parts = sharded_log_prob_parts([x, data])
        return tf.add_n(parts)

      return tfp.math.value_and_gradient(log_prob, x)[1]

    x = tf.constant(1.)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = per_replica_to_tensor(self.strategy.run(run, (x, sharded_data)))

    def true_log_prob(x):
      return (tfd.Normal(0., 1.).log_prob(x) +
              tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data)))

    true_grad = self.evaluate(tfp.math.value_and_gradient(true_log_prob, x)[1])

    self.assertAllEqualNested(self.evaluate(out_grads), tf.ones(4) * true_grad)

  def test_correct_gradient_for_local_variable(self):

    @tf.function(autograph=False)
    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            tfd.Normal(0., 1.).log_prob(x),
            tfd.Normal(x, 1.).log_prob(data)
        ]

      def log_prob(x):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [True, True])
        parts = sharded_log_prob_parts([x, data])
        return tf.add_n(parts)

      return tfp.math.value_and_gradient(log_prob, x)[1]

    x = tf.range(4.)
    sharded_x = self.shard_values(x)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = per_replica_to_tensor(self.strategy.run(run, (sharded_x,
                                                              sharded_data)))

    def true_log_prob(x):
      return (tf.reduce_sum(tfd.Normal(0., 1.).log_prob(x)) +
              tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data)))

    true_grad = self.evaluate(tfp.math.value_and_gradient(true_log_prob, x)[1])

    self.assertAllEqualNested(self.evaluate(out_grads), true_grad)

  def test_correct_gradient_for_global_and_local_variable(self):

    @tf.function(autograph=False)
    def run(w, x, data):

      def log_prob_parts(value):
        w, x, data = value
        return [
            tfd.Normal(0., 1.).log_prob(w),
            tfd.Normal(w, 1.).log_prob(x),
            tfd.Normal(x, 1.).log_prob(data)
        ]

      def log_prob(*value):
        w, x = value
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [False, True, True])
        parts = sharded_log_prob_parts([w, x, data])
        return tf.add_n(parts)

      return tfp.math.value_and_gradient(log_prob, [w, x])[1]

    w = tf.constant(1.)
    x = tf.range(4.)
    sharded_x = self.shard_values(x)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = per_replica_to_tensor(self.strategy.run(run, (w, sharded_x,
                                                              sharded_data)))

    def true_log_prob(*value):
      w, x = value
      return (tfd.Normal(0., 1.).log_prob(w) +
              tf.reduce_sum(tfd.Normal(w, 1.).log_prob(x)) +
              tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data)))

    true_grad = tfp.math.value_and_gradient(true_log_prob, [w, x])[1]
    true_grad[0] = tf.ones(4) * true_grad[0]

    self.assertAllEqualNested(self.evaluate(out_grads),
                              self.evaluate(true_grad))

  def test_correct_gradient_for_global_and_local_variable_dict(self):

    @tf.function(autograph=False)
    def run(w, x, data):

      def log_prob_parts(value):
        return {
            'w': tfd.Normal(0., 1.).log_prob(value['w']),
            'x': tfd.Normal(w, 1.).log_prob(value['x']),
            'data': tfd.Normal(x, 1.).log_prob(value['data']),
        }

      def log_prob(*value):
        w, x = value
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, {'w': False, 'x': True, 'data': True})
        parts = sharded_log_prob_parts({'w': w, 'x': x, 'data': data})
        return tf.add_n(tf.nest.flatten(parts))

      return tfp.math.value_and_gradient(log_prob, [w, x])[1]

    w = tf.constant(1.)
    x = tf.range(4.)
    sharded_x = self.shard_values(x)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = per_replica_to_tensor(self.strategy.run(run, (w, sharded_x,
                                                              sharded_data)))

    def true_log_prob(*value):
      w, x = value
      return (tfd.Normal(0., 1.).log_prob(w) +
              tf.reduce_sum(tfd.Normal(w, 1.).log_prob(x)) +
              tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data)))

    true_grad = tfp.math.value_and_gradient(true_log_prob, [w, x])[1]
    true_grad[0] = tf.ones(4) * true_grad[0]

    self.assertAllEqualNested(self.evaluate(out_grads),
                              self.evaluate(true_grad))

if __name__ == '__main__':
  tf.enable_v2_behavior()
  physical_devices = tf.config.experimental.list_physical_devices()

  num_logical_devices = 4
  tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0],
      [tf.config.experimental.VirtualDeviceConfiguration()] * NUM_DEVICES)
  tf.test.main()
