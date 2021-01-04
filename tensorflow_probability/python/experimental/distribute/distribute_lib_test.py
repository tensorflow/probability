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
from tensorflow_probability.python.experimental.distribute import distribute_test_lib as test_lib
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class LogProbPartsTest(test_lib.DistributedTest):

  @test_util.disable_test_for_backend(
      disable_jax=True, reason='Behavior supported natively')
  def test_can_shard_values_across_logical_devices(self):

    @tf.function(autograph=False)
    def value_fn(ctx):
      return tf.cast(ctx.replica_id_in_sync_group, tf.float32)

    def add_one(x):
      return x + 1.

    values = self.strategy.experimental_distribute_values_from_function(
        value_fn)
    out_values = self.evaluate(
        self.per_replica_to_tensor(self.strategy_run(add_one, (values,))))
    self.assertAllEqual(out_values, [1., 2., 3., 4.])

  @test_util.disable_test_for_backend(
      disable_jax=True,
      reason='Cannot use sharded distributions outside of pmap.')
  def test_correct_log_prob_for_global_variable_no_strategy(self):
    data = tf.ones(4)

    def log_prob_parts(value):
      x, data = value
      return [
          tfd.Normal(0., 1.).log_prob(x),
          tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
      ]

    sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
        log_prob_parts, [False, True], axis_name=None)
    self.assertAllEqualNested(
        self.evaluate(sharded_log_prob_parts([tf.constant(0.), data])),
        self.evaluate([
            tfd.Normal(0., 1.).log_prob(0.),
            tf.reduce_sum(tfd.Normal(0., 1.).log_prob(data))
        ]))

  @test_util.disable_test_for_backend(
      disable_jax=True,
      reason='Cannot use sharded distributions outside of pmap.')
  def test_correct_log_prob_for_local_variable_no_strategy(self):

    data = tf.ones(4)

    def log_prob_parts(value):
      x, data = value
      return [
          tf.reduce_sum(tfd.Normal(0., 1.).log_prob(x)),
          tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
      ]

    sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
        log_prob_parts, [True, True], axis_name=None)
    self.assertAllEqualNested(
        self.evaluate(sharded_log_prob_parts([tf.ones(4), data])),
        self.evaluate([
            tf.reduce_sum(tfd.Normal(0., 1.).log_prob(tf.ones(4))),
            tf.reduce_sum(tfd.Normal(tf.ones(4), 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_global_variable(self):

    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            tfd.Normal(0., 1.).log_prob(x),
            tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
        ]

      sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
          log_prob_parts, [False, True], axis_name=self.axis_name)

      return sharded_log_prob_parts([x, data])

    x = tf.constant(0.)
    data = tf.ones(4)
    sharded_data = self.shard_values(data)
    out_parts = self.per_replica_to_tensor(
        self.strategy_run(run, (x, sharded_data), in_axes=(None, 0)))

    self.assertAllEqualNested(
        self.evaluate(out_parts),
        self.evaluate([
            tf.ones(4) * tfd.Normal(0., 1.).log_prob(0.),
            tf.ones(4) * tf.reduce_sum(tfd.Normal(0., 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_local_variable(self):

    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            tfd.Normal(0., 1.).log_prob(x),
            tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
        ]

      sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
          log_prob_parts, [True, True], axis_name=self.axis_name)

      return sharded_log_prob_parts([x, data])

    x = tf.zeros(4)
    sharded_x = self.shard_values(x)
    data = tf.ones(4)
    sharded_data = self.shard_values(data)
    out_parts = self.per_replica_to_tensor(
        self.strategy_run(run, (sharded_x, sharded_data)))

    self.assertAllEqualNested(
        self.evaluate(out_parts),
        self.evaluate([
            tf.ones(4) * tf.reduce_sum(tfd.Normal(0., 1.).log_prob(x)),
            tf.ones(4) * tf.reduce_sum(tfd.Normal(0., 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_global_and_local_variable(self):

    def run(w, x, data):

      def log_prob_parts(values):
        w, x, data = values
        return [
            tfd.Normal(0., 1.).log_prob(w),
            tfd.Normal(w, 1.).log_prob(x),
            tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
        ]

      sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
          log_prob_parts, [False, True, True], axis_name=self.axis_name)

      return sharded_log_prob_parts([w, x, data])

    w = tf.constant(1.)
    x = 2 * tf.ones(4)
    sharded_x = self.shard_values(x)
    data = 3 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_parts = self.per_replica_to_tensor(
        self.strategy_run(
            run, (w, sharded_x, sharded_data), in_axes=(None, 0, 0)))

    self.assertAllEqualNested(
        self.evaluate(out_parts),
        self.evaluate([
            tf.ones(4) * tfd.Normal(0., 1.).log_prob(w),
            tf.ones(4) * tf.reduce_sum(tfd.Normal(w, 1.).log_prob(x)),
            tf.ones(4) * tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data))
        ]))

  def test_correct_gradient_for_global_variable(self):

    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            tfd.Normal(0., 1.).log_prob(x),
            tfd.Normal(x, 1.).log_prob(data)
        ]

      def log_prob(x):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [False, True], axis_name=self.axis_name)
        parts = sharded_log_prob_parts([x, data])
        return tf.add_n(parts)

      return tfp.math.value_and_gradient(log_prob, x)[1]

    x = tf.constant(1.)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(run, (x, sharded_data), in_axes=(None, 0)))

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
            log_prob_parts, [True, True], axis_name=self.axis_name)
        parts = sharded_log_prob_parts([x, data])
        return tf.add_n(parts)

      return tfp.math.value_and_gradient(log_prob, x)[1]

    x = tf.range(4.)
    sharded_x = self.shard_values(x)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(run, (sharded_x, sharded_data)))

    def true_log_prob(x):
      return (tf.reduce_sum(tfd.Normal(0., 1.).log_prob(x)) +
              tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data)))

    true_grad = self.evaluate(tfp.math.value_and_gradient(true_log_prob, x)[1])

    self.assertAllEqualNested(self.evaluate(out_grads), true_grad)

  def test_correct_gradient_for_global_and_local_variable(self):

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
            log_prob_parts, [False, True, True], axis_name=self.axis_name)
        parts = sharded_log_prob_parts([w, x, data])
        return tf.add_n(parts)

      return tfp.math.value_and_gradient(log_prob, [w, x])[1]

    w = tf.constant(1.)
    x = tf.range(4.)
    sharded_x = self.shard_values(x)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(
            run, (w, sharded_x, sharded_data), in_axes=(None, 0, 0)))

    def true_log_prob(*value):
      w, x = value
      return (tfd.Normal(0., 1.).log_prob(w) +
              tf.reduce_sum(tfd.Normal(w, 1.).log_prob(x)) +
              tf.reduce_sum(tfd.Normal(x, 1.).log_prob(data)))

    true_grad = tfp.math.value_and_gradient(true_log_prob, [w, x])[1]
    true_grad[0] = tf.ones(4) * true_grad[0]

    self.assertAllEqualNested(
        self.evaluate(out_grads), self.evaluate(true_grad))

  def test_correct_gradient_for_global_and_local_variable_batched(self):

    def run(w, x, data):

      def log_prob_parts(value):
        w, x, data = value
        return [
            tf.reduce_sum(tfd.Normal(tf.zeros(2), 1.).log_prob(w), -1),
            # w is non-scalar, to detect spurious broadcasting.
            # The squares are to add some extra non-linearities.
            tfd.Normal(tf.reduce_sum(w, -1)**2, 1.).log_prob(x),
            tfd.Normal(x**2, 1.).log_prob(data)
        ]

      def log_prob(*value):
        w, x = value
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [False, True, True], axis_name=self.axis_name)
        parts = sharded_log_prob_parts([w, x, data])
        return tf.add_n(parts)

      return tfp.math.value_and_gradient(log_prob, [w, x])[1]

    batch_size = 3
    # We'll add an extra dimensions to w to make sure we get non-scalars inside
    # the distributed log_prob function.
    w = tf.ones([batch_size, 2])
    x = tf.tile(tf.range(4.)[tf.newaxis], [batch_size, 1])
    sharded_x = self.shard_values(x, axis=1)
    data = 2 * tf.ones([4])
    sharded_data = self.shard_values(data, axis=0)
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(
            run, (w, sharded_x, sharded_data), in_axes=(None, 1, 0)), axis=1)

    def true_log_prob(*value):
      w, x = value
      return (tf.reduce_sum(tfd.Normal(tf.zeros(2), 1.).log_prob(w), -1) +
              tf.reduce_sum(
                  tfd.Normal(tf.reduce_sum(w, -1, keepdims=True)**2,
                             1.).log_prob(x), -1) +
              tf.reduce_sum(tfd.Normal(x**2, 1.).log_prob(data), -1))

    true_grad = tfp.math.value_and_gradient(true_log_prob, [w, x])[1]
    true_grad[0] = tf.ones([batch_size, 4, 1]) * true_grad[0][:, tf.newaxis]

    self.assertAllEqualNested(
        self.evaluate(out_grads), self.evaluate(true_grad))

  def test_correct_gradient_for_global_and_local_variable_dict(self):

    @tf.function(autograph=False)
    def run(w, x, data):

      def log_prob_parts(value):
        return {
            'w': tfd.Normal(0., 1.).log_prob(value['w']),
            'x': tfd.Normal(value['w'], 1.).log_prob(value['x']),
            'data': tfd.Normal(value['x'], 1.).log_prob(value['data']),
        }

      def log_prob(*value):
        w, x = value
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, {'w': False, 'x': True, 'data': True},
            axis_name=self.axis_name)
        parts = sharded_log_prob_parts({'w': w, 'x': x, 'data': data})
        return tf.add_n(tf.nest.flatten(parts))

      return tfp.math.value_and_gradient(log_prob, [w, x])[1]

    w = tf.constant(1.)
    x = tf.range(4.)
    sharded_x = self.shard_values(x)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(run, (w, sharded_x, sharded_data),
                          in_axes=(None, 0, 0)))

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
  tf.test.main()
