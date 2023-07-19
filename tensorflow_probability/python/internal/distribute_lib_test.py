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
import functools
import itertools

from absl.testing import parameterized

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib as test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient

JAX_MODE = False
NUMPY_MODE = False
TF_MODE = not (JAX_MODE or NUMPY_MODE)


if JAX_MODE:
  from jax import random  # pylint: disable=g-import-not-at-top


def _allow_all_gather(fn):
  return functools.partial(fn, allow_all_gather=True)


@test_util.test_all_tf_execution_regimes
class CollectiveTest(test_lib.DistributedTest):

  def test_tf_should_error_with_more_than_one_named_axis(self):
    if JAX_MODE:
      self.skipTest('Test only applies to TF backend.')
    with self.assertRaisesRegex(
        ValueError, 'TensorFlow backend does not support multiple shard axes'):
      distribute_lib.canonicalize_named_axis(['a', 'b'])

  @parameterized.named_parameters(
      ('sum', tf.reduce_sum, distribute_lib.reduce_sum),
      ('mean', tf.reduce_mean, distribute_lib.reduce_mean),
      ('max', tf.reduce_max, _allow_all_gather(distribute_lib.reduce_max),
       True),
      ('min', tf.reduce_min, _allow_all_gather(distribute_lib.reduce_min),
       True),
      ('logsumexp', tf.reduce_logsumexp,
       _allow_all_gather(distribute_lib.reduce_logsumexp), True))
  def test_distributed_reduce_works_as_normal_with_int_axes(
      self, reduce_op, distributed_op, skip_on_eager=False):
    if skip_on_eager and (tf.executing_eagerly() and TF_MODE):
      self.skipTest('Not supported in Eager.')
    x = tf.reshape(
        tf.range(test_lib.NUM_DEVICES * 6.) / 5., [test_lib.NUM_DEVICES, 3, 2])

    def make_run(axis):
      return lambda x: distributed_op(x, axis=axis)

    for axes in [0, 1, None, [0, 1]]:
      reduce_out = reduce_op(x, axis=axes)
      dist_out = make_run(axes)(x)
      self.assertAllEqual(reduce_out, dist_out)

  @parameterized.named_parameters(*(
      (f'{name} {ax}', *args, ax)  # pylint: disable=g-complex-comprehension
      for (name, *args), ax in itertools.product((
          ('sum', tf.reduce_sum, distribute_lib.reduce_sum, False),
          ('mean', tf.reduce_mean, distribute_lib.reduce_mean, False),
          ('max', tf.reduce_max, _allow_all_gather(distribute_lib.reduce_max),
           True),
          ('min', tf.reduce_min, _allow_all_gather(distribute_lib.reduce_min),
           True),
          ('logsumexp', tf.reduce_logsumexp,
           _allow_all_gather(distribute_lib.reduce_logsumexp), True)), (
               None, 0, 1, 2, [0, 1], [1, 2], [0, 2], [0, 1, 2]))))
  def test_reduce_with_collectives_matches_reduce_without_collectives(
      self, reduce_op, distributed_op, skip_on_eager, axes):
    if skip_on_eager and (tf.executing_eagerly() and TF_MODE):
      self.skipTest('Not supported in Eager.')
    x = tf.reshape(
        tf.range(test_lib.NUM_DEVICES * 6.) / 5., [test_lib.NUM_DEVICES, 3, 2])

    def run(x):
      return distributed_op(
          x, axis=pos_axes, named_axis=named_axes)

    def distributed_run(x):
      return self.per_replica_to_tensor(
          self.strategy_run(run, (self.shard_values(x),)))

    if axes is None:
      pos_axes = list(range(2))
      named_axes = [self.axis_name]
    else:
      axes = [axes] if not isinstance(axes, list) else axes
      pos_axes = [x - 1 for x in axes if x != 0]
      named_axes = [self.axis_name] if 0 in axes else []
    reduce_out = reduce_op(x, axis=axes)
    dist_out = distributed_run(x)
    # If we reduce over the 0 dimension, it will still be present in the
    # distributed op
    if axes is None or 0 in axes:
      for i in range(test_lib.NUM_DEVICES):
        self.assertAllClose(reduce_out, dist_out[i])
    else:
      self.assertAllClose(reduce_out, dist_out)

  @parameterized.named_parameters(
      ('sum', tf.reduce_sum, distribute_lib.reduce_sum, True),
      ('mean', tf.reduce_mean, distribute_lib.reduce_mean, True),
      ('max', tf.reduce_max, _allow_all_gather(
          distribute_lib.reduce_max), False, True),
      ('min', tf.reduce_min, _allow_all_gather(distribute_lib.reduce_min),
       False, True),
      ('logsumexp', tf.reduce_logsumexp,
       _allow_all_gather(distribute_lib.reduce_logsumexp), True, True))
  def test_reduce_with_collective_grads_matches_without_collectives(
      self, reduce_op, distributed_op, is_supported, skip_on_eager=False):
    if not is_supported:
      self.skipTest('Gradient of operation not supported.')
    if skip_on_eager and (tf.executing_eagerly() and TF_MODE):
      self.skipTest('Not supported in Eager.')
    x = tf.reshape(
        tf.range(test_lib.NUM_DEVICES * 6.) / 5., [test_lib.NUM_DEVICES, 3, 2])

    def compute_dist_grads(x):
      return gradient.value_and_gradient(
          lambda x: distributed_op(  # pylint: disable=g-long-lambda
              x,
              axis=[0, 1],
              named_axis=self.axis_name),
          [x])[1][0]

    def distributed_run(x):
      return self.per_replica_to_tensor(
          self.strategy_run(compute_dist_grads, (self.shard_values(x),)))

    reduce_grads = gradient.value_and_gradient(
        lambda x: reduce_op(x, axis=None), [x])[1][0]
    dist_grads = distributed_run(x)
    self.assertAllClose(reduce_grads, dist_grads)


@test_util.test_all_tf_execution_regimes
class ShardedFunctionTest(test_lib.DistributedTest):

  def test_psum_unary_function_applies_psum_to_outputs(self):

    def f(x):
      return x

    f = distribute_lib.make_psum_function(
        f, self.axis_name, self.axis_name, out_dtype=tf.float32)

    x = self.shard_values(tf.ones(4))
    out_parts = self.per_replica_to_tensor(self.strategy_run(f, (x,)))

    self.assertAllEqual(self.evaluate(out_parts), self.evaluate(4 * tf.ones(4)))

  def test_psum_binary_function_applies_psum_to_outputs(self):

    def f(x, y):
      return x + y

    f_psum = distribute_lib.make_psum_function(
        f, (self.axis_name, self.axis_name),
        self.axis_name,
        out_dtype=tf.float32)

    x = self.shard_values(tf.ones(4))
    y = self.shard_values(2 * tf.ones(4))
    out_parts = self.per_replica_to_tensor(self.strategy_run(f_psum, (x, y)))

    self.assertAllEqual(
        self.evaluate(out_parts), self.evaluate(12 * tf.ones(4)))

    f_psum = distribute_lib.make_psum_function(
        f, (self.axis_name, self.axis_name), None, out_dtype=tf.float32)

    x = self.shard_values(tf.ones(4))
    y = self.shard_values(2 * tf.ones(4))
    out_parts = self.per_replica_to_tensor(self.strategy_run(f_psum, (x, y)))

    self.assertAllEqual(self.evaluate(out_parts), self.evaluate(3 * tf.ones(4)))

    f_psum = distribute_lib.make_psum_function(
        f, (self.axis_name, self.axis_name), None, out_dtype=tf.float32)

    x = self.shard_values(tf.ones(4))
    y = self.shard_values(2 * tf.ones(4))
    out_parts = self.per_replica_to_tensor(
        self.strategy_run(f_psum, (x, y), in_axes=(0, 0)))

    self.assertAllEqual(self.evaluate(out_parts), self.evaluate(3 * tf.ones(4)))

    f_psum = distribute_lib.make_psum_function(
        f, (self.axis_name, None), None, out_dtype=tf.float32)

    x = self.shard_values(tf.ones(4))
    y = 2.
    out_parts = self.per_replica_to_tensor(
        self.strategy_run(f_psum, (x, y), in_axes=(0, None)))

    self.assertAllEqual(self.evaluate(out_parts), self.evaluate(3 * tf.ones(4)))

  def test_psum_binary_function_corrects_gradients_to_inputs(self):

    def f(x, y):
      return x * y

    f_psum = distribute_lib.make_psum_function(
        f, (self.axis_name, self.axis_name),
        self.axis_name,
        out_dtype=tf.float32)

    def f_grad(x, y):
      return gradient.value_and_gradient(f_psum, (x, y))[1]

    x = self.shard_values(tf.ones(4))
    y = self.shard_values(2. * tf.range(4.))
    out_grads = self.per_replica_to_tensor(self.strategy_run(f_grad, (x, y)))

    self.assertAllEqual(self.evaluate(out_grads[0]), 2. * tf.range(4.))
    self.assertAllEqual(self.evaluate(out_grads[1]), tf.ones(4))

    f_psum = distribute_lib.make_psum_function(
        f, (self.axis_name, None), self.axis_name, out_dtype=tf.float32)

    def f_grad2(x, y):
      return gradient.value_and_gradient(f_psum, (x, y))[1]

    x = self.shard_values(tf.range(4.))
    y = 2.
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(f_grad2, (x, y), in_axes=(0, None)))

    self.assertAllEqual(self.evaluate(out_grads[0]), 2 * tf.ones(4))
    self.assertAllEqual(self.evaluate(out_grads[1]), 6 * tf.ones(4))

    f_psum = distribute_lib.make_psum_function(
        f, (self.axis_name, self.axis_name), None, out_dtype=tf.float32)

    def f_grad3(x, y):
      return gradient.value_and_gradient(f_psum, (x, y))[1]

    x = self.shard_values(tf.range(4.))
    y = self.shard_values(tf.ones(4))
    out_grads = self.per_replica_to_tensor(self.strategy_run(f_grad3, (x, y)))

    self.assertAllEqual(self.evaluate(out_grads[0]), tf.ones(4))
    self.assertAllEqual(self.evaluate(out_grads[1]), tf.range(4.))

    f_psum = distribute_lib.make_psum_function(
        f, (self.axis_name, None), None, out_dtype=tf.float32)

    def f_grad4(x, y):
      return gradient.value_and_gradient(f_psum, (x, y))[1]

    x = self.shard_values(tf.range(4.))
    y = 2.
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(f_grad4, (x, y), in_axes=(0, None)))

    self.assertAllEqual(self.evaluate(out_grads[0]), 2 * tf.ones(4))
    self.assertAllEqual(self.evaluate(out_grads[1]), tf.range(4.))


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

    values = self.strategy().experimental_distribute_values_from_function(
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
          normal.Normal(0., 1.).log_prob(x),
          tf.reduce_sum(normal.Normal(x, 1.).log_prob(data))
      ]

    sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
        log_prob_parts, [None, True])
    self.assertAllEqualNested(
        self.evaluate(sharded_log_prob_parts([tf.constant(0.), data])),
        self.evaluate([
            normal.Normal(0., 1.).log_prob(0.),
            tf.reduce_sum(normal.Normal(0., 1.).log_prob(data))
        ]))

  @test_util.disable_test_for_backend(
      disable_jax=True,
      reason='Cannot use sharded distributions outside of pmap.')
  def test_correct_log_prob_for_local_variable_no_strategy(self):

    data = tf.ones(4)

    def log_prob_parts(value):
      x, data = value
      return [
          tf.reduce_sum(normal.Normal(0., 1.).log_prob(x)),
          tf.reduce_sum(normal.Normal(x, 1.).log_prob(data))
      ]

    sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
        log_prob_parts, [True, True])
    self.assertAllEqualNested(
        self.evaluate(sharded_log_prob_parts([tf.ones(4), data])),
        self.evaluate([
            tf.reduce_sum(normal.Normal(0., 1.).log_prob(tf.ones(4))),
            tf.reduce_sum(normal.Normal(tf.ones(4), 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_global_variable(self):

    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            normal.Normal(0., 1.).log_prob(x),
            tf.reduce_sum(normal.Normal(x, 1.).log_prob(data))
        ]

      sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
          log_prob_parts, [None, self.axis_name])

      return sharded_log_prob_parts([x, data])

    x = tf.constant(0.)
    data = tf.ones(4)
    sharded_data = self.shard_values(data)
    out_parts = self.per_replica_to_tensor(
        self.strategy_run(run, (x, sharded_data), in_axes=(None, 0)))

    self.assertAllEqualNested(
        self.evaluate(out_parts),
        self.evaluate([
            tf.ones(4) * normal.Normal(0., 1.).log_prob(0.),
            tf.ones(4) * tf.reduce_sum(normal.Normal(0., 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_local_variable(self):

    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            normal.Normal(0., 1.).log_prob(x),
            tf.reduce_sum(normal.Normal(x, 1.).log_prob(data))
        ]

      sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
          log_prob_parts, [self.axis_name, self.axis_name])

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
            tf.ones(4) * tf.reduce_sum(normal.Normal(0., 1.).log_prob(x)),
            tf.ones(4) * tf.reduce_sum(normal.Normal(0., 1.).log_prob(data))
        ]))

  def test_correct_log_prob_for_global_and_local_variable(self):

    def run(w, x, data):

      def log_prob_parts(values):
        w, x, data = values
        return [
            normal.Normal(0., 1.).log_prob(w),
            normal.Normal(w, 1.).log_prob(x),
            tf.reduce_sum(normal.Normal(x, 1.).log_prob(data))
        ]

      sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
          log_prob_parts, [None, self.axis_name, self.axis_name])

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
            tf.ones(4) * normal.Normal(0., 1.).log_prob(w),
            tf.ones(4) * tf.reduce_sum(normal.Normal(w, 1.).log_prob(x)),
            tf.ones(4) * tf.reduce_sum(normal.Normal(x, 1.).log_prob(data))
        ]))

  def test_correct_gradient_for_global_variable(self):

    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            normal.Normal(0., 1.).log_prob(x),
            normal.Normal(x, 1.).log_prob(data)
        ]

      def log_prob(x):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [None, self.axis_name])
        parts = sharded_log_prob_parts([x, data])
        return tf.add_n(parts)

      return gradient.value_and_gradient(log_prob, x)[1]

    x = tf.constant(1.)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(run, (x, sharded_data), in_axes=(None, 0)))

    def true_log_prob(x):
      return (normal.Normal(0., 1.).log_prob(x) +
              tf.reduce_sum(normal.Normal(x, 1.).log_prob(data)))

    true_grad = self.evaluate(gradient.value_and_gradient(true_log_prob, x)[1])

    self.assertAllEqualNested(self.evaluate(out_grads), tf.ones(4) * true_grad)

  def test_correct_gradient_for_local_variable(self):

    @tf.function(autograph=False)
    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            normal.Normal(0., 1.).log_prob(x),
            normal.Normal(x, 1.).log_prob(data)
        ]

      def log_prob(x):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [self.axis_name, self.axis_name])
        parts = sharded_log_prob_parts([x, data])
        return tf.add_n(parts)

      return gradient.value_and_gradient(log_prob, x)[1]

    x = tf.range(4.)
    sharded_x = self.shard_values(x)
    data = 2 * tf.ones(4)
    sharded_data = self.shard_values(data)
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(run, (sharded_x, sharded_data)))

    def true_log_prob(x):
      return (tf.reduce_sum(normal.Normal(0., 1.).log_prob(x)) +
              tf.reduce_sum(normal.Normal(x, 1.).log_prob(data)))

    true_grad = self.evaluate(gradient.value_and_gradient(true_log_prob, x)[1])

    self.assertAllEqualNested(self.evaluate(out_grads), true_grad)

  def test_correct_gradient_for_global_and_local_variable(self):

    def run(w, x, data):

      def log_prob_parts(value):
        w, x, data = value
        return [
            normal.Normal(0., 1.).log_prob(w),
            normal.Normal(w, 1.).log_prob(x),
            normal.Normal(x, 1.).log_prob(data)
        ]

      def log_prob(*value):
        w, x = value
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [None, self.axis_name, self.axis_name])
        parts = sharded_log_prob_parts([w, x, data])
        return tf.add_n(parts)

      return gradient.value_and_gradient(log_prob, [w, x])[1]

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
      return (normal.Normal(0., 1.).log_prob(w) +
              tf.reduce_sum(normal.Normal(w, 1.).log_prob(x)) +
              tf.reduce_sum(normal.Normal(x, 1.).log_prob(data)))

    true_grad = gradient.value_and_gradient(true_log_prob, [w, x])[1]
    true_grad[0] = tf.ones(4) * true_grad[0]

    self.assertAllEqualNested(
        self.evaluate(out_grads), self.evaluate(true_grad))

  def test_correct_gradient_for_global_and_local_variable_batched(self):

    def run(w, x, data):

      def log_prob_parts(value):
        w, x, data = value
        return [
            tf.reduce_sum(normal.Normal(tf.zeros(2), 1.).log_prob(w), -1),
            # w is non-scalar, to detect spurious broadcasting.
            # The squares are to add some extra non-linearities.
            normal.Normal(tf.reduce_sum(w, -1)**2, 1.).log_prob(x),
            normal.Normal(x**2, 1.).log_prob(data)
        ]

      def log_prob(*value):
        w, x = value
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [None, self.axis_name, self.axis_name])
        parts = sharded_log_prob_parts([w, x, data])
        return tf.add_n(parts)

      return gradient.value_and_gradient(log_prob, [w, x])[1]

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
            run, (w, sharded_x, sharded_data), in_axes=(None, 1, 0)),
        axis=1)

    def true_log_prob(*value):
      w, x = value
      return (tf.reduce_sum(normal.Normal(tf.zeros(2), 1.).log_prob(w), -1) +
              tf.reduce_sum(
                  normal.Normal(tf.reduce_sum(w, -1, keepdims=True)**2,
                                1.).log_prob(x), -1) +
              tf.reduce_sum(normal.Normal(x**2, 1.).log_prob(data), -1))

    true_grad = gradient.value_and_gradient(true_log_prob, [w, x])[1]
    true_grad[0] = tf.ones([batch_size, 4, 1]) * true_grad[0][:, tf.newaxis]

    self.assertAllEqualNested(
        self.evaluate(out_grads), self.evaluate(true_grad))

  def test_correct_gradient_for_global_and_local_variable_dict(self):

    @tf.function(autograph=False)
    def run(w, x, data):

      def log_prob_parts(value):
        return {
            'w': normal.Normal(0., 1.).log_prob(value['w']),
            'x': normal.Normal(value['w'], 1.).log_prob(value['x']),
            'data': normal.Normal(value['x'], 1.).log_prob(value['data']),
        }

      def log_prob(*value):
        w, x = value
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, {
                'w': None,
                'x': self.axis_name,
                'data': self.axis_name
            })
        parts = sharded_log_prob_parts({'w': w, 'x': x, 'data': data})
        return tf.add_n(tf.nest.flatten(parts))

      return gradient.value_and_gradient(log_prob, [w, x])[1]

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
      return (normal.Normal(0., 1.).log_prob(w) +
              tf.reduce_sum(normal.Normal(w, 1.).log_prob(x)) +
              tf.reduce_sum(normal.Normal(x, 1.).log_prob(data)))

    true_grad = gradient.value_and_gradient(true_log_prob, [w, x])[1]
    true_grad[0] = tf.ones(4) * true_grad[0]

    self.assertAllEqualNested(
        self.evaluate(out_grads), self.evaluate(true_grad))

  def test_correct_gradient_for_local_integer_variable(self):

    @tf.function(autograph=False)
    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            normal.Normal(0., 1.).log_prob(x),
            bernoulli.Bernoulli(logits=x).log_prob(data)
        ]

      def log_prob(x):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [self.axis_name, self.axis_name])
        parts = sharded_log_prob_parts([x, data])
        return tf.add_n(parts)

      return gradient.value_and_gradient(log_prob, x)[1]

    x = tf.range(4.)
    sharded_x = self.shard_values(x)
    data = tf.ones(4, tf.int32)
    sharded_data = self.shard_values(data)
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(run, (sharded_x, sharded_data)))

    def true_log_prob(x):
      return (tf.reduce_sum(normal.Normal(0., 1.).log_prob(x)) +
              tf.reduce_sum(bernoulli.Bernoulli(logits=x).log_prob(data)))

    true_grad = self.evaluate(gradient.value_and_gradient(true_log_prob, x)[1])

    self.assertAllEqualNested(self.evaluate(out_grads), true_grad)

  def test_correct_gradient_dtype_for_disconnected_variables(self):

    @tf.function(autograph=False)
    def run(x, y):

      def log_prob_parts(value):
        x, y = value
        return [
            # These two RV's do not depend on each other.
            normal.Normal(0., 1.).log_prob(x),
            normal.Normal(0., 1.).log_prob(y),
        ]

      def log_prob(x, y):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [None, self.axis_name])
        parts = sharded_log_prob_parts([x, y])
        return tf.add_n(parts)

      return gradient.value_and_gradient(log_prob, (x, y))[1]

    sharded_x = self.shard_values(tf.range(4.))
    sharded_y = sharded_x
    out_grads = self.per_replica_to_tensor(
        self.strategy_run(run, (sharded_x, sharded_y)))
    self.assertEqual(tf.float32, out_grads[0].dtype)
    self.assertEqual(tf.float32, out_grads[1].dtype)

  def test_multiple_shard_axes(self):
    if not JAX_MODE:
      self.skipTest('Multiple shard axes not supported in TF.')

    other_axis_name = self.axis_name + '_other'

    def run(x, data1, data2):

      def log_prob_parts(value):
        x, data1, data2 = value
        return [
            normal.Normal(0., 1.).log_prob(x),
            normal.Normal(x, 1.).log_prob(data1),
            normal.Normal(x, 1.).log_prob(data2)
        ]

      def log_prob(x, data1, data2):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [None, self.axis_name, other_axis_name])
        parts = sharded_log_prob_parts([x, data1, data2])
        return tf.add_n(parts)

      return gradient.value_and_gradient(log_prob, (x, data1, data2))

    x = tf.constant(1.)
    data1 = 2 * tf.ones(2)
    data2 = 3 * tf.ones(2)

    def outer_run(x, data1):
      return self.strategy_run(
          run, (x, data1, data2),
          in_axes=(None, None, 0),
          axis_name=other_axis_name)

    out_values, out_grads = self.strategy_run(
        outer_run, (x, data1), in_axes=(None, 0), axis_name=self.axis_name)

    def true_log_prob(x, data1, data2):
      return (normal.Normal(0., 1.).log_prob(x) +
              tf.reduce_sum(normal.Normal(x, 1.).log_prob(data1)) +
              tf.reduce_sum(normal.Normal(x, 1.).log_prob(data2)))

    true_values, true_grads = self.evaluate(
        gradient.value_and_gradient(true_log_prob, (x, data1, data2)))

    self.assertAllEqualNested(out_values, tf.ones([2, 2]) * true_values)
    self.assertAllEqualNested(out_grads[0], tf.ones([2, 2]) * true_grads[0])
    self.assertAllEqualNested(out_grads[1], tf.ones([2, 2]) * true_grads[1])
    self.assertAllEqualNested(out_grads[2], tf.ones([2, 2]) * true_grads[2])

  def test_nested_shard_axes(self):
    if not JAX_MODE:
      self.skipTest('Multiple shard axes not supported in TF.')

    other_axis_name = self.axis_name + '_other'

    def run(x, data):

      def log_prob_parts(value):
        x, data = value
        return [
            normal.Normal(0., 1.).log_prob(x),
            normal.Normal(x, 1.).log_prob(data),
        ]

      def log_prob(x, data):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [None, {self.axis_name, other_axis_name}])
        parts = sharded_log_prob_parts([x, data])
        return tf.add_n(parts)

      return gradient.value_and_gradient(log_prob, (x, data))

    x = tf.constant(1.)
    data = 2 * tf.ones([2, 2])

    def outer_run(x, data):
      return self.strategy_run(
          run, (x, data), in_axes=(None, 0), axis_name=other_axis_name)

    out_values, out_grads = self.strategy_run(
        outer_run, (x, data), in_axes=(None, 0))

    def true_log_prob(x, data):
      return (normal.Normal(0., 1.).log_prob(x) +
              tf.reduce_sum(normal.Normal(x, 1.).log_prob(data)))

    true_values, true_grads = self.evaluate(
        gradient.value_and_gradient(true_log_prob, (x, data)))

    self.assertAllEqualNested(out_values, tf.ones([2, 2]) * true_values)
    self.assertAllEqualNested(out_grads[0], tf.ones([2, 2]) * true_grads[0])
    self.assertAllEqualNested(out_grads[1], tf.ones([2, 2]) * true_grads[1])

  def test_gradient_is_correctly_reduced_with_multiple_axes(self):
    if not JAX_MODE:
      self.skipTest('Multiple shard axes not supported in TF.')

    other_axis_name = self.axis_name + '_other'

    def run(x, y, z):

      def log_prob_parts(value):
        x, y, z = value
        return [
            normal.Normal(0., 1.).log_prob(x),
            normal.Normal(0., 1.).log_prob(y),
            normal.Normal(x + y, 1.).log_prob(z),
        ]

      def log_prob(x, y, z):
        sharded_log_prob_parts = distribute_lib.make_sharded_log_prob_parts(
            log_prob_parts, [
                self.axis_name, other_axis_name,
                [self.axis_name, other_axis_name]
            ])
        parts = sharded_log_prob_parts([x, y, z])
        return tf.add_n(parts)

      return gradient.value_and_gradient(log_prob, (x, y, z))

    seed = random.PRNGKey(0)
    x_seed, y_seed, z_seed = samplers.split_seed(seed, n=3)
    x = normal.Normal(0., 1).sample(seed=x_seed, sample_shape=2)
    y = normal.Normal(0., 1.).sample(seed=y_seed, sample_shape=2)
    z = normal.Normal(0., 1.).sample(seed=z_seed, sample_shape=[2, 2])

    def outer_run(x, y, z):
      return self.strategy_run(
          run, (x, y, z), in_axes=(None, 0, 0), axis_name=other_axis_name)

    out_values, out_grads = self.strategy_run(
        outer_run, (x, y, z), in_axes=(0, None, 0))

    def true_log_prob(x, y, z):
      return (
          tf.reduce_sum(normal.Normal(0., 1.).log_prob(x)) +
          tf.reduce_sum(normal.Normal(0., 1.).log_prob(y)) +
          tf.reduce_sum(normal.Normal(x[:, None] + y[None], 1.).log_prob(z)))

    true_values, true_grads = self.evaluate(
        gradient.value_and_gradient(true_log_prob, (x, y, z)))

    self.assertAllClose(
        out_values, tf.ones([2, 2]) * true_values, rtol=1e-6, atol=1e-6)
    self.assertAllEqualNested(out_grads[0],
                              tf.ones([2, 2]) * true_grads[0][:, None])
    self.assertAllEqualNested(out_grads[1],
                              tf.ones([2, 2]) * true_grads[1][None])
    self.assertAllEqualNested(out_grads[2], tf.ones([2, 2]) * true_grads[2])


if __name__ == '__main__':
  test_util.main()
