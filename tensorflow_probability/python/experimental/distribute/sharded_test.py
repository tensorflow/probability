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
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.experimental.distribute import joint_distribution
from tensorflow_probability.python.experimental.distribute import sharded
from tensorflow_probability.python.experimental.distributions import increment_log_prob
from tensorflow_probability.python.internal import distribute_test_lib as test_lib
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class ShardTest(test_lib.DistributedTest):

  def test_sharded_sample_samples_differently_across_shards(self):

    @tf.function(autograph=False)
    def run(key):
      return sharded.Sharded(
          sample_dist_lib.Sample(normal.Normal(0., 1.), 1),
          shard_axis_name=self.axis_name).sample(seed=key)

    sample = self.evaluate(
        self.per_replica_to_tensor(
            self.strategy_run(run, (self.key,), in_axes=None)))
    for i in range(4):
      for j in range(4):
        if i == j:
          continue
        self.assertNotEqual(sample[i], sample[j])

  def test_sharded_samples_differently_with_nested_axes(self):
    if not JAX_MODE:
      self.skipTest('Multiple axes only supported in JAX backend.')

    other_axis_name = self.axis_name + '_other'

    def run(key, _):
      return sharded.Sharded(
          sample_dist_lib.Sample(normal.Normal(0., 1.), 1),
          shard_axis_name=[self.axis_name, other_axis_name]).sample(seed=key)

    def outer_run(key, x):
      return self.strategy_run(run, (key, x),
                               axis_name=other_axis_name,
                               in_axes=(None, 0))
    # Pass a dummy ones([2, 2]) to create the right axis sizes
    sample = self.strategy_run(outer_run, (self.key, tf.ones([2, 2])),
                               in_axes=(None, 0))
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for l in range(2):
            if i == k and j == l:
              continue
            self.assertNotEqual(sample[i, j], sample[k, l])

  def test_nested_sharded_maintains_correct_axis_ordering(self):
    if not JAX_MODE:
      self.skipTest('Multiple axes only supported in JAX backend.')

    other_axis_name = self.axis_name + '_other'

    dist = sharded.Sharded(
        sharded.Sharded(normal.Normal(0., 1.), self.axis_name), other_axis_name)

    self.assertListEqual(
        dist.experimental_shard_axis_names, [self.axis_name, other_axis_name])

  def test_sharded_independent_samples_differently_across_shards(self):

    @tf.function(autograph=False)
    def run(key):
      return sharded.Sharded(
          independent.Independent(normal.Normal(tf.zeros(1), tf.ones(1)), 1),
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
      root = joint_distribution.JointDistributionCoroutine.Root
      _ = yield root(
          sharded.Sharded(
              independent.Independent(
                  normal.Normal(0, tf.ones([7])),
                  reinterpreted_batch_ndims=1,
                  experimental_use_kahan_sum=True),
              shard_axis_name=self.axis_name))

    model = joint_distribution.JointDistributionCoroutine(model_fn)

    samps = self.strategy_run(
        lambda seed: model.sample(seed=seed),
        (test_util.test_seed(sampler_type='stateless'),),
        in_axes=None)

    @tf.function(jit_compile=True)
    def lp_grad(x):
      return gradient.value_and_gradient(model.log_prob, x)

    self.evaluate(
        self.per_replica_to_tensor(self.strategy_run(lp_grad, (samps,))))

  def test_log_prob(self):

    @tf.function
    def lp_grad(x):
      lp1, g1 = gradient.value_and_gradient(
          sharded.Sharded(
              sample_dist_lib.Sample(normal.Normal(0., 1.), 1),
              shard_axis_name=self.axis_name).log_prob, (x,))
      lp2, g2 = gradient.value_and_gradient(
          sharded.Sharded(
              independent.Independent(normal.Normal(tf.zeros([1]), 1.), 1),
              shard_axis_name=self.axis_name).log_prob, (x,))
      return lp1, g1, lp2, g2

    def true_lp_grad(x):
      lp1, g1 = gradient.value_and_gradient(
          sample_dist_lib.Sample(normal.Normal(0., 1.),
                                 test_lib.NUM_DEVICES).log_prob, (x,))
      lp2, g2 = gradient.value_and_gradient(
          independent.Independent(
              normal.Normal(tf.zeros([test_lib.NUM_DEVICES]), 1.), 1).log_prob,
          (x,))
      return lp1, g1, lp2, g2

    x = tf.ones([test_lib.NUM_DEVICES])
    sharded_x = self.shard_values(x)

    lp1, g1, lp2, g2 = self.evaluate(
        self.per_replica_to_tensor(self.strategy_run(lp_grad, (sharded_x,))))
    true_lp1, true_g1, true_lp2, true_g2 = self.evaluate(true_lp_grad(x))

    self.assertAllClose(true_lp1, lp1[0])
    self.assertAllClose(true_g1, g1)
    self.assertAllClose(true_lp2, lp2[0])
    self.assertAllClose(true_g2, g2)

  def test_log_prob_with_multiple_axes(self):
    if not JAX_MODE:
      self.skipTest('Multiple axes only supported in JAX backend.')

    other_axis_name = self.axis_name + '_other'

    @tf.function
    def lp_grad(x):
      lp1, g1 = gradient.value_and_gradient(
          sharded.Sharded(
              sample_dist_lib.Sample(normal.Normal(0., 1.), 1),
              shard_axis_name=[self.axis_name, other_axis_name]).log_prob, (x,))
      lp2, g2 = gradient.value_and_gradient(
          sharded.Sharded(
              independent.Independent(normal.Normal(tf.zeros([1]), 1.), 1),
              shard_axis_name=[self.axis_name, other_axis_name]).log_prob, (x,))
      return lp1, g1, lp2, g2

    def true_lp_grad(x):
      lp1, g1 = gradient.value_and_gradient(
          sample_dist_lib.Sample(normal.Normal(0., 1.), [2, 2]).log_prob, (x,))
      lp2, g2 = gradient.value_and_gradient(
          independent.Independent(normal.Normal(tf.zeros([2, 2]), 1.),
                                  2).log_prob, (x,))
      return lp1, g1, lp2, g2

    x = tf.ones([2, 2])

    def outer_run(x):
      return self.strategy_run(lp_grad, (x,), axis_name=other_axis_name)
    lp1, g1, lp2, g2 = self.strategy_run(outer_run, (x,))
    true_lp1, true_g1, true_lp2, true_g2 = true_lp_grad(x)

    self.assertAllClose(tf.ones([2, 2]) * true_lp1, lp1)
    self.assertAllClose(true_g1[0], g1[0])
    self.assertAllClose(tf.ones([2, 2]) * true_lp2, lp2)
    self.assertAllClose(true_g2[0], g2[0])

  def test_log_prob_with_no_axes(self):
    # The motivation here is to enable writing generic code that works inside
    # and outside a pmap.
    def lp_grad(x, axis_name):
      return gradient.value_and_gradient(
          sharded.Sharded(
              sample_dist_lib.Sample(normal.Normal(0., 1.), x.shape),
              shard_axis_name=axis_name).log_prob, (x,))

    def true_lp_grad(x):
      return gradient.value_and_gradient(
          sample_dist_lib.Sample(normal.Normal(0., 1.), x.shape).log_prob, (x,))

    x = tf.range(1, 1 + test_lib.NUM_DEVICES, dtype=tf.float32)
    lp, g = lp_grad(x, axis_name=[])
    sharded_x = self.shard_values(x)
    sharded_lp, sharded_g = self.per_replica_to_tensor(
        self.strategy_run(lambda x: lp_grad(x, self.axis_name), (sharded_x,)))
    true_lp, true_g = true_lp_grad(x)

    self.assertAllClose(true_lp, lp)
    self.assertAllCloseNested(true_g, g)
    self.assertAllClose(true_lp, sharded_lp[0])
    self.assertAllCloseNested(true_g, sharded_g)

  def test_log_prob_ratio_with_no_axes(self):
    # The motivation here is to enable writing generic code that works inside
    # and outside a pmap.
    def lp_ratio_grad(x0, x1, axis_name):
      dist0 = sharded.Sharded(
          sample_dist_lib.Sample(normal.Normal(0., 1.), x0.shape),
          shard_axis_name=axis_name)
      dist1 = sharded.Sharded(
          sample_dist_lib.Sample(normal.Normal(0., 2.), x0.shape),
          shard_axis_name=axis_name)

      return gradient.value_and_gradient(
          lambda x0, x1: log_prob_ratio.log_prob_ratio(dist0, x0, dist1, x1),
          (x0, x1))

    def true_lp_ratio_grad(x0, x1):
      dist0 = sample_dist_lib.Sample(normal.Normal(0., 1.), x0.shape)
      dist1 = sample_dist_lib.Sample(normal.Normal(0., 2.), x0.shape)

      return gradient.value_and_gradient(
          lambda x0, x1: log_prob_ratio.log_prob_ratio(dist0, x0, dist1, x1),
          (x0, x1))

    x0 = tf.range(1, 1 + test_lib.NUM_DEVICES, dtype=tf.float32)
    x1 = tf.range(5, 5 + test_lib.NUM_DEVICES, dtype=tf.float32)
    lp_ratio, (g0, g1) = lp_ratio_grad(x0, x1, axis_name=[])
    sharded_x0 = self.shard_values(x0)
    sharded_x1 = self.shard_values(x1)
    sharded_lp_ratio, (sharded_g0, sharded_g1) = self.per_replica_to_tensor(
        self.strategy_run(lambda x0, x1: lp_ratio_grad(x0, x1, self.axis_name),
                          (sharded_x0, sharded_x1)))
    true_lp_ratio, (true_g0, true_g1) = true_lp_ratio_grad(x0, x1)

    self.assertAllClose(true_lp_ratio, lp_ratio)
    self.assertAllClose(true_g0, g0)
    self.assertAllClose(true_g1, g1)
    self.assertAllClose(true_lp_ratio, sharded_lp_ratio[0])
    self.assertAllClose(true_g0, sharded_g0)
    self.assertAllClose(true_g1, sharded_g1)

  def test_log_prob_ratio_sample(self):
    dist1 = sample_dist_lib.Sample(normal.Normal(0., 4.), test_lib.NUM_DEVICES)
    dist2 = sample_dist_lib.Sample(normal.Normal(0., 2.), test_lib.NUM_DEVICES)

    x0 = tf.zeros([test_lib.NUM_DEVICES])
    x1 = tf.ones([test_lib.NUM_DEVICES])

    sharded_x0 = self.shard_values(x0)
    sharded_x1 = self.shard_values(x1)

    true_diff, (true_g1, true_g2) = gradient.value_and_gradient(
        lambda x0, x1: log_prob_ratio.log_prob_ratio(dist1, x0, dist2, x1),
        (x0, x1))

    @tf.function
    def test(x0, x1):
      dist_sharded1 = sharded.Sharded(
          sample_dist_lib.Sample(normal.Normal(0., 4.), 1),
          shard_axis_name=self.axis_name)
      dist_sharded2 = sharded.Sharded(
          sample_dist_lib.Sample(normal.Normal(0., 2.), 1),
          shard_axis_name=self.axis_name)
      return (
          gradient.value_and_gradient(
              lambda x0, x1: log_prob_ratio.log_prob_ratio(  # pylint: disable=g-long-lambda
                  dist_sharded1, x0, dist_sharded2, x1),
              (x0, x1)),
          dist_sharded1.log_prob(x0) - dist_sharded2.log_prob(x1))

    (diff1, (g1, g2)), diff2 = self.per_replica_to_tensor(
        self.strategy_run(test, (sharded_x0, sharded_x1)))

    self.assertAllClose(true_diff, diff1[0])
    self.assertAllClose(true_diff, diff2[0])
    self.assertAllClose(true_g1, g1)
    self.assertAllClose(true_g2, g2)

  def test_log_prob_ratio_independent(self):
    dist1 = independent.Independent(
        normal.Normal(tf.zeros([test_lib.NUM_DEVICES]), 2.), 1)
    dist2 = independent.Independent(
        normal.Normal(tf.zeros([test_lib.NUM_DEVICES]), 4.), 1)

    x0 = tf.zeros([test_lib.NUM_DEVICES])
    x1 = tf.ones([test_lib.NUM_DEVICES])

    sharded_x0 = self.shard_values(x0)
    sharded_x1 = self.shard_values(x1)

    true_diff, (true_g1, true_g2) = gradient.value_and_gradient(
        lambda x0, x1: log_prob_ratio.log_prob_ratio(dist1, x0, dist2, x1),
        (x0, x1))

    @tf.function
    def test(x0, x1):
      dist_sharded1 = sharded.Sharded(
          independent.Independent(normal.Normal(tf.zeros([1]), 2.), 1),
          shard_axis_name=self.axis_name)
      dist_sharded2 = sharded.Sharded(
          independent.Independent(normal.Normal(tf.zeros([1]), 4.), 1),
          shard_axis_name=self.axis_name)
      return (
          gradient.value_and_gradient(
              lambda x0, x1: log_prob_ratio.log_prob_ratio(  # pylint: disable=g-long-lambda
                  dist_sharded1, x0, dist_sharded2, x1),
              (x0, x1)),
          dist_sharded1.log_prob(x0) - dist_sharded2.log_prob(x1))

    (diff1, (g1, g2)), diff2 = self.per_replica_to_tensor(
        self.strategy_run(test, (sharded_x0, sharded_x1)))

    self.assertAllClose(true_diff, diff1[0])
    self.assertAllClose(true_diff, diff2[0])
    self.assertAllClose(true_g1, g1)
    self.assertAllClose(true_g2, g2)

  def test_increment_log_prob(self):

    root = jdc.JointDistributionCoroutine.Root
    prior_mean = 3.
    x_size = 100

    def custom_ll(w, x):
      return tf.reduce_sum(normal.Normal(w, 1.).log_prob(x))

    def ulp_grad(w, x):

      @joint_distribution.JointDistributionCoroutine
      def sharded_model():
        w = yield root(normal.Normal(prior_mean, 1.))
        yield root(
            sharded.Sharded(
                increment_log_prob.IncrementLogProb(custom_ll(w, x)),
                shard_axis_name=self.axis_name))

      def ulp_fn(w):
        zeros = tf.zeros([x_size, 0])
        return sharded_model.unnormalized_log_prob(w, zeros)

      ulp, g = gradient.value_and_gradient(ulp_fn, (w,))
      return ulp, g

    def true_ulp_grad(w, x):

      @jdc.JointDistributionCoroutine
      def model():
        w = yield root(normal.Normal(prior_mean, 1.))
        yield root(increment_log_prob.IncrementLogProb(custom_ll(w, x)))

      def ulp_fn(w):
        zeros = tf.zeros([x_size, 0])
        return model.unnormalized_log_prob(w, zeros)

      ulp, g = gradient.value_and_gradient(ulp_fn, (w,))
      return ulp, g

    def test_w_x(w, x):
      sharded_x = self.shard_values(tf.reshape(x, [test_lib.NUM_DEVICES, -1]))

      lp, g = self.evaluate(
          self.per_replica_to_tensor(
              self.strategy_run(ulp_grad, (
                  w,
                  sharded_x,
              ), in_axes=(None, 0))))
      true_lp, true_g = self.evaluate(true_ulp_grad(w, x))

      self.assertAllClose(true_lp, lp[0])
      self.assertAllClose(true_g[0], g[0][0])

    w = tf.constant(4.)
    zeros = tf.zeros([x_size])
    test_w_x(w, zeros)
    random_x = self.evaluate(
        normal.Normal(loc=tf.zeros([x_size]),
                      scale=tf.ones([x_size])).sample(seed=self.key))
    test_w_x(w, random_x)

  def test_default_event_space_bijector(self):
    def sharded_lp_grad(x):
      dist = sharded.Sharded(
          sample_dist_lib.Sample(exponential.Exponential(1.), x.shape),
          shard_axis_name=self.axis_name)
      bij = dist.experimental_default_event_space_bijector()
      dist2 = transformed_distribution.TransformedDistribution(
          dist, invert.Invert(bij))
      return gradient.value_and_gradient(dist2.log_prob, (x,))

    def lp_grad(x):
      dist = sample_dist_lib.Sample(exponential.Exponential(1.), x.shape)
      bij = dist.experimental_default_event_space_bijector()
      dist2 = transformed_distribution.TransformedDistribution(
          dist, invert.Invert(bij))
      return gradient.value_and_gradient(dist2.log_prob, (x,))

    x = tf.range(1, 1 + test_lib.NUM_DEVICES, dtype=tf.float32)
    sharded_x = self.shard_values(x)
    sharded_lp, sharded_g = self.per_replica_to_tensor(
        self.strategy_run(sharded_lp_grad, (sharded_x,)))
    true_lp, true_g = lp_grad(x)

    self.assertAllClose(true_lp, sharded_lp[0])
    self.assertAllCloseNested(true_g, sharded_g)

  def test_none_axis_in_jax_error(self):
    if not JAX_MODE:
      self.skipTest('This error is JAX-only.')
    with self.assertRaisesRegex(
        ValueError, 'Cannot provide a `None` axis name in JAX backend.'):
      sharded.Sharded(normal.Normal(0., 1.))

  def test_none_axis_in_tensorflow(self):
    if JAX_MODE:
      self.skipTest('This feature is TensorFlow-only.')
    dist = sharded.Sharded(normal.Normal(0., 1.))
    self.assertEqual([True], dist.experimental_shard_axis_names)

  def test_multiple_axes_in_tensorflow_error(self):
    if JAX_MODE:
      self.skipTest('This error is TensorFlow-only.')
    dist = sharded.Sharded(normal.Normal(0., 1.), shard_axis_name='i')
    with self.assertRaisesRegex(
        ValueError, 'TensorFlow backend does not support multiple shard axes'):
      sharded.Sharded(dist, shard_axis_name='j')
    with self.assertRaisesRegex(
        ValueError, 'TensorFlow backend does not support multiple shard axes'):
      sharded.Sharded(normal.Normal(0., 1.), shard_axis_name=['i', 'j'])

  def test_duplicate_axes_in_jax(self):
    if not JAX_MODE:
      self.skipTest('This error is JAX-only.')
    dist = sharded.Sharded(normal.Normal(0., 1.), shard_axis_name='i')
    with self.assertRaisesRegex(
        ValueError, 'Found duplicate axis name'):
      sharded.Sharded(dist, shard_axis_name='i')
    with self.assertRaisesRegex(
        ValueError, 'Found duplicate axis name'):
      sharded.Sharded(normal.Normal(0., 1.), shard_axis_name=['i', 'i'])

if __name__ == '__main__':
  test_util.main()
