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
from absl.testing import parameterized

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental.distribute import joint_distribution as jd
from tensorflow_probability.python.experimental.distribute import sharded
from tensorflow_probability.python.internal import distribute_test_lib as test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient

Root = jdc.JointDistributionCoroutine.Root


def true_log_prob_fn(w, x, data):
  return (normal.Normal(0., 1.).log_prob(w) + sample_dist_lib.Sample(
      normal.Normal(w, 1.), (test_lib.NUM_DEVICES, 1)).log_prob(x) +
          independent.Independent(normal.Normal(x, 1.), 2).log_prob(data))


def make_jd_sequential(axis_name):
  return jd.JointDistributionSequential([
      normal.Normal(0., 1.),
      lambda w: sharded.Sharded(  # pylint: disable=g-long-lambda
          sample_dist_lib.Sample(normal.Normal(w, 1.), 1),
          shard_axis_name=axis_name),
      lambda x: sharded.Sharded(  # pylint: disable=g-long-lambda
          independent.Independent(normal.Normal(x, 1.), 1),
          shard_axis_name=axis_name),
  ])


def make_jd_named(axis_name):
  return jd.JointDistributionNamed(  # pylint: disable=g-long-lambda
      dict(
          w=normal.Normal(0., 1.),
          x=lambda w: sharded.Sharded(  # pylint: disable=g-long-lambda
              sample_dist_lib.Sample(normal.Normal(w, 1.), 1),
              shard_axis_name=axis_name),
          data=lambda x: sharded.Sharded(  # pylint: disable=g-long-lambda
              independent.Independent(normal.Normal(x, 1.), 1),
              shard_axis_name=axis_name),
      ))


def make_jd_coroutine(axis_name):

  def model_coroutine():
    w = yield Root(normal.Normal(0., 1.))
    x = yield sharded.Sharded(
        sample_dist_lib.Sample(normal.Normal(w, 1.), 1),
        shard_axis_name=axis_name)
    yield sharded.Sharded(
        independent.Independent(normal.Normal(x, 1.), 1),
        shard_axis_name=axis_name)

  return jd.JointDistributionCoroutine(model_coroutine)


distributions = (
    ('coroutine', make_jd_coroutine),
    ('sequential', make_jd_sequential),
    ('named', make_jd_named),
)


@test_util.test_all_tf_execution_regimes
class JointDistributionTest(test_lib.DistributedTest):

  @test_util.disable_test_for_backend(
      disable_jax=True,
      reason='Cannot call `experimental_is_sharded` outside of pmap.')
  def test_experimental_is_sharded_coroutine(self):
    dist = distributions[0][1](self.axis_name)
    self.assertTupleEqual(dist.experimental_shard_axis_names,
                          ([], [self.axis_name], [self.axis_name]))

  @test_util.disable_test_for_backend(
      disable_jax=True,
      reason='Cannot call `experimental_is_sharded` outside of pmap.')
  def test_experimental_is_sharded_sequential(self):
    dist = distributions[1][1](self.axis_name)
    self.assertListEqual(dist.experimental_shard_axis_names,
                         [[], [self.axis_name], [self.axis_name]])

  @test_util.disable_test_for_backend(
      disable_jax=True,
      reason='Cannot call `experimental_is_sharded` outside of pmap.')
  def test_experimental_is_sharded_named(self):
    dist = distributions[2][1](self.axis_name)
    self.assertDictEqual(dist.experimental_shard_axis_names,
                         dict(w=[], x=[self.axis_name], data=[self.axis_name]))

  @parameterized.named_parameters(*distributions)
  def test_jd(self, dist_fn):
    dist = dist_fn(self.axis_name)

    def run(key):
      sample = dist.sample(seed=key)
      # The identity is to prevent reparameterization gradients from kicking in.
      log_prob, (log_prob_grads,) = gradient.value_and_gradient(
          dist.log_prob, (tf.nest.map_structure(tf.identity, sample),))
      return sample, log_prob, log_prob_grads

    sample, log_prob, log_prob_grads = self.strategy_run(
        run, (self.key,), in_axes=None)
    sample, log_prob, log_prob_grads = self.per_replica_to_tensor(
        (sample, log_prob, log_prob_grads))

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

    true_log_prob, true_log_prob_grads = gradient.value_and_gradient(
        true_log_prob_fn, (w, x, data))

    self.assertAllClose(
        self.evaluate(log_prob),
        self.evaluate(tf.ones(test_lib.NUM_DEVICES) * true_log_prob))
    self.assertAllCloseNested(
        self.evaluate(log_prob_grads), self.evaluate(true_log_prob_grads))

  @parameterized.named_parameters(*distributions)
  def test_jd_log_prob_ratio(self, dist_fn):
    dist = dist_fn(self.axis_name)

    def run(key):
      sample = dist.sample(seed=key)
      log_prob = dist.log_prob(sample)
      return sample, log_prob

    keys = samplers.split_seed(self.key, 2)
    samples = []
    unmapped_samples = []
    log_probs = []
    true_log_probs = []

    for key in keys:
      sample, log_prob = self.per_replica_to_tensor(
          self.strategy_run(run, (key,), in_axes=None))

      if isinstance(dist, jd.JointDistributionNamed):
        # N.B. the global RV 'w' gets replicated, so we grab any single
        # replica's result.
        w, x, data = sample['w'][0], sample['x'], sample['data']
      else:
        w, x, data = sample[0][0], sample[1], sample[2]

      true_log_prob = true_log_prob_fn(w, x, data)

      samples.append(sample)
      unmapped_samples.append((w, x, data))
      log_probs.append(log_prob[0])
      true_log_probs.append(true_log_prob)

    def true_diff(x, y):
      return true_log_prob_fn(*x) - true_log_prob_fn(*y)

    def run_diff(x, y):
      def _lpr(x, y):
        return log_prob_ratio.log_prob_ratio(dist, x, dist, y)

      return gradient.value_and_gradient(_lpr, [x, y])

    dist_lp_diff, dist_lp_diff_grad = self.per_replica_to_tensor(
        self.strategy_run(
            run_diff, tuple(tf.nest.map_structure(self.shard_values, samples))))

    true_lp_diff, true_lp_diff_grad = gradient.value_and_gradient(
        true_diff, unmapped_samples)

    if isinstance(dist, jd.JointDistributionNamed):
      dist_lp_diff_grad[0] = (
          dist_lp_diff_grad[0]['w'][0],
          dist_lp_diff_grad[0]['x'],
          dist_lp_diff_grad[0]['data'])
      dist_lp_diff_grad[1] = (
          dist_lp_diff_grad[1]['w'][0],
          dist_lp_diff_grad[1]['x'],
          dist_lp_diff_grad[1]['data'])
    else:
      true_lp_diff_grad[0] = list(true_lp_diff_grad[0])
      true_lp_diff_grad[1] = list(true_lp_diff_grad[1])
      dist_lp_diff_grad[0] = list(dist_lp_diff_grad[0])
      dist_lp_diff_grad[0][0] = dist_lp_diff_grad[0][0][0]
      dist_lp_diff_grad[1] = list(dist_lp_diff_grad[1])
      dist_lp_diff_grad[1][0] = dist_lp_diff_grad[1][0][0]

    lp_diff = log_probs[0] - log_probs[1]

    self.assertAllClose(
        true_lp_diff, lp_diff,
        rtol=7e-6)  # relaxed tol for fp32 in JAX
    self.assertAllClose(
        true_lp_diff, dist_lp_diff[0])
    self.assertAllCloseNested(
        true_lp_diff_grad, dist_lp_diff_grad)

  def test_jd_has_correct_sample_path_gradients(self):

    def log_prob_fn(x_loc):

      @jdc.JointDistributionCoroutine
      def surrogate():
        x = yield Root(normal.Normal(x_loc, 1.))
        y = yield normal.Normal(x, 1.)
        yield sample_dist_lib.Sample(
            normal.Normal(x + y, 1.), test_lib.NUM_DEVICES)

      @jdc.JointDistributionCoroutine
      def model():
        yield Root(normal.Normal(1., 1.))
        yield Root(normal.Normal(1., 1.))
        yield sample_dist_lib.Sample(
            normal.Normal(1., 1.), test_lib.NUM_DEVICES)
      return tf.reduce_mean(
          model.log_prob(surrogate.sample(sample_shape=1e5, seed=self.key)))

    true_log_prob, true_log_prob_grad = gradient.value_and_gradient(
        log_prob_fn, 0.)

    def run(seed):
      def sharded_log_prob_fn(x_loc):
        @jd.JointDistributionCoroutine
        def surrogate():
          x = yield Root(normal.Normal(x_loc, 1.))
          y = yield normal.Normal(x, 1.)
          yield sharded.Sharded(normal.Normal(x + y, 1.), self.axis_name)

        @jd.JointDistributionCoroutine
        def model():
          yield Root(normal.Normal(1., 1.))
          yield Root(normal.Normal(1., 1.))
          yield sharded.Sharded(normal.Normal(1., 1.), self.axis_name)
        return tf.reduce_mean(
            model.log_prob(surrogate.sample(sample_shape=1e5, seed=seed)))

      sharded_log_prob, sharded_log_prob_grad = gradient.value_and_gradient(
          sharded_log_prob_fn, 0.)
      return sharded_log_prob, sharded_log_prob_grad

    sharded_log_prob, sharded_log_prob_grad = self.per_replica_to_tensor(
        self.strategy_run(
            run, (self.key,), in_axes=None))
    for i in range(test_lib.NUM_DEVICES):
      self.assertAllClose(sharded_log_prob[i], true_log_prob, atol=2e-2)
      self.assertAllClose(sharded_log_prob_grad[i], true_log_prob_grad,
                          atol=2e-2)

  def test_jd_has_correct_sample_path_gradients_with_partial_values(self):

    def run(seed):
      @jd.JointDistributionCoroutine
      def model():
        yield Root(normal.Normal(0., 1., name='x'))
        yield normal.Normal(0., 1., name='y')
        yield sharded.Sharded(normal.Normal(1., 1.), self.axis_name, name='z')

      sample = model.sample(seed=seed)

      def lp_fn1(x, y, z):
        return model.log_prob((x, y, z))

      def lp_fn2(x, z):
        return model.log_prob(model.sample(value=(x, None, z), seed=seed))

      lp_and_grad1 = gradient.value_and_gradient(lp_fn1, [*sample])
      (lp2, grad2) = gradient.value_and_gradient(lp_fn2, [sample.x, sample.z])
      return lp_and_grad1, (lp2, grad2)

    (lp1, grad1), (lp2, grad2) = self.per_replica_to_tensor(
        self.strategy_run(
            run, (self.key,), in_axes=None))
    grad2 = [grad2[0], None, grad2[1]]
    for i in range(test_lib.NUM_DEVICES):
      for j in range(3):
        self.assertAllClose(lp1[i], lp2[i])
        if grad2[j] is not None:
          self.assertAllClose(grad1[j][i], grad2[j][i])

  def test_default_event_space_bijector_non_interacting(self):

    root = jd.JointDistributionCoroutine.Root

    @jdc.JointDistributionCoroutine
    def model():
      x = yield root(lognormal.LogNormal(0., 1., name='x'))
      yield sample_dist_lib.Sample(
          lognormal.LogNormal(0., x), test_lib.NUM_DEVICES, name='y')
      yield sample_dist_lib.Sample(
          scale.Scale(2.)(normal.Normal(0., 1.)),
          test_lib.NUM_DEVICES,
          name='z')

    @jd.JointDistributionCoroutine
    def sharded_model():
      x = yield root(lognormal.LogNormal(0., 1., name='x'))
      yield sharded.Sharded(
          lognormal.LogNormal(0., x), shard_axis_name=self.axis_name, name='y')
      yield sharded.Sharded(
          scale.Scale(2.)(normal.Normal(0., 1.)),
          shard_axis_name=self.axis_name,
          name='z')

    sample = self.evaluate(model.sample(seed=self.key))
    unconstrained_sample = (
        model.experimental_default_event_space_bijector().inverse(sample))

    def unconstrained_lp(model, unconstrained_sample):
      unconstrained_sample = tf.nest.map_structure(tf.identity,
                                                   unconstrained_sample)
      bij = model.experimental_default_event_space_bijector()
      sample = bij(unconstrained_sample)
      lp = model.log_prob(sample)
      fldj = bij.forward_log_det_jacobian(unconstrained_sample)
      return lp + fldj

    true_lp, (true_g,) = gradient.value_and_gradient(
        lambda unconstrained_sample: unconstrained_lp(  # pylint: disable=g-long-lambda
            model, unconstrained_sample),
        (unconstrained_sample,))

    def run(unconstrained_sample):
      return gradient.value_and_gradient(
          lambda unconstrained_sample: unconstrained_lp(  # pylint: disable=g-long-lambda
              sharded_model, unconstrained_sample),
          (unconstrained_sample,))

    sharded_unconstrained_sample = unconstrained_sample._replace(
        y=self.shard_values(unconstrained_sample.y),
        z=self.shard_values(unconstrained_sample.z))

    lp, (g,) = self.per_replica_to_tensor(
        self.strategy_run(
            run, (sharded_unconstrained_sample,),
            in_axes=(model.dtype._replace(x=None, y=0, z=0),)))
    lp = lp[0]
    g = g._replace(x=g.x[0])

    self.assertAllClose(true_lp, lp)
    self.assertAllCloseNested(true_g, g)

  def test_default_event_space_bijector_interacting(self):
    root = jd.JointDistributionCoroutine.Root

    @jdc.JointDistributionCoroutine
    def model():
      x = yield root(lognormal.LogNormal(0., 1., name='x'))
      # Uniform's bijector depends on the global parameter.
      yield sample_dist_lib.Sample(
          uniform.Uniform(0., x), test_lib.NUM_DEVICES, name='y')
      # TransformedDistribution's bijector explicitly depends on the global
      # parameter.
      yield sample_dist_lib.Sample(
          scale.Scale(x)(normal.Normal(0., 1.)), test_lib.NUM_DEVICES, name='z')

    @jd.JointDistributionCoroutine
    def sharded_model():
      x = yield root(lognormal.LogNormal(0., 1., name='x'))
      yield sharded.Sharded(
          uniform.Uniform(0., x), shard_axis_name=self.axis_name, name='y')
      yield sharded.Sharded(
          scale.Scale(x)(normal.Normal(0., 1.)),
          shard_axis_name=self.axis_name,
          name='z')

    sample = model.sample(seed=self.key)
    unconstrained_sample = (
        model.experimental_default_event_space_bijector().inverse(sample))

    def unconstrained_lp(model, unconstrained_sample):
      unconstrained_sample = tf.nest.map_structure(tf.identity,
                                                   unconstrained_sample)
      bij = model.experimental_default_event_space_bijector()
      sample = bij(unconstrained_sample)
      lp = model.log_prob(sample)
      fldj = bij.forward_log_det_jacobian(unconstrained_sample)
      return lp + fldj

    true_lp, (true_g,) = gradient.value_and_gradient(
        lambda unconstrained_sample: unconstrained_lp(  # pylint: disable=g-long-lambda
            model, unconstrained_sample),
        (unconstrained_sample,))

    def run(unconstrained_sample):
      return gradient.value_and_gradient(
          lambda unconstrained_sample: unconstrained_lp(  # pylint: disable=g-long-lambda
              sharded_model, unconstrained_sample),
          (unconstrained_sample,))

    sharded_unconstrained_sample = unconstrained_sample._replace(
        y=self.shard_values(unconstrained_sample.y),
        z=self.shard_values(unconstrained_sample.z))

    lp, (g,) = self.per_replica_to_tensor(
        self.strategy_run(
            run, (sharded_unconstrained_sample,),
            in_axes=(model.dtype._replace(x=None, y=0, z=0),)))
    lp = lp[0]
    g = g._replace(x=g.x[0])

    self.assertAllClose(true_lp, lp)
    self.assertAllCloseNested(true_g, g)

if __name__ == '__main__':
  test_util.main()
