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
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib as test_lib
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


def true_log_prob_fn(w, x, data):
  return (tfd.Normal(0., 1.).log_prob(w) +
          tfd.Sample(tfd.Normal(w, 1.), (test_lib.NUM_DEVICES, 1)).log_prob(x) +
          tfd.Independent(tfd.Normal(x, 1.), 2).log_prob(data))


def make_jd_sequential(axis_name):
  return jd.JointDistributionSequential([
      tfd.Normal(0., 1.),
      lambda w: sharded.Sharded(  # pylint: disable=g-long-lambda
          tfd.Sample(tfd.Normal(w, 1.), 1),
          shard_axis_name=axis_name),
      lambda x: sharded.Sharded(  # pylint: disable=g-long-lambda
          tfd.Independent(tfd.Normal(x, 1.), 1),
          shard_axis_name=axis_name),
  ])


def make_jd_named(axis_name):
  return jd.JointDistributionNamed(  # pylint: disable=g-long-lambda
      dict(
          w=tfd.Normal(0., 1.),
          x=lambda w: sharded.Sharded(  # pylint: disable=g-long-lambda
              tfd.Sample(tfd.Normal(w, 1.), 1),
              shard_axis_name=axis_name),
          data=lambda x: sharded.Sharded(  # pylint: disable=g-long-lambda
              tfd.Independent(tfd.Normal(x, 1.), 1),
              shard_axis_name=axis_name),
      ))


def make_jd_coroutine(axis_name):

  def model_coroutine():
    w = yield tfd.JointDistributionCoroutine.Root(tfd.Normal(0., 1.))
    x = yield sharded.Sharded(
        tfd.Sample(tfd.Normal(w, 1.), 1), shard_axis_name=axis_name)
    yield sharded.Sharded(
        tfd.Independent(tfd.Normal(x, 1.), 1), shard_axis_name=axis_name)

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
      log_prob, (log_prob_grads,) = tfp.math.value_and_gradient(
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

    true_log_prob, true_log_prob_grads = tfp.math.value_and_gradient(
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

    keys = tfp.random.split_seed(self.key, 2)
    samples = []
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
      log_probs.append(log_prob[0])
      true_log_probs.append(true_log_prob)

    def run_diff(x, y):
      return tfp.experimental.distributions.log_prob_ratio(dist, x, dist, y)

    dist_lp_diff = self.per_replica_to_tensor(
        self.strategy_run(
            run_diff, tuple(tf.nest.map_structure(self.shard_values, samples))))

    true_lp_diff = true_log_probs[0] - true_log_probs[1]
    lp_diff = log_probs[0] - log_probs[1]

    self.assertAllClose(
        self.evaluate(true_lp_diff), self.evaluate(lp_diff),
        rtol=7e-6)  # relaxed tol for fp32 in JAX
    self.assertAllClose(
        self.evaluate(true_lp_diff), self.evaluate(dist_lp_diff[0]))

  def test_default_event_space_bijector_non_interacting(self):

    root = jd.JointDistributionCoroutine.Root

    @tfd.JointDistributionCoroutine
    def model():
      x = yield root(tfd.LogNormal(0., 1., name='x'))
      yield tfd.Sample(tfd.LogNormal(0., x), test_lib.NUM_DEVICES, name='y')
      yield tfd.Sample(
          tfb.Scale(2.)(tfd.Normal(0., 1.)), test_lib.NUM_DEVICES, name='z')

    @jd.JointDistributionCoroutine
    def sharded_model():
      x = yield root(tfd.LogNormal(0., 1., name='x'))
      yield sharded.Sharded(
          tfd.LogNormal(0., x), shard_axis_name=self.axis_name, name='y')
      yield sharded.Sharded(
          tfb.Scale(2.)(tfd.Normal(0., 1.)),
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

    true_lp, (true_g,) = tfp.math.value_and_gradient(
        lambda unconstrained_sample: unconstrained_lp(  # pylint: disable=g-long-lambda
            model, unconstrained_sample), (unconstrained_sample,))

    def run(unconstrained_sample):
      return tfp.math.value_and_gradient(
          lambda unconstrained_sample: unconstrained_lp(  # pylint: disable=g-long-lambda
              sharded_model, unconstrained_sample), (unconstrained_sample,))

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

    @tfd.JointDistributionCoroutine
    def model():
      x = yield root(tfd.LogNormal(0., 1., name='x'))
      # Uniform's bijector depends on the global parameter.
      yield tfd.Sample(tfd.Uniform(0., x), test_lib.NUM_DEVICES, name='y')
      # TransformedDistribution's bijector explicitly depends on the global
      # parameter.
      yield tfd.Sample(
          tfb.Scale(x)(tfd.Normal(0., 1.)), test_lib.NUM_DEVICES, name='z')

    @jd.JointDistributionCoroutine
    def sharded_model():
      x = yield root(tfd.LogNormal(0., 1., name='x'))
      yield sharded.Sharded(
          tfd.Uniform(0., x), shard_axis_name=self.axis_name, name='y')
      yield sharded.Sharded(
          tfb.Scale(x)(tfd.Normal(0., 1.)),
          shard_axis_name=self.axis_name,
          name='z')

    @tfd.JointDistributionCoroutine
    def manual_sharded_model():
      # This one has manual pbroadcasts; the goal is to get sharded_model above
      # to do this automatically.
      x = yield root(tfd.LogNormal(0., 1., name='x'))
      x = distribute_lib.pbroadcast(x, axis_name=self.axis_name)
      yield sharded.Sharded(
          tfd.Uniform(0., x), shard_axis_name=self.axis_name, name='y')
      yield sharded.Sharded(
          tfb.Scale(x)(tfd.Normal(0., 1.)),
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

    true_lp, (true_g,) = tfp.math.value_and_gradient(
        lambda unconstrained_sample: unconstrained_lp(  # pylint: disable=g-long-lambda
            model, unconstrained_sample), (unconstrained_sample,))

    def run(unconstrained_sample):
      return tfp.math.value_and_gradient(
          lambda unconstrained_sample: unconstrained_lp(  # pylint: disable=g-long-lambda
              sharded_model, unconstrained_sample), (unconstrained_sample,))

    def manual_run(unconstrained_sample):
      return tfp.math.value_and_gradient(
          lambda unconstrained_sample: unconstrained_lp(  # pylint: disable=g-long-lambda
              manual_sharded_model, unconstrained_sample),
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

    manual_lp, (manual_g,) = self.per_replica_to_tensor(
        self.strategy_run(
            manual_run, (sharded_unconstrained_sample,),
            in_axes=(model.dtype._replace(x=None, y=0, z=0),)))
    manual_lp = manual_lp[0]
    manual_g = manual_g._replace(x=manual_g.x[0])

    self.assertAllClose(true_lp, lp)
    # TODO(b/175084455): This will fail because there are sharded <->
    # non-sharded edges in the gradient graph not accounted for. The edges arise
    # because the sharded bijectors' parameterizations depend non-sharded
    # parameters.
    with self.assertRaises(AssertionError):
      self.assertAllCloseNested(true_g, g)

    self.assertAllClose(true_lp, manual_lp)
    self.assertAllCloseNested(true_g, manual_g)

if __name__ == '__main__':
  tf.test.main()
