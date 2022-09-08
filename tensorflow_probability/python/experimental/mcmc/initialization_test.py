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
"""MCMC initialization tests."""

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import pad
from tensorflow_probability.python.bijectors import sigmoid
from tensorflow_probability.python.distributions import half_cauchy
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import multinomial
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental import distribute
from tensorflow_probability.python.experimental.mcmc import initialization
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import test_util

Root = jdc.JointDistributionCoroutine.Root


@test_util.test_all_tf_execution_regimes
class InitializationTest(test_util.TestCase):

  def testUnconstrainedUniformUsage(self):
    seed = test_util.test_seed(sampler_type='stateless')
    model_dist = jdn.JointDistributionNamed({
        'loc': uniform.Uniform(0., 1.),
        'scale': half_cauchy.HalfCauchy(0., 1.),
        'result': normal.Normal
    })
    init_dist = initialization.init_near_unconstrained_zero(model_dist)
    samples = self.evaluate(init_dist.sample(10, seed=seed))
    assert 'loc' in samples
    assert 'scale' in samples
    assert 'result' in samples
    for _, v in samples.items():
      self.assertAllFinite(v)
    self.assertAllFinite(self.evaluate(model_dist.log_prob(samples)))

  def testUnconstrainedUniformUsageExplicitBijectors(self):
    seed = test_util.test_seed(sampler_type='stateless')
    model_dist = jds.JointDistributionSequential([
        uniform.Uniform(0., 1.),
        half_cauchy.HalfCauchy(0., 1.),
        lambda sigma, mu: normal.Normal(mu, sigma)
    ])
    bijectors = [sigmoid.Sigmoid(), exp.Exp(), identity.Identity()]
    init_dist = initialization.init_near_unconstrained_zero(
        model_dist, constraining_bijector=bijectors)
    samples = self.evaluate(init_dist.sample(10, seed=seed))
    self.assertEqual(3, len(samples))
    for v in samples:
      self.assertAllFinite(v)
    self.assertAllFinite(self.evaluate(model_dist.log_prob(samples)))

  def testUnconstrainedUniformSingleDistribution(self):
    seed = test_util.test_seed(sampler_type='stateless')
    for model_dist in [
        normal.Normal(loc=0, scale=2),
        mvn_diag.MultivariateNormalDiag(loc=[0, 1])
    ]:
      init_dist = initialization.init_near_unconstrained_zero(model_dist)
      samples = self.evaluate(init_dist.sample(10, seed=seed))
      self.assertAllFinite(samples)
      self.assertAllFinite(self.evaluate(model_dist.log_prob(samples)))

  def testFindValidInitialPoints(self):
    seed = test_util.test_seed(sampler_type='stateless')
    model_dist = jds.JointDistributionSequential([
        uniform.Uniform(0., 1.),
        half_cauchy.HalfCauchy(0., 1.),
        lambda sigma, mu: normal.Normal(mu, sigma)
    ])
    # Intentionally bad bijectors lead to a ~25% chance to
    # initialize in the valid region
    bijectors = [identity.Identity(), identity.Identity(), identity.Identity()]
    init_dist = initialization.init_near_unconstrained_zero(
        model_dist, constraining_bijector=bijectors)
    # We expect to be able to fix it with rejection sampling
    init_states = initialization.retry_init(
        init_dist.sample, model_dist.log_prob, sample_shape=[20], seed=seed)
    for s in self.evaluate(init_states):
      self.assertAllFinite(s)
      self.assertEqual((20,), s.shape)
    self.assertAllFinite(self.evaluate(model_dist.log_prob(init_states)))

  def testFailToFindValidInitialPoints(self):
    seed = test_util.test_seed(sampler_type='stateless')
    model_dist = jds.JointDistributionSequential([
        uniform.Uniform(10., 11.),
        half_cauchy.HalfCauchy(10., 1.),
        lambda sigma, mu: normal.Normal(mu, sigma)
    ])
    # Intentionally bad bijectors lead to no chance to
    # initialize in the valid region
    bijectors = [sigmoid.Sigmoid(), identity.Identity(), identity.Identity()]
    init_dist = initialization.init_near_unconstrained_zero(
        model_dist, constraining_bijector=bijectors)
    # We expect bounded rejection sampling to give up with an error
    with self.assertRaisesOpError(
        'Failed to find acceptable initial states'):
      self.evaluate(
          initialization.retry_init(
              init_dist.sample, model_dist.log_prob, seed=seed))

  @test_util.jax_disable_test_missing_functionality('dynamic shape')
  @test_util.numpy_disable_test_missing_functionality('dynamic shape')
  def testPartialShape(self):
    @tf.function
    def f(x):
      model_dist = jds.JointDistributionSequential(
          [sample_dist_lib.Sample(normal.Normal(0, 1), tf.shape(x))])
      return initialization.init_near_unconstrained_zero(model_dist).sample(
          seed=test_util.test_seed())
    g = f.get_concrete_function(tf.TensorSpec([None, 3], tf.float32))
    self.evaluate(g(tf.ones([7, 3])))

  def testBatchShapeJDCoro(self):

    @jdc.JointDistributionCoroutine
    def model():
      x = yield Root(
          mvn_diag.MultivariateNormalDiag(
              tf.zeros([10, 3]), tf.ones(3), name='x'))
      yield multinomial.Multinomial(
          logits=pad.Pad([[0, 1]])(x), total_count=10, name='y')

    p = model.experimental_pin(y=model.sample(seed=test_util.test_seed()).y)

    self.assertEqual((10, 3),
                     initialization.init_near_unconstrained_zero(p).sample(
                         seed=test_util.test_seed()).x.shape)

  @test_util.numpy_disable_test_missing_functionality('vmap')
  def testBatchShapeJDCoroAB(self):
    def model():
      x = yield mvn_diag.MultivariateNormalDiag(
          tf.zeros([10, 3]), tf.ones(3), name='x')
      yield multinomial.Multinomial(
          logits=pad.Pad([[0, 1]])(x), total_count=10, name='y')

    jd = jdab.JointDistributionCoroutineAutoBatched(model, batch_ndims=1)
    p = jd.experimental_pin(y=jd.sample(seed=test_util.test_seed()).y)

    self.assertEqual((10, 3),
                     initialization.init_near_unconstrained_zero(p).sample(
                         seed=test_util.test_seed()).x.shape)


@test_util.disable_test_for_backend(
    disable_numpy=True,
    reason='Sharding not available for NumPy backend.')
class DistributedTest(distribute_test_lib.DistributedTest):

  def test_can_initialize_from_sharded_distribution(self):

    def model():
      x = yield normal.Normal(0., 1.)
      yield distribute.Sharded(normal.Normal(x, 1.), self.axis_name)

    jd = distribute.JointDistributionCoroutine(model)

    def run(seed):
      init_jd = initialization.init_near_unconstrained_zero(jd)
      return init_jd.sample(seed=seed)

    x, y = self.evaluate(
        self.per_replica_to_tensor(
            self.strategy_run(run, args=(test_util.test_seed(),),
                              in_axes=None)))
    for i in range(distribute_test_lib.NUM_DEVICES):
      for j in range(distribute_test_lib.NUM_DEVICES):
        if i == j:
          continue
        self.assertAllClose(x[i], x[j])
        self.assertNotAllClose(y[i], y[j])


if __name__ == '__main__':
  test_util.main()
