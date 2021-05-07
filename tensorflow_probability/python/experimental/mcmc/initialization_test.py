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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class InitializationTest(test_util.TestCase):

  def testUnconstrainedUniformUsage(self):
    seed = test_util.test_seed(sampler_type='stateless')
    model_dist = tfd.JointDistributionNamed({
        'loc': tfd.Uniform(0., 1.),
        'scale': tfd.HalfCauchy(0., 1.),
        'result': tfd.Normal})
    init_dist = tfp.experimental.mcmc.init_near_unconstrained_zero(model_dist)
    samples = self.evaluate(init_dist.sample(10, seed=seed))
    assert 'loc' in samples
    assert 'scale' in samples
    assert 'result' in samples
    for _, v in samples.items():
      self.assertAllFinite(v)
    self.assertAllFinite(self.evaluate(model_dist.log_prob(samples)))

  def testUnconstrainedUniformUsageExplicitBijectors(self):
    seed = test_util.test_seed(sampler_type='stateless')
    model_dist = tfd.JointDistributionSequential([
        tfd.Uniform(0., 1.),
        tfd.HalfCauchy(0., 1.),
        lambda sigma, mu: tfd.Normal(mu, sigma)])
    bijectors = [tfb.Sigmoid(), tfb.Exp(), tfb.Identity()]
    init_dist = tfp.experimental.mcmc.init_near_unconstrained_zero(
        model_dist, constraining_bijector=bijectors)
    samples = self.evaluate(init_dist.sample(10, seed=seed))
    self.assertEqual(3, len(samples))
    for v in samples:
      self.assertAllFinite(v)
    self.assertAllFinite(self.evaluate(model_dist.log_prob(samples)))

  def testUnconstrainedUniformSingleDistribution(self):
    seed = test_util.test_seed(sampler_type='stateless')
    for model_dist in [tfd.Normal(loc=0, scale=2),
                       tfd.MultivariateNormalDiag(loc=[0, 1])]:
      init_dist = tfp.experimental.mcmc.init_near_unconstrained_zero(
          model_dist)
      samples = self.evaluate(init_dist.sample(10, seed=seed))
      self.assertAllFinite(samples)
      self.assertAllFinite(self.evaluate(model_dist.log_prob(samples)))

  def testFindValidInitialPoints(self):
    seed = test_util.test_seed(sampler_type='stateless')
    model_dist = tfd.JointDistributionSequential([
        tfd.Uniform(0., 1.),
        tfd.HalfCauchy(0., 1.),
        lambda sigma, mu: tfd.Normal(mu, sigma)])
    # Intentionally bad bijectors lead to a ~25% chance to
    # initialize in the valid region
    bijectors = [tfb.Identity(), tfb.Identity(), tfb.Identity()]
    init_dist = tfp.experimental.mcmc.init_near_unconstrained_zero(
        model_dist, constraining_bijector=bijectors)
    # We expect to be able to fix it with rejection sampling
    init_states = tfp.experimental.mcmc.retry_init(
        init_dist.sample,
        model_dist.log_prob,
        sample_shape=[20],
        seed=seed)
    for s in self.evaluate(init_states):
      self.assertAllFinite(s)
      self.assertEqual((20,), s.shape)
    self.assertAllFinite(self.evaluate(model_dist.log_prob(init_states)))

  def testFailToFindValidInitialPoints(self):
    seed = test_util.test_seed(sampler_type='stateless')
    model_dist = tfd.JointDistributionSequential([
        tfd.Uniform(10., 11.),
        tfd.HalfCauchy(10., 1.),
        lambda sigma, mu: tfd.Normal(mu, sigma)])
    # Intentionally bad bijectors lead to no chance to
    # initialize in the valid region
    bijectors = [tfb.Sigmoid(), tfb.Identity(), tfb.Identity()]
    init_dist = tfp.experimental.mcmc.init_near_unconstrained_zero(
        model_dist, constraining_bijector=bijectors)
    # We expect bounded rejection sampling to give up with an error
    with self.assertRaisesOpError(
        'Failed to find acceptable initial states'):
      self.evaluate(tfp.experimental.mcmc.retry_init(
          init_dist.sample,
          model_dist.log_prob,
          seed=seed))

  @test_util.jax_disable_test_missing_functionality('dynamic shape')
  @test_util.numpy_disable_test_missing_functionality('dynamic shape')
  def testPartialShape(self):
    @tf.function
    def f(x):
      model_dist = tfd.JointDistributionSequential([
          tfd.Sample(tfd.Normal(0, 1), tf.shape(x))])
      return tfp.experimental.mcmc.init_near_unconstrained_zero(
          model_dist).sample(seed=test_util.test_seed())
    g = f.get_concrete_function(tf.TensorSpec([None, 3], tf.float32))
    self.evaluate(g(tf.ones([7, 3])))


if __name__ == '__main__':
  tf.test.main()
