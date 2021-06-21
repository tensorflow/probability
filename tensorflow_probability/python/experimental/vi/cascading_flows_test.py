# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for structured surrogate posteriors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util
from tensorflow.python.util import nest

# Dependency imports

tfb = tfp.bijectors
tfd = tfp.distributions
Root = tfd.JointDistributionCoroutine.Root

@test_util.test_all_tf_execution_regimes
class CascadingFlowTests(test_util.TestCase):

  def test_shapes(self):

    def test_shapes_model():
      # Matrix-valued random variable with batch shape [3].
      A = yield Root(
        tfd.WishartTriL(df=2, scale_tril=tf.eye(2, batch_shape=[3]), name='A'))
      # Vector-valued random variable with batch shape [3] (inherited from `A`)
      x = yield tfd.MultivariateNormalTriL(loc=tf.zeros([2]),
                                           scale_tril=tf.linalg.cholesky(A),
                                           name='x')
      # Scalar-valued random variable, with batch shape `[3]`.
      y = yield tfd.Normal(loc=tf.reduce_sum(x, axis=-1), scale=tf.ones([3]))

    prior = tfd.JointDistributionCoroutineAutoBatched(test_shapes_model, batch_ndims=1)
    surrogate_posterior = tfp.experimental.vi.build_cascading_flow_surrogate_posterior(prior) #num_auxiliary_variables=10)

    x1 = surrogate_posterior.sample()
    x2 = nest.map_structure_up_to(
      x1,
      # Strip auxiliary variables.
      lambda *rv_and_aux: rv_and_aux[0],
      surrogate_posterior.sample())

    # Assert that samples from the surrogate have the same shape as the prior.
    get_shapes = lambda x: tf.nest.map_structure(lambda xp: xp.shape, x)
    self.assertAllEqualNested(get_shapes(x1), get_shapes(x2))


@test_util.test_all_tf_execution_regimes
class _TrainableCFSurrogate(object):

  def _expected_num_trainable_variables(self, prior_dist, num_layers):
    """Infers the expected number of trainable variables for a non-nested JD."""
    prior_dists = prior_dist._get_single_sample_distributions()  # pylint: disable=protected-access
    expected_num_trainable_variables = 0

    # For each distribution in the prior, we will have one highway flow with
    # `num_layers` blocks, and each block has 4 trainable variables:
    # `residual_fraction`, `lower_diagonal_weights_matrix`,
    # `upper_diagonal_weights_matrix` and `bias`.
    for original_dist in prior_dists:
      expected_num_trainable_variables += (4 * num_layers)
    return expected_num_trainable_variables

  def test_dims_and_gradients(self):
    prior_dist = self.make_prior_dist()
    num_layers = 3
    surrogate_posterior = tfp.experimental.vi.build_cascading_flow_surrogate_posterior(
      prior=prior_dist, num_layers=num_layers)

    # Test that the correct number of trainable variables are being tracked
    self.assertLen(surrogate_posterior.trainable_variables,
                   self._expected_num_trainable_variables(prior_dist,
                                                          num_layers))

    # Test that the sample shape is correct
    three_posterior_samples = surrogate_posterior.sample(
      3, seed=test_util.test_seed(sampler_type='stateless'))
    three_prior_samples = prior_dist.sample(
      3, seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllEqualNested(
      [s.shape for s in tf.nest.flatten(three_prior_samples)],
      [s.shape for s in tf.nest.flatten(three_posterior_samples)])

    # Test that gradients are available wrt the variational parameters.
    with tf.GradientTape() as tape:
      posterior_sample = surrogate_posterior.sample(
        seed=test_util.test_seed(sampler_type='stateless'))
      posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))

  def test_initialization_is_deterministic_following_seed(self):
    prior_dist = self.make_prior_dist()

    surrogate_posterior = tfp.experimental.vi.build_cascading_flow_surrogate_posterior(
      prior=prior_dist,
      seed=test_util.test_seed(sampler_type='stateless'))
    self.evaluate(
      [v.initializer for v in surrogate_posterior.trainable_variables])
    posterior_sample = surrogate_posterior.sample(
      seed=test_util.test_seed(sampler_type='stateless'))

    surrogate_posterior2 = tfp.experimental.vi.build_cascading_flow_surrogate_posterior(
      prior=prior_dist,
      seed=test_util.test_seed(sampler_type='stateless'))
    self.evaluate(
      [v.initializer for v in surrogate_posterior2.trainable_variables])
    posterior_sample2 = surrogate_posterior2.sample(
      seed=test_util.test_seed(sampler_type='stateless'))

    self.assertAllEqualNested(posterior_sample, posterior_sample2)

  def test_surrogate_and_prior_have_same_domain(self):
    prior_dist = self.make_prior_dist()
    surrogate_posterior = tfp.experimental.vi.build_cascading_flow_surrogate_posterior(
      prior=prior_dist,
      seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllFinite(prior_dist.log_prob(
      surrogate_posterior.sample(10, seed=test_util.test_seed())))

@test_util.test_all_tf_execution_regimes
class CFSurrogatePosteriorTestBrownianMotion(test_util.TestCase,
                                             _TrainableCFSurrogate):

  def make_prior_dist(self):

    def _prior_model_fn():
      innovation_noise = 0.1
      prior_loc = 0.
      new = yield tfd.Normal(loc=prior_loc, scale=innovation_noise)
      for _ in range(4):
        new = yield tfd.Normal(loc=new, scale=innovation_noise)

    return tfd.JointDistributionCoroutineAutoBatched(_prior_model_fn)

  def make_likelihood_model(self, x, observation_noise):

    def _likelihood_model():
      for i in range(5):
        yield tfd.Normal(loc=x[i], scale=observation_noise)

    return tfd.JointDistributionCoroutineAutoBatched(_likelihood_model)

  def get_observations(self, prior_dist):
    observation_noise = 0.15
    ground_truth = prior_dist.sample()
    likelihood = self.make_likelihood_model(
      x=ground_truth, observation_noise=observation_noise)

    return likelihood.sample(1)

  def get_target_log_prob(self, observations, prior_dist):

    def target_log_prob(*x):
      observation_noise = 0.15
      likelihood_dist = self.make_likelihood_model(
        x=x, observation_noise=observation_noise)
      return likelihood_dist.log_prob(observations) + prior_dist.log_prob(
        x)

    return target_log_prob

  def test_fitting_surrogate_posterior(self):

    prior_dist = self.make_prior_dist()
    observations = self.get_observations(prior_dist)
    surrogate_posterior = tfp.experimental.vi.build_cascading_flow_surrogate_posterior(
      prior=prior_dist)
    target_log_prob = self.get_target_log_prob(observations, prior_dist)

    # Test vi fit surrogate posterior works
    losses = tfp.vi.fit_surrogate_posterior(
      target_log_prob,
      surrogate_posterior,
      num_steps=5,  # Don't optimize to completion.
      optimizer=tf.optimizers.Adam(0.1),
      sample_size=10)

    # Compute posterior statistics.
    with tf.control_dependencies([losses]):
      posterior_samples = surrogate_posterior.sample(100)
      posterior_mean = tf.nest.map_structure(tf.reduce_mean,
                                             posterior_samples)
      posterior_stddev = tf.nest.map_structure(tf.math.reduce_std,
                                               posterior_samples)

    self.evaluate(tf1.global_variables_initializer())
    _ = self.evaluate(losses)
    _ = self.evaluate(posterior_mean)
    _ = self.evaluate(posterior_stddev)


@test_util.test_all_tf_execution_regimes
class CFSurrogatePosteriorTestEightSchools(test_util.TestCase,
                                           _TrainableCFSurrogate):
  def make_prior_dist(self):
    treatment_effects = tf.constant([28, 8, -3, 7, -1, 1, 18, 12],
                                    dtype=tf.float32)
    num_schools = ps.shape(treatment_effects)[-1]

    return tfd.JointDistributionNamed({
      'avg_effect':
        tfd.Normal(loc=0., scale=10., name='avg_effect'),
      'log_stddev':
        tfd.Normal(loc=5., scale=1., name='log_stddev'),
      'school_effects':
        lambda log_stddev, avg_effect: (
          # pylint: disable=g-long-lambda
          tfd.Independent(
            tfd.Normal(
              loc=avg_effect[..., None] * tf.ones(num_schools),
              scale=tf.exp(log_stddev[..., None]) * tf.ones(
                num_schools),
              name='school_effects'),
            reinterpreted_batch_ndims=1))
    })


@test_util.test_all_tf_execution_regimes
class CFSurrogatePosteriorTestEightSchoolsSample(test_util.TestCase,
                                                 _TrainableCFSurrogate):

  def make_prior_dist(self):
    return tfd.JointDistributionNamed({
      'avg_effect':
        tfd.Normal(loc=0., scale=10., name='avg_effect'),
      'log_stddev':
        tfd.Normal(loc=5., scale=1., name='log_stddev'),
      'school_effects':
        lambda log_stddev, avg_effect: (
          # pylint: disable=g-long-lambda
          tfd.Sample(
            tfd.Normal(
              loc=avg_effect[..., None],
              scale=tf.exp(log_stddev[..., None]),
              name='school_effects'),
            sample_shape=[8]))
    })


@test_util.test_all_tf_execution_regimes
class CFSurrogatePosteriorTestHalfNormal(test_util.TestCase,
                                         _TrainableCFSurrogate):

  def make_prior_dist(self):
    def _prior_model_fn():
      innovation_noise = 1.
      yield tfd.HalfNormal(
        scale=innovation_noise, validate_args=True,
        allow_nan_stats=False)

    return tfd.JointDistributionCoroutineAutoBatched(_prior_model_fn)

@test_util.test_all_tf_execution_regimes
class CFSurrogatePosteriorTestNesting(test_util.TestCase,
                                      _TrainableCFSurrogate):

  def make_prior_dist(self):
    def nested_model():
      a = yield tfd.Sample(
        tfd.Sample(
          tfd.Normal(0., 1.),
          sample_shape=4),
        sample_shape=[2],
        name='a')
      b = yield tfb.Sigmoid()(
        tfb.Square()(
          tfd.Exponential(rate=tf.exp(a))),
        name='b')
      # pylint: disable=g-long-lambda
      yield tfd.JointDistributionSequential(
        [tfd.Laplace(loc=a, scale=b),
         lambda c1: tfd.Independent(
           tfd.Beta(concentration1=1.,
                    concentration0=tf.nn.softplus(c1)),
           reinterpreted_batch_ndims=1),
         lambda c1, c2: tfd.JointDistributionNamed({
           'x': tfd.Gamma(concentration=tf.nn.softplus(c1), rate=c2)})
         ], name='c')
      # pylint: enable=g-long-lambda

    return tfd.JointDistributionCoroutineAutoBatched(nested_model)


if __name__ == '__main__':
  tf.test.main()
