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

# Dependency imports

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.vi import cascading_flows
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions
 # test_util.test_seed(sampler_type='stateless'))

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
    surrogate_posterior = tfp.experimental.vi.build_cf_surrogate_posterior(
        prior=prior_dist, num_layers=num_layers)

    # Test that the correct number of trainable variables are being tracked
    self.assertLen(surrogate_posterior.trainable_variables,
                   self._expected_num_trainable_variables(prior_dist, num_layers))

    # Test that the sample shape is correct
    three_posterior_samples = surrogate_posterior.sample(
        3, seed=1)
    three_prior_samples = prior_dist.sample(
        3, seed=1)
    self.assertAllEqualNested(
        [s.shape for s in tf.nest.flatten(three_prior_samples)],
        [s.shape for s in tf.nest.flatten(three_posterior_samples)])

  def test_initialization_is_deterministic_following_seed(self):
    prior_dist = self.make_prior_dist()

    surrogate_posterior = tfp.experimental.vi.build_cf_surrogate_posterior(
        prior=prior_dist,
        seed=1)
    self.evaluate(
        [v.initializer for v in surrogate_posterior.trainable_variables])
    posterior_sample = surrogate_posterior.sample(
        seed=1)

    surrogate_posterior2 = tfp.experimental.vi.build_cf_surrogate_posterior(
        prior=prior_dist,
        seed=1)
    self.evaluate(
        [v.initializer for v in surrogate_posterior2.trainable_variables])
    posterior_sample2 = surrogate_posterior2.sample(
        seed=1)

    self.assertAllEqualNested(posterior_sample, posterior_sample2)


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
      return likelihood_dist.log_prob(observations) + prior_dist.log_prob(x)

    return target_log_prob

  '''def test_fitting_surrogate_posterior(self):

    prior_dist = self.make_prior_dist()
    observations = self.get_observations(prior_dist)
    surrogate_posterior = tfp.experimental.vi.build_cf_surrogate_posterior(
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
      posterior_mean = tf.nest.map_structure(tf.reduce_mean, posterior_samples)
      posterior_stddev = tf.nest.map_structure(tf.math.reduce_std,
                                               posterior_samples)

    self.evaluate(tf1.global_variables_initializer())
    _ = self.evaluate(losses)
    _ = self.evaluate(posterior_mean)
    _ = self.evaluate(posterior_stddev)'''

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
            lambda log_stddev, avg_effect: (  # pylint: disable=g-long-lambda
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
            lambda log_stddev, avg_effect: (  # pylint: disable=g-long-lambda
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
          scale=innovation_noise, validate_args=True, allow_nan_stats=False)

    return tfd.JointDistributionCoroutineAutoBatched(_prior_model_fn)


'''@test_util.test_all_tf_execution_regimes
class CFSurrogatePosteriorTestDiscreteLatent(
    test_util.TestCase, _TrainableCFSurrogate):

  def make_prior_dist(self):

    def _prior_model_fn():
      a = yield tfd.Bernoulli(logits=0.5, name='a')
      yield tfd.Normal(loc=2. * tf.cast(a, tf.float32) - 1.,
                       scale=1., name='b')

    return tfd.JointDistributionCoroutineAutoBatched(_prior_model_fn)


@test_util.test_all_tf_execution_regimes
class CFSurrogatePosteriorTestNesting(test_util.TestCase,
                                        _TrainableCFSurrogate):

  def _expected_num_trainable_variables(self, _):
    # Nested distributions have total of 10 params after Exponential->Gamma
    # substitution, multiplied by 2 variables per param.
    return 20

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


@test_util.test_all_tf_execution_regimes
class TestCFDistributionSubstitution(test_util.TestCase):

  def test_default_substitutes_trainable_families(self):

    @tfd.JointDistributionCoroutineAutoBatched
    def model():
      yield tfd.Sample(
          tfd.Uniform(low=-2., high=7.),
          sample_shape=[2],
          name='a')
      yield tfd.HalfNormal(1., name='b')
      yield tfd.Exponential(rate=[1., 2.], name='c')
      yield tfd.Chi2(df=3., name='d')

    surrogate = tfp.experimental.vi.build_cf_surrogate_posterior(
        model)
    self.assertAllEqualNested(model.event_shape, surrogate.event_shape)

    surrogate_dists, _ = surrogate.sample_distributions()
    self.assertIsInstance(surrogate_dists.a, tfd.Independent)
    self.assertIsInstance(surrogate_dists.a.distribution,
                          tfd.TransformedDistribution)
    self.assertIsInstance(surrogate_dists.a.distribution.distribution,
                          tfd.Beta)
    self.assertIsInstance(surrogate_dists.b, tfd.TruncatedNormal)
    self.assertIsInstance(surrogate_dists.c, tfd.Gamma)
    self.assertIsInstance(surrogate_dists.d, tfd.Gamma)

  def test_can_specify_custom_substitution(self):

    @tfd.JointDistributionCoroutineAutoBatched
    def centered_horseshoe(ndims=100):
      global_scale = yield tfd.HalfCauchy(
          loc=0., scale=1., name='global_scale')
      local_scale = yield tfd.HalfCauchy(
          loc=0., scale=tf.ones([ndims]), name='local_scale')
      yield tfd.Normal(
          loc=0., scale=tf.sqrt(global_scale * local_scale), name='weights')

    tfp.experimental.vi.register_asvi_substitution_rule(
        condition=tfd.HalfCauchy,
        substitution_fn=(
            lambda d: tfb.Softplus(1e-6)(tfd.Normal(loc=d.loc, scale=d.scale))))
    surrogate = tfp.experimental.vi.build_cf_surrogate_posterior(
        centered_horseshoe)
    self.assertAllEqualNested(centered_horseshoe.event_shape,
                              surrogate.event_shape)

    # If the surrogate was built with names or structure differing from the
    # model, so that it had to be `tfb.Restructure`'d, then this
    # sample_distributions call will fail because the surrogate isn't an
    # instance of tfd.JointDistribution.
    surrogate_dists, _ = surrogate.sample_distributions()
    self.assertIsInstance(surrogate_dists.global_scale.distribution,
                          tfd.Normal)
    self.assertIsInstance(surrogate_dists.local_scale.distribution,
                          tfd.Normal)
    self.assertIsInstance(surrogate_dists.weights, tfd.Normal)'''

# TODO(kateslin): Add an ASVI surrogate posterior test for gamma distributions.
# TODO(kateslin): Add an ASVI surrogate posterior test with for a model with
#  missing observations.

if __name__ == '__main__':
  tf.test.main()
