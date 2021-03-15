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
from tensorflow_probability.python.experimental.vi import automatic_structured_vi
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class _TrainableASVISurrogate(object):

  def test_dims_and_gradients(self):

    prior_dist = self.make_prior_dist()

    surrogate_posterior = tfp.experimental.vi.build_asvi_surrogate_posterior(
        prior=prior_dist)

    # Test that the correct number of trainable variables are being tracked
    prior_dists = prior_dist._get_single_sample_distributions()  # pylint: disable=protected-access
    expected_num_trainable_vars = 0
    for original_dist in prior_dists:
      try:
        original_dist = original_dist.distribution
      except AttributeError:
        pass
      dist = automatic_structured_vi._as_trainable_family(original_dist)
      dist_params = dist.parameters
      for param, value in dist_params.items():
        if (param not in automatic_structured_vi._NON_STATISTICAL_PARAMS
            and value is not None and param not in ('low', 'high')):
          expected_num_trainable_vars += 2  # prior_weight, mean_field_parameter

    self.assertLen(surrogate_posterior.trainable_variables,
                   expected_num_trainable_vars)

    # Test that the sample shape is correct
    three_posterior_samples = surrogate_posterior.sample(3)
    three_prior_samples = prior_dist.sample(3)
    self.assertAllEqualNested(
        [s.shape for s in tf.nest.flatten(three_prior_samples)],
        [s.shape for s in tf.nest.flatten(three_posterior_samples)])

    # Test that gradients are available wrt the variational parameters.
    posterior_sample = surrogate_posterior.sample()
    with tf.GradientTape() as tape:
      posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))

  def test_make_asvi_trainable_variables(self):
    prior_dist = self.make_prior_dist()
    trained_vars = automatic_structured_vi._make_asvi_trainable_variables(
        prior=prior_dist)

    # Confirm that there is one dictionary per distribution.
    prior_dists = prior_dist._get_single_sample_distributions()  # pylint: disable=protected-access
    self.assertEqual(len(trained_vars), len(prior_dists))

    # Confirm that there exists correct number of trainable variables.
    for (prior_distribution, trained_vars_dict) in zip(prior_dists,
                                                       trained_vars):
      substituted_dist = automatic_structured_vi._as_trainable_family(
          prior_distribution)
      try:
        posterior_distribution = substituted_dist.distribution
      except AttributeError:
        posterior_distribution = substituted_dist

      for param_name, prior_value in posterior_distribution.parameters.items():
        if (param_name not in automatic_structured_vi._NON_STATISTICAL_PARAMS
            and prior_value is not None and param_name not in ('low', 'high')):
          self.assertIsInstance(trained_vars_dict[param_name],
                                automatic_structured_vi.ASVIParameters)


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestBrownianMotion(test_util.TestCase,
                                               _TrainableASVISurrogate):

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

  def test_fitting_surrogate_posterior(self):

    prior_dist = self.make_prior_dist()
    observations = self.get_observations(prior_dist)
    surrogate_posterior = tfp.experimental.vi.build_asvi_surrogate_posterior(
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
    _ = self.evaluate(posterior_stddev)


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestEightSchools(test_util.TestCase,
                                             _TrainableASVISurrogate):

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
class ASVISurrogatePosteriorTestEightSchoolsSample(test_util.TestCase,
                                                   _TrainableASVISurrogate):

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
class ASVISurrogatePosteriorTestHalfNormal(test_util.TestCase,
                                           _TrainableASVISurrogate):

  def make_prior_dist(self):

    def _prior_model_fn():
      innovation_noise = 1.
      yield tfd.HalfNormal(
          scale=innovation_noise, validate_args=True, allow_nan_stats=False)

    return tfd.JointDistributionCoroutineAutoBatched(_prior_model_fn)


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestDiscreteLatent(
    test_util.TestCase, _TrainableASVISurrogate):

  def make_prior_dist(self):

    def _prior_model_fn():
      a = yield tfd.Bernoulli(logits=0.5, name='a')
      yield tfd.Normal(loc=2. * tf.cast(a, tf.float32) - 1.,
                       scale=1., name='b')

    return tfd.JointDistributionCoroutineAutoBatched(_prior_model_fn)


# TODO(kateslin): Add an ASVI surrogate posterior test for gamma distributions.
# TODO(kateslin): Add an ASVI surrogate posterior test with for a model with
#  missing observations.
# TODO(kateslin): Add an ASVI surrogate posterior test for Uniform distribution
# to check that Beta substitution works properly

if __name__ == '__main__':
  tf.test.main()
