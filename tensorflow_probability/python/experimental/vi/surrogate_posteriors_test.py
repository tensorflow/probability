# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for surrogate posteriors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.vi import surrogate_posteriors
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class TrainableLocationScale(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'ScalarLaplace',
       'event_shape': [],
       'batch_shape': [],
       'distribution_fn': tfd.Laplace,
       'dtype': np.float64},
      {'testcase_name': 'VectorNormal',
       'event_shape': [2],
       'batch_shape': [3, 1],
       'distribution_fn': tfd.Normal,
       'dtype': np.float32},)
  def test_has_correct_ndims_and_gradients(
      self, event_shape, batch_shape, distribution_fn, dtype):

    initial_loc = np.ones(batch_shape + event_shape)
    dist = tfp.experimental.vi.build_trainable_location_scale_distribution(
        initial_loc=_build_tensor(initial_loc, dtype=dtype,
                                  use_static_shape=True),
        initial_scale=1e-6,
        event_ndims=len(event_shape),
        distribution_fn=distribution_fn,
        validate_args=True)
    self.evaluate([v.initializer for v in dist.trainable_variables])
    self.assertAllClose(self.evaluate(dist.sample()),
                        initial_loc,
                        atol=1e-4)  # Much larger than initial_scale.
    self.assertAllEqual(dist.event_shape.as_list(), event_shape)
    self.assertAllEqual(dist.batch_shape.as_list(), batch_shape)
    for v in dist.trainable_variables:
      self.assertAllEqual(v.shape.as_list(), batch_shape + event_shape)

    # Test that gradients are available wrt the variational parameters.
    self.assertNotEmpty(dist.trainable_variables)
    with tf.GradientTape() as tape:
      posterior_logprob = dist.log_prob(initial_loc)
    grad = tape.gradient(posterior_logprob,
                         dist.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))


@test_util.test_all_tf_execution_regimes
class FactoredSurrogatePosterior(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEvent',
       'event_shape': tf.TensorShape([4]),
       'constraining_bijectors': [tfb.Sigmoid()],
       'dtype': np.float64, 'use_static_shape': True},
      {'testcase_name': 'ListEvent',
       'event_shape': [tf.TensorShape([3]),
                       tf.TensorShape([]),
                       tf.TensorShape([2, 2])],
       'constraining_bijectors': [tfb.Softplus(), None, tfb.FillTriangular()],
       'dtype': np.float32, 'use_static_shape': False},
      {'testcase_name': 'DictEvent',
       'event_shape': {'x': tf.TensorShape([1]), 'y': tf.TensorShape([])},
       'constraining_bijectors': None,
       'dtype': np.float64, 'use_static_shape': True},
      {'testcase_name': 'NestedEvent',
       'event_shape': {'x': [tf.TensorShape([1]), tf.TensorShape([1, 2])],
                       'y': tf.TensorShape([])},
       'constraining_bijectors': {
           'x': [tfb.Identity(), tfb.Softplus()], 'y': tfb.Sigmoid()},
       'dtype': np.float32, 'use_static_shape': True},
  )
  def test_specifying_event_shape(self, event_shape,
                                  constraining_bijectors,
                                  dtype, use_static_shape):
    seed = test_util.test_seed_stream()
    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=tf.nest.map_structure(
                functools.partial(_build_tensor,
                                  dtype=np.int32,
                                  use_static_shape=use_static_shape),
                event_shape),
            constraining_bijectors=constraining_bijectors,
            initial_unconstrained_loc=functools.partial(
                tf.random.uniform, minval=-2., maxval=2., dtype=dtype),
            seed=seed(),
            validate_args=True))
    self.evaluate([v.initializer
                   for v in surrogate_posterior.trainable_variables])
    posterior_sample_ = self.evaluate(surrogate_posterior.sample(seed=seed()))
    posterior_logprob_ = self.evaluate(
        surrogate_posterior.log_prob(posterior_sample_))
    posterior_event_shape = self.evaluate(
        surrogate_posterior.event_shape_tensor())

    # Test that the posterior has the specified event shape(s).
    tf.nest.map_structure(
        self.assertAllEqual, event_shape, posterior_event_shape)

    # Test that all sample Tensors have the expected shapes.
    check_shape = lambda s, x: self.assertAllEqual(s, x.shape)
    tf.nest.map_structure(check_shape, event_shape, posterior_sample_)

    self.assertAllEqual([], posterior_logprob_.shape)

    # Test that gradients are available wrt the variational parameters.
    self.assertNotEmpty(surrogate_posterior.trainable_variables)
    with tf.GradientTape() as tape:
      posterior_logprob = surrogate_posterior.log_prob(posterior_sample_)
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEvent',
       'event_shape': [4],
       'initial_loc': np.array([[[0.9, 0.1, 0.5, 0.7]]]),
       'implicit_batch_shape': [1, 1],
       'constraining_bijectors': tfb.Sigmoid(),
       'dtype': np.float32, 'use_static_shape': False},
      {'testcase_name': 'ListEvent',
       'event_shape': [[3], [], [2, 2]],
       'initial_loc': [np.array([0.1, 7., 3.]),
                       0.1,
                       np.array([[1., 0], [-4., 2.]])],
       'implicit_batch_shape': [],
       'constraining_bijectors': [tfb.Softplus(), None, tfb.FillTriangular()],
       'dtype': np.float64, 'use_static_shape': True},
      {'testcase_name': 'DictEvent',
       'event_shape': {'x': [2], 'y': []},
       'initial_loc': {'x': np.array([[0.9, 1.2]]),
                       'y': np.array([-4.1])},
       'implicit_batch_shape': [1],
       'constraining_bijectors': None,
       'dtype': np.float32, 'use_static_shape': False},
  )
  def test_specifying_initial_loc(self, event_shape, initial_loc,
                                  implicit_batch_shape,
                                  constraining_bijectors,
                                  dtype, use_static_shape):
    initial_loc = tf.nest.map_structure(
        lambda s: _build_tensor(s, dtype=dtype,  # pylint: disable=g-long-lambda
                                use_static_shape=use_static_shape),
        initial_loc)

    if constraining_bijectors is not None:
      initial_unconstrained_loc = tf.nest.map_structure(
          lambda x, b: x if b is None else b.inverse(x),
          initial_loc, constraining_bijectors)
    else:
      initial_unconstrained_loc = initial_loc

    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=event_shape,
            initial_unconstrained_loc=initial_unconstrained_loc,
            initial_unconstrained_scale=1e-6,
            constraining_bijectors=constraining_bijectors,
            validate_args=True))
    self.evaluate([v.initializer
                   for v in surrogate_posterior.trainable_variables])
    posterior_sample_ = self.evaluate(surrogate_posterior.sample(
        seed=test_util.test_seed()))
    posterior_logprob_ = self.evaluate(
        surrogate_posterior.log_prob(posterior_sample_))

    self.assertAllEqual(implicit_batch_shape, posterior_logprob_.shape)

    # Check that the samples have the correct structure and that the sampled
    # values are close to the initial locs.
    tf.nest.map_structure(functools.partial(self.assertAllClose, atol=1e-4),
                          self.evaluate(initial_loc),
                          posterior_sample_)

  def test_that_gamma_fitting_example_runs(self):

    Root = tfd.JointDistributionCoroutine.Root  # pylint: disable=invalid-name

    # Build model.
    def model_fn():
      concentration = yield Root(tfd.Exponential(1.))
      rate = yield Root(tfd.Exponential(1.))
      y = yield tfd.Sample(  # pylint: disable=unused-variable
          tfd.Gamma(concentration=concentration, rate=rate),
          sample_shape=4)
    model = tfd.JointDistributionCoroutine(model_fn)

    # Build surrogate posterior.
    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=model.event_shape_tensor()[:-1],
            constraining_bijectors=[tfb.Softplus(), tfb.Softplus()]))

    # Fit model.
    y = [0.2, 0.5, 0.3, 0.7]
    losses = tfp.vi.fit_surrogate_posterior(
        lambda rate, concentration: model.log_prob((rate, concentration, y)),
        surrogate_posterior,
        num_steps=5,  # Don't optimize to completion.
        optimizer=tf.optimizers.Adam(0.1),
        sample_size=10)

    # Compute posterior statistics.
    with tf.control_dependencies([losses]):
      posterior_samples = surrogate_posterior.sample(100)
      posterior_mean = [tf.reduce_mean(x) for x in posterior_samples]
      posterior_stddev = [tf.math.reduce_std(x) for x in posterior_samples]

    self.evaluate(tf1.global_variables_initializer())
    _ = self.evaluate(losses)
    _ = self.evaluate(posterior_mean)
    _ = self.evaluate(posterior_stddev)

  def test_multipart_bijector(self):
    dist = tfd.JointDistributionNamed({
        'a': tfd.Exponential(1.),
        'b': tfd.Normal(0., 1.),
        'c': lambda b, a: tfd.Sample(tfd.Normal(b, a), sample_shape=[5])})

    seed = test_util.test_seed_stream()
    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=dist.event_shape,
            constraining_bijectors=(
                dist.experimental_default_event_space_bijector()),
            initial_unconstrained_loc=functools.partial(
                tf.random.uniform, minval=-2., maxval=2.),
            seed=seed(),
            validate_args=True))
    self.evaluate([v.initializer
                   for v in surrogate_posterior.trainable_variables])

    # Test that the posterior has the specified event shape(s).
    self.assertAllEqualNested(
        self.evaluate(dist.event_shape_tensor()),
        self.evaluate(surrogate_posterior.event_shape_tensor()))

    posterior_sample_ = self.evaluate(surrogate_posterior.sample(seed=seed()))
    posterior_logprob_ = self.evaluate(
        surrogate_posterior.log_prob(posterior_sample_))

    # Test that all sample Tensors have the expected shapes.
    check_shape = lambda s, x: self.assertAllEqual(s, x.shape)
    self.assertAllAssertsNested(
        check_shape, dist.event_shape, posterior_sample_)

    # Test that samples are finite and not NaN.
    self.assertAllAssertsNested(self.assertAllFinite, posterior_sample_)

    # Test that logprob is scalar, finite, and not NaN.
    self.assertEmpty(posterior_logprob_.shape)
    self.assertAllFinite(posterior_logprob_)


def _build_tensor(ndarray, dtype, use_static_shape):
  # Enforce parameterized dtype and static/dynamic testing.
  ndarray = np.asarray(ndarray).astype(dtype)
  return tf1.placeholder_with_default(
      input=ndarray, shape=ndarray.shape if use_static_shape else None)


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
      dist = surrogate_posteriors._as_trainable_family(original_dist)
      dist_params = dist.parameters
      for param, value in dist_params.items():
        if (param not in surrogate_posteriors._NON_STATISTICAL_PARAMS
            and value is not None and param not in ('low', 'high')):
          expected_num_trainable_vars += 2  # prior_weight, mean_field_parameter

    self.assertLen(surrogate_posterior.trainable_variables,
                   expected_num_trainable_vars)

    # Test that gradients are available wrt the variational parameters.
    posterior_sample = surrogate_posterior.sample()
    with tf.GradientTape() as tape:
      posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))

    # Test that the sample shape is correct
    three_posterior_samples = surrogate_posterior.sample(3)
    three_prior_samples = prior_dist.sample(3)
    self.assertAllEqualNested(
        [s.shape for s in tf.nest.flatten(three_prior_samples)],
        [s.shape for s in tf.nest.flatten(three_posterior_samples)])

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

  def test_make_asvi_trainable_variables(self):
    prior_dist = self.make_prior_dist()
    trained_vars = surrogate_posteriors._make_asvi_trainable_variables(
        prior=prior_dist)

    # Confirm that there is one dictionary per distribution.
    prior_dists = prior_dist._get_single_sample_distributions()  # pylint: disable=protected-access
    self.assertEqual(len(trained_vars), len(prior_dists))

    # Confirm that there exists correct number of trainable variables.
    for (prior_distribution, trained_vars_dict) in zip(prior_dists,
                                                       trained_vars):
      substituted_dist = surrogate_posteriors._as_trainable_family(
          prior_distribution)
      try:
        posterior_distribution = substituted_dist.distribution
      except AttributeError:
        posterior_distribution = substituted_dist

      for param_name, prior_value in posterior_distribution.parameters.items():
        if (param_name not in surrogate_posteriors._NON_STATISTICAL_PARAMS
            and prior_value is not None and param_name not in ('low', 'high')):
          self.assertIsInstance(trained_vars_dict[param_name],
                                surrogate_posteriors.ASVIParameters)


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

  def make_likelihood_model(self, x, observation_noise=None):
    treatment_stddevs = tf.constant([15, 10, 16, 11, 9, 11, 10, 18],
                                    dtype=tf.float32)

    return tfd.Independent(
        tfd.Normal(loc=x['school_effects'], scale=treatment_stddevs),
        reinterpreted_batch_ndims=1)

  def get_observations(self, prior_dist):
    ground_truth = self.evaluate(prior_dist.sample())
    likelihood = self.make_likelihood_model(x=ground_truth)
    return likelihood.sample(1)

  def get_target_log_prob(self, observations, prior_dist):

    def target_log_prob(**x):
      likelihood_dist = self.make_likelihood_model(x=x)
      return likelihood_dist.log_prob(observations) + prior_dist.log_prob(x)

    return target_log_prob


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestHalfNormal(test_util.TestCase,
                                           _TrainableASVISurrogate):

  def make_prior_dist(self):

    def _prior_model_fn():
      innovation_noise = 1.
      yield tfd.HalfNormal(
          scale=innovation_noise, validate_args=True, allow_nan_stats=False)

    return tfd.JointDistributionCoroutineAutoBatched(_prior_model_fn)

  def make_likelihood_model(self, x, observation_noise):

    def _likelihood_model():
      yield tfd.Normal(
          loc=x,
          scale=observation_noise,
          validate_args=True,
          allow_nan_stats=False)

    return tfd.JointDistributionCoroutineAutoBatched(_likelihood_model)

  def get_observations(self, prior_dist):
    observation_noise = 1.
    ground_truth = prior_dist.sample()
    likelihood = self.make_likelihood_model(
        x=ground_truth, observation_noise=observation_noise)
    return likelihood.sample(1)

  def get_target_log_prob(self, observations, prior_dist):

    obs = observations
    def target_log_prob(*x):
      observation_noise = 0.15
      likelihood_dist = self.make_likelihood_model(
          x=x, observation_noise=observation_noise)

      return likelihood_dist.log_prob(obs) + prior_dist.log_prob(x)

    return target_log_prob

# TODO(kateslin): Add an ASVI surrogate posterior test for gamma distributions.
# TODO(kateslin): Add an ASVI surrogate posterior test with for a model with
#  missing observations.
# TODO(kateslin): Add an ASVI surrogate posterior test for Uniform distribution
# to check that Beta substitution works properly

if __name__ == '__main__':
  tf.test.main()
