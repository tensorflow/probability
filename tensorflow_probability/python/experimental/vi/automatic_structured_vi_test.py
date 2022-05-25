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

# Dependency imports

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.experimental.vi import automatic_structured_vi
from tensorflow_probability.python.internal import custom_gradient
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions

JAX_MODE = False


@test_util.test_all_tf_execution_regimes
class _TrainableASVISurrogate(object):

  @test_util.jax_disable_variable_test
  def _expected_num_trainable_variables(self, prior_dist):
    """Infers the expected number of trainable variables for a non-nested JD."""
    prior_dists = prior_dist._get_single_sample_distributions()  # pylint: disable=protected-access
    expected_num_trainable_variables = 0
    for original_dist in prior_dists:
      try:
        original_dist = original_dist.distribution
      except AttributeError:
        pass
      dist = automatic_structured_vi._as_substituted_distribution(
          original_dist,
          prior_substitution_rules=(
              automatic_structured_vi.ASVI_DEFAULT_PRIOR_SUBSTITUTION_RULES))
      dist_params = dist.parameters
      for param, value in dist_params.items():
        if (param not in automatic_structured_vi._NON_STATISTICAL_PARAMS
            and value is not None and param not in ('low', 'high')):
          # One variable each for prior_weight, mean_field_parameter.
          expected_num_trainable_variables += 2
    return expected_num_trainable_variables

  @test_util.jax_disable_variable_test
  def test_dims_and_gradients(self):

    prior_dist = self.make_prior_dist()

    surrogate_posterior = tfp.experimental.vi.build_asvi_surrogate_posterior(
        prior=prior_dist)

    # Test that the correct number of trainable variables are being tracked
    self.assertLen(surrogate_posterior.trainable_variables,
                   self._expected_num_trainable_variables(prior_dist))

    # Test that the sample shape is correct
    three_posterior_samples = surrogate_posterior.sample(
        3, seed=test_util.test_seed(sampler_type='stateless'))
    three_prior_samples = prior_dist.sample(
        3, seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllEqualNested(
        [s.shape for s in tf.nest.flatten(three_prior_samples)],
        [s.shape for s in tf.nest.flatten(three_posterior_samples)])

    # Test that gradients are available wrt the variational parameters.
    posterior_sample = surrogate_posterior.sample(
        seed=test_util.test_seed(sampler_type='stateless'))
    with tf.GradientTape() as tape:
      posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))

  def test_dims_and_gradients_stateless(self):

    prior_dist = self.make_prior_dist()

    surrogate_init_fn, surrogate_apply_fn = (
        tfp.experimental.vi.build_asvi_surrogate_posterior_stateless(
            prior=prior_dist))
    raw_params = surrogate_init_fn(
        seed=test_util.test_seed(sampler_type='stateless'))

    # Test that the correct number of trainable variables are being tracked
    self.assertLen(tf.nest.flatten(raw_params),
                   self._expected_num_trainable_variables(prior_dist))

    # Test that the sample shape is correct.
    surrogate_posterior = surrogate_apply_fn(raw_params)
    three_posterior_samples = surrogate_posterior.sample(
        3, seed=test_util.test_seed(sampler_type='stateless'))
    three_prior_samples = prior_dist.sample(
        3, seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllEqualNested(
        [s.shape for s in tf.nest.flatten(three_prior_samples)],
        [s.shape for s in tf.nest.flatten(three_posterior_samples)])

    # Test that gradients are available wrt the variational parameters.
    posterior_sample = surrogate_posterior.sample(
        seed=test_util.test_seed(sampler_type='stateless'))
    _, grad = tfp.math.value_and_gradient(
        lambda params: surrogate_apply_fn(params).log_prob(posterior_sample),
        [raw_params])
    self.assertTrue(
        all(custom_gradient.is_valid_gradient(g)
            for g in tf.nest.flatten(grad)))

  @test_util.jax_disable_variable_test
  def test_initialization_is_deterministic_following_seed(self):
    prior_dist = self.make_prior_dist()
    seed = test_util.test_seed(sampler_type='stateless')
    init_seed, sample_seed = tfp.random.split_seed(seed)

    surrogate_posterior = tfp.experimental.vi.build_asvi_surrogate_posterior(
        prior=prior_dist, seed=init_seed)
    self.evaluate(
        [v.initializer for v in surrogate_posterior.trainable_variables])
    posterior_sample = surrogate_posterior.sample(seed=sample_seed)

    surrogate_posterior2 = tfp.experimental.vi.build_asvi_surrogate_posterior(
        prior=prior_dist, seed=init_seed)
    self.evaluate(
        [v.initializer for v in surrogate_posterior2.trainable_variables])
    posterior_sample2 = surrogate_posterior2.sample(seed=sample_seed)

    self.assertAllEqualNested(posterior_sample, posterior_sample2)


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
    ground_truth = prior_dist.sample(
        seed=test_util.test_seed(sampler_type='stateless'))
    likelihood = self.make_likelihood_model(
        x=ground_truth, observation_noise=observation_noise)
    return likelihood.sample(
        1, seed=test_util.test_seed(sampler_type='stateless'))

  def get_target_log_prob(self, observations, prior_dist):

    def target_log_prob(*x):
      observation_noise = 0.15
      likelihood_dist = self.make_likelihood_model(
          x=x, observation_noise=observation_noise)
      return likelihood_dist.log_prob(observations) + prior_dist.log_prob(x)

    return target_log_prob

  @test_util.jax_disable_variable_test
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
      posterior_samples = surrogate_posterior.sample(
          100, seed=test_util.test_seed(sampler_type='stateless'))
      posterior_mean = tf.nest.map_structure(tf.reduce_mean, posterior_samples)
      posterior_stddev = tf.nest.map_structure(tf.math.reduce_std,
                                               posterior_samples)

    self.evaluate(tf1.global_variables_initializer())
    _ = self.evaluate(losses)
    _ = self.evaluate(posterior_mean)
    _ = self.evaluate(posterior_stddev)

  def test_fitting_surrogate_posterior_stateless(self):
    if not JAX_MODE:
      self.skipTest('Requires optax.')
    import optax  # pylint: disable=g-import-not-at-top

    prior_dist = self.make_prior_dist()
    observations = self.get_observations(prior_dist)
    init_fn, build_surrogate_posterior_fn = (
        tfp.experimental.vi.build_asvi_surrogate_posterior_stateless(
            prior=prior_dist))
    target_log_prob = self.get_target_log_prob(observations, prior_dist)

    def loss_fn(*params, seed=None):
      surrogate_posterior = build_surrogate_posterior_fn(*params)
      zs, q_lp = surrogate_posterior.experimental_sample_and_log_prob(
          10, seed=seed)
      return tf.reduce_mean(q_lp - target_log_prob(*zs), axis=0)

    # Test vi fit surrogate posterior works
    optimized_params, _ = tfp.math.minimize_stateless(
        loss_fn,
        init=init_fn(seed=test_util.test_seed()),
        num_steps=5,  # Don't optimize to completion.
        optimizer=optax.adam(0.1),
        seed=test_util.test_seed(sampler_type='stateless'))
    surrogate_posterior = build_surrogate_posterior_fn(optimized_params)
    surrogate_posterior.sample(
        100, seed=test_util.test_seed(sampler_type='stateless'))


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


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestNesting(test_util.TestCase,
                                        _TrainableASVISurrogate):

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
class ASVISurrogatePosteriorTestMarkovChain(test_util.TestCase,
                                            _TrainableASVISurrogate):

  def _expected_num_trainable_variables(self, _):
    return 16

  def make_prior_dist(self):
    num_timesteps = 10
    def stochastic_volatility_prior_fn():
      """Generative process for a stochastic volatility model."""
      persistence_of_volatility = 0.9
      mean_log_volatility = yield tfd.Cauchy(
          loc=0., scale=5., name='mean_log_volatility')
      white_noise_shock_scale = yield tfd.HalfCauchy(
          loc=0., scale=2., name='white_noise_shock_scale')
      _ = yield tfd.MarkovChain(
          initial_state_prior=tfd.Normal(
              loc=mean_log_volatility,
              scale=white_noise_shock_scale / tf.math.sqrt(
                  tf.ones([]) - persistence_of_volatility**2)),
          transition_fn=lambda _, x_t: tfd.Normal(  # pylint: disable=g-long-lambda
              loc=persistence_of_volatility * (
                  x_t -  mean_log_volatility) + mean_log_volatility,
              scale=white_noise_shock_scale),
          num_steps=num_timesteps,
          name='log_volatility')

    return tfd.JointDistributionCoroutineAutoBatched(
        stochastic_volatility_prior_fn)


@test_util.test_all_tf_execution_regimes
class TestASVISubstitutionAndSurrogateRules(test_util.TestCase):

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

    init_fn, apply_fn = (
        tfp.experimental.vi.build_asvi_surrogate_posterior_stateless(model))
    surrogate = apply_fn(init_fn(seed=test_util.test_seed()))
    self.assertAllEqualNested(model.event_shape, surrogate.event_shape)

    surrogate_dists, _ = surrogate.sample_distributions(
        seed=test_util.test_seed(sampler_type='stateless'))
    self.assertIsInstance(surrogate_dists.a, independent._Independent)
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

    init_fn, apply_fn = (
        tfp.experimental.vi.build_asvi_surrogate_posterior_stateless(
            centered_horseshoe,
            prior_substitution_rules=tuple([
                (tfd.HalfCauchy,
                 lambda d: tfb.Softplus(1e-6)(  # pylint: disable=g-long-lambda
                     tfd.Normal(loc=d.loc, scale=d.scale)))
            ]) + automatic_structured_vi.ASVI_DEFAULT_PRIOR_SUBSTITUTION_RULES))

    surrogate = apply_fn(init_fn(seed=test_util.test_seed()))
    self.assertAllEqualNested(centered_horseshoe.event_shape,
                              surrogate.event_shape)

    # If the surrogate was built with names or structure differing from the
    # model, so that it had to be `tfb.Restructure`'d, then this
    # sample_distributions call will fail because the surrogate isn't an
    # instance of tfd.JointDistribution.
    surrogate_dists, _ = surrogate.sample_distributions(
        seed=test_util.test_seed(sampler_type='stateless'))
    self.assertIsInstance(surrogate_dists.global_scale.distribution,
                          tfd.Normal)
    self.assertIsInstance(surrogate_dists.local_scale.distribution,
                          tfd.Normal)
    self.assertIsInstance(surrogate_dists.weights, tfd.Normal)

  def test_can_specify_custom_surrogate(self):

    def student_t_surrogate(
        dist, build_nested_surrogate, sample_shape=None):
      del build_nested_surrogate  # Unused.
      del sample_shape  # Unused.
      return tfp.experimental.util.make_trainable_stateless(
          tfd.StudentT, batch_and_event_shape=dist.batch_shape)

    init_fn, apply_fn = (
        tfp.experimental.vi.build_asvi_surrogate_posterior_stateless(
            tfd.Normal(2., scale=[3., 1.]),
            surrogate_rules=(
                (tfd.Normal, student_t_surrogate),
                ) + automatic_structured_vi.ASVI_DEFAULT_SURROGATE_RULES))
    surrogate = apply_fn(init_fn(seed=test_util.test_seed()))
    self.assertIsInstance(surrogate, tfd.StudentT)


# TODO(kateslin): Add an ASVI surrogate posterior test for gamma distributions.
# TODO(kateslin): Add an ASVI surrogate posterior test with for a model with
#  missing observations.

if __name__ == '__main__':
  test_util.main()
