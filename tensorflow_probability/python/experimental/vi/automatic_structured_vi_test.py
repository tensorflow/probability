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

from absl.testing import parameterized
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import sigmoid
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.bijectors import square
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import cauchy
from tensorflow_probability.python.distributions import chi2
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import half_cauchy
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import laplace
from tensorflow_probability.python.distributions import markov_chain
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import truncated_normal
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental.util import trainable
from tensorflow_probability.python.experimental.vi import automatic_structured_vi
from tensorflow_probability.python.internal import custom_gradient
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.minimize import minimize_stateless
from tensorflow_probability.python.vi import optimization


JAX_MODE = False


class _TrainableASVISurrogate(test_util.TestCase):

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

  @parameterized.named_parameters(
      ('_32', tf.float32),
      ('_64', tf.float64),
  )
  @test_util.jax_disable_variable_test
  def test_dims_and_gradients(self, dtype):

    prior_dist = self.make_prior_dist(dtype)

    surrogate_posterior = automatic_structured_vi.build_asvi_surrogate_posterior(
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

    @tf.function
    def log_prob():
      return surrogate_posterior.log_prob(posterior_sample)

    with tf.GradientTape() as tape:
      posterior_logprob = log_prob()
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))

  @parameterized.named_parameters(
      ('_32', tf.float32),
      ('_64', tf.float64),
  )
  def test_dims_and_gradients_stateless(self, dtype):

    prior_dist = self.make_prior_dist(dtype)

    surrogate_init_fn, surrogate_apply_fn = (
        automatic_structured_vi.build_asvi_surrogate_posterior_stateless(
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

    @tf.function
    def log_prob(params):
      return surrogate_apply_fn(params).log_prob(posterior_sample)

    _, grad = gradient.value_and_gradient(log_prob, [raw_params])
    self.assertTrue(
        all(custom_gradient.is_valid_gradient(g)
            for g in tf.nest.flatten(grad)))

  @test_util.jax_disable_variable_test
  def test_initialization_is_deterministic_following_seed(self):
    prior_dist = self.make_prior_dist(tf.float32)
    seed = test_util.test_seed(sampler_type='stateless')
    init_seed, sample_seed = samplers.split_seed(seed)

    surrogate_posterior = automatic_structured_vi.build_asvi_surrogate_posterior(
        prior=prior_dist, seed=init_seed)
    self.evaluate(
        [v.initializer for v in surrogate_posterior.trainable_variables])
    posterior_sample = surrogate_posterior.sample(seed=sample_seed)

    surrogate_posterior2 = automatic_structured_vi.build_asvi_surrogate_posterior(
        prior=prior_dist, seed=init_seed)
    self.evaluate(
        [v.initializer for v in surrogate_posterior2.trainable_variables])
    posterior_sample2 = surrogate_posterior2.sample(seed=sample_seed)

    self.assertAllEqualNested(posterior_sample, posterior_sample2)


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestBrownianMotion(_TrainableASVISurrogate):

  def make_prior_dist(self, dtype):

    def _prior_model_fn():
      innovation_noise = tf.constant(0.1, dtype)
      prior_loc = tf.constant(0., dtype)
      new = yield normal.Normal(loc=prior_loc, scale=innovation_noise)
      for _ in range(4):
        new = yield normal.Normal(loc=new, scale=innovation_noise)

    return jdab.JointDistributionCoroutineAutoBatched(_prior_model_fn)

  def make_likelihood_model(self, x, observation_noise):

    def _likelihood_model():
      for i in range(5):
        yield normal.Normal(loc=x[i], scale=observation_noise)

    return jdab.JointDistributionCoroutineAutoBatched(_likelihood_model)

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

  @parameterized.named_parameters(
      ('_32', tf.float32),
      ('_64', tf.float64),
  )
  @test_util.jax_disable_variable_test
  def test_fitting_surrogate_posterior(self, dtype):

    prior_dist = self.make_prior_dist(dtype)
    observations = self.get_observations(prior_dist)
    surrogate_posterior = automatic_structured_vi.build_asvi_surrogate_posterior(
        prior=prior_dist)
    target_log_prob = self.get_target_log_prob(observations, prior_dist)

    # Test vi fit surrogate posterior works
    losses = optimization.fit_surrogate_posterior(
        target_log_prob,
        surrogate_posterior,
        num_steps=3,  # Don't optimize to completion.
        optimizer=tf.optimizers.Adam(0.1),
        sample_size=5)

    # Compute posterior statistics.
    with tf.control_dependencies([losses]):
      posterior_samples = surrogate_posterior.sample(
          20, seed=test_util.test_seed(sampler_type='stateless'))
      posterior_mean = tf.nest.map_structure(tf.reduce_mean, posterior_samples)
      posterior_stddev = tf.nest.map_structure(tf.math.reduce_std,
                                               posterior_samples)

    self.evaluate(tf1.global_variables_initializer())
    _ = self.evaluate(losses)
    _ = self.evaluate(posterior_mean)
    _ = self.evaluate(posterior_stddev)

  @parameterized.named_parameters(
      ('_32', tf.float32),
      ('_64', tf.float64),
  )
  def test_fitting_surrogate_posterior_stateless(self, dtype):
    if not JAX_MODE:
      self.skipTest('Requires optax.')
    import optax  # pylint: disable=g-import-not-at-top

    prior_dist = self.make_prior_dist(dtype)
    observations = self.get_observations(prior_dist)
    init_fn, build_surrogate_posterior_fn = (
        automatic_structured_vi.build_asvi_surrogate_posterior_stateless(
            prior=prior_dist))
    target_log_prob = self.get_target_log_prob(observations, prior_dist)

    def loss_fn(*params, seed=None):
      surrogate_posterior = build_surrogate_posterior_fn(*params)
      zs, q_lp = surrogate_posterior.experimental_sample_and_log_prob(
          10, seed=seed)
      return tf.reduce_mean(q_lp - target_log_prob(*zs), axis=0)

    # Test vi fit surrogate posterior works
    optimized_params, _ = minimize_stateless(
        loss_fn,
        init=init_fn(seed=test_util.test_seed()),
        num_steps=3,  # Don't optimize to completion.
        optimizer=optax.adam(0.1),
        seed=test_util.test_seed(sampler_type='stateless'))
    surrogate_posterior = build_surrogate_posterior_fn(optimized_params)
    surrogate_posterior.sample(
        20, seed=test_util.test_seed(sampler_type='stateless'))


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestEightSchools(_TrainableASVISurrogate):

  def make_prior_dist(self, dtype):
    treatment_effects = tf.constant([28, 8, -3, 7, -1, 1, 18, 12],
                                    dtype=dtype)
    num_schools = ps.shape(treatment_effects)[-1]

    return jdn.JointDistributionNamed({
        'avg_effect':
            normal.Normal(
                loc=tf.constant(0., dtype), scale=10., name='avg_effect'),
        'log_stddev':
            normal.Normal(
                loc=tf.constant(5., dtype), scale=1., name='log_stddev'),
        'school_effects':
            lambda log_stddev, avg_effect: (  # pylint: disable=g-long-lambda
                independent.Independent(
                    normal.Normal(
                        loc=avg_effect[..., None] * tf.ones(num_schools, dtype),
                        scale=tf.exp(log_stddev[..., None]) * tf.ones(
                            num_schools, dtype),
                        name='school_effects'),
                    reinterpreted_batch_ndims=1))
    })


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestEightSchoolsSample(_TrainableASVISurrogate):

  def make_prior_dist(self, dtype):

    return jdn.JointDistributionNamed({
        'avg_effect':
            normal.Normal(
                loc=tf.constant(0., dtype), scale=10., name='avg_effect'),
        'log_stddev':
            normal.Normal(
                loc=tf.constant(5., dtype), scale=1., name='log_stddev'),
        'school_effects':
            lambda log_stddev, avg_effect: (  # pylint: disable=g-long-lambda
                sample.Sample(
                    normal.Normal(
                        loc=avg_effect[..., None],
                        scale=tf.exp(log_stddev[..., None]),
                        name='school_effects'),
                    sample_shape=[8]))
    })


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestHalfNormal(_TrainableASVISurrogate):

  def make_prior_dist(self, dtype):

    def _prior_model_fn():
      innovation_noise = tf.constant(1., dtype)
      yield half_normal.HalfNormal(
          scale=innovation_noise, validate_args=True, allow_nan_stats=False)

    return jdab.JointDistributionCoroutineAutoBatched(_prior_model_fn)


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestDiscreteLatent(_TrainableASVISurrogate):

  def make_prior_dist(self, dtype):

    def _prior_model_fn():
      a = yield bernoulli.Bernoulli(logits=tf.constant(0.5, dtype), name='a')
      yield normal.Normal(
          loc=2. * tf.cast(a, dtype) - 1., scale=1., name='b')

    return jdab.JointDistributionCoroutineAutoBatched(_prior_model_fn)


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestNesting(_TrainableASVISurrogate):

  def _expected_num_trainable_variables(self, _):
    # Nested distributions have total of 10 params after Exponential->Gamma
    # substitution, multiplied by 2 variables per param.
    return 20

  def make_prior_dist(self, dtype):

    def nested_model():
      a = yield sample.Sample(
          sample.Sample(
              normal.Normal(tf.constant(0., dtype), 1.), sample_shape=4),
          sample_shape=[2],
          name='a')
      b = yield sigmoid.Sigmoid()(
          square.Square()(exponential.Exponential(rate=tf.exp(a))), name='b')
      # pylint: disable=g-long-lambda
      yield jds.JointDistributionSequential(
          [
              laplace.Laplace(
                  loc=a, scale=b), lambda c1: independent.Independent(
                      beta.Beta(
                          concentration1=1., concentration0=tf.nn.softplus(c1)),
                      reinterpreted_batch_ndims=1),
              lambda c1, c2: jdn.JointDistributionNamed(
                  {'x': gamma.Gamma(concentration=tf.nn.softplus(c1), rate=c2)})
          ],
          name='c',
      )
      # pylint: enable=g-long-lambda

    return jdab.JointDistributionCoroutineAutoBatched(nested_model)


@test_util.test_all_tf_execution_regimes
class ASVISurrogatePosteriorTestMarkovChain(_TrainableASVISurrogate):

  def _expected_num_trainable_variables(self, _):
    return 16

  def make_prior_dist(self, dtype):
    num_timesteps = 10
    def stochastic_volatility_prior_fn():
      """Generative process for a stochastic volatility model."""
      persistence_of_volatility = tf.constant(0.9, dtype)
      mean_log_volatility = yield cauchy.Cauchy(
          loc=tf.constant(0., dtype), scale=5., name='mean_log_volatility')
      white_noise_shock_scale = yield half_cauchy.HalfCauchy(
          loc=tf.constant(0., dtype), scale=2., name='white_noise_shock_scale')
      _ = yield markov_chain.MarkovChain(
          initial_state_prior=normal.Normal(
              loc=mean_log_volatility,
              scale=white_noise_shock_scale /
              tf.math.sqrt(1. - persistence_of_volatility**2)),
          transition_fn=lambda _, x_t: normal.Normal(  # pylint: disable=g-long-lambda
              loc=persistence_of_volatility *
              (x_t - mean_log_volatility) + mean_log_volatility,
              scale=white_noise_shock_scale),
          num_steps=num_timesteps,
          name='log_volatility')

    return jdab.JointDistributionCoroutineAutoBatched(
        stochastic_volatility_prior_fn)


@test_util.test_all_tf_execution_regimes
class TestASVISubstitutionAndSurrogateRules(test_util.TestCase):

  @parameterized.named_parameters(
      ('_32', tf.float32),
      ('_64', tf.float64),
  )
  def test_default_substitutes_trainable_families(self, dtype):

    @jdab.JointDistributionCoroutineAutoBatched
    def model():
      yield sample.Sample(
          uniform.Uniform(low=tf.constant(-2., dtype), high=7.),
          sample_shape=[2],
          name='a')
      yield half_normal.HalfNormal(tf.constant(1., dtype), name='b')
      yield exponential.Exponential(rate=tf.constant([1., 2.], dtype), name='c')
      yield chi2.Chi2(df=tf.constant(3., dtype), name='d')

    init_fn, apply_fn = (
        automatic_structured_vi.build_asvi_surrogate_posterior_stateless(model))
    surrogate = apply_fn(init_fn(seed=test_util.test_seed()))
    self.assertAllEqualNested(model.event_shape, surrogate.event_shape)

    surrogate_dists, _ = surrogate.sample_distributions(
        seed=test_util.test_seed(sampler_type='stateless'))
    self.assertIsInstance(surrogate_dists.a, independent.Independent)
    self.assertIsInstance(surrogate_dists.a.distribution,
                          transformed_distribution.TransformedDistribution)
    self.assertIsInstance(surrogate_dists.a.distribution.distribution,
                          beta.Beta)
    self.assertIsInstance(surrogate_dists.b, truncated_normal.TruncatedNormal)
    self.assertIsInstance(surrogate_dists.c, gamma.Gamma)
    self.assertIsInstance(surrogate_dists.d, gamma.Gamma)

  @parameterized.named_parameters(
      ('_32', tf.float32),
      ('_64', tf.float64),
  )
  def test_can_specify_custom_substitution(self, dtype):

    @jdab.JointDistributionCoroutineAutoBatched
    def centered_horseshoe(ndims=100):
      global_scale = yield half_cauchy.HalfCauchy(
          loc=tf.constant(0., dtype), scale=1., name='global_scale')
      local_scale = yield half_cauchy.HalfCauchy(
          loc=0.,
          scale=tf.ones([ndims], dtype),
          name='local_scale')
      yield normal.Normal(
          loc=0.,
          scale=tf.sqrt(global_scale * local_scale),
          name='weights')

    init_fn, apply_fn = (
        automatic_structured_vi.build_asvi_surrogate_posterior_stateless(
            centered_horseshoe,
            prior_substitution_rules=tuple([(
                half_cauchy.HalfCauchy,
                lambda d: softplus.Softplus(tf.constant(1e-6, dtype))(  # pylint: disable=g-long-lambda
                    normal.Normal(loc=d.loc, scale=d.scale)))]) +
            automatic_structured_vi.ASVI_DEFAULT_PRIOR_SUBSTITUTION_RULES))

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
                          normal.Normal)
    self.assertIsInstance(surrogate_dists.local_scale.distribution,
                          normal.Normal)
    self.assertIsInstance(surrogate_dists.weights, normal.Normal)

  @parameterized.named_parameters(
      ('_32', tf.float32),
      ('_64', tf.float64),
  )
  def test_can_specify_custom_surrogate(self, dtype):

    def student_t_surrogate(dist, build_nested_surrogate, sample_shape=None):
      del build_nested_surrogate  # Unused.
      del sample_shape  # Unused.
      return trainable.make_trainable_stateless(
          student_t.StudentT, batch_and_event_shape=dist.batch_shape)

    init_fn, apply_fn = (
        automatic_structured_vi.build_asvi_surrogate_posterior_stateless(
            normal.Normal(tf.constant(2., dtype), scale=[3., 1.]),
            surrogate_rules=((normal.Normal, student_t_surrogate),) +
            automatic_structured_vi.ASVI_DEFAULT_SURROGATE_RULES))
    surrogate = apply_fn(init_fn(seed=test_util.test_seed()))
    self.assertIsInstance(surrogate, student_t.StudentT)


del _TrainableASVISurrogate

# TODO(kateslin): Add an ASVI surrogate posterior test for gamma distributions.
# TODO(kateslin): Add an ASVI surrogate posterior test with for a model with
#  missing observations.

if __name__ == '__main__':
  test_util.main()
