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
"""Tests for ImportanceResample."""

from absl.testing import parameterized

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfed = tfp.experimental.distributions


@test_util.test_graph_and_eager_modes
class ImportanceResampleTest(test_util.TestCase):

  def test_shapes_match_proposal_distribution(self):
    target = tfd.MultivariateNormalTriL(loc=[-2., 1.],
                                        scale_tril=[[1., 0.], [-3., 2.]])
    # Proposal with batch shape.
    proposal = tfd.Independent(
        tfd.StudentT(df=2,
                     loc=0.,
                     scale=[[5., 2.],
                            [1., 3.]]),
        reinterpreted_batch_ndims=1)

    seed = test_util.test_seed(sampler_type='stateless')
    resampled = tfed.ImportanceResample(
        proposal,
        target_log_prob_fn=target.log_prob,
        importance_sample_size=10,
        stochastic_approximation_seed=seed)
    self.assertAllEqual(resampled.event_shape, proposal.event_shape)
    self.assertAllEqual(resampled.event_shape_tensor(),
                        proposal.event_shape_tensor())
    self.assertAllEqual(resampled.batch_shape, proposal.batch_shape)
    self.assertAllEqual(resampled.batch_shape_tensor(),
                        proposal.batch_shape_tensor())

    self.assertAllEqual(resampled.sample(7, seed=seed).shape,
                        proposal.sample(7, seed=seed).shape)

    xs = tf.range(2, dtype=proposal.dtype)
    self.assertAllEqual(resampled.log_prob(xs).shape,
                        proposal.log_prob(xs).shape)

  def test_sample_size_one_reproduces_proposal_distribution(self):
    target = tfd.Normal(loc=0., scale=1.)
    proposal = tfd.Normal(loc=-4., scale=5.)
    resampled = tfed.ImportanceResample(proposal,
                                        target_log_prob_fn=target.log_prob,
                                        importance_sample_size=1)
    xs = self.evaluate(
        resampled.sample(10000,
                         seed=test_util.test_seed(sampler_type='stateless')))
    # Check that samples have statistics matching the proposal distribution.
    self.assertAllClose(tf.reduce_mean(xs), proposal.mean(), atol=0.1)
    self.assertAllClose(tf.math.reduce_std(xs), proposal.stddev(), atol=0.1)
    # Log probs should also match up to numerics.
    self.assertAllClose(resampled.log_prob(xs),
                        proposal.log_prob(xs),
                        rtol=1e-4)

  def test_log_prob_is_stochastic_lower_bound(self):
    target = tfd.Normal(loc=0., scale=2.)
    proposal = tfd.StudentT(df=2, loc=-4., scale=5.)
    resampled = tfed.ImportanceResample(proposal,
                                        target_log_prob_fn=target.log_prob,
                                        importance_sample_size=2)
    seed = test_util.test_seed(sampler_type='stateless')
    xs, lp_upper_bound = self.evaluate(
        resampled.experimental_sample_and_log_prob(500, seed=seed))
    lp_lower_bound = resampled.log_prob(xs, sample_size=3, seed=seed)
    lp_tight = resampled.log_prob(xs, sample_size=1000, seed=seed)
    self.assertAllGreater(tf.reduce_sum(lp_upper_bound - lp_tight), 0.)
    self.assertAllLess(tf.reduce_sum(lp_lower_bound - lp_tight), 0.)

  def test_samples_approach_target_distribution(self):
    num_samples = 10000
    seeds = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'), n=6)

    target = tfd.Normal(loc=0., scale=2.)
    proposal = tfd.StudentT(df=2, loc=-1., scale=3.)
    resampled2 = tfed.ImportanceResample(proposal,
                                         target_log_prob_fn=target.log_prob,
                                         importance_sample_size=2)
    resampled20 = tfed.ImportanceResample(proposal,
                                          target_log_prob_fn=target.log_prob,
                                          importance_sample_size=20)

    xs = self.evaluate(target.sample(num_samples, seed=seeds[0]))
    xs2 = self.evaluate(resampled2.sample(num_samples, seed=seeds[1]))
    xs20 = self.evaluate(resampled20.sample(num_samples, seed=seeds[2]))
    for statistic_fn in (lambda x: x, lambda x: x**2):
      true_statistic = tf.reduce_mean(statistic_fn(xs))
      # Statistics should approach those of the target distribution as
      # `importance_sample_size` increases.
      self.assertAllGreater(
          tf.abs(tf.reduce_mean(statistic_fn(xs2)) - true_statistic),
          tf.abs(tf.reduce_mean(statistic_fn(xs20)) - true_statistic))

      expectation_no_resampling = resampled2.self_normalized_expectation(
          statistic_fn, importance_sample_size=10000, seed=seeds[3])
      self.assertAllClose(true_statistic, expectation_no_resampling, atol=0.1)

  def test_log_prob_approaches_target_distribution(self):
    seed = test_util.test_seed(sampler_type='stateless')
    target = tfd.Normal(loc=0., scale=2.)
    proposal = tfd.StudentT(df=2, loc=-4., scale=5.)
    resampled = tfed.ImportanceResample(proposal,
                                        target_log_prob_fn=target.log_prob,
                                        importance_sample_size=1000)

    xs, target_lp = self.evaluate(
        target.experimental_sample_and_log_prob(100, seed=seed))
    self.assertAllClose(target_lp,
                        resampled.log_prob(xs, seed=seed),
                        atol=0.1)

  @test_util.numpy_disable_test_missing_functionality('vectorized_map')
  def test_supports_joint_events(self):

    @tfd.JointDistributionCoroutineAutoBatched
    def target():
      x = yield tfd.Normal(-1., 1.0, name='x')
      yield tfd.MultivariateNormalTriL(loc=[x + 2],
                                       scale_tril=[[0.5]],
                                       name='y')

    @tfd.JointDistributionCoroutineAutoBatched
    def proposal():
      yield tfd.StudentT(df=2, loc=0., scale=2., name='x')
      yield tfd.StudentT(df=2, loc=[0.], scale=[2.], name='y')

    resampled = tfed.ImportanceResample(
        proposal,
        target_log_prob_fn=target.log_prob,
        importance_sample_size=5)
    self.assertAllEqual(resampled.dtype, proposal.dtype)
    self.assertAllEqual(resampled.event_shape, proposal.event_shape)

    seed = test_util.test_seed(sampler_type='stateless')
    x = self.evaluate(resampled.sample(2, seed=seed))
    self.evaluate(resampled.log_prob(x, sample_size=10, seed=seed))

    # Expectation of a structured statistic.
    estimated_mean = resampled.self_normalized_expectation(lambda x: x,
                                                           sample_size=10000,
                                                           seed=seed)
    self.assertAllClose(estimated_mean.x, -1., atol=0.2)
    self.assertAllClose(estimated_mean.y, [1.], atol=0.2)

  @parameterized.named_parameters(
      ('_static_shape', False),
      ('_unknown_shape', True))
  def test_error_when_proposal_has_deficient_batch_shape(self, unknown_shape):
    if unknown_shape and tf.executing_eagerly():
      self.skipTest('Eager execution.')

    as_tensor = tf.convert_to_tensor
    if unknown_shape:
      as_tensor = lambda x: tf1.placeholder_with_default(  # pylint: disable=g-long-lambda
          x, shape=[None for _ in tf.convert_to_tensor(x).shape])

    resampled = tfed.ImportanceResample(
        proposal_distribution=tfd.Normal(loc=as_tensor(0.),
                                         scale=as_tensor(2.)),
        target_log_prob_fn=tfd.Normal(loc=as_tensor([0., 0.]),
                                      scale=as_tensor([1., 0.5])).log_prob,
        importance_sample_size=2,
        validate_args=True)

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError if unknown_shape else ValueError,
        'Shape of importance weights does not match the batch'):
      self.evaluate(resampled.sample(seed=test_util.test_seed()))

  @test_util.numpy_disable_test_missing_functionality('tfp.vi')
  @test_util.jax_disable_test_missing_functionality('tfp.vi')
  def test_importance_resampled_surrogate_is_equivalent_to_iwae(self):
    importance_sample_size = 5
    sample_size = 1e4
    target = tfd.MultivariateNormalTriL(loc=[1., -1.],
                                        scale_tril=[[1., 0.], [-2., 0.2]])
    proposal = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])

    iwae_bound = self.evaluate(tfp.vi.monte_carlo_variational_loss(
        target_log_prob_fn=target.log_prob,
        surrogate_posterior=proposal,
        importance_sample_size=5,
        sample_size=sample_size,
        seed=test_util.test_seed(sampler_type='stateless')))

    elbo_with_resampled_surrogate = self.evaluate(
        tfp.vi.monte_carlo_variational_loss(
            target_log_prob_fn=target.log_prob,
            surrogate_posterior=tfed.ImportanceResample(
                proposal,
                target_log_prob_fn=target.log_prob,
                importance_sample_size=importance_sample_size),
            sample_size=sample_size,
            seed=test_util.test_seed(sampler_type='stateless')))
    # Passes with `atol=0.01` and `sample_size = 1e6`.
    self.assertAllClose(iwae_bound, elbo_with_resampled_surrogate, atol=0.1)

  def test_docstring_example_runs(self):

    def target_log_prob_fn(x):
      prior = tfd.Normal(loc=0., scale=1.).log_prob(x)
      likelihood = tfd.MixtureSameFamily(  # Multimodal likelihood.
          mixture_distribution=tfd.Categorical(probs=[0.4, 0.6]),
          components_distribution=tfd.Normal(loc=[-1., 1.], scale=0.1)
          ).log_prob(x)
      return prior + likelihood

    # Use importance sampling to infer an approximate posterior.
    seed = test_util.test_seed(sampler_type='stateless')
    approximate_posterior = tfed.ImportanceResample(
        proposal_distribution=tfd.Normal(loc=0., scale=2.),
        target_log_prob_fn=target_log_prob_fn,
        importance_sample_size=3,
        stochastic_approximation_seed=seed)

    # Directly compute expectations under the posterior via importance weights.
    posterior_mean = approximate_posterior.self_normalized_expectation(
        lambda x: x, seed=seed)
    approximate_posterior.self_normalized_expectation(
        lambda x: (x - posterior_mean)**2, seed=seed)

    posterior_samples = approximate_posterior.sample(5, seed=seed)
    tf.reduce_mean(posterior_samples)
    tf.math.reduce_variance(posterior_samples)

    posterior_mean_efficient = (
        approximate_posterior.self_normalized_expectation(
            lambda x: x, sample_size=10, seed=seed))
    approximate_posterior.self_normalized_expectation(
        lambda x: (x - posterior_mean_efficient)**2, sample_size=10, seed=seed)

    # Approximate the posterior density.
    xs = tf.linspace(-3., 3., 101)
    approximate_posterior.prob(xs, sample_size=10, seed=seed)

if __name__ == '__main__':
  test_util.main()
