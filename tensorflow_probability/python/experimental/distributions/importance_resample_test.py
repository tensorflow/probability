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

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import mixture_same_family
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.experimental.distributions import importance_resample
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.vi import csiszar_divergence


@test_util.test_graph_and_eager_modes
class ImportanceResampleTest(test_util.TestCase):

  def test_shapes_match_proposal_distribution(self):
    target = mvn_tril.MultivariateNormalTriL(
        loc=[-2., 1.], scale_tril=[[1., 0.], [-3., 2.]])
    # Proposal with batch shape.
    proposal = independent.Independent(
        student_t.StudentT(df=2, loc=0., scale=[[5., 2.], [1., 3.]]),
        reinterpreted_batch_ndims=1)

    seed = test_util.test_seed(sampler_type='stateless')
    resampled = importance_resample.ImportanceResample(
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
    target = normal.Normal(loc=0., scale=1.)
    proposal = normal.Normal(loc=-4., scale=5.)
    resampled = importance_resample.ImportanceResample(
        proposal, target_log_prob_fn=target.log_prob, importance_sample_size=1)
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
    target = normal.Normal(loc=0., scale=2.)
    proposal = student_t.StudentT(df=2, loc=-4., scale=5.)
    resampled = importance_resample.ImportanceResample(
        proposal, target_log_prob_fn=target.log_prob, importance_sample_size=2)
    seed = test_util.test_seed(sampler_type='stateless')
    xs, lp_upper_bound = self.evaluate(
        resampled.experimental_sample_and_log_prob(500, seed=seed))
    lp_lower_bound = resampled.log_prob(xs, sample_size=3, seed=seed)
    lp_tight = resampled.log_prob(xs, sample_size=1000, seed=seed)
    self.assertAllGreater(tf.reduce_sum(lp_upper_bound),
                          tf.reduce_sum(lp_tight))
    self.assertAllLess(tf.reduce_sum(lp_lower_bound),
                       tf.reduce_sum(lp_tight))

  def test_samples_approach_target_distribution(self):
    num_samples = 10_000
    seeds = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'), n=6)

    target = normal.Normal(loc=0., scale=2.)
    proposal = student_t.StudentT(df=2, loc=-1., scale=3.)
    resampled2 = importance_resample.ImportanceResample(
        proposal, target_log_prob_fn=target.log_prob, importance_sample_size=2)
    resampled20 = importance_resample.ImportanceResample(
        proposal, target_log_prob_fn=target.log_prob, importance_sample_size=20)

    xs = self.evaluate(target.sample(num_samples, seed=seeds[0]))
    xs2 = self.evaluate(resampled2.sample(num_samples, seed=seeds[1]))
    xs20 = self.evaluate(resampled20.sample(num_samples, seed=seeds[2]))
    for statistic_fn, seed in zip([lambda x: x, lambda x: x**2], seeds[3:]):
      true_statistic = tf.reduce_mean(statistic_fn(xs))
      # Statistics should approach those of the target distribution as
      # `importance_sample_size` increases.
      self.assertAllGreater(
          tf.abs(tf.reduce_mean(statistic_fn(xs2)) - true_statistic),
          tf.abs(tf.reduce_mean(statistic_fn(xs20)) - true_statistic))

      expectation_no_resampling = resampled2.self_normalized_expectation(
          statistic_fn, importance_sample_size=10000, seed=seed)
      self.assertAllClose(true_statistic, expectation_no_resampling, atol=0.15)

  def test_log_prob_approaches_target_distribution(self):
    seed = test_util.test_seed(sampler_type='stateless')
    target = normal.Normal(loc=0., scale=2.)
    proposal = student_t.StudentT(df=2, loc=-4., scale=5.)
    resampled = importance_resample.ImportanceResample(
        proposal,
        target_log_prob_fn=target.log_prob,
        importance_sample_size=1000)

    xs, target_lp = self.evaluate(
        target.experimental_sample_and_log_prob(100, seed=seed))
    self.assertAllClose(target_lp,
                        resampled.log_prob(xs, seed=seed),
                        atol=0.17)

  def test_supports_joint_events(self):
    root = jdc.JointDistributionCoroutine.Root

    @jdc.JointDistributionCoroutine
    def target():
      x = yield root(normal.Normal(-1., 1.0, name='x'))
      yield mvn_tril.MultivariateNormalTriL(
          loc=(x + 2)[..., tf.newaxis], scale_tril=[[0.5]], name='y')

    @jdc.JointDistributionCoroutine
    def proposal():
      yield root(student_t.StudentT(df=2, loc=0., scale=2., name='x'))
      yield root(
          independent.Independent(
              student_t.StudentT(df=2, loc=[0.], scale=[2.]), 1, name='y'))

    resampled = importance_resample.ImportanceResample(
        proposal, target_log_prob_fn=target.log_prob, importance_sample_size=5)
    self.assertAllEqual(resampled.dtype, proposal.dtype)
    self.assertAllEqualNested(
        list(resampled.event_shape), list(proposal.event_shape))

    seed = test_util.test_seed(sampler_type='stateless')
    x = self.evaluate(resampled.sample(2, seed=seed))
    self.evaluate(resampled.log_prob(x, sample_size=10, seed=seed))

    # Expectation of a structured statistic.
    estimated_mean = resampled.self_normalized_expectation(lambda x: x,
                                                           sample_size=10000,
                                                           seed=seed)
    self.assertAllClose(estimated_mean.x, -1., atol=0.2)
    self.assertAllClose(estimated_mean.y, [1.], atol=0.22)

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

    resampled = importance_resample.ImportanceResample(
        proposal_distribution=normal.Normal(
            loc=as_tensor(0.), scale=as_tensor(2.)),
        target_log_prob_fn=normal.Normal(
            loc=as_tensor([0., 0.]), scale=as_tensor([1., 0.5])).log_prob,
        importance_sample_size=2,
        validate_args=True)

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError if unknown_shape else ValueError,
        'Shape of importance weights does not match the batch'):
      self.evaluate(resampled.sample(seed=test_util.test_seed()))

  def test_importance_resampled_surrogate_is_equivalent_to_iwae(self):
    importance_sample_size = 10
    sample_size = 1e4
    target = mvn_tril.MultivariateNormalTriL(
        loc=[1., -1.], scale_tril=[[1., 0.], [-2., 0.2]])
    proposal = mvn_diag.MultivariateNormalDiag(
        loc=[0., 0.], scale_diag=[1., 1.])

    stream = test_util.test_seed_stream()
    iwae_bound = self.evaluate(
        csiszar_divergence.monte_carlo_variational_loss(
            target_log_prob_fn=target.log_prob,
            surrogate_posterior=proposal,
            importance_sample_size=importance_sample_size,
            sample_size=sample_size,
            seed=stream()))

    elbo_with_resampled_surrogate = self.evaluate(
        csiszar_divergence.monte_carlo_variational_loss(
            target_log_prob_fn=target.log_prob,
            surrogate_posterior=importance_resample.ImportanceResample(
                proposal,
                target_log_prob_fn=target.log_prob,
                importance_sample_size=importance_sample_size),
            sample_size=sample_size,
            seed=stream()))
    # Passes with `atol=0.015` and `sample_size = 1e6`.
    self.assertAllClose(iwae_bound, elbo_with_resampled_surrogate, atol=0.15)

  def test_docstring_example_runs(self):

    def target_log_prob_fn(x):
      prior = normal.Normal(loc=0., scale=1.).log_prob(x)
      # Multimodal likelihood.
      likelihood = mixture_same_family.MixtureSameFamily(
          mixture_distribution=categorical.Categorical(probs=[0.4, 0.6]),
          components_distribution=normal.Normal(loc=[-1., 1.],
                                                scale=0.1)).log_prob(x)
      return prior + likelihood

    # Use importance sampling to infer an approximate posterior.
    seed = lambda: test_util.test_seed(sampler_type='stateless')
    approximate_posterior = importance_resample.ImportanceResample(
        proposal_distribution=normal.Normal(loc=0., scale=2.),
        target_log_prob_fn=target_log_prob_fn,
        importance_sample_size=3,
        stochastic_approximation_seed=seed())

    # Directly compute expectations under the posterior via importance weights.
    posterior_mean = approximate_posterior.self_normalized_expectation(
        lambda x: x, seed=seed())
    approximate_posterior.self_normalized_expectation(
        lambda x: (x - posterior_mean)**2, seed=seed())

    posterior_samples = approximate_posterior.sample(5, seed=seed())
    tf.reduce_mean(posterior_samples)
    tf.math.reduce_variance(posterior_samples)

    posterior_mean_efficient = (
        approximate_posterior.self_normalized_expectation(
            lambda x: x, sample_size=10, seed=seed()))
    approximate_posterior.self_normalized_expectation(
        lambda x: (
            x - posterior_mean_efficient)**2, sample_size=10, seed=seed())

    # Approximate the posterior density.
    xs = tf.linspace(-3., 3., 101)
    approximate_posterior.prob(xs, sample_size=10, seed=seed())

  def test_log_prob_independence_per_x(self):
    dist = importance_resample.ImportanceResample(
        proposal_distribution=normal.Normal(loc=0., scale=1.),
        target_log_prob_fn=normal.Normal(loc=0.85, scale=0.1).log_prob,
        importance_sample_size=20)

    seed = test_util.test_seed(sampler_type='stateless')
    xs = np.linspace(0, 1.6, 100)
    aucs = []
    for s in samplers.split_seed(seed, n=30):
      aucs.append(np.trapezoid(
          self.evaluate(dist.prob(xs, seed=s, sample_size=100)), xs))

    self.assertAllClose(aucs, np.ones_like(aucs), atol=.05)


if __name__ == '__main__':
  test_util.main()
