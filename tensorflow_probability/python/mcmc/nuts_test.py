# Copyright 2018 The TensorFlow Probability Authors.
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
"""Tests of the No U-Turn Sampler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import test_util


@tf.function(autograph=False)
def run_nuts_chain(event_size, batch_size, num_steps, initial_state=None,
                   seed=None):
  if seed is None:
    seed = 1
  def target_log_prob_fn(event):
    with tf.name_scope('nuts_test_target_log_prob'):
      return tfd.MultivariateNormalDiag(
          tf.zeros(event_size),
          scale_identity_multiplier=1.).log_prob(event)

  if initial_state is None:
    initial_state = tf.zeros([batch_size, event_size])

  kernel = tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn,
      step_size=[0.3],
      unrolled_leapfrog_steps=2,
      max_tree_depth=4,
      parallel_iterations=1,
      seed=seed)

  chain_state, leapfrogs_taken = tfp.mcmc.sample_chain(
      num_results=num_steps,
      num_burnin_steps=0,
      # Intentionally pass a list argument to test that singleton lists are
      # handled reasonably (c.f. assert_univariate_target_conservation, which
      # uses an unwrapped singleton).
      current_state=[initial_state],
      kernel=kernel,
      trace_fn=lambda _, pkr: pkr.leapfrogs_taken,
      parallel_iterations=1)

  return chain_state, leapfrogs_taken


def assert_univariate_target_conservation(test, target_d, step_size):
  # Sample count limited partly by memory reliably available on Forge.  The test
  # remains reasonable even if the nuts recursion limit is severely curtailed
  # (e.g., 3 or 4 levels), so use that to recover some memory footprint and bump
  # the sample count.
  num_samples = int(5e4)
  num_steps = 1
  strm = test_util.test_seed_stream()
  initialization = target_d.sample([num_samples], seed=strm())

  @tf.function(autograph=False)
  def run_chain():
    nuts = tfp.mcmc.NoUTurnSampler(
        target_d.log_prob,
        step_size=step_size,
        max_tree_depth=3,
        unrolled_leapfrog_steps=2,
        parallel_iterations=1,
        seed=strm())
    result = tfp.mcmc.sample_chain(
        num_results=num_steps,
        num_burnin_steps=0,
        current_state=initialization,
        trace_fn=None,
        parallel_iterations=1,
        kernel=nuts)
    return result

  result = run_chain()
  test.assertAllEqual([num_steps, num_samples], result.shape)
  answer = result[0]
  check_cdf_agrees = st.assert_true_cdf_equal_by_dkwm(
      answer, target_d.cdf, false_fail_rate=1e-6)
  check_enough_power = assert_util.assert_less(
      st.min_discrepancy_of_true_cdfs_detectable_by_dkwm(
          num_samples, false_fail_rate=1e-6, false_pass_rate=1e-6), 0.025)
  movement = tf.abs(answer - initialization)
  test.assertAllEqual([num_samples], movement.shape)
  # This movement distance (1 * step_size) was selected by reducing until 100
  # runs with independent seeds all passed.
  check_movement = assert_util.assert_greater_equal(
      tf.reduce_mean(movement), 1 * step_size)
  return (check_cdf_agrees, check_enough_power, check_movement)


def assert_mvn_target_conservation(event_size, batch_size, **kwargs):
  strm = test_util.test_seed_stream()
  initialization = tfd.MultivariateNormalFullCovariance(
      loc=tf.zeros(event_size),
      covariance_matrix=tf.eye(event_size)).sample(
          batch_size, seed=strm())
  samples, _ = run_nuts_chain(
      event_size, batch_size, num_steps=1,
      initial_state=initialization, **kwargs)
  answer = samples[0][-1]
  check_cdf_agrees = (
      st.assert_multivariate_true_cdf_equal_on_projections_two_sample(
          answer, initialization, num_projections=100, false_fail_rate=1e-6))
  check_sample_shape = assert_util.assert_equal(
      tf.shape(answer)[0], batch_size)
  movement = tf.linalg.norm(answer - initialization, axis=-1)
  # This movement distance (0.3) was copied from the univariate case.
  check_movement = assert_util.assert_greater_equal(
      tf.reduce_mean(movement), 0.3)
  check_enough_power = assert_util.assert_less(
      st.min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample(
          batch_size, batch_size, false_fail_rate=1e-8, false_pass_rate=1e-6),
      0.055)
  return (
      check_cdf_agrees,
      check_sample_shape,
      check_movement,
      check_enough_power,
  )


@test_util.test_graph_and_eager_modes
class NutsTest(test_util.TestCase):

  def testLogAcceptRatio(self):
    """Ensure that `log_accept_ratio` is close to 0 if step size is small."""
    seed = test_util.test_seed()
    @tf.function(autograph=False)
    def sample_from_banana():
      def banana_model():
        x0 = yield tfd.JointDistributionCoroutine.Root(tfd.Normal(0., 10.))
        _ = yield tfd.Normal(0.03 * (tf.square(x0) - 100.), 1.)
      banana = tfd.JointDistributionCoroutine(banana_model)
      kernel = tfp.mcmc.NoUTurnSampler(banana.log_prob,
                                       step_size=0.35,
                                       seed=seed,
                                       parallel_iterations=1)
      trace_fn = lambda _, pkr: pkr.log_accept_ratio
      return tfp.mcmc.sample_chain(50,
                                   [0., 0.],
                                   kernel=kernel,
                                   trace_fn=trace_fn,
                                   parallel_iterations=1)[1]
    log_accept_ratio_trace = self.evaluate(sample_from_banana())
    self.assertAllGreater(log_accept_ratio_trace, -0.2)

  def testReproducibility(self):
    seed = test_util.test_seed()
    s1 = self.evaluate(run_nuts_chain(2, 5, 10, seed=seed)[0])
    if tf.executing_eagerly():
      tf.random.set_seed(seed)
    s2 = self.evaluate(run_nuts_chain(2, 5, 10, seed=seed)[0])
    self.assertAllEqual(s1, s2)

  def testCorrectReadWriteInstruction(self):
    mocknuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=lambda x: x,
        max_tree_depth=4,
        step_size=1.)

    self.assertAllEqual(
        mocknuts._write_instruction,
        np.array([0, 4, 1, 4, 1, 4, 2, 4, 1, 4, 2, 4, 2, 4, 3, 4]))
    self.assertAllEqual(
        mocknuts._read_instruction,
        np.array([[0, 0],
                  [0, 1],
                  [0, 0],
                  [0, 2],
                  [0, 0],
                  [1, 2],
                  [0, 0],
                  [0, 3],
                  [0, 0],
                  [1, 2],
                  [0, 0],
                  [1, 3],
                  [0, 0],
                  [2, 3],
                  [0, 0],
                  [0, 4]]))

  def testUnivariateNormalTargetConservation(self):
    normal_dist = tfd.Normal(loc=1., scale=2.)
    self.evaluate(assert_univariate_target_conservation(
        self, normal_dist, step_size=0.2))

  def testLogitBetaTargetConservation(self):
    logit_beta_dist = tfb.Invert(tfb.Sigmoid())(
        tfd.Beta(concentration0=1., concentration1=2.))
    self.evaluate(assert_univariate_target_conservation(
        self, logit_beta_dist, step_size=0.2))

  def testSigmoidBetaTargetConservation(self):
    # Not inverting the sigmoid bijector makes a kooky distribution, but nuts
    # should still conserve it (with a smaller step size).
    sigmoid_beta_dist = tfb.Identity(tfb.Sigmoid())(
        tfd.Beta(concentration0=1., concentration1=2.))
    self.evaluate(assert_univariate_target_conservation(
        self, sigmoid_beta_dist, step_size=1e-3))

  @parameterized.parameters(
      (3, 50000,),
      # (5, 2,),
  )
  def testMultivariateNormalNd(self, event_size, batch_size):
    strm = test_util.test_seed_stream()
    self.evaluate(assert_mvn_target_conservation(event_size, batch_size,
                                                 seed=strm()))

  @parameterized.parameters(
      ([], 100),      # test scalar case
      ([1], 100),     # test size 1 case
      ([5], 100),
      ([2, 5], 100),  # test rank 2 case
  )
  def testLatentsOfMixedRank(self, batch_shape, num_steps):
    strm = test_util.test_seed_stream()

    init0 = [tf.ones(batch_shape + [6])]
    init1 = [tf.ones(batch_shape + []),
             tf.ones(batch_shape + [1]),
             tf.ones(batch_shape + [2, 2])]

    @tf.function(autograph=False)
    def run_two_chains(init0, init1):

      def log_prob0(x):
        return tf.squeeze(tfd.Independent(
            tfd.Normal(tf.range(6, dtype=tf.float32),
                       tf.constant(1.)),
            reinterpreted_batch_ndims=1).log_prob(x))
      kernel0 = tfp.mcmc.NoUTurnSampler(
          log_prob0,
          step_size=0.3,
          parallel_iterations=1,
          seed=strm())
      [results0] = tfp.mcmc.sample_chain(
          num_results=num_steps,
          num_burnin_steps=10,
          current_state=init0,
          kernel=kernel0,
          trace_fn=None,
          parallel_iterations=1)

      def log_prob1(state0, state1, state2):
        return tf.squeeze(
            tfd.Normal(tf.constant(0.), tf.constant(1.)).log_prob(state0)
            + tfd.Independent(
                tfd.Normal(tf.constant([1.]), tf.constant(1.)),
                reinterpreted_batch_ndims=1).log_prob(state1)
            + tfd.Independent(
                tfd.Normal(tf.constant([[2., 3.], [4., 5.]]), tf.constant(1.)),
                reinterpreted_batch_ndims=2).log_prob(state2)
        )
      kernel1 = tfp.mcmc.NoUTurnSampler(
          log_prob1,
          step_size=0.3,
          parallel_iterations=1,
          seed=strm())
      results1_ = tfp.mcmc.sample_chain(
          num_results=num_steps,
          num_burnin_steps=10,
          current_state=init1,
          kernel=kernel1,
          trace_fn=None,
          parallel_iterations=1)
      results1 = tf.concat(
          [tf.reshape(x, [num_steps] + batch_shape + [-1]) for x in results1_],
          axis=-1)

      return results0, results1

    results0, results1 = run_two_chains(init0, init1)

    self.evaluate(
        st.assert_true_cdf_equal_by_dkwm_two_sample(results0, results1))

  @parameterized.parameters(
      (1000, 5, 3),
      # (500, 1000, 20),
  )
  def testMultivariateNormalNdConvergence(self, nsamples, nchains, nd):
    strm = test_util.test_seed_stream()
    theta0 = np.zeros((nchains, nd))
    mu = np.arange(nd)
    w = np.random.randn(nd, nd) * 0.1
    cov = w * w.T + np.diagflat(np.arange(nd) + 1.)
    step_size = np.random.rand(nchains, 1) * 0.1 + 1.

    @tf.function(autograph=False)
    def run_chain_and_get_summary(mu, scale_tril, step_size, nsamples, state):
      def target_log_prob_fn(event):
        with tf.name_scope('nuts_test_target_log_prob'):
          return tfd.MultivariateNormalTriL(
              loc=tf.cast(mu, dtype=tf.float64),
              scale_tril=tf.cast(scale_tril, dtype=tf.float64)).log_prob(event)

      nuts = tfp.mcmc.NoUTurnSampler(
          target_log_prob_fn,
          step_size=[step_size],
          max_tree_depth=5,
          parallel_iterations=1,
          seed=strm())

      def trace_fn(_, pkr):
        return (pkr.is_accepted, pkr.leapfrogs_taken)

      [x], (is_accepted, leapfrogs_taken) = tfp.mcmc.sample_chain(
          num_results=nsamples,
          num_burnin_steps=0,
          current_state=[tf.cast(state, dtype=tf.float64)],
          kernel=nuts,
          trace_fn=trace_fn,
          parallel_iterations=1)

      return (
          tf.shape(x),
          # We'll average over samples (dim=0) and chains (dim=1).
          tf.reduce_mean(x, axis=[0, 1]),
          tfp.stats.covariance(x, sample_axis=[0, 1]),
          leapfrogs_taken[is_accepted])

    sample_shape, sample_mean, sample_cov, leapfrogs_taken = self.evaluate(
        run_chain_and_get_summary(
            mu, np.linalg.cholesky(cov), step_size, nsamples, theta0))

    self.assertAllEqual(sample_shape, [nsamples, nchains, nd])
    self.assertAllClose(mu, sample_mean, atol=0.1, rtol=0.1)
    self.assertAllClose(cov, sample_cov, atol=0.15, rtol=0.15)
    # Test early stopping in tree building
    self.assertTrue(
        np.any(np.isin(np.asarray([5, 9, 11, 13]), np.unique(leapfrogs_taken))))

  def testCorrelated2dNormalwithinMCError(self):
    strm = test_util.test_seed_stream()
    nchains = 100
    num_steps = 1000
    mu = np.asarray([0., 3.], dtype=np.float32)
    rho = 0.75
    sigma1 = np.float32(1.)
    sigma2 = np.float32(2.)
    cov = np.asarray([[sigma1 * sigma1, rho * sigma1 * sigma2],
                      [rho * sigma1 * sigma2, sigma2 * sigma2]],
                     dtype=np.float32)
    true_param = np.hstack([mu, np.array([sigma1**2, sigma2**2, rho])])
    scale_tril = np.linalg.cholesky(cov)
    initial_state = np.zeros((nchains, 2), np.float32)

    @tf.function(autograph=False)
    def run_chain_and_get_estimation_error():
      chain_state = tfp.mcmc.sample_chain(
          num_results=num_steps,
          num_burnin_steps=0,
          current_state=initial_state,
          kernel=tfp.mcmc.NoUTurnSampler(
              tfd.MultivariateNormalTriL(loc=mu,
                                         scale_tril=scale_tril).log_prob,
              step_size=np.asarray([sigma1, sigma2]),
              parallel_iterations=1,
              seed=strm()),
          parallel_iterations=1,
          trace_fn=None)
      variance_est = tf.square(chain_state - mu)
      correlation_est = tf.reduce_prod(
          chain_state - mu, axis=-1, keepdims=True) / (sigma1 * sigma2)
      mcmc_samples = tf.concat([chain_state, variance_est, correlation_est],
                               axis=-1)

      expected = tf.reduce_mean(mcmc_samples, axis=[0, 1])

      ess = tf.reduce_sum(tfp.mcmc.effective_sample_size(mcmc_samples), axis=0)
      avg_monte_carlo_standard_error = tf.reduce_mean(
          tf.math.reduce_std(mcmc_samples, axis=0),
          axis=0) / tf.sqrt(ess)
      scaled_error = (
          tf.abs(expected - true_param) / avg_monte_carlo_standard_error)

      return tfd.Normal(loc=0., scale=1.).survival_function(scaled_error)

    # Probability of getting an error more extreme than this
    error_prob = self.evaluate(run_chain_and_get_estimation_error())

    # Check convergence using Markov chain central limit theorem, this is a
    # z-test at p=.01
    self.assertAllGreater(error_prob, 0.005)

  @parameterized.parameters(
      (7, 5, 3, None),
      (7, 5, 1, tf.TensorShape([None, 1])),
  )
  def testDynamicShape(self, nsample, batch_size, nd, dynamic_shape):
    dtype = np.float32

    kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=tfd.Independent(
            tfd.Normal(tf.zeros(nd, dtype=dtype), 1.), 1).log_prob,
        parallel_iterations=1,
        step_size=.1)
    x_ = np.zeros([batch_size, nd], dtype=dtype)
    x = tf1.placeholder_with_default(x_, shape=dynamic_shape)
    mcmc_trace_ = tfp.mcmc.sample_chain(
        num_results=nsample,
        current_state=x,
        kernel=kernel,
        parallel_iterations=1,
        trace_fn=None)
    mcmc_trace = self.evaluate(mcmc_trace_)
    self.assertAllEqual(mcmc_trace.shape, [nsample, batch_size, nd])

  def testDivergence(self):
    """Neals funnel with large step size."""
    strm = test_util.test_seed_stream()
    neals_funnel = tfd.JointDistributionSequential(
        [
            tfd.Normal(loc=0., scale=3.),  # b0
            lambda y: tfd.Sample(  # pylint: disable=g-long-lambda
                tfd.Normal(loc=0., scale=tf.math.exp(y / 2)),
                sample_shape=9),
        ],
        validate_args=True
    )

    @tf.function(autograph=False)
    def run_chain_and_get_divergence():
      nchains = 5
      init_states = neals_funnel.sample(nchains, seed=strm())
      _, has_divergence = tfp.mcmc.sample_chain(
          num_results=100,
          kernel=tfp.mcmc.NoUTurnSampler(
              target_log_prob_fn=lambda *args: neals_funnel.log_prob(args),
              step_size=[1., 1.],
              parallel_iterations=1,
              seed=strm()),
          current_state=init_states,
          trace_fn=lambda _, pkr: pkr.has_divergence,
          parallel_iterations=1)
      return tf.reduce_sum(tf.cast(has_divergence, dtype=tf.int32))

    divergence_count = self.evaluate(run_chain_and_get_divergence())

    # Test that we observe a fair among of divergence.
    self.assertAllGreater(divergence_count, 100)

  def testSampleEndtoEndXLA(self):
    """An end-to-end test of sampling using NUTS."""
    if tf.executing_eagerly() or tf.config.experimental_functions_run_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    strm = test_util.test_seed_stream()
    predictors = tf.cast([
        201., 244., 47., 287., 203., 58., 210., 202., 198., 158., 165., 201.,
        157., 131., 166., 160., 186., 125., 218., 146.
    ], tf.float32)
    obs = tf.cast([
        592., 401., 583., 402., 495., 173., 479., 504., 510., 416., 393., 442.,
        317., 311., 400., 337., 423., 334., 533., 344.
    ], tf.float32)
    y_sigma = tf.cast([
        61., 25., 38., 15., 21., 15., 27., 14., 30., 16., 14., 25., 52., 16.,
        34., 31., 42., 26., 16., 22.
    ], tf.float32)

    # Robust linear regression model
    robust_lm = tfd.JointDistributionSequential(
        [
            tfd.Normal(loc=0., scale=1.),  # b0
            tfd.Normal(loc=0., scale=1.),  # b1
            tfd.HalfNormal(5.),  # df
            lambda df, b1, b0: tfd.Independent(  # pylint: disable=g-long-lambda
                tfd.StudentT(  # Likelihood
                    df=df[..., tf.newaxis],
                    loc=(b0[..., tf.newaxis] +
                         b1[..., tf.newaxis] * predictors[tf.newaxis]),
                    scale=y_sigma)),
        ],
        validate_args=True)

    log_prob = lambda b0, b1, df: robust_lm.log_prob([b0, b1, df, obs])
    init_step_size = [1., .2, .5]
    step_size0 = [tf.cast(x, dtype=tf.float32) for x in init_step_size]

    number_of_steps, burnin, nchain = 200, 50, 10

    @tf.function(autograph=False, experimental_compile=True)
    def run_chain_and_get_diagnostic():
      # Ensure we're really in graph mode.
      assert hasattr(tf.constant([]), 'graph')

      # random initialization of the starting postion of each chain
      b0, b1, df, _ = robust_lm.sample(nchain, seed=strm())

      # bijector to map contrained parameters to real
      unconstraining_bijectors = [
          tfb.Identity(),
          tfb.Identity(),
          tfb.Exp(),
      ]

      def trace_fn(_, pkr):
        return (pkr.inner_results.inner_results.step_size,
                pkr.inner_results.inner_results.log_accept_ratio)

      kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
          tfp.mcmc.TransformedTransitionKernel(
              inner_kernel=tfp.mcmc.NoUTurnSampler(
                  target_log_prob_fn=log_prob,
                  step_size=step_size0,
                  parallel_iterations=1,
                  seed=strm()),
              bijector=unconstraining_bijectors),
          target_accept_prob=.8,
          num_adaptation_steps=burnin,
      )

      # Sampling from the chain and get diagnostics
      mcmc_trace, (step_size, log_accept_ratio) = tfp.mcmc.sample_chain(
          num_results=number_of_steps,
          num_burnin_steps=burnin,
          current_state=[b0, b1, df],
          kernel=kernel,
          trace_fn=trace_fn,
          parallel_iterations=1)
      rhat = tfp.mcmc.potential_scale_reduction(mcmc_trace)
      return (
          [s[-1] for s in step_size],  # final step size
          tf.math.exp(tfp.math.reduce_logmeanexp(log_accept_ratio)),
          [tf.reduce_mean(rhat_) for rhat_ in rhat],  # average rhat
      )

    # Sample from posterior distribution and get diagnostic
    [
        final_step_size, average_accept_ratio, average_rhat
    ] = self.evaluate(run_chain_and_get_diagnostic())

    # Check that step size adaptation reduced the initial step size
    self.assertAllLess(
        np.asarray(final_step_size) - np.asarray(init_step_size), 0.)
    # Check that average acceptance ratio is close to target
    self.assertAllClose(
        average_accept_ratio,
        .8 * np.ones_like(average_accept_ratio),
        atol=0.1, rtol=0.1)
    # Check that mcmc sample quality is acceptable with tuning
    self.assertAllClose(
        average_rhat, np.ones_like(average_rhat), atol=0.05, rtol=0.05)


if __name__ == '__main__':
  tf.test.main()
