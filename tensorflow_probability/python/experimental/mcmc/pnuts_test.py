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

import collections

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import sigmoid
from tensorflow_probability.python.distributions import beta
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import wishart
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.experimental.distributions import mvn_precision_factor_linop as mvnpflo
from tensorflow_probability.python.experimental.mcmc import preconditioned_nuts
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.mcmc import diagnostic
from tensorflow_probability.python.mcmc import dual_averaging_step_size_adaptation as dassa
from tensorflow_probability.python.mcmc import sample
from tensorflow_probability.python.mcmc import simple_step_size_adaptation as sssa
from tensorflow_probability.python.mcmc import transformed_kernel
from tensorflow_probability.python.stats import sample_stats

JAX_MODE = False


@tf.function(autograph=False)
def run_nuts_chain(event_size, batch_size, num_steps, initial_state=None,
                   seed=None):
  if seed is None:
    seed = test_util.test_seed()
  def target_log_prob_fn(event):
    with tf.name_scope('nuts_test_target_log_prob'):
      return mvn_diag.MultivariateNormalDiag(
          tf.zeros(event_size)).log_prob(event)

  if initial_state is None:
    initial_state = tf.zeros([batch_size, event_size])

  kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
      target_log_prob_fn,
      step_size=[0.3],
      unrolled_leapfrog_steps=2,
      max_tree_depth=4)

  chain_state, leapfrogs_taken = sample.sample_chain(
      num_results=num_steps,
      num_burnin_steps=0,
      # Intentionally pass a list argument to test that singleton lists are
      # handled reasonably (c.f. assert_univariate_target_conservation, which
      # uses an unwrapped singleton).
      current_state=[initial_state],
      kernel=kernel,
      trace_fn=lambda _, pkr: pkr.leapfrogs_taken,
      seed=seed)

  return chain_state, leapfrogs_taken


def assert_univariate_target_conservation(test, target_d, step_size):
  # Sample count limited partly by memory reliably available on Forge.  The test
  # remains reasonable even if the nuts recursion limit is severely curtailed
  # (e.g., 3 or 4 levels), so use that to recover some memory footprint and bump
  # the sample count.
  num_samples = int(5e4)
  num_steps = 1
  strm = test_util.test_seed_stream()
  # We wrap the initial values in `tf.identity` to avoid broken gradients
  # resulting from a bijector cache hit, since bijectors of the same
  # type/parameterization now share a cache.
  # TODO(b/72831017): Fix broken gradients caused by bijector caching.
  initialization = tf.identity(target_d.sample([num_samples], seed=strm()))

  @tf.function(autograph=False)
  def run_chain():
    nuts = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_d.log_prob,
        step_size=step_size,
        max_tree_depth=3,
        unrolled_leapfrog_steps=2)
    result = sample.sample_chain(
        num_results=num_steps,
        num_burnin_steps=0,
        current_state=initialization,
        trace_fn=None,
        kernel=nuts,
        seed=strm())
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
  initialization = mvn_diag.MultivariateNormalDiag(
      loc=tf.zeros(event_size)).sample(
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
        x0 = yield jdc.JointDistributionCoroutine.Root(normal.Normal(0., 10.))
        _ = yield normal.Normal(0.03 * (tf.square(x0) - 100.), 1.)

      banana = jdc.JointDistributionCoroutine(banana_model)
      kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
          banana.log_prob, step_size=0.35)
      trace_fn = lambda _, pkr: pkr.log_accept_ratio
      return sample.sample_chain(
          50, [0., 0.], kernel=kernel, trace_fn=trace_fn, seed=seed)[1]
    log_accept_ratio_trace = self.evaluate(sample_from_banana())
    self.assertAllGreater(log_accept_ratio_trace, -0.35)

  def testReproducibility(self):
    seed = test_util.test_seed()
    s1 = self.evaluate(run_nuts_chain(2, 5, 10, seed=seed)[0])
    if tf.executing_eagerly():
      tf.random.set_seed(seed)
    s2 = self.evaluate(run_nuts_chain(2, 5, 10, seed=seed)[0])
    self.assertAllEqual(s1, s2)

  def testCorrectReadWriteInstruction(self):
    mocknuts = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_log_prob_fn=lambda x: x, max_tree_depth=4, step_size=1.)

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
    normal_dist = normal.Normal(loc=1., scale=2.)
    self.evaluate(assert_univariate_target_conservation(
        self, normal_dist, step_size=0.2))

  def testSigmoidBetaTargetConservation(self):
    sigmoid_beta_dist = invert.Invert(sigmoid.Sigmoid())(
        beta.Beta(concentration0=1., concentration1=2.))
    self.evaluate(assert_univariate_target_conservation(
        self, sigmoid_beta_dist, step_size=0.2))

  def testBetaTargetConservation(self):
    # Not that we expect NUTS to do a good job without an unconstraining
    # bijector, but...
    beta_dist = beta.Beta(concentration0=1., concentration1=2.)
    self.evaluate(assert_univariate_target_conservation(
        self, beta_dist, step_size=1e-3))

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
        return tf.squeeze(
            independent.Independent(
                normal.Normal(tf.range(6, dtype=tf.float32), tf.constant(1.)),
                reinterpreted_batch_ndims=1).log_prob(x))

      kernel0 = preconditioned_nuts.PreconditionedNoUTurnSampler(
          log_prob0, step_size=0.3)
      [results0] = sample.sample_chain(
          num_results=num_steps,
          num_burnin_steps=10,
          current_state=init0,
          kernel=kernel0,
          trace_fn=None,
          seed=strm())

      def log_prob1(state0, state1, state2):
        return tf.squeeze(
            normal.Normal(tf.constant(0.), tf.constant(1.)).log_prob(state0) +
            independent.Independent(
                normal.Normal(tf.constant([1.]), tf.constant(1.)),
                reinterpreted_batch_ndims=1).log_prob(state1) +
            independent.Independent(
                normal.Normal(
                    tf.constant([[2., 3.], [4., 5.]]), tf.constant(1.)),
                reinterpreted_batch_ndims=2).log_prob(state2))

      kernel1 = preconditioned_nuts.PreconditionedNoUTurnSampler(
          log_prob1, step_size=0.3)
      results1_ = sample.sample_chain(
          num_results=num_steps,
          num_burnin_steps=10,
          current_state=init1,
          kernel=kernel1,
          trace_fn=None,
          seed=strm())
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
          return mvn_tril.MultivariateNormalTriL(
              loc=tf.cast(mu, dtype=tf.float64),
              scale_tril=tf.cast(scale_tril, dtype=tf.float64)).log_prob(event)

      nuts = preconditioned_nuts.PreconditionedNoUTurnSampler(
          target_log_prob_fn, step_size=[step_size], max_tree_depth=4)

      def trace_fn(_, pkr):
        return (pkr.is_accepted, pkr.leapfrogs_taken)

      [x], (is_accepted, leapfrogs_taken) = sample.sample_chain(
          num_results=nsamples,
          num_burnin_steps=0,
          current_state=[tf.cast(state, dtype=tf.float64)],
          kernel=nuts,
          trace_fn=trace_fn,
          seed=strm())

      return (
          tf.shape(x),
          # We'll average over samples (dim=0) and chains (dim=1).
          tf.reduce_mean(x, axis=[0, 1]),
          sample_stats.covariance(x, sample_axis=[0, 1]),
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
    dtype = np.float64

    # We run nreplica independent test to improve test robustness.
    nreplicas = 20
    nchains = 100
    num_steps = 1000
    mu = np.asarray([0., 3.], dtype=dtype)
    rho = 0.75
    sigma1 = 1.
    sigma2 = 2.
    cov = np.asarray([[sigma1 * sigma1, rho * sigma1 * sigma2],
                      [rho * sigma1 * sigma2, sigma2 * sigma2]],
                     dtype=dtype)
    true_param = np.hstack([mu, np.array([sigma1**2, sigma2**2, rho])])
    scale_tril = np.linalg.cholesky(cov)
    initial_state = np.zeros((nchains, nreplicas, 2), dtype)

    @tf.function(autograph=False)
    def run_chain_and_get_estimation_error():
      target_log_prob = mvn_tril.MultivariateNormalTriL(
          loc=mu, scale_tril=scale_tril).log_prob
      nuts_kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
          target_log_prob, step_size=tf.constant([sigma1, sigma2], dtype))
      chain_state = sample.sample_chain(
          num_results=num_steps,
          num_burnin_steps=25,
          current_state=initial_state,
          kernel=dassa.DualAveragingStepSizeAdaptation(nuts_kernel, 25, .8),
          seed=strm(),
          trace_fn=None)
      variance_est = tf.square(chain_state - mu)
      correlation_est = tf.reduce_prod(
          chain_state - mu, axis=-1, keepdims=True) / (sigma1 * sigma2)
      mcmc_samples = tf.concat([chain_state, variance_est, correlation_est],
                               axis=-1)

      expected = tf.reduce_mean(mcmc_samples, axis=[0, 1])

      ess = tf.reduce_sum(
          diagnostic.effective_sample_size(mcmc_samples), axis=0)
      avg_monte_carlo_standard_error = tf.reduce_mean(
          tf.math.reduce_std(mcmc_samples, axis=0),
          axis=0) / tf.sqrt(ess)
      scaled_error = (
          tf.abs(expected - true_param) / avg_monte_carlo_standard_error)

      return normal.Normal(
          loc=tf.zeros([], dtype), scale=1.).survival_function(scaled_error)

    # Run chains, compute the error, and compute the probability of getting
    # a more extreme error. `error_prob` has shape (nreplica * 5)
    error_prob = run_chain_and_get_estimation_error()

    # Check convergence using Markov chain central limit theorem, this is a
    # z-test at p=.01
    is_converged = error_prob > .005
    # Test at most 5% test fail out of total number of independent tests.
    n_total_tests = nreplicas * len(true_param)
    num_test_failed = self.evaluate(
        tf.math.reduce_sum(tf.cast(is_converged, dtype)))
    self.assertLessEqual(
        n_total_tests - num_test_failed, np.round(n_total_tests * .05))

  @parameterized.parameters(
      # (7, 5, 3, None),  TODO(b/182886159): Re-enable this test.
      (7, 5, 1, tf.TensorShape([None, 1])),
  )
  def testDynamicShape(self, nsample, batch_size, nd, dynamic_shape):
    dtype = np.float32

    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_log_prob_fn=independent.Independent(
            normal.Normal(tf.zeros(nd, dtype=dtype), 1.), 1).log_prob,
        step_size=.1)
    x_ = np.zeros([batch_size, nd], dtype=dtype)
    x = tf1.placeholder_with_default(x_, shape=dynamic_shape)
    mcmc_trace_ = sample.sample_chain(
        num_results=nsample,
        current_state=x,
        kernel=kernel,
        trace_fn=None,
        seed=test_util.test_seed())
    mcmc_trace = self.evaluate(mcmc_trace_)
    self.assertAllEqual(mcmc_trace.shape, [nsample, batch_size, nd])

  def testDivergence(self):
    """Neals funnel with large step size."""
    strm = test_util.test_seed_stream()
    neals_funnel = jds.JointDistributionSequential(
        [
            normal.Normal(loc=0., scale=3.),  # b0
            lambda y: sample_dist_lib.Sample(  # pylint: disable=g-long-lambda
                normal.Normal(loc=0., scale=tf.math.exp(y / 2)),
                sample_shape=9),
        ],
        validate_args=True)

    @tf.function(autograph=False)
    def run_chain_and_get_divergence():
      nchains = 5
      init_states = neals_funnel.sample(nchains, seed=strm())
      _, has_divergence = sample.sample_chain(
          num_results=100,
          kernel=preconditioned_nuts.PreconditionedNoUTurnSampler(
              target_log_prob_fn=lambda *args: neals_funnel.log_prob(args),
              step_size=[1., 1.]),
          current_state=init_states,
          trace_fn=lambda _, pkr: pkr.has_divergence,
          seed=strm())
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
    robust_lm = jds.JointDistributionSequential(
        [
            normal.Normal(loc=0., scale=1.),  # b0
            normal.Normal(loc=0., scale=1.),  # b1
            half_normal.HalfNormal(5.),  # df
            lambda df, b1, b0: independent.Independent(  # pylint: disable=g-long-lambda
                student_t.StudentT(  # Likelihood
                    df=df[..., tf.newaxis],
                    loc=(b0[..., tf.newaxis] + b1[..., tf.newaxis] * predictors[
                        tf.newaxis]),
                    scale=y_sigma)),
        ],
        validate_args=True)

    log_prob = lambda b0, b1, df: robust_lm.log_prob([b0, b1, df, obs])
    init_step_size = [1., .2, .5]
    step_size0 = [tf.cast(x, dtype=tf.float32) for x in init_step_size]

    number_of_steps, burnin, nchain = 200, 50, 10

    @tf.function(autograph=False, jit_compile=True)
    def run_chain_and_get_diagnostic():
      # Ensure we're really in graph mode.
      assert hasattr(tf.constant([]), 'graph')

      # random initialization of the starting postion of each chain
      b0, b1, df, _ = robust_lm.sample(nchain, seed=strm())

      # bijector to map contrained parameters to real
      unconstraining_bijectors = [
          identity.Identity(),
          identity.Identity(),
          exp.Exp(),
      ]

      def trace_fn(_, pkr):
        return (pkr.inner_results.inner_results.step_size,
                pkr.inner_results.inner_results.log_accept_ratio)

      kernel = dassa.DualAveragingStepSizeAdaptation(
          transformed_kernel.TransformedTransitionKernel(
              inner_kernel=preconditioned_nuts.PreconditionedNoUTurnSampler(
                  target_log_prob_fn=log_prob, step_size=step_size0),
              bijector=unconstraining_bijectors),
          target_accept_prob=.8,
          num_adaptation_steps=burnin,
      )

      # Sampling from the chain and get diagnostics
      mcmc_trace, (step_size, log_accept_ratio) = sample.sample_chain(
          num_results=number_of_steps,
          num_burnin_steps=burnin,
          current_state=[b0, b1, df],
          kernel=kernel,
          trace_fn=trace_fn,
          seed=strm())
      rhat = diagnostic.potential_scale_reduction(mcmc_trace)
      return (
          [s[-1] for s in step_size],  # final step size
          tf.math.exp(generic.reduce_logmeanexp(log_accept_ratio)),
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

  def test_step_size_trace(self):
    dist = normal.Normal(0., 1.)
    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        dist.log_prob, step_size=1.)
    _, _, fkr = sample.sample_chain(
        10,
        0.,
        kernel=kernel,
        return_final_kernel_results=True,
        seed=test_util.test_seed())
    self.assertAlmostEqual(1., self.evaluate(fkr.step_size))

  def test_zero_sized_event(self):
    tlp_fn = lambda x, y: x[:, 0] + tf.pad(y, [[0, 0], [0, 1]])[:, 0]
    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        tlp_fn, step_size=0.1)
    xy = [tf.ones([1, 1]), tf.ones([1, 0])]
    results = kernel.bootstrap_results(xy)
    self.evaluate(kernel.one_step(xy, results, seed=test_util.test_seed())[0])


# Allowed type of preconditioning schemes to use.
# See code for details.
PRECONDITION_SCHEMES = frozenset([
    'direct', 'precision_factor', 'sqrtm', 'scale',
    # `None` ==> No preconditioner. This is different than a "bad"
    # preconditioner. We will be able to check asymptotics with "None".
    'no_preconditioner'])


RunNUTSResults = collections.namedtuple('RunNUTSResults', [
    'draws',
    'step_size',
    'final_step_size',
    'accept_prob',
    'mean_accept_prob',
    'min_ess',
    'sample_mean',
    'sample_cov',
    'sample_var',
    'mean_atol',
    'cov_atol',
    'var_rtol',
])


@test_util.test_graph_mode_only
class PreconditionedNUTSCorrectnessTest(test_util.TestCase):
  """More careful tests that sampling/preconditioning is actually working."""

  def _run_nuts_with_step_size(
      self,
      target_mvn,
      precondition_scheme,
      target_accept=0.75,
      num_results=2000,
      num_adaptation_steps=20,
  ):
    """Run NUTS with step_size adaptation, and return RunNUTSResults."""
    assert precondition_scheme in PRECONDITION_SCHEMES

    target_cov = target_mvn.covariance()

    cov_linop = tf.linalg.LinearOperatorFullMatrix(
        target_cov,
        is_self_adjoint=True,
        is_positive_definite=True)

    if precondition_scheme == 'no_preconditioner':
      momentum_distribution = None
    elif precondition_scheme == 'direct':
      momentum_distribution = mvn_linear_operator.MultivariateNormalLinearOperator(
          # The covariance of momentum is inv(covariance of position), and we
          # parameterize distributions by a square root of the covariance.
          scale=cov_linop.inverse().cholesky(),)
    elif precondition_scheme == 'precision_factor':
      momentum_distribution = mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
          # The precision of momentum is the covariance of position.
          # The "factor" is the cholesky factor.
          precision_factor=cov_linop.cholesky(),)
    elif precondition_scheme == 'sqrtm':
      if JAX_MODE:
        self.skipTest('`sqrtm` is not yet implemented in JAX.')
      momentum_distribution = mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
          # The symmetric square root is a perfectly valid "factor".
          precision_factor=tf.linalg.LinearOperatorFullMatrix(
              tf.linalg.sqrtm(target_cov)),)
    elif precondition_scheme == 'scale':
      momentum_distribution = mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
          # Nothing wrong with using "scale", since the scale should be the
          # same as cov_linop.cholesky().
          precision_factor=target_mvn.scale,)
    else:
      raise RuntimeError(
          'Unhandled precondition_scheme: {}'.format(precondition_scheme))

    nuts_kernel = dassa.DualAveragingStepSizeAdaptation(
        preconditioned_nuts.PreconditionedNoUTurnSampler(
            target_log_prob_fn=target_mvn.log_prob,
            momentum_distribution=momentum_distribution,
            max_tree_depth=4,
            step_size=1.),
        num_adaptation_steps=num_adaptation_steps,
        target_accept_prob=target_accept)

    def trace_fn(_, pkr):
      results = pkr.inner_results
      return {
          'accept_prob':
              tf.exp(tf.minimum(0., results.log_accept_ratio)),
          'step_size':
              results.step_size,
      }

    strm = test_util.test_seed_stream()
    @tf.function
    def do_run_run_run():
      """Do a run, return RunNUTSResults."""
      states, trace = sample.sample_chain(
          num_results,
          current_state=tf.identity(target_mvn.sample(seed=strm())),
          kernel=nuts_kernel,
          num_burnin_steps=num_adaptation_steps,
          seed=strm(),
          trace_fn=trace_fn)

      # If we had some number of chain dimensions, we would change sample_axis.
      sample_axis = 0

      sample_cov = sample_stats.covariance(states, sample_axis=sample_axis)
      max_variance = tf.reduce_max(tf.linalg.diag_part(sample_cov))
      max_stddev = tf.sqrt(max_variance)
      min_ess = tf.reduce_min(diagnostic.effective_sample_size(states))
      mean_accept_prob = tf.reduce_mean(trace['accept_prob'])

      return RunNUTSResults(
          draws=states,
          step_size=trace['step_size'],
          final_step_size=trace['step_size'][-1],
          accept_prob=trace['accept_prob'],
          mean_accept_prob=mean_accept_prob,
          min_ess=tf.reduce_min(diagnostic.effective_sample_size(states)),
          sample_mean=tf.reduce_mean(states, axis=sample_axis),
          sample_cov=sample_cov,
          sample_var=tf.linalg.diag_part(sample_cov),

          # Standard error in variance estimation is related to standard
          # deviation of variance estimates. For a Normal, this is just Sqrt(2)
          # times variance divided by sqrt sample size (or so my old notes say).
          # So a relative tolerance is useful.
          # Add in a factor of 5 as a buffer.
          var_rtol=5 * tf.sqrt(2.) / tf.sqrt(min_ess),

          # For covariance matrix estimates, there can be terms that have
          # expectation = 0 (e.g. off diagonal entries). So the above doesn't
          # hold. So use an atol.
          cov_atol=5 * max_variance / tf.sqrt(min_ess),

          # Standard error in mean estimation is stddev divided by sqrt
          # sample size. This is an absolute tolerance.
          # Add in a factor of 5 as a buffer.
          mean_atol=5 * max_stddev / tf.sqrt(min_ess),
      )

    # Evaluate now, to ensure that states/accept_prob/etc... all match up with
    # the same graph evaluation. This is a gotcha about TFP MCMC in graph mode.
    return self.evaluate(do_run_run_run())

  def _check_correctness_of_moments_and_preconditioning(
      self,
      target_mvn,
      num_results,
      precondition_scheme,
  ):
    """Test that step size adaptation finds the theoretical optimal step size.

    See _caclulate_expected_step_size for formula details, but roughly, for a
    high dimensional Gaussian posterior, we can calculate the approximate step
    size to achieve a given target accept rate. For such a posterior,
    `PreconditionedNoUTurnSampler` mimics the dynamics of sampling from a
    standard normal distribution, and so should adapt to the step size where
    the scales are all ones.

    In the example below, `expected_step` is around 0.00002, so there is
    significantly different behavior when conditioning.

    Args:
      target_mvn: Multivariate normal instance to sample from.
      num_results: Number of samples to collect (post burn-in).
      precondition_scheme: String telling how to do preconditioning.
        Should be in PRECONDITION_SCHEMES.

    Returns:
      RunNUTSResults
    """
    results = self._run_nuts_with_step_size(
        target_mvn, precondition_scheme=precondition_scheme)

    self.assertAllClose(
        results.sample_mean, target_mvn.mean(), atol=results.mean_atol)
    self.assertAllClose(
        results.sample_var, target_mvn.variance(), rtol=results.var_rtol)
    self.assertAllClose(
        results.sample_cov, target_mvn.covariance(), atol=results.cov_atol)

    return results

  @parameterized.named_parameters(
      dict(testcase_name='_' + str(scheme), precondition_scheme=scheme)
      for scheme in PRECONDITION_SCHEMES)
  def test_correctness_with_2d_mvn_tril(self, precondition_scheme):
    # Low dimensional test to help people who want to step through and debug.
    target_mvn = mvn_tril.MultivariateNormalTriL(
        loc=tf.constant([0., 0.]),
        scale_tril=[[1., 0.], [0.5, 2.]],
    )
    self._check_correctness_of_moments_and_preconditioning(
        target_mvn,
        # Lots of results, to test tight tolerance.
        # We're using a small dims here, so this isn't a big deal.
        num_results=2000,
        precondition_scheme=precondition_scheme)

  @parameterized.named_parameters(
      dict(testcase_name='_' + str(scheme), precondition_scheme=scheme)
      for scheme in PRECONDITION_SCHEMES)
  def test_correctness_with_20d_mvn_tril(self, precondition_scheme):
    # This is an almost complete check of the Gaussian case.
    dims = 20
    scale_wishart = wishart.WishartLinearOperator(
        # Important that df is just slightly bigger than dims. This makes the
        # scale_wishart ill condtioned. The result is that tests fail if we do
        # not handle transposes correctly.
        df=1.1 * dims,
        scale=tf.linalg.LinearOperatorIdentity(dims),
        input_output_cholesky=True,
        name='wishart_for_samples',
    )

    # evaluate right here to avoid working with a random target_mvn in graph
    # mode....that would cause issues, since we read off expected statistics
    # from looking at the mvn properties, so it would be bad if these properties
    # changed with every graph eval.
    scale_tril = self.evaluate(scale_wishart.sample(seed=test_util.test_seed()))

    target_mvn = mvn_tril.MultivariateNormalTriL(
        # Non-trivial "loc" ensures we do not rely on being centered at 0.
        loc=tf.range(0., dims),
        scale_tril=scale_tril,
    )

    self._check_correctness_of_moments_and_preconditioning(
        target_mvn,
        # Lots of results, to test tight tolerance.
        num_results=1000,
        precondition_scheme=precondition_scheme)


@test_util.test_graph_mode_only
@parameterized.named_parameters(
    dict(testcase_name='_default', use_default=True),
    dict(testcase_name='_explicit', use_default=False))
class PreconditionedNUTSTest(test_util.TestCase):

  def test_f64(self, use_default):
    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = normal.Normal(0.,
                                            tf.constant(.5, dtype=tf.float64))
    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        lambda x: -x**2,
        step_size=.5,
        max_tree_depth=4,
        momentum_distribution=momentum_distribution)
    kernel = sssa.SimpleStepSizeAdaptation(kernel, num_adaptation_steps=3)
    self.evaluate(
        sample.sample_chain(
            1,
            kernel=kernel,
            current_state=tf.ones([], tf.float64),
            num_burnin_steps=5,
            seed=test_util.test_seed(),
            trace_fn=None))

  # TODO(b/175787154): Enable this test
  def DISABLED_test_f64_multichain(self, use_default):
    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = normal.Normal(
          0.0, tf.constant(0.5, dtype=tf.float64)
      )
    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        lambda x: -x**2,
        step_size=.5,
        max_tree_depth=2,
        momentum_distribution=momentum_distribution)
    kernel = sssa.SimpleStepSizeAdaptation(kernel, num_adaptation_steps=3)
    nchains = 7
    self.evaluate(
        sample.sample_chain(
            1,
            kernel=kernel,
            current_state=tf.ones([nchains], tf.float64),
            num_burnin_steps=5,
            seed=test_util.test_seed(),
            trace_fn=None))

  def test_diag(self, use_default):
    """Test that a diagonal multivariate normal can be effectively sampled from.

    Args:
      use_default: bool, whether to use a custom momentum distribution, or
        the default.
    """
    mvn = mvn_diag.MultivariateNormalDiag(
        loc=[1., 2., 3.], scale_diag=[0.1, 1., 10.])

    if use_default:
      momentum_distribution = None
      step_size = 0.1
    else:
      momentum_distribution = mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
          precision_factor=mvn.scale,)
      step_size = 1.1
    nuts_kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=step_size,
        max_tree_depth=4)
    draws = sample.sample_chain(
        110,
        tf.zeros(3),
        kernel=nuts_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = diagnostic.effective_sample_size(
        draws[-100:], filter_threshold=0, filter_beyond_positive_pairs=False)

    if not use_default:
      self.assertGreaterEqual(self.evaluate(tf.reduce_min(ess)), 40.)
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 100.)

  def test_tril(self, use_default):
    if tf.executing_eagerly():
      self.skipTest('b/169882656 Too many warnings are issued in eager logs')
    cov = 0.9 * tf.ones([3, 3]) + 0.1 * tf.eye(3)
    scale = tf.linalg.cholesky(cov)
    mv_tril = mvn_tril.MultivariateNormalTriL(
        loc=[1., 2., 3.], scale_tril=scale)

    if use_default:
      momentum_distribution = None
      step_size = 0.3
    else:
      momentum_distribution = mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
          # TODO(b/170015229) Don't use the covariance as inverse scale,
          # it is the wrong preconditioner.
          precision_factor=tf.linalg.LinearOperatorFullMatrix(cov),)
      step_size = 1.1
    nuts_kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_log_prob_fn=mv_tril.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=step_size,
        max_tree_depth=4)
    draws = sample.sample_chain(
        120,
        tf.zeros(3),
        kernel=nuts_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = diagnostic.effective_sample_size(
        draws[-100:], filter_threshold=0, filter_beyond_positive_pairs=False)

    # TODO(b/170015229): These and other tests like it, which assert ess is
    # greater than some number, were all passing, even though the preconditioner
    # was the wrong one. Why is that? A guess is that since there are *many*
    # ways to have larger ess, these tests don't really test correctness.
    # Perhaps remove all tests like these.
    if not use_default:
      self.assertGreaterEqual(self.evaluate(tf.reduce_min(ess)), 40.)
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 100.)

  def test_multi_state_part(self, use_default):
    mvn = jds.JointDistributionSequential([
        normal.Normal(1., 0.1),
        normal.Normal(2., 1.),
        independent.Independent(normal.Normal(3 * tf.ones([2, 3, 4]), 10.), 3)
    ])

    if use_default:
      momentum_distribution = None
      step_size = 0.1
    else:
      reshape_to_scalar = reshape.Reshape(event_shape_out=[])
      reshape_to_234 = reshape.Reshape(event_shape_out=[2, 3, 4])
      momentum_distribution = jds.JointDistributionSequential([
          reshape_to_scalar(
              mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag([0.1]))),
          reshape_to_scalar(
              mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag([1.]))),
          reshape_to_234(
              mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag(
                      tf.fill([24], 10.))))
      ])
      step_size = 0.3
    nuts_kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=step_size,
        max_tree_depth=4)

    draws = sample.sample_chain(
        100, [0., 0., tf.zeros((2, 3, 4))],
        kernel=nuts_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = diagnostic.effective_sample_size(
        draws, filter_threshold=0, filter_beyond_positive_pairs=False)
    if not use_default:
      self.assertGreaterEqual(
          self.evaluate(
              tf.reduce_min(tf.nest.map_structure(tf.reduce_min, ess))),
          40.)
    else:
      self.assertLess(
          self.evaluate(
              tf.reduce_min(tf.nest.map_structure(tf.reduce_min, ess))),
          50.)

  def test_batched_state(self, use_default):
    mvn = mvn_diag.MultivariateNormalDiag(
        loc=[1., 2., 3.], scale_diag=[0.1, 1., 10.])
    batch_shape = [2, 4]
    if use_default:
      step_size = 0.1
      momentum_distribution = None
    else:
      step_size = 1.0
      momentum_distribution = mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
          tf.zeros((2, 4, 3)), precision_factor=mvn.scale)

    nuts_kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=step_size,
        max_tree_depth=5)

    draws = sample.sample_chain(
        110,
        tf.zeros(batch_shape + [3]),
        kernel=nuts_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = diagnostic.effective_sample_size(
        draws[10:],
        cross_chain_dims=[1, 2],
        filter_threshold=0,
        filter_beyond_positive_pairs=False)
    if not use_default:
      self.assertGreaterEqual(self.evaluate(tf.reduce_min(ess)), 40.)
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 100.)

  def test_batches(self, use_default):
    mvn = jds.JointDistributionSequential(
        [normal.Normal(1., 0.1),
         normal.Normal(2., 1.),
         normal.Normal(3., 10.)])
    n_chains = 10
    if use_default:
      momentum_distribution = None
      step_size = 0.1
    else:
      reshape_to_scalar = reshape.Reshape(event_shape_out=[])
      momentum_distribution = jds.JointDistributionSequential([
          reshape_to_scalar(
              mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag(
                      tf.fill([n_chains, 1], 0.1)))),
          reshape_to_scalar(
              mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag(
                      tf.fill([n_chains, 1], 1.)))),
          reshape_to_scalar(
              mvnpflo.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag(
                      tf.fill([n_chains, 1], 10.)))),
      ])
      step_size = 1.1

    nuts_kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=step_size,
        max_tree_depth=4)

    draws = sample.sample_chain(
        100, [tf.zeros([n_chains]) for _ in range(3)],
        kernel=nuts_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = diagnostic.effective_sample_size(
        draws,
        cross_chain_dims=[1 for _ in draws],
        filter_threshold=0,
        filter_beyond_positive_pairs=False)
    if not use_default:
      self.assertGreaterEqual(self.evaluate(tf.reduce_min(ess)), 40.)
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 100.)


@test_util.test_all_tf_execution_regimes
class DistributedNutsTest(distribute_test_lib.DistributedTest):

  def test_pnuts_kernel_tracks_axis_names(self):
    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        normal.Normal(0, 1).log_prob, step_size=1.9)
    self.assertIsNone(kernel.experimental_shard_axis_names)
    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        normal.Normal(0, 1).log_prob,
        step_size=1.9,
        experimental_shard_axis_names=['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])
    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        normal.Normal(0, 1).log_prob,
        step_size=1.9).experimental_with_shard_axes(['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])

  def test_takes_same_number_leapfrog_steps_with_sharded_state(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) + distribute_lib.psum(
          normal.Normal(distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b),
          'foo'))

    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_log_prob, step_size=1.9)
    sharded_kernel = kernel.experimental_with_shard_axes([None, ['foo']])

    def run(seed):
      state = [tf.convert_to_tensor(0.), tf.convert_to_tensor(0.)]
      kr = sharded_kernel.bootstrap_results(state)
      _, kr = sharded_kernel.one_step(state, kr, seed=seed)
      return kr.leapfrogs_taken

    leapfrogs_taken = self.evaluate(
        self.per_replica_to_tensor(
            self.strategy_run(
                run,
                args=(samplers.zeros_seed(),),
                in_axes=None,
                axis_name='foo'), 0))

    for i in range(distribute_test_lib.NUM_DEVICES):
      self.assertAllClose(leapfrogs_taken[i], leapfrogs_taken[0])

  def test_unsharded_state_remains_synchronized_across_devices(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) + distribute_lib.psum(
          normal.Normal(distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b),
          'foo'))

    kernel = preconditioned_nuts.PreconditionedNoUTurnSampler(
        target_log_prob, step_size=1e-1)
    sharded_kernel = kernel.experimental_with_shard_axes([None, ['foo']])

    def run(seed):
      state = [tf.convert_to_tensor(-10.), tf.convert_to_tensor(-10.)]
      kr = sharded_kernel.bootstrap_results(state)
      state, _ = sharded_kernel.one_step(state, kr, seed=seed)
      return state

    state = self.evaluate(
        self.per_replica_to_tensor(
            self.strategy_run(
                run,
                args=(samplers.zeros_seed(),),
                in_axes=None,
                axis_name='foo'), 0))

    for i in range(distribute_test_lib.NUM_DEVICES):
      self.assertAllClose(state[0][i], state[0][0])


if __name__ == '__main__':
  test_util.main()
