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
"""Tests for HamiltonianMonteCarlo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings

# Dependency imports

import numpy as np
from scipy import stats

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc.hmc import _compute_log_acceptance_correction


def _set_seed(seed):
  """Helper which uses graph seed if using eager."""
  # TODO(b/68017812): Deprecate once eager correctly supports seed.
  if tf.executing_eagerly():
    tf.random.set_seed(seed)
    return None
  return seed


@test_util.test_all_tf_execution_regimes
class HMCTest(test_util.TestCase):

  def setUp(self):
    self._shape_param = 5.
    self._rate_param = 10.

    super(HMCTest, self).setUp()
    tf.random.set_seed(10003)
    np.random.seed(10003)

  def assertAllFinite(self, x):
    self.assertAllEqual(np.ones_like(x).astype(bool), np.isfinite(x))

  def _log_gamma_log_prob(self, x, event_dims=()):
    """Computes log-pdf of a log-gamma random variable.

    Args:
      x: Value of the random variable.
      event_dims: Dimensions not to treat as independent.

    Returns:
      log_prob: The log-pdf up to a normalizing constant.
    """
    return tf.reduce_sum(
        self._shape_param * x - self._rate_param * tf.math.exp(x),
        axis=event_dims)

  def testSampleChainSeedReproducibleWorksCorrectly(self):
    num_results = 10
    independent_chain_ndims = 1

    def log_gamma_log_prob(x):
      event_dims = tf.range(independent_chain_ndims, tf.rank(x))
      return self._log_gamma_log_prob(x, event_dims)

    current_state = np.random.rand(4, 3, 2)

    samples0, kernel_results0 = tfp.mcmc.sample_chain(
        num_results=2 * num_results,
        num_steps_between_results=0,
        # Following args are identical to below.
        current_state=current_state,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_gamma_log_prob,
            step_size=0.1,
            num_leapfrog_steps=2,
            seed=_set_seed(52)),
        num_burnin_steps=150,
        parallel_iterations=1)

    samples1, kernel_results1 = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_steps_between_results=1,
        # Following args are identical to above.
        current_state=current_state,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_gamma_log_prob,
            step_size=0.1,
            num_leapfrog_steps=2,
            seed=_set_seed(52)),
        num_burnin_steps=150,
        parallel_iterations=1)

    [
        samples0_,
        samples1_,
        target_log_prob0_,
        target_log_prob1_,
    ] = self.evaluate([
        samples0,
        samples1,
        kernel_results0.accepted_results.target_log_prob,
        kernel_results1.accepted_results.target_log_prob,
    ])
    self.assertAllClose(samples0_[::2], samples1_,
                        atol=1e-5, rtol=1e-5)
    self.assertAllClose(target_log_prob0_[::2], target_log_prob1_,
                        atol=1e-5, rtol=1e-5)

  def _chain_gets_correct_expectations(self, x, independent_chain_ndims):
    counter = collections.Counter()
    def log_gamma_log_prob(x):
      counter['target_calls'] += 1
      event_dims = tf.range(independent_chain_ndims, tf.rank(x))
      return self._log_gamma_log_prob(x, event_dims)

    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=150,
        current_state=x,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_gamma_log_prob,
            step_size=0.05,
            num_leapfrog_steps=2,
            seed=_set_seed(42)),
        num_burnin_steps=150,
        parallel_iterations=1)

    if tf.executing_eagerly():
      # TODO(b/79991421): Figure out why this is approx twice as many as it
      # should be. I.e., `expected_calls = (150 + 150) * 2 + 1`.
      expected_calls = 1202
    else:
      expected_calls = 4
    self.assertAllEqual(dict(target_calls=expected_calls), counter)

    expected_x = (tf.math.digamma(self._shape_param) - np.log(self._rate_param))

    expected_exp_x = self._shape_param / self._rate_param

    log_accept_ratio_, samples_, expected_x_ = self.evaluate(
        [kernel_results.log_accept_ratio, samples, expected_x])

    actual_x = samples_.mean()
    actual_exp_x = np.exp(samples_).mean()
    acceptance_probs = np.exp(np.minimum(log_accept_ratio_, 0.))

    tf1.logging.vlog(
        1, 'True      E[x, exp(x)]: {}\t{}'.format(expected_x_, expected_exp_x))
    tf1.logging.vlog(
        1, 'Estimated E[x, exp(x)]: {}\t{}'.format(actual_x, actual_exp_x))
    self.assertNear(actual_x, expected_x_, 3e-2)
    self.assertNear(actual_exp_x, expected_exp_x, 2e-2)
    self.assertAllEqual(np.ones_like(acceptance_probs, np.bool),
                        acceptance_probs > 0.5)
    self.assertAllEqual(np.ones_like(acceptance_probs, np.bool),
                        acceptance_probs <= 1.)

  def _chain_gets_correct_expectations_wrapper(self, independent_chain_ndims):
    x = tf.constant(np.random.rand(50, 10, 2), np.float32, name='x')
    self._chain_gets_correct_expectations(x, independent_chain_ndims)

  def testHMCChainExpectationsNullShape(self):
    self._chain_gets_correct_expectations_wrapper(0)

  def testHMCChainExpectations1(self):
    self._chain_gets_correct_expectations_wrapper(1)

  def testHMCChainExpectations2(self):
    self._chain_gets_correct_expectations_wrapper(2)

  def testKernelResultsUsingTruncatedDistribution(self):
    def log_prob(x):
      return tf.where(x < 0., tf.constant(-np.inf, x.dtype), -x - x**2)
    # This log_prob has the property that it is likely to attract
    # the flow toward, and below, zero...but for x <=0,
    # log_prob(x) = -inf, which should result in rejection, as well
    # as a non-finite log_prob.  Thus, this distribution gives us an opportunity
    # to test out the kernel results ability to correctly capture rejections due
    # to finite AND non-finite reasons.
    # Why use a non-constant gradient?  This ensures the leapfrog integrator
    # will not be exact.

    num_results = 1000
    # Large step size, will give rejections due to integration error in addition
    # to rejection due to going into a region of log_prob = -inf.
    step_size = 0.2
    num_leapfrog_steps = 5
    num_chains = 2

    # Start multiple independent chains.
    initial_state = tf.convert_to_tensor([0.1] * num_chains)

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=initial_state,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            seed=_set_seed(42)),
        parallel_iterations=1)

    states_, kernel_results_ = self.evaluate([states, kernel_results])
    pstates_ = kernel_results_.proposed_state

    neg_inf_mask = np.isneginf(
        kernel_results_.proposed_results.target_log_prob)

    # First:  Test that the mathematical properties of the above log prob
    # function in conjunction with HMC show up as expected in kernel_results_.

    # We better have log_prob = -inf some of the time.
    self.assertLess(0, neg_inf_mask.sum())
    # We better have some rejections due to something other than -inf.
    self.assertLess(neg_inf_mask.sum(), (~kernel_results_.is_accepted).sum())
    # We better have accepted a decent amount, even near end of the chain.
    self.assertLess(
        0.1, kernel_results_.is_accepted[int(0.9 * num_results):].mean())
    # We better not have any NaNs in states or log_prob.
    # We may have some NaN in grads, which involve multiplication/addition due
    # to gradient rules.  This is the known "NaN grad issue with tf.where."
    self.assertAllEqual(
        np.zeros_like(states_),
        np.isnan(kernel_results_.proposed_results.target_log_prob))
    self.assertAllEqual(
        np.zeros_like(states_),
        np.isnan(states_))
    # We better not have any +inf in states, grads, or log_prob.
    self.assertAllEqual(
        np.zeros_like(states_),
        np.isposinf(kernel_results_.proposed_results.target_log_prob))
    self.assertAllEqual(
        np.zeros_like(states_),
        np.isposinf(
            kernel_results_.proposed_results.grads_target_log_prob[0]))
    self.assertAllEqual(np.zeros_like(states_),
                        np.isposinf(states_))

    # Second:  Test that kernel_results is congruent with itself and
    # acceptance/rejection of states.

    # Proposed state is negative iff proposed target log prob is -inf.
    np.testing.assert_array_less(pstates_[neg_inf_mask], 0.)
    np.testing.assert_array_less(0., pstates_[~neg_inf_mask])

    # Acceptance probs are zero whenever proposed state is negative.
    acceptance_probs = np.exp(np.minimum(
        kernel_results_.log_accept_ratio, 0.))
    self.assertAllEqual(
        np.zeros_like(pstates_[neg_inf_mask]),
        acceptance_probs[neg_inf_mask])

    # The move is accepted ==> state = proposed state.
    self.assertAllEqual(
        states_[kernel_results_.is_accepted],
        pstates_[kernel_results_.is_accepted],
    )
    # The move was rejected <==> state[t] == state[t - 1].
    for t in range(1, num_results):
      for i in range(num_chains):
        if kernel_results_.is_accepted[t, i]:
          self.assertNotEqual(states_[t, i], states_[t - 1, i])
        else:
          self.assertEqual(states_[t, i], states_[t - 1, i])

  def _kernel_leaves_target_invariant(self, initial_draws,
                                      independent_chain_ndims):
    def log_gamma_log_prob(x):
      event_dims = tf.range(independent_chain_ndims, tf.rank(x))
      return self._log_gamma_log_prob(x, event_dims)

    def fake_log_prob(x):
      """Cooled version of the target distribution."""
      return 1.1 * log_gamma_log_prob(x)

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_gamma_log_prob,
        step_size=0.4,
        num_leapfrog_steps=5,
        seed=_set_seed(43))
    sample, kernel_results = hmc.one_step(
        current_state=initial_draws,
        previous_kernel_results=hmc.bootstrap_results(initial_draws))

    bad_hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=fake_log_prob,
        step_size=0.4,
        num_leapfrog_steps=5,
        seed=_set_seed(44))
    bad_sample, bad_kernel_results = bad_hmc.one_step(
        current_state=initial_draws,
        previous_kernel_results=bad_hmc.bootstrap_results(initial_draws))

    [
        log_accept_ratio_,
        bad_log_accept_ratio_,
        initial_draws_,
        updated_draws_,
        fake_draws_,
    ] = self.evaluate([
        kernel_results.log_accept_ratio,
        bad_kernel_results.log_accept_ratio,
        initial_draws,
        sample,
        bad_sample,
    ])

    # Confirm step size is small enough that we usually accept.
    acceptance_probs = np.exp(np.minimum(log_accept_ratio_, 0.))
    bad_acceptance_probs = np.exp(np.minimum(bad_log_accept_ratio_, 0.))
    self.assertGreater(acceptance_probs.mean(), 0.5)
    self.assertGreater(bad_acceptance_probs.mean(), 0.5)

    # Confirm step size is large enough that we sometimes reject.
    self.assertLess(acceptance_probs.mean(), 0.99)
    self.assertLess(bad_acceptance_probs.mean(), 0.99)

    _, ks_p_value_true = stats.ks_2samp(initial_draws_.flatten(),
                                        updated_draws_.flatten())
    _, ks_p_value_fake = stats.ks_2samp(initial_draws_.flatten(),
                                        fake_draws_.flatten())

    tf1.logging.vlog(
        1,
        'acceptance rate for true target: {}'.format(acceptance_probs.mean()))
    tf1.logging.vlog(
        1, 'acceptance rate for fake target: {}'.format(
            bad_acceptance_probs.mean()))
    tf1.logging.vlog(
        1, 'K-S p-value for true target: {}'.format(ks_p_value_true))
    tf1.logging.vlog(
        1, 'K-S p-value for fake target: {}'.format(ks_p_value_fake))
    # Make sure that the MCMC update hasn't changed the empirical CDF much.
    self.assertGreater(ks_p_value_true, 1e-3)
    # Confirm that targeting the wrong distribution does
    # significantly change the empirical CDF.
    self.assertLess(ks_p_value_fake, 1e-6)

  def _kernel_leaves_target_invariant_wrapper(self, independent_chain_ndims):
    """Tests that the kernel leaves the target distribution invariant.

    Draws some independent samples from the target distribution,
    applies an iteration of the MCMC kernel, then runs a
    Kolmogorov-Smirnov test to determine if the distribution of the
    MCMC-updated samples has changed.

    We also confirm that running the kernel with a different log-pdf
    does change the target distribution. (And that we can detect that.)

    Args:
      independent_chain_ndims: Python `int` scalar representing the number of
        dims associated with independent chains.
    """
    initial_draws = np.log(np.random.gamma(self._shape_param,
                                           size=[50000, 2, 2]))
    initial_draws -= np.log(self._rate_param)
    x = tf.constant(initial_draws, np.float32)
    self._kernel_leaves_target_invariant(x, independent_chain_ndims)

  def testKernelLeavesTargetInvariant1(self):
    self._kernel_leaves_target_invariant_wrapper(1)

  def testKernelLeavesTargetInvariant2(self):
    self._kernel_leaves_target_invariant_wrapper(2)

  def testKernelLeavesTargetInvariant3(self):
    self._kernel_leaves_target_invariant_wrapper(3)

  def testNanRejection(self):
    """Tests that an update that yields NaN potentials gets rejected.

    We run HMC with a target distribution that returns NaN
    log-likelihoods if any element of x < 0, and unit-scale
    exponential log-likelihoods otherwise. The exponential potential
    pushes x towards 0, ensuring that any reasonably large update will
    push us over the edge into NaN territory.
    """
    def _unbounded_exponential_log_prob(x):
      """An exponential distribution with log-likelihood NaN for x < 0."""
      per_element_potentials = tf.where(
          x < 0., tf.constant(np.nan, x.dtype), -x)
      return tf.reduce_sum(per_element_potentials)

    initial_x = tf.linspace(0.01, 5, 10)
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=_unbounded_exponential_log_prob,
        step_size=2.,
        num_leapfrog_steps=5,
        seed=_set_seed(46))
    updated_x, kernel_results = hmc.one_step(
        current_state=initial_x,
        previous_kernel_results=hmc.bootstrap_results(initial_x))
    initial_x_, updated_x_, log_accept_ratio_ = self.evaluate(
        [initial_x, updated_x, kernel_results.log_accept_ratio])
    acceptance_probs = np.exp(np.minimum(log_accept_ratio_, 0.))

    tf1.logging.vlog(1, 'initial_x = {}'.format(initial_x_))
    tf1.logging.vlog(1, 'updated_x = {}'.format(updated_x_))
    tf1.logging.vlog(1, 'log_accept_ratio = {}'.format(log_accept_ratio_))

    self.assertAllEqual(initial_x_, updated_x_)
    self.assertEqual(acceptance_probs, 0.)

  def testNanFromGradsDontPropagate(self):
    """Test that update with NaN gradients does not cause NaN in results."""
    if tf1.control_flow_v2_enabled():
      self.skipTest('b/138796859')
    if tf.executing_eagerly(): return
    def _nan_log_prob_with_nan_gradient(x):
      return np.nan * tf.reduce_sum(x)

    initial_x = tf.linspace(0.01, 5, 10)
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=_nan_log_prob_with_nan_gradient,
        step_size=2.,
        num_leapfrog_steps=5,
        seed=_set_seed(47))
    updated_x, kernel_results = hmc.one_step(
        current_state=initial_x,
        previous_kernel_results=hmc.bootstrap_results(initial_x))
    initial_x_, updated_x_, log_accept_ratio_ = self.evaluate(
        [initial_x, updated_x, kernel_results.log_accept_ratio])
    acceptance_probs = np.exp(np.minimum(log_accept_ratio_, 0.))

    tf1.logging.vlog(1, 'initial_x = {}'.format(initial_x_))
    tf1.logging.vlog(1, 'updated_x = {}'.format(updated_x_))
    tf1.logging.vlog(1, 'log_accept_ratio = {}'.format(log_accept_ratio_))

    self.assertAllEqual(initial_x_, updated_x_)
    self.assertEqual(acceptance_probs, 0.)

    self.assertAllEqual([True], [
        g is None for g in tf.gradients(
            ys=kernel_results.proposed_results.grads_target_log_prob,
            xs=initial_x)
    ])
    self.assertAllFinite(
        self.evaluate(tf.gradients(ys=updated_x, xs=initial_x)[0]))

    # Gradients of the acceptance probs and new log prob are not finite.
    # self.assertAllFinite(
    #     self.evaluate(tf.gradients(acceptance_probs, initial_x)[0]))
    # self.assertAllFinite(
    #     self.evaluate(tf.gradients(new_log_prob, initial_x)[0]))

  def _testChainWorksDtype(self, dtype):
    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=10,
        current_state=np.zeros(5).astype(dtype),
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=lambda x: -tf.reduce_sum(x**2., axis=-1),
            step_size=0.01,
            num_leapfrog_steps=10,
            seed=_set_seed(48)),
        parallel_iterations=1)
    states_, log_accept_ratio_ = self.evaluate(
        [states, kernel_results.log_accept_ratio])
    self.assertEqual(dtype, states_.dtype)
    self.assertEqual(dtype, log_accept_ratio_.dtype)

  def testChainWorksIn64Bit(self):
    self._testChainWorksDtype(np.float64)

  def testChainWorksIn16Bit(self):
    self._testChainWorksDtype(np.float16)

  def testChainWorksCorrelatedMultivariate(self):
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5],
                      [0.5, 1]])
    num_results = 1500
    counter = collections.Counter()
    def target_log_prob(x, y):
      counter['target_calls'] += 1
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      z = tf.stack([x, y], axis=-1) - true_mean
      z = tf.squeeze(
          tf.linalg.triangular_solve(
              np.linalg.cholesky(true_cov),
              z[..., tf.newaxis]),
          axis=-1)
      return -0.5 * tf.reduce_sum(z**2., axis=-1)

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=[dtype(-2), dtype(2)],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            step_size=[1.23, 1.23],
            num_leapfrog_steps=2,
            seed=_set_seed(54)),
        num_burnin_steps=200,
        parallel_iterations=1)

    if tf.executing_eagerly():
      # TODO(b/79991421): Figure out why this is approx twice as many as it
      # should be. I.e., `expected_calls = (num_results + 200) * 2 * 2 + 1`.
      expected_calls = 6802
    else:
      expected_calls = 4
    self.assertAllEqual(dict(target_calls=expected_calls), counter)

    states = tf.stack(states, axis=-1)
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    x = states - sample_mean
    sample_cov = tf.matmul(x, x, transpose_a=True) / dtype(num_results)
    [sample_mean_, sample_cov_, is_accepted_] = self.evaluate([
        sample_mean, sample_cov, kernel_results.is_accepted])
    self.assertNear(0.6, is_accepted_.mean(), err=0.05)
    self.assertAllClose(true_mean, sample_mean_,
                        atol=0.06, rtol=0.)
    self.assertAllClose(true_cov, sample_cov_,
                        atol=0., rtol=0.2)

  def testUncalibratedHMCPreservesStaticShape(self):
    uncal_hmc = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -tf.reduce_sum(x**2., axis=-1),
        step_size=0.5,
        num_leapfrog_steps=2,
        seed=_set_seed(1042))
    x0 = tf.constant([[-1., 0.5],
                      [0., 0.],
                      [1., 1.25]])
    r0 = uncal_hmc.bootstrap_results(x0)
    x1, r1 = uncal_hmc.one_step(x0, r0)
    self.assertAllEqual([3, 2], x0.shape)
    self.assertAllEqual([3], r0.target_log_prob.shape)
    self.assertAllEqual([3, 2], x1.shape)
    self.assertAllEqual([3], r1.target_log_prob.shape)

  def testHMCPreservesStaticShape(self):
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -tf.reduce_sum(x**2., axis=-1),
        step_size=0.5,
        num_leapfrog_steps=2,
        seed=_set_seed(1042))
    x0 = tf.constant([[-1., 0.5],
                      [0., 0.],
                      [1., 1.25]])
    r0 = hmc.bootstrap_results(x0)
    x1, r1 = hmc.one_step(x0, r0)
    self.assertAllEqual([3, 2], x0.shape)
    self.assertAllEqual([3], r0.accepted_results.target_log_prob.shape)
    self.assertAllEqual([3, 2], x1.shape)
    self.assertAllEqual([3], r1.accepted_results.target_log_prob.shape)

  def testHMCIsCalibrated(self):
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,
        step_size=0.5,
        num_leapfrog_steps=2,
    )
    self.assertTrue(hmc.is_calibrated)

  def testUncalibratedHMCIsNotCalibrated(self):
    uncal_hmc = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -tf.square(x) / 2.,
        step_size=0.5,
        num_leapfrog_steps=2,
    )
    self.assertFalse(uncal_hmc.is_calibrated)

  def testAdaptiveParameters(self):
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -x**2.,
        step_size=0.5,
        num_leapfrog_steps=2,
        seed=_set_seed(1042),
        store_parameters_in_results=True)
    x0 = tf.zeros(2)
    r0 = hmc.bootstrap_results(x0)
    x1, r1 = hmc.one_step(x0, r0)
    r1_zero = r1._replace(
        accepted_results=r1.accepted_results._replace(
            step_size=tf.constant(0.)))
    x2, r2 = hmc.one_step(x1, r1_zero)

    r0_, r1_, r2_, x1_, x2_ = self.evaluate([r0, r1, r2, x1, x2])

    self.assertAllEqual(0.5, r0_.accepted_results.step_size)
    self.assertAllEqual(2, r0_.accepted_results.num_leapfrog_steps)
    self.assertAllEqual(0.5, r1_.accepted_results.step_size)
    self.assertAllEqual(2, r1_.accepted_results.num_leapfrog_steps)
    self.assertAllEqual(0., r2_.accepted_results.step_size)
    self.assertAllEqual(2, r2_.accepted_results.num_leapfrog_steps)
    # Since step size is 0, we shouldn't have moved despite being accepted.
    self.assertAllEqual(x2_, x1_)
    self.assertAllEqual([True, True], r2_.is_accepted)

  def testWarnMutableParameters(self):
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as triggered:
      tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=lambda x: -x**2.,
          num_leapfrog_steps=tf.Variable(2.),
          step_size=tf.Variable(0.1),
          store_parameters_in_results=False)
    self.assertTrue(
        any('Please consult the docstring' in str(warning.message)
            for warning in triggered))

  def testSoftplusCreatedOutsideKernelDoesNotKillGradients(self):
    softplus = tfp.bijectors.Softplus()
    def target_log_prob_fn(x):
      x = softplus.forward(x)
      return -tf.reduce_sum(x**2., axis=-1)

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.5,
        num_leapfrog_steps=2,
        seed=_set_seed(1042))
    x0 = tf.constant([[-1., 0.5], [0., 0.], [1., 1.25]])
    # Simply calling hmc.bootstrap_results(x0) used to fail with
    #   ValueError: Encountered `None` gradient.
    r0 = hmc.bootstrap_results(x0)
    # Also ensure eval doesn't crash things.
    self.evaluate(r0)


class _LogCorrectionTest(object):

  def testHandlesNanFromPotential(self):
    tlp = [1, np.inf, -np.inf, np.nan]
    target_log_prob, proposed_target_log_prob = [
        self.dtype(x.flatten()) for x in np.meshgrid(tlp, tlp)]
    num_chains = len(target_log_prob)
    x0 = np.zeros(num_chains, dtype=self.dtype)

    def make_trick_fun(f):
      f_x = tf.convert_to_tensor(f)

      def _fn(x):
        # We'll make the gradient be `1` regardless of input.
        return f_x + (x - tf.stop_gradient(x))
      return _fn

    # Use trick fun to get "current" results.
    pkr = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=make_trick_fun(target_log_prob),
        step_size=1.,
        num_leapfrog_steps=1).bootstrap_results(x0)

    # Use trick fun to inject "proposed" results.
    _, results = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=make_trick_fun(proposed_target_log_prob),
        step_size=1.,
        num_leapfrog_steps=1).one_step(x0, pkr)

    [actual_log_accept_ratio_, actual_grads_target_log_prob_] = self.evaluate([
        results.log_accept_ratio,
        results.accepted_results.grads_target_log_prob])

    # First log(accept_ratio) is finite, rest are weird so reject them.
    self.assertTrue(np.isfinite(actual_log_accept_ratio_[0]))
    self.assertAllEqual(self.dtype([-np.inf]*(num_chains - 1)),
                        actual_log_accept_ratio_[1:])

    # Ensure gradient is finite.
    self.assertAllEqual(
        np.ones_like(actual_grads_target_log_prob_).astype(np.bool),
        np.isfinite(actual_grads_target_log_prob_))

  def testHandlesNanFromKinetic(self):
    if tf.executing_eagerly(): return
    x = [1, np.inf, -np.inf, np.nan]
    momentums, proposed_momentums = [
        [np.reshape(self.dtype(x), [-1, 1])]
        for x in np.meshgrid(x, x)]
    num_chains = len(momentums[0])

    momentums = [tf.convert_to_tensor(momentums[0])]
    proposed_momentums = [tf.convert_to_tensor(proposed_momentums[0])]

    log_acceptance_correction = _compute_log_acceptance_correction(
        momentums,
        proposed_momentums,
        independent_chain_ndims=1)
    grads = tf.gradients(ys=log_acceptance_correction, xs=momentums)

    [actual_log_acceptance_correction, grads_] = self.evaluate([
        log_acceptance_correction, grads])

    # Ensure log_acceptance_correction is `inf` (note: that's positive inf) in
    # weird cases and finite otherwise.
    expected_log_acceptance_correction = -(
        self.dtype([0] + [np.inf]*(num_chains - 1)))
    self.assertAllEqual(expected_log_acceptance_correction,
                        actual_log_acceptance_correction)

    # Ensure gradient is finite.
    g = grads_[0].reshape([len(x), len(x)])[:, 0]
    self.assertAllEqual(np.ones_like(g).astype(np.bool), np.isfinite(g))

    # The remaining gradients are nan because the momentum was itself nan or
    # inf.
    g = grads_[0].reshape([len(x), len(x)])[:, 1:]
    self.assertAllEqual(np.ones_like(g).astype(np.bool), np.isnan(g))


@test_util.test_all_tf_execution_regimes
class LogCorrectionTest16(test_util.TestCase, _LogCorrectionTest):
  dtype = np.float16


@test_util.test_all_tf_execution_regimes
class LogCorrectionTest32(test_util.TestCase, _LogCorrectionTest):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class LogCorrectionTest64(test_util.TestCase, _LogCorrectionTest):
  dtype = np.float64


class _HMCHandlesLists(object):

  def testStateParts(self):
    cast = lambda x: np.array(x, self.dtype)
    dist_x = tfd.Normal(loc=cast(0), scale=cast(1))
    dist_y = tfd.Independent(
        tfd.Gamma(concentration=cast([1, 2]),
                  rate=cast([0.5, 0.75])),
        reinterpreted_batch_ndims=1)
    def target_log_prob(x, y):
      return dist_x.log_prob(x) + dist_y.log_prob(y)
    x0 = [dist_x.sample(seed=_set_seed(61)), dist_y.sample(seed=_set_seed(62))]
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=1.,
        num_leapfrog_steps=1,
        seed=_set_seed(49))
    # We are using bijectors to sample from a transformed density defined on
    # an unbounded domain. The samples returned are from the original bounded
    # domain.
    unconstraining_bijectors = [
        tfb.Identity(),      # Maps R to R.
        tfb.Exp(),           # Maps R to a positive real.
    ]
    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=kernel, bijector=unconstraining_bijectors)
    samples, _ = tfp.mcmc.sample_chain(
        num_results=2000,
        current_state=x0,
        kernel=transformed_kernel,
        num_burnin_steps=500,
        parallel_iterations=1)
    actual_means = [tf.reduce_mean(s, axis=0) for s in samples]
    actual_vars = [tf.math.reduce_variance(s, axis=0) for s in samples]
    expected_means = [dist_x.mean(), dist_y.mean()]
    expected_vars = [dist_x.variance(), dist_y.variance()]
    [
        actual_means_,
        actual_vars_,
        expected_means_,
        expected_vars_,
    ] = self.evaluate([
        actual_means,
        actual_vars,
        expected_means,
        expected_vars,
    ])
    self.assertAllClose(expected_means_, actual_means_, atol=0.1, rtol=0.16)
    self.assertAllClose(expected_vars_, actual_vars_, atol=0., rtol=0.5)


@test_util.test_all_tf_execution_regimes
class HMCHandlesLists32(_HMCHandlesLists, test_util.TestCase):
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class HMCHandlesLists64(_HMCHandlesLists, test_util.TestCase):
  dtype = np.float64


@test_util.test_all_tf_execution_regimes
class HMCAdaptiveStepSize(test_util.TestCase):

  def setUp(self):
    super(HMCAdaptiveStepSize, self).setUp()
    tf.random.set_seed(10014)
    np.random.seed(10014)

  def test_multiple_step_sizes_different_ranks(self):
    num_results = 5

    # Ignoring adaptation (or assuming that adaptation is performed using
    # a wrapper class like SimpleStepSizeAdaptation), test that we can
    # construct and run an HMC kernel with state parts and matching per-element
    # step sizes of varying rank.
    initial_step_sizes = [1e-5, [1e-4, 1e-3]]  # Scalar and vector state parts.
    initial_state = [0., [0., 0.]]
    dtype = np.float32

    def target_log_prob_fn(x1, x2):
      d = tfd.Normal(dtype(0.), dtype(1.))
      return d.log_prob(x1) + tf.reduce_sum(d.log_prob(x2))

    samples, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=0,
        current_state=[dtype(x) for x in initial_state],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            num_leapfrog_steps=2,
            step_size=initial_step_sizes,
            state_gradients_are_stopped=True,
            seed=_set_seed(252)),
        parallel_iterations=1)

    init_op = tf1.global_variables_initializer()
    self.evaluate(init_op)

    _ = self.evaluate(samples)

  def test_multiple_step_sizes_different_dtype(self):
    num_results = 5
    initial_step_sizes = [1e-5, 1e-4]
    initial_state = [0., 0.]
    # Non-float32 dtype.
    dtype = np.float64

    step_size = [
        tf.constant(initial_step_size, dtype=dtype, name='step_size')
        for initial_step_size in initial_step_sizes]

    def target_log_prob_fn(x1, x2):
      return tf.reduce_sum(
          tfd.Normal(dtype(0), dtype(1)).log_prob([x1, x2]),
          axis=-1)

    _, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=0,
        current_state=[dtype(x) for x in initial_state],
        kernel=tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                num_leapfrog_steps=2,
                step_size=step_size,
                state_gradients_are_stopped=True,
                seed=_set_seed(252)),
            num_adaptation_steps=2),
        parallel_iterations=1)

    step_size_ = self.evaluate(kernel_results.new_step_size)

    # We apply the same adjustment to each step size in the list, so
    # the starting ratio of step sizes should match the final ratio.
    self.assertNear(step_size_[0][0]/step_size_[1][0],
                    step_size_[0][-1]/step_size_[1][-1], err=1e-4)

  def test_finite_adaptation(self):

    # Test that the adaptation runs for the specified number of steps.
    # We set up a chain with a tiny initial step size, so every step accepts,
    # and test that the final step size is incremented exactly
    # `num_adaptation_steps` times.
    num_results = 10
    num_adaptation_steps = 3
    initial_step_size = 1e-5

    _, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=0,
        current_state=tf.constant(0.),
        kernel=tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=lambda x: tfd.Normal(0., 1.).log_prob(x),
                num_leapfrog_steps=2,
                step_size=initial_step_size,
                state_gradients_are_stopped=True,
                seed=_set_seed(252)),
            num_adaptation_steps=num_adaptation_steps,
            adaptation_rate=1.),
        parallel_iterations=1)

    init_op = tf1.global_variables_initializer()
    self.evaluate(init_op)

    [_, step_size_] = self.evaluate([
        kernel_results, kernel_results.new_step_size])

    # Test that we've incremented the step size every time. This verifies
    # that adaptation ran on each of the first `num_adaptation_steps` steps.
    self.assertNear(initial_step_size * 2**num_adaptation_steps,
                    step_size_[num_adaptation_steps], err=1e-6)

    # Test that the step size does not change after the first
    # `num_adaptation_steps` steps.
    self.assertEqual(step_size_[num_adaptation_steps:].min(),
                     step_size_[num_adaptation_steps:].max())


@test_util.test_all_tf_execution_regimes
class HMCEMAdaptiveStepSize(test_util.TestCase):
  """This test verifies that the docstring example works as advertised."""

  def setUp(self):
    super(HMCEMAdaptiveStepSize, self).setUp()
    tf.random.set_seed(10014)
    np.random.seed(10014)

  def make_training_data(self, num_samples, dims, sigma):
    dt = np.asarray(sigma).dtype
    x = np.random.randn(dims, num_samples).astype(dt)
    w = sigma * np.random.randn(1, dims).astype(dt)
    noise = np.random.randn(num_samples).astype(dt)
    y = w.dot(x) + noise
    return y[0], x, w[0]

  def make_weights_prior(self, dims, sigma):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([dims], dtype=sigma.dtype),
        scale_identity_multiplier=sigma)

  def make_response_likelihood(self, w, x):
    if tensorshape_util.rank(w.shape) == 1:
      y_bar = tf.matmul(w[tf.newaxis], x)[0]
    else:
      y_bar = tf.matmul(w, x)
    return tfd.Normal(loc=y_bar, scale=tf.ones_like(y_bar))  # [n]

  def test_mcem_converges(self):
    # Setup assumptions.
    dtype = np.float32
    num_samples = 500
    dims = 10

    weights_prior_true_scale = np.array(0.3, dtype)
    y, x, w0 = self.make_training_data(num_samples, dims,
                                       weights_prior_true_scale)
    tf1.logging.vlog(1, 'w0: %s', w0)

    sigma = tfp.util.TransformedVariable(
        name='sigma', initial_value=np.array(1, dtype), bijector=tfb.Exp())

    optimizer = tf.optimizers.SGD(learning_rate=0.01)

    # TODO(b/144045420): eliminate the need for this tf.function decorator. The
    # reason it was added was that the test code is written to work in both
    # eager and graph modes, and in order to do so, calls this funtion
    # repeatedly in an optimization loop. In graph mode, that results in the
    # graph growing during optimization which results in runtime quadratic in
    # number of optimization steps. Decorating with tf.function means the graph
    # doesn't grow, but is hacky. Instead, we should ensure the code is written
    # performantly in eager and graph modes, potentially by forking the
    # implementation based on tf.executing_eagerly().
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[dims], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
    ])
    def mcem_iter(weights_chain_start, step_size):
      prior = self.make_weights_prior(dims, sigma)

      def unnormalized_posterior_log_prob(w):
        likelihood = self.make_response_likelihood(w, x)
        return (prior.log_prob(w) +
                tf.reduce_sum(likelihood.log_prob(y), axis=-1))  # [m]

      def trace_fn(_, pkr):
        return (pkr.inner_results.log_accept_ratio,
                pkr.inner_results.accepted_results.step_size)

      num_results = 2
      weights, (log_accept_ratio, step_size) = tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=0,
          current_state=weights_chain_start,
          kernel=tfp.mcmc.SimpleStepSizeAdaptation(
              tfp.mcmc.HamiltonianMonteCarlo(
                  target_log_prob_fn=unnormalized_posterior_log_prob,
                  num_leapfrog_steps=2,
                  step_size=step_size,
                  state_gradients_are_stopped=True,
              ),
              # Adapt for the entirety of the trajectory.
              num_adaptation_steps=2),
          trace_fn=trace_fn,
          parallel_iterations=1)

      # We do an optimization step to propagate `sigma` after two HMC
      # steps to propagate `weights`.
      with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(unnormalized_posterior_log_prob(weights))

      avg_acceptance_ratio = tf.math.exp(tfp.math.reduce_logmeanexp(
          tf.minimum(log_accept_ratio, 0.)))

      optimizer.apply_gradients([[
          tape.gradient(loss, sigma.pretransformed_input),
          sigma.pretransformed_input
      ]])

      weights_prior_estimated_scale = tf.identity(sigma)
      return (weights_prior_estimated_scale, weights[-1], loss,
              step_size[-1], avg_acceptance_ratio)

    if not tf.executing_eagerly():
      # To create the variables.
      mcem_iter(np.zeros(dims, dtype), 0.)
    self.evaluate(tf1.global_variables_initializer())

    num_iters = int(40)

    weights_prior_estimated_scale_ = np.zeros(num_iters, dtype)
    weights_ = np.zeros([num_iters + 1, dims], dtype)
    loss_ = np.zeros([num_iters], dtype)
    weights_[0] = np.random.randn(dims).astype(dtype)
    step_size_ = 0.03

    for iter_ in range(num_iters):
      [
          weights_prior_estimated_scale_[iter_],
          weights_[iter_ + 1],
          loss_[iter_],
          step_size_,
          avg_acceptance_ratio_,
      ] = self.evaluate(mcem_iter(weights_[iter_], step_size_))
      # Enable using bazel flags:
      # `--test_arg="--logtostderr" --test_arg="--vmodule=hmc_test=2"`,
      # E.g.,
      # bazel test --test_output=streamed -c opt :hmc_test \
      # --test_filter=HMCEMAdaptiveStepSize \
      # --test_arg="--logtostderr" --test_arg="--vmodule=hmc_test=2"
      tf1.logging.vlog(
          1, ('iter:{:>2}  loss:{: 9.3f}  scale:{:.3f}  '
              'step_size:{:.4f}  avg_acceptance_ratio:{:.4f}').format(
                  iter_, loss_[iter_], weights_prior_estimated_scale_[iter_],
                  step_size_, avg_acceptance_ratio_))

    # Loss had better decrease....
    self.assertGreater(loss_[:10].mean(), loss_[-10:].mean())
    self.assertNear(0.22,  # Actually smaller than weights_prior_true_scale,
                    weights_prior_estimated_scale_[-5:].mean(),
                    err=0.02)

  def test_step_size_adapts(self):
    dtype = np.float32

    def unnormalized_log_prob(x):
      return -x - x**2

    def trace_fn(_, pkr):
      return [pkr.inner_results.is_accepted,
              pkr.inner_results.accepted_results.step_size]

    num_results = 1000
    num_burnin_steps = 100
    _, [is_accepted, step_size] = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=tf.zeros([], dtype),
        kernel=tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_prob,
                num_leapfrog_steps=2,
                step_size=0.05,
                seed=_set_seed(252)),
            num_adaptation_steps=num_results + num_burnin_steps),
        trace_fn=trace_fn,
        parallel_iterations=1)

    is_accepted_, step_size_ = self.evaluate([is_accepted, step_size])

    # The important thing is that the new step_size does not equal the original,
    # 0.05. However, we're not using `self.assertNotEqual` because testing for
    # `1.25` reveals just how much the step_size has changed.
    self.assertNear(1.25, step_size_[-100:].mean(), err=0.07)
    self.assertNear(0., step_size_[-100:].std(), err=0.04)
    # Anything in [0.6, 0.9] is sufficient. https://arxiv.org/abs/1411.6669
    self.assertNear(0.75, is_accepted_.mean(), err=0.07)


if __name__ == '__main__':
  tf.test.main()
