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

import itertools

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.experimental.auto_batching import instructions as inst
from tensorflow_probability.python.internal import test_util


def run_nuts_chain(
    event_size, batch_size, num_steps,
    initial_state=None, dry_run=False, stackless=False, seed=None):
  if seed is None:
    seed = 1
  def target_log_prob_fn(event):
    with tf1.name_scope('nuts_test_target_log_prob', values=[event]):
      return tfd.MultivariateNormalDiag(
          tf.zeros(event_size),
          scale_identity_multiplier=1.).log_prob(event)

  if initial_state is not None:
    state = initial_state
  else:
    state = tf.zeros([batch_size, event_size])
  # kernel = tfp.mcmc.HamiltonianMonteCarlo(
  #     target_log_prob_fn,
  #     num_leapfrog_steps=3,
  #     step_size=0.3,
  #     seed=seed)
  kernel = tfp.experimental.mcmc.NoUTurnSampler(
      target_log_prob_fn,
      step_size=[0.3],
      use_auto_batching=not dry_run,
      stackless=stackless,
      unrolled_leapfrog_steps=2,
      max_tree_depth=4,
      seed=seed)
  chain_state, extra = tfp.mcmc.sample_chain(
      num_results=num_steps,
      num_burnin_steps=0,
      # Intentionally pass a list argument to test that singleton lists are
      # handled reasonably (c.f. assert_univariate_target_conservation, which
      # uses an unwrapped singleton).
      current_state=[state],
      kernel=kernel,
      parallel_iterations=1)
  return chain_state, extra.leapfrogs_taken


def assert_univariate_target_conservation(
    test, mk_target, step_size, stackless):
  # Sample count limited partly by memory reliably available on Forge.  The test
  # remains reasonable even if the nuts recursion limit is severely curtailed
  # (e.g., 3 or 4 levels), so use that to recover some memory footprint and bump
  # the sample count.
  num_samples = int(5e4)
  num_steps = 1
  target_d = mk_target()
  strm = tfp.util.SeedStream(salt='univariate_nuts_test', seed=1)
  initialization = target_d.sample([num_samples], seed=strm())
  def target(*args):
    # TODO(axch): Just use target_d.log_prob directly, and accept target_d
    # itself as an argument instead of a maker function.  Blocked by
    # b/128932888.  It would then also be nice not to eta-expand
    # target_d.log_prob; that was blocked by b/122414321, but maybe tfp's port
    # of value_and_gradients_function fixed that bug.
    return mk_target().log_prob(*args)
  operator = tfp.experimental.mcmc.NoUTurnSampler(
      target,
      step_size=step_size,
      max_tree_depth=3,
      use_auto_batching=True,
      stackless=stackless,
      unrolled_leapfrog_steps=2,
      seed=strm())
  result, extra = tfp.mcmc.sample_chain(
      num_results=num_steps,
      num_burnin_steps=0,
      current_state=initialization,
      kernel=operator)
  # Note: sample_chain puts the chain history on top, not the (independent)
  # chains.
  test.assertAllEqual([num_steps, num_samples], result.shape)
  answer = result[0]
  check_cdf_agrees = st.assert_true_cdf_equal_by_dkwm(
      answer, target_d.cdf, false_fail_rate=1e-6)
  check_enough_power = tf1.assert_less(
      st.min_discrepancy_of_true_cdfs_detectable_by_dkwm(
          num_samples, false_fail_rate=1e-6, false_pass_rate=1e-6), 0.025)
  test.assertAllEqual([num_samples], extra.leapfrogs_taken[0].shape)
  unique, _ = tf.unique(extra.leapfrogs_taken[0])
  check_leapfrogs_vary = tf1.assert_greater_equal(
      tf.shape(input=unique)[0], 3)
  avg_leapfrogs = tf.math.reduce_mean(input_tensor=extra.leapfrogs_taken[0])
  check_leapfrogs = tf1.assert_greater_equal(
      avg_leapfrogs, tf.constant(4, dtype=avg_leapfrogs.dtype))
  movement = tf.abs(answer - initialization)
  test.assertAllEqual([num_samples], movement.shape)
  # This movement distance (1 * step_size) was selected by reducing until 100
  # runs with independent seeds all passed.
  check_movement = tf1.assert_greater_equal(
      tf.reduce_mean(input_tensor=movement), 1 * step_size)
  return (check_cdf_agrees, check_enough_power, check_leapfrogs_vary,
          check_leapfrogs, check_movement)


def assert_mvn_target_conservation(event_size, batch_size, **kwargs):
  initialization = tfd.MultivariateNormalFullCovariance(
      loc=tf.zeros(event_size),
      covariance_matrix=tf.eye(event_size)).sample(
          batch_size, seed=4)
  samples, leapfrogs = run_nuts_chain(
      event_size, batch_size, num_steps=1,
      initial_state=initialization, **kwargs)
  answer = samples[0][-1]
  check_cdf_agrees = (
      st.assert_multivariate_true_cdf_equal_on_projections_two_sample(
          answer, initialization, num_projections=100, false_fail_rate=1e-6))
  check_sample_shape = tf1.assert_equal(
      tf.shape(input=answer)[0], batch_size)
  unique, _ = tf.unique(leapfrogs[0])
  check_leapfrogs_vary = tf1.assert_greater_equal(
      tf.shape(input=unique)[0], 3)
  avg_leapfrogs = tf.math.reduce_mean(input_tensor=leapfrogs[0])
  check_leapfrogs = tf1.assert_greater_equal(
      avg_leapfrogs, tf.constant(4, dtype=avg_leapfrogs.dtype))
  movement = tf.linalg.norm(tensor=answer - initialization, axis=-1)
  # This movement distance (0.3) was copied from the univariate case.
  check_movement = tf1.assert_greater_equal(
      tf.reduce_mean(input_tensor=movement), 0.3)
  check_enough_power = tf1.assert_less(
      st.min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample(
          batch_size, batch_size, false_fail_rate=1e-8, false_pass_rate=1e-6),
      0.055)
  return (check_cdf_agrees, check_sample_shape, check_leapfrogs_vary,
          check_leapfrogs, check_movement, check_enough_power)


@test_util.test_all_tf_execution_regimes
class NutsTest(test_util.TestCase):

  @parameterized.parameters(itertools.product([2, 3], [1, 2, 3]))
  def testLeapfrogStepCounter(self, tree_depth, unrolled_leapfrog_steps):
    def never_u_turns_log_prob(x):
      return 1e-6 * x
    kernel = tfp.experimental.mcmc.NoUTurnSampler(
        never_u_turns_log_prob,
        step_size=[0.3],
        use_auto_batching=True,
        stackless=False,
        unrolled_leapfrog_steps=unrolled_leapfrog_steps,
        max_tree_depth=tree_depth,
        seed=1)
    _, extra = tfp.mcmc.sample_chain(
        num_results=1,
        num_burnin_steps=0,
        current_state=[tf.ones([1])],
        kernel=kernel,
        parallel_iterations=1)
    self.assertEqual([(2**tree_depth - 1) * unrolled_leapfrog_steps],
                     self.evaluate(extra.leapfrogs_taken))

  def testReproducibility(self):
    seed = test_util.test_seed()
    s1 = self.evaluate(run_nuts_chain(2, 5, 10, seed=seed)[0])
    if tf.executing_eagerly():
      tf.random.set_seed(seed)
    s2 = self.evaluate(run_nuts_chain(2, 5, 10, seed=seed)[0])
    self.assertAllEqual(s1, s2)

  def testUnivariateNormalTargetConservation(self):
    def mk_normal():
      return tfd.Normal(loc=1., scale=2.)
    self.evaluate(assert_univariate_target_conservation(
        self, mk_normal, step_size=0.2, stackless=False))

  def testLogitBetaTargetConservation(self):
    def mk_logit_beta():
      beta = tfd.Beta(concentration0=1., concentration1=2.)
      return tfb.Invert(tfb.Sigmoid())(beta)
    self.evaluate(assert_univariate_target_conservation(
        self, mk_logit_beta, step_size=0.2, stackless=False))

  def testSigmoidBetaTargetConservation(self):
    def mk_sigmoid_beta():
      beta = tfd.Beta(concentration0=1., concentration1=2.)
      # Not inverting the sigmoid bijector makes a kooky distribution, but nuts
      # should still conserve it (with a smaller step size).
      return tfb.Sigmoid()(beta)
    self.evaluate(assert_univariate_target_conservation(
        self, mk_sigmoid_beta, step_size=1e-4, stackless=False))

  @parameterized.parameters(
      (3, 50000,),
      # (5, 2,),
  )
  def testMultivariateNormalNd(self, event_size, batch_size):
    self.evaluate(assert_mvn_target_conservation(event_size, batch_size))

  def testLatentsOfMixedRank(self):
    batch_size = 4
    init = [tf.ones([batch_size, 3, 2]), tf.zeros([batch_size, 7])]
    def batched_synthetic_log_prob(x, y):
      pt1 = tf.reduce_sum(input_tensor=x, axis=[-1, -2])
      pt2 = tf.reduce_sum(input_tensor=y, axis=-1)
      return pt1 + pt2
    kernel = tfp.experimental.mcmc.NoUTurnSampler(
        batched_synthetic_log_prob,
        step_size=0.3,
        use_auto_batching=True,
        stackless=False,
        unrolled_leapfrog_steps=2,
        max_tree_depth=4,
        seed=1)
    results, extra = tfp.mcmc.sample_chain(
        num_results=1,
        num_burnin_steps=0,
        current_state=init,
        kernel=kernel,
        parallel_iterations=1)
    self.evaluate([results, extra])

  def testDryRunMode(self):
    if not tf.executing_eagerly(): return
    _, ev_leapfrogs = self.evaluate(run_nuts_chain(
        event_size=3, batch_size=1, num_steps=1, dry_run=True))
    ev_leapfrogs = ev_leapfrogs[0]
    self.assertTrue(all(ev_leapfrogs > 1))

  def testStacklessMode(self):
    if not tf.executing_eagerly(): return
    self.evaluate(assert_mvn_target_conservation(
        3, 50000, dry_run=False, stackless=True))

  def testStacklessUnivariateNormalTargetConservation(self):
    if not tf.executing_eagerly(): return
    def mk_normal():
      return tfd.Normal(loc=1., scale=2.)
    self.evaluate(assert_univariate_target_conservation(
        self, mk_normal, step_size=0.2, stackless=True))

  def testStacklessLogitBetaTargetConservation(self):
    if not tf.executing_eagerly(): return
    def mk_logit_beta():
      beta = tfd.Beta(concentration0=1., concentration1=2.)
      return tfb.Invert(tfb.Sigmoid())(beta)
    self.evaluate(assert_univariate_target_conservation(
        self, mk_logit_beta, step_size=0.02, stackless=True))

  def testStacklessSigmoidBetaTargetConservation(self):
    if not tf.executing_eagerly(): return
    def mk_sigmoid_beta():
      beta = tfd.Beta(concentration0=1., concentration1=2.)
      # Not inverting the sigmoid bijector makes a kooky distribution, but
      # nuts should still conserve it (with a smaller step size).
      return tfb.Sigmoid()(beta)
    self.evaluate(assert_univariate_target_conservation(
        self, mk_sigmoid_beta, step_size=1e-4, stackless=True))

  def _correlated_mvn_nuts(self, dim, step_size, num_steps):
    # The correlated MVN example is taken from the NUTS paper
    # https://arxiv.org/pdf/1111.4246.pdf.
    # This implementation in terms of MVNCholPrecisionTril follows
    # tfp/examples/jupyter_notebooks/Bayesian_Gaussian_Mixture_Model.ipynb

    class MVNCholPrecisionTriL(tfd.TransformedDistribution):
      """MVN from loc and (Cholesky) precision matrix."""

      def __init__(self, loc, chol_precision_tril, name=None):
        super(MVNCholPrecisionTriL, self).__init__(
            distribution=tfd.Independent(tfd.Normal(tf.zeros_like(loc),
                                                    scale=tf.ones_like(loc)),
                                         reinterpreted_batch_ndims=1),
            bijector=tfb.Chain([
                tfb.Affine(shift=loc),
                tfb.Invert(tfb.Affine(scale_tril=chol_precision_tril,
                                      adjoint=True)),
            ]),
            name=name)

    strm = test_util.test_seed_stream()
    wishart = tfd.WishartTriL(
        dim, scale_tril=tf.eye(dim), input_output_cholesky=True)
    chol_precision = wishart.sample(seed=strm())
    mvn = MVNCholPrecisionTriL(
        loc=tf.zeros(dim), chol_precision_tril=chol_precision)
    kernel = tfp.experimental.mcmc.NoUTurnSampler(
        mvn.log_prob,
        step_size=[step_size],
        num_trajectories_per_step=num_steps,
        use_auto_batching=True,
        stackless=False,
        max_tree_depth=7,
        seed=strm())
    return kernel

  def testCorrelatedMVNOneStep(self):
    # Assert that we get a diversity of leapfrogs taken after one step
    kernel = self._correlated_mvn_nuts(dim=10, step_size=0.1, num_steps=1)
    _, extra_ = tfp.mcmc.sample_chain(
        num_results=1,
        num_burnin_steps=0,
        current_state=[tf.zeros([30, 10])],
        kernel=kernel,
        parallel_iterations=1)
    extra = self.evaluate(extra_)
    self.assertLess(extra.leapfrogs_taken.min(), 25)
    self.assertGreater(extra.leapfrogs_taken.max(), 40)
    # Also sanity-check that leapfrogs_computed is computed consistently
    for t, c in zip(extra.leapfrogs_taken[-1], extra.leapfrogs_computed[-1]):
      self.assertLessEqual(t, c)

  def testCorrelatedMVNChain(self):
    # Assert that naive sample_chain gets bad batch utilization.
    kernel = self._correlated_mvn_nuts(dim=10, step_size=0.4, num_steps=1)
    _, extra_ = tfp.mcmc.sample_chain(
        num_results=1,
        num_burnin_steps=10,
        current_state=[tf.zeros([20, 10])],
        kernel=kernel,
        parallel_iterations=1)
    extra = self.evaluate(extra_)
    utilization = extra.leapfrogs_taken[-1] / extra.leapfrogs_computed[-1]
    # Average utilization is bad
    self.assertAllLess(np.mean(utilization), 0.65)
    # Even best-member utilization isn't 100%
    self.assertAllLess(utilization, 0.9)

  def testCorrelatedMVNManySteps(self):
    # Assert that thinning inside the autobatched nuts gives better optimal
    # utilization, in the sense that the number of leapfrogs computed is forced
    # by the length of some batch member's computation.
    kernel = self._correlated_mvn_nuts(dim=10, step_size=0.4, num_steps=10)
    _, extra_ = tfp.mcmc.sample_chain(
        num_results=1,
        num_burnin_steps=0,
        current_state=[tf.zeros([20, 10])],
        kernel=kernel,
        parallel_iterations=1)
    extra = self.evaluate(extra_)
    for c in extra.leapfrogs_computed[-1]:
      self.assertEqual(c, extra.leapfrogs_taken.max())

  def testProgramProperties(self):
    def target(x):
      return x
    operator = tfp.experimental.mcmc.NoUTurnSampler(
        target,
        step_size=0,
        use_auto_batching=True)
    program = operator.autobatch_context.program_lowered('evolve_trajectory')
    def full_var(var):
      return program.var_alloc[var] == inst.VariableAllocation.FULL

    # Check that the number of FULL variables doesn't accidentally grow.  This
    # is an equality rather than comparison to remind the maintainer to reduce
    # the expected number when they implement any pass that manages to reduce
    # the count.
    num_full_vars = 0
    for var in program.var_alloc:
      if full_var(var):
        num_full_vars += 1
    self.assertEqual(10, num_full_vars)

    # Check that the number of stack pushes doesn't accidentally grow.
    num_full_stack_pushes = 0
    for i in range(program.graph.exit_index()):
      block = program.graph.block(i)
      for op in block.instructions:
        if hasattr(op, 'vars_out'):
          for var in inst.pattern_traverse(op.vars_out):
            if full_var(var):
              if (not hasattr(op, 'skip_push_mask')
                  or var not in op.skip_push_mask):
                num_full_stack_pushes += 1
      if isinstance(block.terminator, inst.PushGotoOp):
        if full_var(inst.pc_var):
          num_full_stack_pushes += 1
    self.assertEqual(20, num_full_stack_pushes)


if __name__ == '__main__':
  tf.test.main()
