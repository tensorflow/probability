# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for DiagonalMassMatrixAdaptation kernel."""

import collections

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.experimental.mcmc import preconditioning_utils as pu
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

JAX_MODE = False


RunHMCResults = collections.namedtuple('RunHMCResults', [
    'draws',
    'final_mean',
    'final_precision_factor',
    'final_precision',
    'empirical_mean',
    'empirical_variance',
    'true_mean',
    'true_variance'])


@test_util.test_all_tf_execution_regimes
class DiagonalMassMatrixAdaptationShapesTest(test_util.TestCase):

  @parameterized.named_parameters([
      {'testcase_name': '_two_batches_of_three',
       'state_part_shape': (2, 3),
       'variance_part_shape': (2, 3),
       'log_prob_shape': (2,)},
      {'testcase_name': '_no_batches_of_two',
       'state_part_shape': (2,),
       'variance_part_shape': (2,),
       'log_prob_shape': ()},
      {'testcase_name': '_batch_of_matrix_batches',
       'state_part_shape': (2, 3, 4, 5, 6),
       'variance_part_shape': (4, 5, 6),
       'log_prob_shape': (2, 3, 4)},
      {'testcase_name': '_batch_of_scalars',
       'state_part_shape': (2, 3),
       'variance_part_shape': (3,),
       'log_prob_shape': (2, 3)},
  ])
  def testBatches(self, state_part_shape, variance_part_shape, log_prob_shape):
    dist = tfd.Independent(
        tfd.Normal(tf.zeros(state_part_shape), tf.ones(state_part_shape)),
        reinterpreted_batch_ndims=len(state_part_shape) - len(log_prob_shape))
    state_part = tf.zeros(state_part_shape)

    running_variance = tfp.experimental.stats.RunningVariance.from_stats(
        num_samples=10.,
        mean=tf.zeros(variance_part_shape),
        variance=tf.ones(variance_part_shape))

    kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=dist.log_prob,
        num_leapfrog_steps=2,
        step_size=1.)
    kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
        inner_kernel=kernel,
        initial_running_variance=running_variance)

    num_results = 5
    draws = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=state_part,
        kernel=kernel,
        seed=test_util.test_seed(),
        trace_fn=None)

    # Make sure the result has the correct shape
    self.assertEqual(draws.shape, (num_results,) + state_part_shape)

  def testBatchBroadcast(self):
    n_batches = 8
    dist = tfd.MultivariateNormalDiag(tf.zeros(3), tf.ones(3))
    target_log_prob_fn = dist.log_prob
    kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=2,
        step_size=1.)
    initial_running_variance = (
        tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=1.,
            mean=tf.zeros(3),
            variance=tf.ones(3)))
    kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
        inner_kernel=kernel,
        initial_running_variance=initial_running_variance)

    num_results = 5
    draws = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=tf.zeros([n_batches, 3]),
        kernel=kernel,
        seed=test_util.test_seed(),
        trace_fn=None)

    # Make sure the result has the correct shape
    self.assertEqual(draws.shape, (num_results, n_batches, 3))

  def testMultipleStateParts(self):
    dist = tfd.JointDistributionSequential([
        tfd.MultivariateNormalDiag(tf.zeros(3), tf.ones(3)),
        tfd.MultivariateNormalDiag(tf.zeros(2), tf.ones(2))])
    target_log_prob_fn = dist.log_prob
    kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=2,
        step_size=1.)
    initial_running_variance = [
        tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=1., mean=tf.zeros(3), variance=tf.ones(3)),
        tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=1., mean=tf.zeros(2), variance=tf.ones(2))]
    kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
        inner_kernel=kernel,
        initial_running_variance=initial_running_variance)

    num_results = 5
    draws = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=[tf.zeros(3), tf.zeros(2)],
        kernel=kernel,
        seed=test_util.test_seed(),
        trace_fn=None)

    # Make sure the result has the correct shape
    self.assertEqual(len(draws), 2)
    self.assertEqual(draws[0].shape, (num_results, 3))
    self.assertEqual(draws[1].shape, (num_results, 2))


@test_util.test_graph_and_eager_modes
class DiagonalMassMatrixAdaptationTest(test_util.TestCase):

  def setUp(self):
    self.mvn_mean = [0., 0., 0.]
    self.mvn_scale = [0.1, 1., 10.]
    super(DiagonalMassMatrixAdaptationTest, self).setUp()

  def testTurnOnStoreParametersInKernelResults(self):
    mvn = tfd.MultivariateNormalDiag(self.mvn_mean, scale_diag=self.mvn_scale)
    target_log_prob_fn = mvn.log_prob
    kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=2,
        step_size=1.)
    self.assertFalse(kernel.parameters['store_parameters_in_results'])
    initial_running_variance = (
        tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=1., mean=tf.zeros(3), variance=tf.ones(3)))
    kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
        inner_kernel=kernel,
        initial_running_variance=initial_running_variance)
    self.assertTrue(
        kernel.inner_kernel.parameters['store_parameters_in_results'])

  def _run_hmc(self, num_results, initial_running_variance):
    mvn = tfd.MultivariateNormalDiag(self.mvn_mean, scale_diag=self.mvn_scale)
    target_log_prob_fn = mvn.log_prob
    kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=32,
        step_size=0.001)
    kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
        inner_kernel=kernel,
        initial_running_variance=initial_running_variance)

    @tf.function
    def do_sample(seed):

      def trace_fn(_, pkr):
        return (pkr.running_variance,
                pkr.inner_results.accepted_results.momentum_distribution)

      sample_seed, chain_seed = tfp.random.split_seed(seed, 2)
      draws, (rv_state, dist) = tfp.mcmc.sample_chain(
          num_results=num_results,
          current_state=tf.zeros(3),
          kernel=kernel,
          seed=chain_seed,
          trace_fn=trace_fn)

      # sample_distributions returns `[dists], [samples]`, so the 0th
      # distribution corresponds to the 0th, and only, state part
      # The distribution is a BatchBroadcast containing a transformed
      # distribution, so we need to use .distribution.distribution
      batched_transformed_dist = dist.sample_distributions(
          seed=sample_seed)[0][0]
      momentum_dist = batched_transformed_dist.distribution.distribution
      final_precision_factor = momentum_dist.precision_factor.diag[-1]
      # Evaluate here so we can check the value twice later
      final_precision = momentum_dist.precision.diag[-1]
      final_mean = rv_state[0].mean[-1]
      empirical_mean = tf.reduce_mean(draws, axis=0)
      # The final_precision is taken directly from the momentum distribution,
      # which never "sees" the last sample.
      empirical_variance = tf.math.reduce_variance(draws[:-1], axis=0)
      return RunHMCResults(draws=draws,
                           final_mean=final_mean,
                           final_precision_factor=final_precision_factor,
                           final_precision=final_precision,
                           empirical_mean=empirical_mean,
                           empirical_variance=empirical_variance,
                           true_mean=mvn.mean(),
                           true_variance=mvn.variance())
    return self.evaluate(do_sample(test_util.test_seed()))

  def testUpdatesCorrectly(self):
    running_variance = tfp.experimental.stats.RunningVariance.from_shape((3,))
    # This is more straightforward than doing the math, but need at least
    # two observations to get a start.
    pseudo_observations = [-tf.ones(3), tf.ones(3)]
    for pseudo_observation in pseudo_observations:
      running_variance = running_variance.update(pseudo_observation)

    results = self._run_hmc(
        num_results=5,
        initial_running_variance=running_variance)
    draws = tf.concat([tf.stack(pseudo_observations), results.draws], axis=0)
    self.assertAllClose(results.final_precision_factor**2,
                        results.final_precision)
    self.assertAllClose(results.final_mean, tf.reduce_mean(draws, axis=0))
    self.assertAllClose(results.final_precision,
                        tf.math.reduce_variance(draws, axis=0))

  def testDoesRegularize(self):
    # Make sure that using regularization makes the final estimate closer to
    # the initial state than the empirical result.
    init_mean = tf.zeros(3)
    init_variance = tf.ones(3)
    initial_running_variance = (
        tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=10., mean=init_mean, variance=init_variance))
    results = self._run_hmc(
        num_results=50,
        initial_running_variance=initial_running_variance)

    # the `final_mean` should be a weighted average
    self.assertAllClose(
        results.final_mean,
        10. / 60. * init_mean + 50. / 60. * results.empirical_mean)

    # the `final_precision` is not quite a weighted average, since the
    # estimate of the mean also gets updated, but it is close-ish
    self.assertAllClose(
        results.final_precision,
        10. / 60. * init_variance + 50. / 60. * results.empirical_variance,
        rtol=0.2)

  def testVarGoesInRightDirection(self):
    # This online updating scheme violates detailed balance, and in general
    # will not leave the target distribution invariant. We test a weaker
    # property, which is that the variance gets closer to the target variance,
    # assuming we start at the correct mean. This test does not pass reliably
    # when the mean is not near the true mean.
    error_factor = 5.
    init_variance = error_factor * tf.convert_to_tensor(self.mvn_scale)**2
    init_mean = tf.convert_to_tensor(self.mvn_mean)
    initial_running_variance = (
        tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=10., mean=init_mean, variance=init_variance))
    results = self._run_hmc(
        num_results=1000,
        initial_running_variance=initial_running_variance)

    # This number started off at `error_factor`, and should be smaller now
    # This makes sure it is 90% closer to equal. The intention is that the
    # precision of the momentum should eventually equal the variance of the
    # state. We test elsewhere that the precision of the momentum faithfully
    # updates according to the draws it makes. This makes sure that those draws
    # are also getting closer to the underlying variance.
    new_error_factor = 1. + 0.1 * (error_factor - 1.)

    final_var_ratio = results.final_precision / results.true_variance
    self.assertAllLess(final_var_ratio, new_error_factor)

  def testMeanGoesInRightDirection(self):
    # As with `testVarGoesInRightDirection`, this makes sure the mean gets
    # closer. Pleasantly, we do not even need that the variance starts very
    # close to the true variance.
    mvn_scale = tf.convert_to_tensor(self.mvn_scale)
    error_factor = 5. * mvn_scale
    init_variance = error_factor * mvn_scale**2
    init_mean = tf.convert_to_tensor(self.mvn_mean) + error_factor
    initial_running_variance = (
        tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=10., mean=init_mean, variance=init_variance))
    results = self._run_hmc(
        num_results=1000,
        initial_running_variance=initial_running_variance)

    # This number started at `error_factor`. Make sure the mean is now at least
    # 75% closer.
    final_mean_diff = tf.abs(results.final_mean - results.true_mean)
    np.testing.assert_array_less(
        self.evaluate(final_mean_diff),
        self.evaluate(0.25 * error_factor))

  def testDoesNotGoesInWrongDirection(self):
    # As above, we test a weaker property, which is that the variance and
    # mean estimates do not get too away if initialized at the true variance
    # and mean.
    initial_running_variance = (
        tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=10., mean=self.mvn_mean,
            variance=tf.convert_to_tensor(self.mvn_scale)**2))
    results = self._run_hmc(
        num_results=1000,
        initial_running_variance=initial_running_variance)

    # Allow the large scale dimension to be a little further off
    fudge_factor = tf.sqrt(results.true_variance)
    final_mean_diff = tf.abs(results.final_mean - results.true_mean)
    np.testing.assert_array_less(self.evaluate(final_mean_diff),
                                 self.evaluate(fudge_factor))

    final_std_diff = tf.abs(results.final_precision_factor -
                            tf.sqrt(results.true_variance))
    np.testing.assert_array_less(self.evaluate(final_std_diff),
                                 self.evaluate(fudge_factor))

  def test_momentum_dists(self):
    state_parts = [
        tf.ones([13, 5, 3]), tf.ones([13, 5]), tf.ones([13, 5, 2, 4])]
    batch_shape = [13, 5]
    md = pu.make_momentum_distribution(state_parts, batch_shape)
    md = pu.update_momentum_distribution(
        md,
        tf.nest.map_structure(
            lambda s: tf.reduce_sum(s, (0, 1)), state_parts))
    self.evaluate(tf.nest.flatten(md, expand_composites=True))


@test_util.test_all_tf_execution_regimes
class DistributedDiagonalMMATest(distribute_test_lib.DistributedTest):

  def test_dmma_kernel_tracks_axis_names(self):

    def _make_kernel(**kwargs):
      running_variance = tfp.experimental.stats.RunningVariance.from_stats(
          num_samples=10.,
          mean=tf.zeros(5),
          variance=tf.ones(5))

      kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
          target_log_prob_fn=tfd.Sample(tfd.Normal(0., 1.), 5).log_prob,
          num_leapfrog_steps=2,
          step_size=1.)
      kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
          inner_kernel=kernel,
          initial_running_variance=running_variance, **kwargs)
      return kernel

    kernel = _make_kernel()
    self.assertIsNone(kernel.experimental_shard_axis_names)
    kernel = _make_kernel(experimental_shard_axis_names=['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])
    kernel = _make_kernel().experimental_with_shard_axes(['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])

  def test_momentum_distribution_has_right_shard_axis_names(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (
          tfd.Normal(0., 1.).log_prob(a)
          + distribute_lib.psum(tfd.Normal(
              distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b), 'foo'))

    running_variance = [tfp.experimental.stats.RunningVariance.from_stats(
        num_samples=10.,
        mean=tf.zeros([]),
        variance=tf.ones([]))] * 2

    kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        num_leapfrog_steps=2,
        step_size=1.)
    kernel = tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
        inner_kernel=kernel,
        initial_running_variance=running_variance)
    kernel = kernel.experimental_with_shard_axes([[], ['foo']])

    def run(seed):
      state = [tf.convert_to_tensor(-10.), tf.convert_to_tensor(-10.)]
      kr = kernel.bootstrap_results(state)
      state, _ = kernel.one_step(state, kr, seed=seed)
      inner_results = kr.inner_results.proposed_results
      axis_names = (
          inner_results.momentum_distribution.experimental_shard_axis_names)
      self.assertListEqual(
          list(axis_names),
          [[], ['foo']])
      return state

    self.evaluate(self.per_replica_to_tensor(
        self.strategy_run(run, args=(samplers.zeros_seed(),),
                          in_axes=None, axis_name='foo'), 0))


if __name__ == '__main__':
  distribute_test_lib.main()
