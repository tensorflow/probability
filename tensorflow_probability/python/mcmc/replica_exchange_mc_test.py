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
"""Tests for ReplicaExchangeMC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


def init_tfp_randomwalkmetropolis(
    target_log_prob_fn,
    seed,
    step_size,
    store_parameters_in_results=False, num_leapfrog_steps=None):  # pylint: disable=unused-argument
  return tfp.mcmc.RandomWalkMetropolis(
      target_log_prob_fn,
      new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size),
      seed=seed)


def effective_sample_size(x, **kwargs):
  """tfp.mcmc.effective_sample_size, with a maximum appropriate for HMC."""
  # Since ESS is an estimate, it can go wrong...  E.g. we can have negatively
  # correlated samples, which *do* have ESS > N, but this ESS is only applicable
  # for variance reduction power for estimation of the mean.  We want to
  # (blindly) use ESS everywhere (e.g. variance estimates)....and so...
  ess = tfp.mcmc.effective_sample_size(x, **kwargs)
  n = tf.cast(prefer_static.size0(x), x.dtype)
  return tf.minimum(ess, n)


def _set_seed():
  """Helper which uses graph seed if using TFE."""
  # TODO(b/68017812): Deprecate once TFE supports seed.
  seed_stream = test_util.test_seed_stream()
  if tf.executing_eagerly():
    tf.random.set_seed(seed_stream())
    return None
  return seed_stream()


@test_util.test_graph_and_eager_modes
class DefaultSwapProposedFnTest(test_util.TestCase):

  @parameterized.named_parameters(
      ('prob1p0_n1', 1.0, 1),
      ('prob1p0_n2', 1.0, 2),
      ('prob1p0_n4', 1.0, 4),
      ('prob1p0_n5', 1.0, 5),
      ('prob0p5_n1', 0.5, 1),
      ('prob0p5_n4', 0.5, 4),
      ('prob0p5_n7', 0.5, 7),
      ('prob0p0_n1', 0.0, 1),
      ('prob0p0_n2', 0.0, 2),
      ('prob0p0_n5', 0.0, 5),
  )
  def testProbSwapNumReplicaNoBatch(self, prob_swap, num_replica):
    fn = tfp.mcmc.default_swap_proposal_fn(prob_swap)
    num_results = 100
    swaps = tf.stack(
        [fn(num_replica, seed=i) for i in range(num_results)],
        axis=0)

    self.assertAllEqual((num_results, num_replica), swaps.shape)
    self.check_swaps_with_no_batch_shape(self.evaluate(swaps), prob_swap)

  @parameterized.named_parameters(
      ('prob1p0_n1', 1.0, 1),
      ('prob1p0_n2', 1.0, 2),
      ('prob1p0_n5', 1.0, 5),
      ('prob0p5_n1', 0.5, 1),
      ('prob0p5_n2', 0.5, 2),
      ('prob0p5_n3', 0.5, 3),
      ('prob0p0_n1', 0.0, 1),
      ('prob0p0_n2', 0.0, 2),
      ('prob0p0_n5', 0.0, 5),
  )
  def testProbSwapNumReplicaWithBatch(self, prob_swap, num_replica):
    fn = tfp.mcmc.default_swap_proposal_fn(prob_swap)
    num_results = 100
    swaps = tf.stack(
        [fn(num_replica, batch_shape=[2], seed=i) for i in range(num_results)],
        axis=0)

    self.assertAllEqual((num_results, num_replica, 2), swaps.shape)
    swaps_ = self.evaluate(swaps)

    # Batch members should have distinct swaps in most cases.
    frac_same = np.mean(swaps_[..., 0] == swaps_[..., 1])

    # If prob_swap == 0, swap is the null_swap always.
    if (prob_swap == 0 or
        # If num_replica == 1, swap = [0] always.
        num_replica == 1 or
        # In this case, we always swap and it's always [1, 0].
        (num_replica == 2 and prob_swap == 1)):
      self.assertEqual(1.0, frac_same)
    else:
      self.assertLess(frac_same, 0.9)

    # Check that each batch member has proper statistics.
    for i in range(swaps_.shape[-1]):
      self.check_swaps_with_no_batch_shape(swaps_[..., i], prob_swap)

  def check_swaps_with_no_batch_shape(self, swaps_, prob_swap):
    assert swaps_.ndim == 2, 'Expected shape [num_results, num_replica]'
    num_results, num_replica = swaps_.shape

    null_swaps = np.arange(num_replica)

    # Check that we propose at least one swap, prob_swap fraction of the
    # time.
    # An exception is made for when num_replica == 1, since in this case the
    # only swap is the null swap.
    expected_prob_swap = prob_swap * np.float32(num_replica > 1)
    observed_prob_swap = np.mean(np.any(swaps_ != null_swaps, axis=1))
    self.assertAllClose(
        expected_prob_swap,
        observed_prob_swap,
        rtol=0,
        # Accurate to 4 standard errors.
        atol=4 * np.sqrt(prob_swap * (1 - prob_swap) / num_results))

    # Verify the swap is "once only."
    for n in range(20):
      self.assertAllEqual(null_swaps, np.take(swaps_[n], swaps_[n]))


@test_util.test_graph_and_eager_modes
class REMCTest(test_util.TestCase):

  def setUp(self):
    tf.random.set_seed(123)
    super(REMCTest, self).setUp()

  @parameterized.named_parameters([
      dict(  # pylint: disable=g-complex-comprehension
          testcase_name=testcase_name + kernel_name,
          tfp_transition_kernel=tfp_transition_kernel,
          inverse_temperatures=inverse_temperatures,
          store_parameters_in_results=store_param)
      for kernel_name, tfp_transition_kernel, store_param in [
          ('HMC', tfp.mcmc.HamiltonianMonteCarlo, True),
          ('RWMH', init_tfp_randomwalkmetropolis, False),
      ]
      for testcase_name, inverse_temperatures in [
          ('OddNumReplicas', [1.0, 0.8, 0.6]),
          ('EvenNumReplicas', [1.0, 0.8, 0.7, 0.6]),
          ('HighTemperatureOnly', [0.5]),
          ('LowTemperatureOnly', [2.0]),
      ]
  ])
  def testNormal(self,
                 tfp_transition_kernel,
                 inverse_temperatures,
                 store_parameters_in_results,
                 prob_swap=1.0,
                 dtype=np.float32):
    """Sampling from standard normal with REMC."""
    num_results = 500 if tf.executing_eagerly() else 2000

    target = tfd.Normal(dtype(0.), dtype(1.))
    inverse_temperatures = dtype(inverse_temperatures)
    num_replica = len(inverse_temperatures)

    step_size = 0.51234 / np.sqrt(inverse_temperatures)
    num_leapfrog_steps = 3

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp_transition_kernel(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=step_size,
          store_parameters_in_results=store_parameters_in_results,
          num_leapfrog_steps=num_leapfrog_steps)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tf.function(target.log_prob, autograph=False),
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn,
        swap_proposal_fn=tfp.mcmc.default_swap_proposal_fn(
            prob_swap),
        seed=_set_seed())

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=target.sample(seed=_set_seed()),
        kernel=remc,
        num_burnin_steps=50,
        trace_fn=lambda _, results: results,
        parallel_iterations=1)  # For determinism.

    self.assertAllEqual((num_results,), states.shape)

    states_, kr_, replica_ess_ = self.evaluate([
        states,
        kernel_results,
        # Get the first (and only) state part for all replicas.
        effective_sample_size(kernel_results.post_swap_replica_states[0]),
    ])

    logging.vlog(
        2, '---- execution:{}  mean:{}  stddev:{}'.format(
            'eager' if tf.executing_eagerly() else 'graph',
            states_.mean(), states_.std()))

    # Some shortened names.
    replica_log_accept_ratio = (
        kr_.post_swap_replica_results.log_accept_ratio)
    replica_states_ = kr_.post_swap_replica_states[0]  # Get rid of "parts"

    # Target state is at index 0.
    self.assertAllClose(states_, replica_states_[:, 0])

    # Check that *each* replica has correct marginal.
    def _check_sample_stats(replica_idx):
      x = replica_states_[:, replica_idx]
      ess = replica_ess_[replica_idx]

      err_msg = 'replica_idx={}'.format(replica_idx)

      mean_atol = 5 * 1.0 / np.sqrt(ess)
      self.assertAllClose(x.mean(), 0.0, atol=mean_atol, msg=err_msg)

      # For a tempered Normal, Variance = T.
      expected_var = 1 / inverse_temperatures[replica_idx]
      var_atol = 5 * expected_var * np.sqrt(2) / np.sqrt(ess)
      self.assertAllClose(np.var(x), expected_var, atol=var_atol, msg=err_msg)

    for replica_idx in range(num_replica):
      _check_sample_stats(replica_idx)

    # Test log_accept_ratio and replica_log_accept_ratio.
    self.assertAllEqual((num_results, num_replica),
                        replica_log_accept_ratio.shape)
    replica_mean_accept_ratio = np.mean(
        np.exp(np.minimum(0, replica_log_accept_ratio)), axis=0)
    for accept_ratio in replica_mean_accept_ratio:
      # Every single replica should have a decent P[Accept]
      self.assertBetween(accept_ratio, 0.2, 0.99)

    # Check swap probabilities for adjacent swaps.
    self.assertAllEqual((num_results, num_replica - 1),
                        kr_.is_swap_accepted_adjacent.shape)
    conditional_swap_prob = (
        np.sum(kr_.is_swap_accepted_adjacent, axis=0) /
        np.sum(kr_.is_swap_proposed_adjacent, axis=0)
    )
    if num_replica > 1 and prob_swap > 0:
      # If temperatures are reasonable, this should be the case.
      # Ideally conditional_swap_prob is near 30%, but we're not tuning here
      self.assertGreater(np.min(conditional_swap_prob), 0.01)
      self.assertLess(np.max(conditional_swap_prob), 0.99)

    # Check swap probabilities for all swaps.
    def _check_swap_matrix(matrix):
      self.assertAllEqual((num_results, num_replica, num_replica),
                          matrix.shape)
      # Matrix is stochastic (since you either get swapd with another
      # replica, or yourself), and symmetric, since we do once-only swaps.
      self.assertAllEqual(np.ones((num_results, num_replica)),
                          matrix.sum(axis=-1))
      self.assertAllEqual(matrix, np.transpose(matrix, (0, 2, 1)))
      # By default, all swaps are between adjacent replicas.
      for i in range(num_replica):
        for j in range(i + 2, num_replica):
          self.assertEqual(0.0, np.max(np.abs(matrix[..., i, j])))
    _check_swap_matrix(kr_.is_swap_proposed)
    _check_swap_matrix(kr_.is_swap_accepted)

    # Check inverse_temperatures never change.
    self.assertAllEqual(
        np.repeat([inverse_temperatures], axis=0, repeats=num_results),
        kr_.inverse_temperatures)

    if store_parameters_in_results:
      # Check that store_parameters_in_results=True worked for HMC.
      self.assertAllEqual(
          np.repeat([step_size], axis=0, repeats=num_results),
          kr_.post_swap_replica_results.accepted_results.step_size)

      self.assertAllEqual(
          np.repeat([num_leapfrog_steps], axis=0, repeats=num_results),
          kr_.post_swap_replica_results.accepted_results.num_leapfrog_steps)

  @parameterized.named_parameters([
      ('HMC', tfp.mcmc.HamiltonianMonteCarlo),
      ('RWMH', init_tfp_randomwalkmetropolis),
  ])
  def testRWM2DMixNormal(self, tfp_transition_kernel):
    """Sampling from a 2-D Mixture Normal Distribution."""
    dtype = np.float32

    # By symmetry, target has mean [0, 0]
    # Therefore, Var = E[X^2] = E[E[X^2 | c]], where c is the component.
    # Now..., for the first component,
    #   E[X1^2] =  Var[X1] + Mean[X1]^2
    #           =  0.3^2 + 1^2,
    # and similarly for the second. As a result, Var[mixture] = 1.09.
    target = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[-1., -1], [1., 1.]],
            scale_identity_multiplier=0.3))

    inverse_temperatures = 10.**tf.linspace(start=0., stop=-1., num=4)
    # We need to pad the step_size so it broadcasts against MCMC samples. In
    # this case we have 1 replica dim, 0 batch dims, and 1 event dim hence need
    # to right pad the step_size by one dim (for the event).
    step_size = 0.2 / tf.math.sqrt(inverse_temperatures[:, tf.newaxis])
    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp_transition_kernel(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=step_size,
          num_leapfrog_steps=5)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tf.function(target.log_prob, autograph=False),
        # Verified that test fails if inverse_temperatures = [1.]
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed())

    def trace_fn(state, results):  # pylint: disable=unused-argument
      return results.post_swap_replica_results.log_accept_ratio

    num_results = 500 if tf.executing_eagerly() else 2000
    states, replica_log_accept_ratio = tfp.mcmc.sample_chain(
        num_results=num_results,
        # Start at one of the modes, in order to make mode jumping necessary
        # if we want to pass test.
        current_state=tf.ones(2, dtype=dtype),
        kernel=remc,
        num_burnin_steps=50,
        trace_fn=trace_fn,
        parallel_iterations=1)  # For determinism.
    self.assertAllEqual((num_results, 2), states.shape)
    replica_accept_ratio = tf.reduce_mean(
        tf.math.exp(tf.minimum(0., replica_log_accept_ratio)),
        axis=0)

    [
        sample_mean_,
        sample_variance_,
        replica_accept_ratio_,
        expected_mean_,
        expected_stddev_,
        expected_variance_,
        ess_,
    ] = self.evaluate([
        tf.reduce_mean(states, axis=0),
        tfp.stats.variance(states),
        replica_accept_ratio,
        target.mean(),
        target.stddev(),
        target.variance(),
        effective_sample_size(states),
    ])

    logging.vlog(
        2, '---- execution:{}  accept_ratio:{}  mean:{}'.format(
            'eager' if tf.executing_eagerly() else 'graph',
            replica_accept_ratio_, sample_mean_))

    self.assertAllClose(
        expected_mean_,
        sample_mean_,
        atol=5 * expected_stddev_ / np.sqrt(np.min(ess_)))
    self.assertAllClose(
        expected_variance_,
        sample_variance_,
        atol=5 * expected_variance_ / np.sqrt(np.min(ess_)))

  def testMultipleCorrelatedStatesWithNoBatchDims(self):
    dtype = np.float32
    num_results = 500 if tf.executing_eagerly() else 2000
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    # Use LinearOperatorLowerTriangular to get broadcasting ability.
    linop = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(true_cov))

    # Its ok to decorate this since we only need to stress the TransitionKernel.
    @tf.function(autograph=False)
    def target_log_prob(x, y):
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      xy = tf.stack([x, y], axis=-1) - true_mean
      z = linop.solvevec(xy)
      return -0.5 * tf.reduce_sum(z**2., axis=-1)

    inverse_temperatures = tf.constant([1., 0.75, 0.5])
    # We need to pad the step_size so it broadcasts against MCMC samples. In
    # this case we have 1 replica dim, 0 batch dims, and 0 event dims (per each
    # of 2 state parts) hence no padding is needed.
    # We do however supply a step size for each state part.
    step_sizes = [0.9 / tf.math.sqrt(inverse_temperatures)]*2

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=step_sizes,
          num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target_log_prob,
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed())

    def trace_fn(state, results):  # pylint: disable=unused-argument
      return results.post_swap_replica_results.log_accept_ratio

    [samples_x,
     samples_y], replica_log_accept_ratio = (tfp.mcmc.sample_chain(
         num_results=num_results,
         num_burnin_steps=200,
         current_state=[1., 1.],
         kernel=remc,
         trace_fn=trace_fn,
         parallel_iterations=1))  # For determinism.
    samples = tf.stack([samples_x, samples_y], axis=-1)
    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_cov = tfp.stats.covariance(samples, sample_axis=0)

    replica_accept_ratio = tf.reduce_mean(
        tf.math.exp(tf.minimum(0., replica_log_accept_ratio)),
        axis=0)

    [
        sample_mean_,
        sample_cov_,
        replica_accept_ratio_,
        ess_,
    ] = self.evaluate([
        sample_mean,
        sample_cov,
        replica_accept_ratio,
        effective_sample_size(samples),
    ])
    logging.vlog(
        2, '---- execution:{}  accept_ratio:{}  mean:{}  cov:{}'.format(
            'eager' if tf.executing_eagerly() else 'graph',
            replica_accept_ratio_, sample_mean_, sample_cov_))

    self.assertAllEqual([num_results], samples_x.shape)
    self.assertAllEqual([num_results], samples_y.shape)

    max_scale = np.sqrt(np.max(true_cov))

    self.assertAllClose(
        true_mean, sample_mean_, atol=5 * max_scale / np.sqrt(np.min(ess_)))
    self.assertAllClose(
        true_cov, sample_cov_, atol=5 * max_scale**2 / np.sqrt(np.min(ess_)))

  @parameterized.named_parameters([
      dict(  # pylint: disable=g-complex-comprehension
          testcase_name=testcase_name + kernel_name,
          tfp_transition_kernel=tfp_transition_kernel,
          inverse_temperatures=inverse_temperatures,
          step_size_fn=step_size_fn,
          ess_scaling=ess_scaling)
      for kernel_name, tfp_transition_kernel, ess_scaling in [
          ('HMC', tfp.mcmc.HamiltonianMonteCarlo, .1),
          ('RWMH', init_tfp_randomwalkmetropolis, .02),
      ]
      for testcase_name, inverse_temperatures, step_size_fn in [
          ('1DTemperatureScalarStep', np.float32([1.0, 0.5, 0.25]),
           lambda x: 0.5),
          ('1DTemperature1DStep', np.float32([1.0, 0.5, 0.25]),
           lambda x: 0.5 / np.sqrt(x).reshape(3, 1, 1)),
          (
              '1DTemperature2DStep',
              np.float32([1.0, 0.5, 0.25]),
              lambda x: np.stack(  # pylint: disable=g-long-lambda
                  [0.5 / np.sqrt(x), 0.5 / np.sqrt(x)],
                  axis=-1).reshape(3, 2, 1)),
          (
              '2DTemperature1DStep',
              np.float32(
                  np.stack([[1.0, 0.5, 0.25], [1.0, 0.25, 0.05]], axis=-1)),
              lambda x: 0.5 / np.sqrt(  # pylint: disable=g-long-lambda
                  x.mean(axis=-1).reshape(3, 1, 1))),
          ('2DTemperature2DStep',
           np.float32(np.stack([[1.0, 0.5, 0.25], [1.0, 0.25, 0.05]], axis=-1)),
           lambda x: 0.5 / np.sqrt(x).reshape(3, 2, 1))
      ]
  ])
  def test1EventDim2BatchDim3Replica(self,
                                     tfp_transition_kernel,
                                     inverse_temperatures,
                                     step_size_fn,
                                     ess_scaling):
    """Sampling from two batch diagonal multivariate normal."""
    step_size = step_size_fn(inverse_temperatures) + np.exp(
        np.pi) / 100  # Prevent resonances.

    # Small scale and well-separated modes mean we need replica swap to
    # work or else tests fail.
    loc = np.array(
        [
            # Use 3-D normals, ensuring batch and event sizes don't broadcast.
            [-1., -0.5, 0.],  # loc of first batch
            [1., 0.5, 0.],  # loc of second batch
        ],
        dtype=np.float32)
    scale_identity_multiplier = [0.5, 0.8]
    target = tfd.MultivariateNormalDiag(
        loc=loc, scale_identity_multiplier=scale_identity_multiplier)

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp_transition_kernel(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=step_size,
          num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tf.function(
            lambda x: target.copy().log_prob(x), autograph=False),
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed())

    def trace_fn(state, results):  # pylint: disable=unused-argument
      return [
          results.post_swap_replica_results.log_accept_ratio,
          results.post_swap_replica_states
      ]

    num_results = 500 if tf.executing_eagerly() else 2000
    states, (log_accept_ratio, replica_states) = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=loc[::-1],  # Batch members far from their mode!
        kernel=remc,
        num_burnin_steps=100,
        trace_fn=trace_fn,
        parallel_iterations=1)  # For determinism.

    num_replica = inverse_temperatures.shape[0]

    self.assertLen(replica_states, 1)  # One state part
    replica_states = replica_states[0]

    self.assertAllEqual((num_results, num_replica) + loc.shape,
                        replica_states.shape)
    self.assertAllEqual((num_results,) + loc.shape, states.shape)

    (
        states_,
        replica_states_,
        replica_mean_,
        replica_cov_,
        accept_probs_,
        ess_,
    ) = self.evaluate([
        states,
        replica_states,
        tf.reduce_mean(replica_states, axis=0),
        tfp.stats.covariance(replica_states),
        tf.math.exp(tf.minimum(0., log_accept_ratio)),
        effective_sample_size(replica_states),
    ])

    logging.vlog(
        2, '---- execution:{} Min[ESS]: {}  mean_accept: {}'.format(
            'eager' if tf.executing_eagerly() else 'graph',
            np.min(ess_), np.mean(accept_probs_, axis=0)))

    self.assertAllEqual(states_, replica_states_[:, 0])

    def _check_stats(replica_idx, batch_idx, ess_scaling):
      err_msg = 'Failure in replica {}, batch {}'.format(replica_idx, batch_idx)
      assert inverse_temperatures.ndim in [1, 2]
      if inverse_temperatures.ndim == 1:
        temperature = 1 / inverse_temperatures[replica_idx]
      elif inverse_temperatures.ndim == 2:
        temperature = 1 / inverse_temperatures[replica_idx, batch_idx]

      expected_scale = (
          scale_identity_multiplier[batch_idx] * np.sqrt(temperature))

      ess = np.min(ess_[replica_idx, batch_idx])  # Conservative estimate.
      self.assertGreater(ess, num_results * ess_scaling, msg='Bad sampling!')

      self.assertAllClose(
          replica_mean_[replica_idx, batch_idx],
          loc[batch_idx],
          # 5 standard errors of a mean estimate.
          atol=5 * expected_scale / np.sqrt(ess),
          msg=err_msg)
      self.assertAllClose(
          expected_scale**2 * np.eye(loc.shape[1]),
          replica_cov_[replica_idx, batch_idx],
          # 10 standard errors of a variance estimate.
          atol=10 * np.sqrt(2) * expected_scale**2 / np.sqrt(ess),
          msg=err_msg)

    for replica_idx in range(num_replica):
      for batch_idx in range(loc.shape[0]):
        _check_stats(replica_idx, batch_idx, ess_scaling)

  def testMultipleCorrelatedStatesWithOneBatchDim(self):
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    # Use LinearOperatorLowerTriangular to get broadcasting ability.
    linop = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(true_cov))
    num_results = 250 if tf.executing_eagerly() else 2000

    def target_log_prob(x, y):
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      xy = tf.stack([x, y], axis=-1) - true_mean
      z = linop.solvevec(xy)
      return -0.5 * tf.reduce_sum(z**2., axis=-1)

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=[0.75, 0.75],
          num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tf.function(target_log_prob, autograph=False),
        inverse_temperatures=[1., 0.9, 0.8],
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed())

    states = tfp.mcmc.sample_chain(
        num_results=num_results,
        # batch_shape = [4] for each initial state
        current_state=[tf.ones(4), tf.ones(4)],
        kernel=remc,
        num_burnin_steps=400,
        trace_fn=None,
        parallel_iterations=1)  # For determinism.

    states = tf.stack(states, axis=-1)
    self.assertAllEqual((num_results, 4, 2), states.shape)

    states_, ess_, cov_ = self.evaluate([
        states,
        effective_sample_size(states),
        tfp.stats.covariance(states)
    ])

    self.assertGreater(np.min(ess_), num_results / 10, 'Bad sampling found!')

    # 5 standard errors for mean/variance estimates.
    mean_atol = 5 / np.sqrt(np.min(ess_))
    cov_atol = 5 * np.sqrt(2) / np.sqrt(np.min(ess_))

    self.assertAllClose(
        true_mean, states_[:, 0, :].mean(axis=0), atol=mean_atol)
    self.assertAllClose(
        true_mean, states_[:, 1, :].mean(axis=0), atol=mean_atol)
    self.assertAllClose(true_cov, cov_[0], atol=cov_atol)
    self.assertAllClose(true_cov, cov_[1], atol=cov_atol)

  def testInversePermutationError(self):
    """Using invalid `inverse_temperatures`."""
    dtype = np.float32
    def bad_swap_fn(num_replica, batch_shape=(), seed=None):  # pylint: disable=unused-argument
      return [1, 2, 0]
    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tfd.Normal(loc=dtype(0), scale=dtype(1)).log_prob,
        inverse_temperatures=dtype([1., 0.5, 0.25]),
        make_kernel_fn=lambda tlp, seed: tfp.mcmc.HamiltonianMonteCarlo(  # pylint: disable=g-long-lambda
            target_log_prob_fn=tlp,
            seed=seed,
            step_size=1.,
            num_leapfrog_steps=3),
        # Fun fact: of the six length-3 permutations, only two are not
        # "one-time swap" permutations: [1, 2, 0], [2, 0, 1]
        swap_proposal_fn=bad_swap_fn,
        validate_args=True,
        seed=_set_seed())
    with self.assertRaisesRegexp(
        tf.errors.OpError, 'must be.*self-inverse permutation'):
      self.evaluate(tfp.mcmc.sample_chain(
          num_results=10,
          num_burnin_steps=2,
          current_state=[dtype(1)],
          kernel=remc,
          trace_fn=None,
          parallel_iterations=1))  # For determinism.

  def testKernelResultsHaveCorrectShapeWhenMultipleStatesAndBatchDims(self):
    def target_log_prob(x, y):
      xy = tf.concat([x, y], axis=-1)
      return -0.5 * tf.reduce_sum(xy**2, axis=-1)

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=[0.3, 0.1],
          num_leapfrog_steps=3)

    inverse_temperatures = [1., 0.5, 0.25, 0.1]
    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tf.function(target_log_prob, autograph=False),
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed())

    num_results = 6
    n_batch = 5
    n_events = 3
    n_states = 2  # Set by target_log_prob.
    num_replica = len(inverse_temperatures)

    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=[tf.zeros((n_batch, n_events))] * n_states,
        kernel=remc,
        num_burnin_steps=2,
        trace_fn=lambda _, results: results)

    self.assertLen(samples, n_states)
    self.assertAllEqual((num_results, n_batch, n_events), samples[0].shape)
    self.assertAllEqual((num_results, n_batch, n_events), samples[1].shape)

    kr_ = self.evaluate(kernel_results)

    # Boring checks of existence/shape.
    self.assertEqual(
        (num_results, num_replica, n_batch, n_states, n_events),
        tf.stack(kr_.post_swap_replica_states, axis=-2).shape)

    self.assertEqual(
        (num_results, num_replica, n_batch),
        kr_.pre_swap_replica_results.log_accept_ratio.shape)

    self.assertEqual(
        (num_results, num_replica, n_batch),
        kr_.post_swap_replica_results.log_accept_ratio.shape)

    self.assertEqual(
        (num_results, num_replica, num_replica, n_batch),
        kr_.is_swap_proposed.shape)
    self.assertEqual(
        (num_results, num_replica, num_replica, n_batch),
        kr_.is_swap_accepted.shape)

    self.assertEqual(
        (num_results, num_replica - 1, n_batch),
        kr_.is_swap_proposed_adjacent.shape)
    self.assertEqual(
        (num_results, num_replica - 1, n_batch),
        kr_.is_swap_accepted_adjacent.shape)

    self.assertEqual(
        (num_results, num_replica),
        tf.stack(kr_.inverse_temperatures, axis=1).shape)

    self.assertEqual(
        (num_results, num_replica, n_batch),
        kr_.swaps.shape)


if __name__ == '__main__':
  tf.test.main()
