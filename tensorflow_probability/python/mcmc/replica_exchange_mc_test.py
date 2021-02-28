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
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_combinations
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import unnest

tfd = tfp.distributions

JAX_MODE = False


def init_tfp_randomwalkmetropolis(
    target_log_prob_fn,
    step_size,
    store_parameters_in_results=False, num_leapfrog_steps=None):  # pylint: disable=unused-argument
  return tfp.mcmc.RandomWalkMetropolis(
      target_log_prob_fn,
      new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size))


def init_tfp_adaptive_hmc(
    target_log_prob_fn,
    step_size,
    num_leapfrog_steps=None, store_parameters_in_results=False):  # pylint: disable=unused-argument
  return tfp.mcmc.simple_step_size_adaptation.SimpleStepSizeAdaptation(
      tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn,
          step_size=step_size,
          num_leapfrog_steps=num_leapfrog_steps,
          store_parameters_in_results=store_parameters_in_results),
      target_accept_prob=0.75, num_adaptation_steps=250)


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
  seed = test_util.test_seed()
  if tf.executing_eagerly() and not JAX_MODE:
    tf.random.set_seed(seed)
    return None
  return seed


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
    seeds = samplers.split_seed(test_util.test_seed(), n=num_results)
    swaps = tf.stack(
        [fn(num_replica, seed=seeds[i]) for i in range(num_results)],
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
    seeds = samplers.split_seed(test_util.test_seed(), n=num_results)
    swaps = tf.stack(
        [fn(num_replica, batch_shape=[2], seed=seeds[i])
         for i in range(num_results)],
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
class EvenOddSwapProposedFnTest(test_util.TestCase):

  def testYesBatchThreeReplicas(self):
    # There are three replicas, so we get...
    #   null swaps [0, 1, 2],
    #   even swaps [1, 0, 2],
    #   odd swaps [0, 2, 1].
    # Each type is identically repeated across all batch members.
    fn = tfp.mcmc.even_odd_swap_proposal_fn(swap_frequency=0.5)

    num_batch = 2
    num_replica = 3

    null_swap = [[0] * num_batch,
                 [1] * num_batch,
                 [2] * num_batch]

    even_swap = [[1] * num_batch,
                 [0] * num_batch,
                 [2] * num_batch]

    odd_swap = [[0] * num_batch,
                [2] * num_batch,
                [1] * num_batch]

    with self.subTest(
        'Even swap since count=0 is an even multiple of period=2'):
      swaps = fn(num_replica=num_replica, batch_shape=[num_batch], step_count=0)
      self.assertAllEqual((num_replica, num_batch), swaps.shape)
      self.assertAllEqual(even_swap, self.evaluate(swaps))

    with self.subTest(
        'No swap since count=1 is not a multiple of period=2'):
      swaps = fn(num_replica=num_replica, batch_shape=[num_batch], step_count=1)
      self.assertAllEqual((num_replica, num_batch), swaps.shape)
      self.assertAllEqual(null_swap, self.evaluate(swaps))

    with self.subTest(
        'Odd swap since count=2 is an odd multiple of period=2'):
      swaps = fn(num_replica=num_replica, batch_shape=[num_batch], step_count=2)
      self.assertAllEqual((num_replica, num_batch), swaps.shape)
      self.assertAllEqual(odd_swap, self.evaluate(swaps))

    with self.subTest(
        'No swap since count=3 is not a multiple of period=2'):
      swaps = fn(num_replica=num_replica, batch_shape=[num_batch], step_count=3)
      self.assertAllEqual(null_swap, self.evaluate(swaps))

    with self.subTest(
        'Even swap since count=4 is an even multiple of period=2'):
      swaps = fn(num_replica=num_replica, batch_shape=[num_batch], step_count=4)
      self.assertAllEqual(even_swap, self.evaluate(swaps))

  def testNoBatchOneReplica(self):
    # There is only one replica, so in all cases swaps = [0].
    for swap_frequency in [0., 0.5, 1.]:
      fn = tfp.mcmc.even_odd_swap_proposal_fn(swap_frequency=swap_frequency)
      for step_count in range(5):
        swaps = fn(num_replica=1, step_count=step_count)
        self.assertAllEqual((1,), swaps.shape)
        swaps_ = self.evaluate(swaps)
        self.assertAllEqual(swaps_, [0])

  def testNoBatchTwoReplicas(self):
    # There are two replicas, so we get null swaps [0, 1], and swaps [1, 0]
    for swap_frequency in [0., 0.5, 1.]:
      fn = tfp.mcmc.even_odd_swap_proposal_fn(swap_frequency=swap_frequency)
      for step_count in range(10):
        swaps = fn(num_replica=2, step_count=step_count)
        swaps_ = self.evaluate(swaps)

        if (
            # swap_frequency == 0 means never swap.
            not swap_frequency or
            # Not at swap_period := 1 / swap_frequency.
            step_count % int(np.ceil(1 / swap_frequency))
        ):
          self.assertAllEqual(swaps_, [0, 1])  # No swap
        else:
          self.assertAllEqual(swaps_, [1, 0])

  def testNoBatchThreeReplicas(self):
    # There are three replicas, so we get...
    #   null swaps [0, 1, 2],
    #   even swaps [1, 0, 2],
    #   odd swaps [0, 2, 1].
    for swap_frequency in [0., 0.5, 1.]:
      fn = tfp.mcmc.even_odd_swap_proposal_fn(swap_frequency=swap_frequency)
      for step_count in range(10):
        swaps = fn(num_replica=3, step_count=step_count)
        swaps_ = self.evaluate(swaps)

        if (
            # swap_frequency == 0 means never swap.
            not swap_frequency or
            # Not at swap_period := 1 / swap_frequency.
            step_count % int(np.ceil(1 / swap_frequency))
        ):
          self.assertAllEqual(swaps_, [0, 1, 2])  # No swap
        else:  # Else swapping
          period_count = step_count // int(np.ceil(1 / swap_frequency))
          if period_count % 2:  # Odd parity
            self.assertAllEqual(swaps_, [0, 2, 1])
          else:  # Even parity
            self.assertAllEqual(swaps_, [1, 0, 2])


@test_util.test_graph_and_eager_modes
class REMCTest(test_util.TestCase):

  def setUp(self):
    tf.random.set_seed(123)
    super(REMCTest, self).setUp()

  def assertRaisesError(self, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(Exception, msg)
    return self.assertRaisesOpError(msg)

  @parameterized.named_parameters([
      dict(  # pylint: disable=g-complex-comprehension
          testcase_name=(
              testcase_name + kernel_name +
              ['', '_state_includes_replicas'][state_includes_replicas] +
              ['_fast_execute_only', '_slow_asserts'][asserts]),
          tfp_transition_kernel=tfp_transition_kernel,
          inverse_temperatures=inverse_temperatures,
          state_includes_replicas=state_includes_replicas,
          store_parameters_in_results=store_param,
          asserts=asserts) for state_includes_replicas in [False, True]
      for asserts in [True, False]
      for kernel_name, tfp_transition_kernel, store_param in [
          ('HMC', tfp.mcmc.HamiltonianMonteCarlo, True),  # NUMPY_DISABLE
          ('AdaptiveHMC', init_tfp_adaptive_hmc, True),  # NUMPY_DISABLE
          ('RWMH', init_tfp_randomwalkmetropolis, False),
      ] for testcase_name, inverse_temperatures in [
          ('OddNumReplicas', [1.0, 0.8, 0.6]),
          ('EvenNumReplicas', [1.0, 0.8, 0.7, 0.6]),
          ('HighTemperatureOnly', [0.5]),
          ('LowTemperatureOnly', [2.0]),
      ]
  ])
  def testNormal(self,
                 tfp_transition_kernel,
                 inverse_temperatures,
                 state_includes_replicas,
                 store_parameters_in_results,
                 asserts,
                 prob_swap=1.0,
                 dtype=np.float32):
    """Sampling from standard normal with REMC."""

    target = tfd.Normal(dtype(0.), dtype(1.))
    inverse_temperatures = dtype(inverse_temperatures)
    num_replica = len(inverse_temperatures)

    step_size = 0.51234 / np.sqrt(inverse_temperatures)
    num_leapfrog_steps = 3

    def make_kernel_fn(target_log_prob_fn):
      return tfp_transition_kernel(
          target_log_prob_fn=target_log_prob_fn,
          step_size=step_size,
          store_parameters_in_results=store_parameters_in_results,
          num_leapfrog_steps=num_leapfrog_steps)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target.log_prob,
        inverse_temperatures=inverse_temperatures,
        state_includes_replicas=state_includes_replicas,
        make_kernel_fn=make_kernel_fn,
        swap_proposal_fn=tfp.mcmc.default_swap_proposal_fn(prob_swap))

    num_results = 17
    if asserts:
      num_results = 2000
      remc.one_step = tf.function(remc.one_step, autograph=False)

    if state_includes_replicas:
      current_state = target.sample(num_replica, seed=_set_seed())
    else:
      current_state = target.sample(seed=_set_seed())

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=current_state,
        kernel=remc,
        num_burnin_steps=50,
        trace_fn=lambda _, results: results,
        seed=_set_seed())

    if state_includes_replicas:
      self.assertAllEqual((num_results, num_replica), states.shape)
    else:
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
    replica_log_accept_ratio = unnest.get_innermost(
        kr_.post_swap_replica_results, 'log_accept_ratio')
    replica_states_ = kr_.post_swap_replica_states[0]  # Get rid of "parts"

    # Target state is at index 0.
    if state_includes_replicas:
      self.assertAllClose(states_, replica_states_)
    else:
      self.assertAllClose(states_, replica_states_[:, 0])

    # Check that *each* replica has correct marginal.
    def _check_sample_stats(replica_idx):
      x = replica_states_[:, replica_idx]
      ess = replica_ess_[replica_idx]

      err_msg = 'replica_idx={}'.format(replica_idx)

      mean_atol = 6 * 1.0 / np.sqrt(ess)
      self.assertAllClose(x.mean(), 0.0, atol=mean_atol, msg=err_msg)

      # For a tempered Normal, Variance = T.
      expected_var = 1 / inverse_temperatures[replica_idx]
      var_atol = 6 * expected_var * np.sqrt(2) / np.sqrt(ess)
      self.assertAllClose(np.var(x), expected_var, atol=var_atol, msg=err_msg)

    if not asserts:
      return

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
      # Matrix is stochastic (since you either get swapped with another
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
      if not isinstance(
          kr_.post_swap_replica_results,
          tfp.mcmc.simple_step_size_adaptation.SimpleStepSizeAdaptationResults):
        self.assertAllEqual(
            np.repeat([step_size], axis=0, repeats=num_results),
            unnest.get_innermost(kr_.post_swap_replica_results, 'step_size'))

      self.assertAllEqual(
          np.repeat([num_leapfrog_steps], axis=0, repeats=num_results),
          unnest.get_innermost(
              kr_.post_swap_replica_results, 'num_leapfrog_steps'))

  @test_util.numpy_disable_gradient_test('HMC')
  @test_combinations.generate(
      test_combinations.combine(
          use_untempered_lp=[False, True],
          use_xla=[False, True],
      )
  )
  def testAdaptingPerReplicaStepSize(self, use_untempered_lp, use_xla):
    num_chains, num_events = 4, 2
    num_results = 250
    target = tfd.MultivariateNormalDiag(loc=[0.] * num_events)

    # The inverse_temperatures are decaying rapidly enough to force
    # significantly different adapted step sizes.
    # It's important not to make inverse_temperatures[-1] too small, or else
    # the last few temperatures could be essentially the same when using an
    # untempered_log_prob_fn.
    inverse_temperatures = 0.25**np.arange(3, dtype=np.float32)
    num_replica = len(inverse_temperatures)

    # step_size is...
    #   Too large == 3!
    #   A shape that will broadcast across the chains, and (after adaptation)
    #   give a different value for each replica.
    initial_step_size = 3 * np.ones((num_replica, 1, 1), dtype=np.float32)
    target_accept_prob = 0.9

    def _get_states_and_trace():
      # Whether using the untempered_log_prob_fn setup or not, the target is
      # target.log_prob.
      # However, the untempered_log_prob_fn is very dispersed...it is as though
      # it has a temperature of 100.
      if use_untempered_lp:
        target_log_prob_fn = None
        untempered_log_prob_fn = lambda x: target.log_prob(x) / 100.
        tempered_log_prob_fn = lambda x: target.log_prob(x) * 99 / 100.
      else:
        target_log_prob_fn = target.log_prob
        untempered_log_prob_fn = None
        tempered_log_prob_fn = None

      def make_kernel_fn(target_log_prob_fn):
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=initial_step_size,
            num_leapfrog_steps=3,
        )
        return tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc,
            target_accept_prob=target_accept_prob,
            adaptation_rate=0.01,
            num_adaptation_steps=num_results,
        )

      adaptive_remc = tfp.mcmc.ReplicaExchangeMC(
          target_log_prob_fn=target_log_prob_fn,
          untempered_log_prob_fn=untempered_log_prob_fn,
          tempered_log_prob_fn=tempered_log_prob_fn,
          inverse_temperatures=inverse_temperatures,
          make_kernel_fn=make_kernel_fn,
          state_includes_replicas=True,
      )

      def trace_fn(_, results):
        step_adapt_results = results.post_swap_replica_results
        hmc_results = step_adapt_results.inner_results
        return {
            'step_size':
                step_adapt_results.new_step_size,
            'prob_accept':
                tf.math.exp(tf.math.minimum(hmc_results.log_accept_ratio, 0.)),
        }

      return tfp.mcmc.sample_chain(
          num_results=num_results,
          current_state=tf.zeros((num_replica, num_chains, num_events)),
          kernel=adaptive_remc,
          trace_fn=trace_fn,
          num_burnin_steps=0,
          seed=_set_seed(),
      )

    if use_xla:
      states, trace = tf.function(
          _get_states_and_trace, jit_compile=True)()
    else:
      states, trace = _get_states_and_trace()

    states_, final_step_size_, mean_accept_, ess_ = self.evaluate([
        states,
        trace['step_size'][-1],
        tf.reduce_mean(trace['prob_accept'][num_results // 2:], axis=0),
        effective_sample_size(states, cross_chain_dims=-2),
    ])

    self.assertAllEqual((num_replica, 1, 1), final_step_size_.shape)

    self.assertAllClose(
        target_accept_prob * np.ones_like(mean_accept_), mean_accept_, atol=0.2)

    # Step size should be increasing (with temperature).
    np.testing.assert_array_less(0.0, np.diff(final_step_size_.ravel()))

    # Step size for the Temperature = 1 replica should have decreased
    # significantly from the large initial value.
    self.assertEqual(initial_step_size[0, 0, 0], 3)
    self.assertLess(final_step_size_[0, 0, 0], 2)

    # The mean shouldn't be ridiculous.
    for r in range(num_replica):
      x = states_[:, r]
      mean = np.mean(x)
      stddev = np.std(x)
      n = np.min(ess_[r])
      self.assertAllClose(np.zeros_like(mean),
                          mean,
                          atol=4 * stddev / np.sqrt(n),
                          msg='Failed at replica {}'.format(r))

  @parameterized.named_parameters([
      # pylint: disable=line-too-long
      ('_HMC_default', tfp.mcmc.HamiltonianMonteCarlo, False, 'default'),  # NUMPY_DISABLE
      ('_HMC_scr_default', tfp.mcmc.HamiltonianMonteCarlo, True, 'default'),  # NUMPY_DISABLE
      ('_RWMH_default', init_tfp_randomwalkmetropolis, False, 'default'),
      ('_adaptive_HMC_default', init_tfp_adaptive_hmc, False, 'default'),  # NUMPY_DISABLE
      ('_HMC_even_odd', tfp.mcmc.HamiltonianMonteCarlo, False, 'even_odd'),  # NUMPY_DISABLE
      ('_RWMH_even_odd', init_tfp_randomwalkmetropolis, False, 'default'),  # NUMPY_DISABLE
      # pylint: enable=line-too-long
  ])
  def test2DMixNormal(self, tfp_transition_kernel, state_includes_replicas,
                      swap_proposal_fn_name):
    """Sampling from a 2-D Mixture Normal Distribution."""
    swap_proposal_fn = {
        'default': tfp.mcmc.default_swap_proposal_fn(1.),
        'even_odd': tfp.mcmc.even_odd_swap_proposal_fn(1.),
    }[swap_proposal_fn_name]

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

    num_replica = 4
    inverse_temperatures = 10.**tf.linspace(start=0., stop=-1., num=num_replica)
    # We need to pad the step_size so it broadcasts against MCMC samples. In
    # this case we have 1 replica dim, 0 batch dims, and 1 event dim hence need
    # to right pad the step_size by one dim (for the event).
    step_size = 0.2 / tf.math.sqrt(inverse_temperatures[:, tf.newaxis])
    def make_kernel_fn(target_log_prob_fn):
      return tfp_transition_kernel(
          target_log_prob_fn=target_log_prob_fn,
          step_size=step_size,
          num_leapfrog_steps=5)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target.log_prob,
        # Verified that test fails if inverse_temperatures = [1.]
        inverse_temperatures=inverse_temperatures,
        swap_proposal_fn=swap_proposal_fn,
        make_kernel_fn=make_kernel_fn)
    remc.one_step = tf.function(remc.one_step, autograph=False)

    def trace_fn(state, results):  # pylint: disable=unused-argument
      return unnest.get_innermost(
          results.post_swap_replica_results, 'log_accept_ratio')

    if state_includes_replicas:
      current_state = tf.ones((num_replica, 2), dtype=dtype)
    else:
      current_state = tf.ones(2, dtype=dtype)

    num_results = 2000
    states, replica_log_accept_ratio = tfp.mcmc.sample_chain(
        num_results=num_results,
        # Start at one of the modes, in order to make mode jumping necessary
        # if we want to pass test.
        current_state=current_state,
        kernel=remc,
        num_burnin_steps=50,
        trace_fn=trace_fn,
        seed=test_util.test_seed())

    if state_includes_replicas:
      self.assertAllEqual((num_results, num_replica, 2), states.shape)
      states = states[:, 0]  # Keep only replica 0
    else:
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

    mean_atol = 6 * expected_stddev_ / np.sqrt(np.min(ess_))
    var_atol = 6 * expected_variance_ / np.sqrt(np.min(ess_))
    for i in range(mean_atol.shape[0]):
      self.assertAllClose(
          expected_mean_[i],
          sample_mean_[i],
          atol=mean_atol[i],
          msg='position {}'.format(i))
      self.assertAllClose(
          expected_variance_[i],
          sample_variance_[i],
          atol=var_atol[i],
          msg=i)

  @test_util.numpy_disable_gradient_test('HMC')
  def testMultipleCorrelatedStatesWithNoBatchDims(self):
    dtype = np.float32
    num_results = 2000
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    # Use LinearOperatorLowerTriangular to get broadcasting ability.
    linop = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(true_cov))

    # Its ok to decorate this since we only need to stress the TransitionKernel.
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

    def make_kernel_fn(target_log_prob_fn):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          step_size=step_sizes,
          num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target_log_prob,
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn)
    remc.one_step = tf.function(remc.one_step, autograph=False)

    def trace_fn(state, results):  # pylint: disable=unused-argument
      return results.post_swap_replica_results.log_accept_ratio

    [samples_x, samples_y], replica_log_accept_ratio = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=200,
        current_state=[1., 1.],
        kernel=remc,
        trace_fn=trace_fn,
        seed=test_util.test_seed())
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
        true_mean, sample_mean_, atol=6 * max_scale / np.sqrt(np.min(ess_)))
    self.assertAllClose(
        true_cov, sample_cov_, atol=6 * max_scale**2 / np.sqrt(np.min(ess_)))

  @parameterized.named_parameters([
      dict(  # pylint: disable=g-complex-comprehension
          testcase_name=testcase_name + kernel_name + ulpname,
          tfp_transition_kernel=tfp_transition_kernel,
          inverse_temperatures=inverse_temperatures,
          step_size_fn=step_size_fn,
          ess_scaling=ess_scaling,
          use_untempered_log_prob_fn=use_untempered_log_prob_fn)
      for kernel_name, tfp_transition_kernel, ess_scaling in [
          ('HMC', tfp.mcmc.HamiltonianMonteCarlo, .1),  # NUMPY_DISABLE
          ('RWMH', init_tfp_randomwalkmetropolis, .009),
      ]
      for testcase_name, inverse_temperatures, step_size_fn in [
          ('1DTemperatureScalarStep',
           np.float32([1.0, 0.5, 0.25]),
           lambda x: 0.5),
          ('1DTemperature1DStep',
           np.float32([1.0, 0.5, 0.25]),
           lambda x: 0.5 / np.sqrt(x).reshape(3, 1, 1)),
          ('1DTemperature2DStep',
           np.float32([1.0, 0.5, 0.25]),
           lambda x: np.stack(  # pylint: disable=g-long-lambda
               [0.5 / np.sqrt(x), 0.5 / np.sqrt(x)],
               axis=-1).reshape(3, 2, 1)),
          ('2DTemperature1DStep',
           np.float32(np.stack([[1.0, 0.5, 0.25], [1.0, 0.25, 0.05]], axis=-1)),
           lambda x: 0.5 / np.sqrt(  # pylint: disable=g-long-lambda
               x.mean(axis=-1).reshape(3, 1, 1))),
          ('2DTemperature2DStep',
           np.float32(np.stack([[1.0, 0.5, 0.25], [1.0, 0.25, 0.05]], axis=-1)),
           lambda x: 0.5 / np.sqrt(x).reshape(3, 2, 1))
      ]
      for ulpname, use_untempered_log_prob_fn in [
          ('NoUntemperedLogProb', False), ('YesUntemperedLogProb', True),
      ]
  ])
  def test1EventDim2BatchDim3Replica(self,
                                     tfp_transition_kernel,
                                     inverse_temperatures,
                                     step_size_fn,
                                     ess_scaling,
                                     use_untempered_log_prob_fn):
    """Sampling from two batch diagonal multivariate normal."""
    step_size = (step_size_fn(inverse_temperatures) +
                 np.exp(np.pi) / 100).astype(np.float32)  # Prevent resonances.

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

    def make_kernel_fn(target_log_prob_fn):
      return tfp_transition_kernel(
          target_log_prob_fn=target_log_prob_fn,
          step_size=step_size,
          num_leapfrog_steps=3)

    target_log_prob_fn = tf.function(target.log_prob, autograph=False)

    if use_untempered_log_prob_fn:
      # This untempered_log_prob_fn should not change any of the results.
      untempered_log_prob_fn = lambda x: tf.zeros_like(loc[..., 0])
      tempered_log_prob_fn = target_log_prob_fn
      target_log_prob_fn = None
    else:
      untempered_log_prob_fn = None
      tempered_log_prob_fn = None

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target_log_prob_fn,
        untempered_log_prob_fn=untempered_log_prob_fn,
        tempered_log_prob_fn=tempered_log_prob_fn,
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn)
    remc.one_step = tf.function(remc.one_step, autograph=False)

    def trace_fn(state, results):  # pylint: disable=unused-argument
      return [
          unnest.get_innermost(
              results.post_swap_replica_results, 'log_accept_ratio'),
          results.post_swap_replica_states
      ]

    num_results = 2000
    states, (log_accept_ratio, replica_states) = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=loc[::-1],  # Batch members far from their mode!
        kernel=remc,
        num_burnin_steps=100,
        trace_fn=trace_fn,
        seed=test_util.test_seed())

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
          # 6 standard errors of a mean estimate.
          atol=6 * expected_scale / np.sqrt(ess),
          msg=err_msg)
      self.assertAllClose(
          expected_scale**2 * np.eye(loc.shape[1]),
          replica_cov_[replica_idx, batch_idx],
          # 12 standard errors of a variance estimate.
          atol=12 * np.sqrt(2) * expected_scale**2 / np.sqrt(ess),
          msg=err_msg)

    for replica_idx in range(num_replica):
      for batch_idx in range(loc.shape[0]):
        _check_stats(replica_idx, batch_idx, ess_scaling)

  @parameterized.named_parameters(
      [
          dict(
              testcase_name='_slow_asserts',
              asserts=True,
              use_untempered_fn=False),
          dict(
              testcase_name='_fast_execute_only',
              asserts=False,
              use_untempered_fn=False),
          dict(
              testcase_name='_fast_with_untempered',
              asserts=False,
              use_untempered_fn=True)
      ],)
  @test_util.numpy_disable_gradient_test('HMC')
  def testMultipleCorrelatedStatesWithOneBatchDim(self, asserts,
                                                  use_untempered_fn):
    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    # Use LinearOperatorLowerTriangular to get broadcasting ability.
    linop = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(true_cov))

    def target_log_prob(x, y):
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      xy = tf.stack([x, y], axis=-1) - true_mean
      z = linop.solvevec(xy)
      return -0.5 * tf.reduce_sum(z**2., axis=-1)

    if use_untempered_fn:
      # Should not change results at all
      untempered_log_prob_fn = lambda x, unused_y: tf.zeros_like(x)
      tempered_log_prob_fn = target_log_prob
      target_log_prob = None
    else:
      untempered_log_prob_fn = None
      tempered_log_prob_fn = None

    def make_kernel_fn(target_log_prob_fn):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          step_size=[0.75, 0.75],
          num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target_log_prob,
        inverse_temperatures=[1., 0.9, 0.8],
        make_kernel_fn=make_kernel_fn,
        untempered_log_prob_fn=untempered_log_prob_fn,
        tempered_log_prob_fn=tempered_log_prob_fn)

    num_results = 13
    if asserts:
      num_results = 2000
      remc.one_step = tf.function(remc.one_step, autograph=False)

    states = tfp.mcmc.sample_chain(
        num_results=num_results,
        # batch_shape = [4] for each initial state
        current_state=[tf.ones(4), tf.ones(4)],
        kernel=remc,
        num_burnin_steps=400,
        trace_fn=None,
        seed=test_util.test_seed())

    states = tf.stack(states, axis=-1)
    self.assertAllEqual((num_results, 4, 2), states.shape)

    states_, ess_, cov_ = self.evaluate([
        states,
        effective_sample_size(states),
        tfp.stats.covariance(states)
    ])

    if not asserts:
      return

    self.assertGreater(np.min(ess_), num_results / 10, 'Bad sampling found!')

    # 6 standard errors for mean/variance estimates.
    mean_atol = 8 / np.sqrt(np.min(ess_))
    cov_atol = 8 * np.sqrt(2) / np.sqrt(np.min(ess_))

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
        make_kernel_fn=lambda tlp: tfp.mcmc.RandomWalkMetropolis(  # pylint: disable=g-long-lambda
            target_log_prob_fn=tlp,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=1.)),
        # Fun fact: of the six length-3 permutations, only two are not
        # "one-time swap" permutations: [1, 2, 0], [2, 0, 1]
        swap_proposal_fn=bad_swap_fn,
        validate_args=True)
    with self.assertRaisesOpError('must be.*self-inverse permutation'):
      self.evaluate(tfp.mcmc.sample_chain(
          num_results=10,
          num_burnin_steps=2,
          current_state=[dtype(1)],
          kernel=remc,
          trace_fn=None,
          seed=test_util.test_seed()))

  @parameterized.named_parameters([
      dict(testcase_name='_no_state_includes_replicas',
           state_includes_replicas=False),
      dict(testcase_name='_yes_state_includes_replicas',
           state_includes_replicas=True),
  ])
  def testKernelResultsHaveCorrectShapeWhenMultipleStatesAndBatchDims(
      self, state_includes_replicas):
    def target_log_prob(x, y):
      xy = tf.concat([x, y], axis=-1)
      return -0.5 * tf.reduce_sum(xy**2, axis=-1)

    def make_kernel_fn(target_log_prob_fn):
      return tfp.mcmc.RandomWalkMetropolis(
          target_log_prob_fn=target_log_prob_fn,
          new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=[0.3, 0.1]))

    inverse_temperatures = [1., 0.5, 0.25, 0.1]
    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target_log_prob,
        inverse_temperatures=inverse_temperatures,
        state_includes_replicas=state_includes_replicas,
        make_kernel_fn=make_kernel_fn)

    num_results = 6
    n_batch = 5
    n_events = 3
    n_states = 2  # Set by target_log_prob.
    num_replica = len(inverse_temperatures)

    if state_includes_replicas:
      current_state = [tf.zeros((num_replica, n_batch, n_events))] * n_states
    else:
      current_state = [tf.zeros((n_batch, n_events))] * n_states

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=current_state,
        kernel=remc,
        num_burnin_steps=2,
        trace_fn=lambda _, results: results,
        seed=test_util.test_seed())

    self.assertLen(states, n_states)

    if state_includes_replicas:
      state_shape = (num_results, num_replica, n_batch, n_events)
    else:
      state_shape = (num_results, n_batch, n_events)

    self.assertAllEqual(state_shape, states[0].shape)
    self.assertAllEqual(state_shape, states[1].shape)

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

  def testInconsistantInverseTemperaturesAndStateSizeRaises(self):
    target = tfd.Normal(0., 1.)
    inverse_temperatures = [1., 0.5, 0.25]  # Implies num_replica = 3
    current_state = tf.zeros((4,))  # Implies num_replica = 4

    def make_kernel_fn(target_log_prob_fn):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          step_size=0.1,
          num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target.log_prob,
        inverse_temperatures=inverse_temperatures,
        state_includes_replicas=True,
        make_kernel_fn=make_kernel_fn,
        swap_proposal_fn=tfp.mcmc.default_swap_proposal_fn(prob_swap=1.))

    with self.assertRaisesRegex(ValueError, 'Number of replicas'):
      tfp.mcmc.sample_chain(
          num_results=5,
          current_state=current_state,
          kernel=remc,
          seed=_set_seed())

  def testSpecifyingWrongCombinationOfLogProbArgsRaises(self):

    def make_kernel_fn(target_log_prob_fn):
      return tfp.mcmc.RandomWalkMetropolis(
          target_log_prob_fn=target_log_prob_fn,
          new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=0.1))

    with self.subTest('provided tempered but not untempered'):
      with self.assertRaisesRegex(ValueError, 'either neither or both'):
        tfp.mcmc.ReplicaExchangeMC(
            target_log_prob_fn=None,
            untempered_log_prob_fn=None,
            tempered_log_prob_fn=lambda x: 1.0,  # provided
            inverse_temperatures=[1.],
            make_kernel_fn=make_kernel_fn)

    with self.subTest('provided untempered but not tempered'):
      with self.assertRaisesRegex(ValueError, 'either neither or both'):
        tfp.mcmc.ReplicaExchangeMC(
            target_log_prob_fn=None,
            untempered_log_prob_fn=lambda x: 1.0,  # provided
            tempered_log_prob_fn=None,
            inverse_temperatures=[1.],
            make_kernel_fn=make_kernel_fn)

    with self.subTest('provided all three'):
      with self.assertRaisesRegex(ValueError, 'Exactly one'):
        tfp.mcmc.ReplicaExchangeMC(
            target_log_prob_fn=lambda x: 1.0,  # provided
            untempered_log_prob_fn=lambda x: 1.0,  # provided
            tempered_log_prob_fn=lambda x: 1.0,  # provided
            inverse_temperatures=[1.],
            make_kernel_fn=make_kernel_fn)

  def testInvalidInverseTemperaturesRaises(self):

    def make_kernel_fn(target_log_prob_fn):
      return tfp.mcmc.RandomWalkMetropolis(
          target_log_prob_fn=target_log_prob_fn,
          new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=0.1))

    with self.subTest('Nothing raised if valid (even with validate_args)'):
      tfp.mcmc.ReplicaExchangeMC(
          target_log_prob_fn=None,
          untempered_log_prob_fn=lambda x: 1.0,  # provided
          tempered_log_prob_fn=lambda x: 1.0,  # provided
          inverse_temperatures=[1., 0.5],
          make_kernel_fn=make_kernel_fn,
          validate_args=True,
      )

    with self.subTest('Negative temperatures not allowed'):
      with self.assertRaisesError('must be non-negative'):
        tfp.mcmc.ReplicaExchangeMC(
            target_log_prob_fn=None,
            untempered_log_prob_fn=lambda x: 1.0,  # provided
            tempered_log_prob_fn=lambda x: 1.0,  # provided
            inverse_temperatures=[-1., -0.5],
            make_kernel_fn=make_kernel_fn,
            validate_args=True,
        )

    with self.subTest('Zero temperatures not allowed without split tempering'):
      with self.assertRaisesError('must be positive'):
        tfp.mcmc.ReplicaExchangeMC(
            target_log_prob_fn=lambda x: 1.0,  # provided
            untempered_log_prob_fn=None,
            tempered_log_prob_fn=None,
            inverse_temperatures=[1., 0.5, 0.0],
            make_kernel_fn=make_kernel_fn,
            validate_args=True,
        )

  def testWithUntemperedLPemperatureGapNearOne(self):
    inverse_temperatures = tf.convert_to_tensor([1., 0.0005, 0.])

    if tf.executing_eagerly():
      num_results = 50
    else:
      num_results = 1000

    results = self.checkAndMakeResultsForTestingUntemperedLogProbFn(
        likelihood_variance=tf.convert_to_tensor([0.05] * 4),
        prior_variance=tf.convert_to_tensor([1.] * 4),
        inverse_temperatures=inverse_temperatures,
        num_results=num_results,
    )

    # Temperatures 0 and 1 are widely separated, so don't expect any swapping.
    self.assertLess(results['conditional_swap_prob'][0], 0.05)

    # Temperatures 1 and 2 are close, so they should swap.
    self.assertGreater(results['conditional_swap_prob'][1], 0.95)

  def testWithUntemperedLPTemperatureGapNearZero(self):
    inverse_temperatures = tf.convert_to_tensor([1., 0.9999, 0.0])

    if tf.executing_eagerly():
      num_results = 50
    else:
      num_results = 10000

    n_events = 5

    results = self.checkAndMakeResultsForTestingUntemperedLogProbFn(
        likelihood_variance=tf.convert_to_tensor([0.05] * n_events),
        prior_variance=tf.convert_to_tensor([1.] * n_events),
        inverse_temperatures=inverse_temperatures,
        num_results=num_results,
    )

    # Temperatures 0 and 1 are close, so usually swap.
    self.assertGreater(results['conditional_swap_prob'][0], 0.9)

    # Temperatures 1 and 2 are far apart, so seldom swap.
    self.assertLess(results['conditional_swap_prob'][1], 0.05)

  def testWithUntemperedLPGeometricTemperaturesGivesReasonableSwaps(self):
    inverse_temperatures = tf.convert_to_tensor([1., 0.5, 0.25])

    if tf.executing_eagerly():
      num_results = 50
      tol_multiplier = 25
      expected_max_conditional_swap_prob = 0.99
    else:
      num_results = 1000
      tol_multiplier = 5
      expected_max_conditional_swap_prob = 0.8

    self.checkAndMakeResultsForTestingUntemperedLogProbFn(
        likelihood_variance=tf.convert_to_tensor([0.25] * 4),
        prior_variance=tf.convert_to_tensor([1.] * 4),
        inverse_temperatures=inverse_temperatures,
        num_results=num_results,
        # The temperatures were dispersed geometrically, so expect a reasonable
        # range of conditional swap probs.
        expected_min_conditional_swap_prob=0.2,
        expected_max_conditional_swap_prob=expected_max_conditional_swap_prob,
        tol_multiplier=tol_multiplier,
    )

  def testWithUntemperedLPGeometricAndStateParts(self):
    inverse_temperatures = tf.convert_to_tensor([1., 0.5, 0.25])

    # The tolerance on these is not very tight...
    # However, I (langmore) tried running these with num_results much higher,
    # and we get progressively better results...so the algorithm is correct, but
    # the sampling is inefficient (no preconditioning).
    if tf.executing_eagerly():
      num_results = 50
      tol_multiplier = 25
      expected_max_conditional_swap_prob = 0.99
    else:
      num_results = 5000
      tol_multiplier = 30
      expected_max_conditional_swap_prob = 0.9

    self.checkAndMakeResultsForTestingUntemperedLogProbFn(
        likelihood_variance=[
            tf.convert_to_tensor([0.25] * 2),
            tf.convert_to_tensor([0.45] * 2),
        ],
        prior_variance=[
            tf.convert_to_tensor([0.5] * 2),
            tf.convert_to_tensor([1.0] * 2),
        ],
        inverse_temperatures=inverse_temperatures,
        num_results=num_results,
        # The temperatures were dispersed geometrically, so expect a reasonable
        # range of conditional swap probs.
        expected_min_conditional_swap_prob=0.2,
        expected_max_conditional_swap_prob=expected_max_conditional_swap_prob,
        tol_multiplier=tol_multiplier,
    )

  @test_util.numpy_disable_gradient_test('HMC')
  def checkAndMakeResultsForTestingUntemperedLogProbFn(
      self,
      likelihood_variance,
      prior_variance,
      inverse_temperatures,
      num_results,
      expected_min_conditional_swap_prob=None,
      expected_max_conditional_swap_prob=None,
      tol_multiplier=10,
  ):
    """Make dictionary of results and do some standard checks."""
    # With an untempered log prob fn we can use beta = 0. This unfortunately
    # requires a different setup in terms of step size than the usual tests, and
    # we expect slightly different results. So we made our own special test.
    inverse_temperatures = tf.convert_to_tensor(inverse_temperatures)
    num_replica = inverse_temperatures.shape[0]
    dtype = inverse_temperatures.dtype

    is_list_like = isinstance(likelihood_variance, (list, tuple))
    # If is_list_like, we will concatenate tensors along the way.
    # The log prob fns we build will do this concatenation to inputs
    # The "theoretical" outputs will be concatenated.
    # It's up to the user to split back into "parts" if required.

    if is_list_like:
      assert isinstance(prior_variance, (list, tuple))
      assert [len(v.shape) == 1 for v in likelihood_variance]
      assert [len(v.shape) == 1 for v in prior_variance]
      n_events = [v.shape[-1] for v in likelihood_variance]
      likelihood_variance = tf.concat(likelihood_variance, axis=-1)
      prior_variance = tf.concat(prior_variance, axis=-1)
      current_state = [tf.zeros((num_replica, n)) for n in n_events]
    else:
      assert len(likelihood_variance.shape) == 1
      assert len(prior_variance.shape) == 1
      n_events = likelihood_variance.shape[-1]
      likelihood_variance = tf.convert_to_tensor(likelihood_variance,
                                                 dtype=dtype)
      prior_variance = tf.convert_to_tensor(prior_variance, dtype=dtype)
      current_state = tf.zeros((num_replica, n_events))

    likelihood = tfd.MultivariateNormalDiag(
        scale_diag=tf.sqrt(likelihood_variance))
    prior = tfd.MultivariateNormalDiag(scale_diag=tf.sqrt(prior_variance))

    def likelihood_log_prob(*x):
      if is_list_like:
        x = tf.concat(x, axis=-1)
      else:
        x = x[0]
      return likelihood.log_prob(x)

    def prior_log_prob(*x):
      if is_list_like:
        x = tf.concat(x, axis=-1)
      else:
        x = x[0]
      return prior.log_prob(x)

    # Precision is inverse of covariance. The (untempered) prior and
    # (tempered) likelihood precisions add.
    replica_precision = tf.linalg.LinearOperatorDiag(tf.stack([
        beta / likelihood_variance + 1. / prior_variance
        for beta in tf.unstack(inverse_temperatures)
    ]))
    replica_covariance = replica_precision.inverse()

    replica_smallest_scale_length = tf.reduce_min(
        tf.sqrt(replica_covariance.diag_part()), axis=-1)
    step_size = 0.41234 * replica_smallest_scale_length[:, tf.newaxis]
    num_leapfrog_steps = tf.cast(
        tf.math.ceil(1.75 / tf.reduce_min(step_size)), tf.int32)

    def make_kernel_fn(target_log_prob_fn):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          step_size=step_size,
          num_leapfrog_steps=num_leapfrog_steps,
      )

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=None,
        untempered_log_prob_fn=prior_log_prob,
        tempered_log_prob_fn=likelihood_log_prob,
        inverse_temperatures=inverse_temperatures,
        state_includes_replicas=True,
        make_kernel_fn=make_kernel_fn,
        swap_proposal_fn=tfp.mcmc.even_odd_swap_proposal_fn(1.),
    )

    def trace_fn(state, results):  # pylint: disable=unused-argument
      return {
          'replica_log_accept_ratio':
              unnest.get_innermost(results.post_swap_replica_results,
                                   'log_accept_ratio'),
          'is_swap_accepted_adjacent':
              results.is_swap_accepted_adjacent,
          'is_swap_proposed_adjacent':
              results.is_swap_proposed_adjacent,
      }

    replica_states, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        # Start at one of the modes, in order to make mode jumping necessary
        # if we want to pass test.
        current_state=current_state,
        kernel=remc,
        num_burnin_steps=50,
        trace_fn=trace_fn,
        seed=test_util.test_seed())

    # Make sure replicas are swapping. If they are not, then the individual
    # replicas will converge even if the swap probabilities are incorrect.
    conditional_swap_prob = (
        tf.reduce_sum(
            tf.cast(trace['is_swap_accepted_adjacent'], dtype), axis=0) /
        tf.reduce_sum(
            tf.cast(trace['is_swap_proposed_adjacent'], dtype), axis=0))
    replica_prob_accept = tf.reduce_mean(
        tf.exp(tf.minimum(trace['replica_log_accept_ratio'], 0.)), axis=0)

    if is_list_like:
      replica_states = tf.concat(replica_states, axis=-1)

    results = self.evaluate({
        'replica_states':
            replica_states,
        'replica_prob_accept':
            replica_prob_accept,
        'conditional_swap_prob':
            conditional_swap_prob,
        'theoretical_replica_mean':
            tf.zeros_like(replica_covariance.diag_part()),
        'theoretical_replica_variance':
            replica_covariance.diag_part(),
        'sample_replica_variance':
            tfp.stats.variance(replica_states, sample_axis=0),
        'sample_replica_mean':
            tf.reduce_mean(replica_states, axis=0),
        'replica_ess':
            effective_sample_size(replica_states),
    })

    results['min_ess'] = np.min(results['replica_ess'])

    # All replicas should be mixing well, individually.
    self.assertGreater(
        results['min_ess'],
        num_results / 10,
        msg='Bad sampling found!')
    self.assertAllGreater(results['replica_prob_accept'], 0.6)

    # If replicas are not swapping properly, other results may look nice
    # even if swap probabilities are incorrect.
    if expected_min_conditional_swap_prob is not None:
      self.assertAllGreater(results['conditional_swap_prob'],
                            expected_min_conditional_swap_prob)
    if expected_max_conditional_swap_prob is not None:
      self.assertAllLess(results['conditional_swap_prob'],
                         expected_max_conditional_swap_prob)

    mean_atol = tol_multiplier / np.sqrt(results['min_ess'])
    cov_rtol = tol_multiplier * np.sqrt(2) / np.sqrt(results['min_ess'])

    self.assertAllClose(
        results['sample_replica_variance'],
        results['theoretical_replica_variance'],
        rtol=cov_rtol)

    self.assertAllClose(
        results['sample_replica_mean'],
        results['theoretical_replica_mean'],
        atol=mean_atol)

    return results


if __name__ == '__main__':
  tf.test.main()
