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

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions


def _set_seed(seed):
  """Helper which uses graph seed if using TFE."""
  # TODO(b/68017812): Deprecate once TFE supports seed.
  if tf.executing_eagerly():
    return None
  return seed


@test_util.run_all_in_graph_and_eager_modes
class DefaultExchangeProposedFnTest(tf.test.TestCase):

  def setUp(self):
    tf.compat.v1.set_random_seed(123)

  def generate_exchanges(self, exchange_proposed_fn, num_replica, seed):

    def _scan_fn(*_):
      exchange = exchange_proposed_fn(num_replica, seed)
      flat_replicas = tf.reshape(exchange, [-1])
      with tf.control_dependencies([
          tf.compat.v1.assert_equal(
              tf.size(input=flat_replicas),
              tf.size(input=tf.unique(flat_replicas)[0])),
          tf.compat.v1.assert_greater_equal(flat_replicas, 0),
          tf.compat.v1.assert_less(flat_replicas, num_replica),
      ]):
        return tf.shape(input=exchange)[0]

    return self.evaluate(
        tf.scan(_scan_fn, tf.range(1000), initializer=0, parallel_iterations=1))

  def testProbExchange0p5NumReplica2(self):
    prob_exchange = 0.5
    num_replica = 2
    fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)

    exchanges_lens = self.generate_exchanges(
        fn, num_replica=num_replica, seed=_set_seed(123))

    # All exchanges_lens, if proposed, will be 1.
    self.assertAllClose(
        prob_exchange, np.mean([e == 1 for e in exchanges_lens]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange, np.mean([e == 0 for e in exchanges_lens]), atol=0.05)

  def testProbExchange0p5NumReplica4(self):
    prob_exchange = 0.5
    num_replica = 4
    fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)

    exchanges_lens = self.generate_exchanges(
        fn, num_replica=num_replica, seed=_set_seed(312))

    # No exchanges_lens 1 - prob_exchange of the time.
    self.assertAllClose(
        1 - prob_exchange, np.mean([e == 0 for e in exchanges_lens]), atol=0.05)

    # All exchanges_lens, if proposed, will be 0 or 1.
    self.assertAllClose(
        prob_exchange / 2, np.mean([e == 1 for e in exchanges_lens]), atol=0.05)
    self.assertAllClose(
        prob_exchange / 2, np.mean([e == 2 for e in exchanges_lens]), atol=0.05)

  def testProbExchange0p5NumReplica3(self):
    prob_exchange = 0.5
    num_replica = 3
    fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)

    exchanges_lens = self.generate_exchanges(
        fn, num_replica=num_replica, seed=_set_seed(42))

    # All exchanges_lens, if proposed, will be 1.
    self.assertAllClose(
        prob_exchange, np.mean([e == 1 for e in exchanges_lens]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange, np.mean([e == 0 for e in exchanges_lens]), atol=0.05)

  def testProbExchange0p5NumReplica5(self):
    prob_exchange = 0.5
    num_replica = 5
    fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)

    exchanges_lens = self.generate_exchanges(
        fn, num_replica=num_replica, seed=_set_seed(1))

    # All exchanges_lens, if proposed, will be 2.
    self.assertAllClose(
        prob_exchange, np.mean([e == 2 for e in exchanges_lens]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange, np.mean([e == 0 for e in exchanges_lens]), atol=0.05)

  def testProbExchange1p0(self):
    prob_exchange = 1.0
    num_replica = 15
    fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)

    exchanges_lens = self.generate_exchanges(
        fn, num_replica=num_replica, seed=_set_seed(667))

    # All exchanges_lens, if proposed, will be 7.  And prob_exchange is 1.
    self.assertAllClose(
        prob_exchange, np.mean([e == 7 for e in exchanges_lens]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange, np.mean([e == 0 for e in exchanges_lens]), atol=0.05)

  def testProbExchange0p0(self):
    prob_exchange = 0.0
    num_replica = 15
    fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)

    exchanges_lens = self.generate_exchanges(
        fn, num_replica=num_replica, seed=_set_seed(665))

    # All exchanges_lens, if proposed, will be 7.  And prob_exchange is 0.
    self.assertAllClose(
        prob_exchange, np.mean([e == 7 for e in exchanges_lens]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange, np.mean([e == 0 for e in exchanges_lens]), atol=0.05)


@test_util.run_all_in_graph_and_eager_modes
class REMCTest(tf.test.TestCase):

  def setUp(self):
    tf.compat.v1.set_random_seed(123)

  def _getNormalREMCSamples(self,
                            inverse_temperatures,
                            num_results=1000,
                            step_size=1.,
                            dtype=np.float32):
    """Sampling from standard normal with REMC."""

    target = tfd.Normal(dtype(0.), dtype(1.))

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=step_size,
          num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tf.function(target.log_prob, autograph=False),
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed(1))

    samples = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=target.sample(seed=_set_seed(1)),
        kernel=remc,
        num_burnin_steps=50,
        trace_fn=None,
        parallel_iterations=1)  # For determinism.

    self.assertAllEqual((num_results,), samples.shape)

    return self.evaluate(samples)

  def testNormalOddNumReplicas(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(
        inverse_temperatures=[1., 0.3, 0.1, 0.03, 0.01])

    self.assertAllClose(samps_.mean(), 0., atol=0.1, rtol=0.1)
    self.assertAllClose(samps_.std(), 1., atol=0.1, rtol=0.1)

  def testNormalEvenNumReplicas(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(
        inverse_temperatures=[1., 0.9, 0.8, 0.7],)

    self.assertAllClose(samps_.mean(), 0., atol=0.1, rtol=0.1)
    self.assertAllClose(samps_.std(), 1., atol=0.1, rtol=0.1)

  def testNormalOddNumReplicasLowTolerance(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(
        inverse_temperatures=[1., 0.3, 0.1, 0.03, 0.01], num_results=500)

    self.assertAllClose(samps_.mean(), 0., atol=0.3, rtol=0.1)
    self.assertAllClose(samps_.std(), 1., atol=0.3, rtol=0.1)

  def testNormalEvenNumReplicasLowTolerance(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(
        inverse_temperatures=[1., 0.9, 0.8, 0.7], num_results=500)

    self.assertAllClose(samps_.mean(), 0., atol=0.3, rtol=0.1)
    self.assertAllClose(samps_.std(), 1., atol=0.3, rtol=0.1)

  def testNormalHighTemperatureOnlyHasLargerStddev(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(
        inverse_temperatures=[0.2], step_size=3.)

    self.assertAllClose(samps_.mean(), 0., atol=0.2, rtol=0.1)
    self.assertGreater(samps_.std(), 2.)

  def testNormalLowTemperatureOnlyHasSmallerStddev(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(
        inverse_temperatures=[6.0], step_size=0.5)

    self.assertAllClose(samps_.mean(), 0., atol=0.2, rtol=0.1)
    self.assertLess(samps_.std(), 0.6)

  def testRWM2DMixNormal(self):
    """Sampling from a 2-D Mixture Normal Distribution."""
    dtype = np.float32

    # By symmetry, target has mean [0, 0]
    # Therefore, Var = E[X^2] = E[E[X^2 | c]], where c is the component.
    # Now..., for the first component,
    #   E[X1^2] =  Var[X1] + Mean[X1]^2
    #           =  0.3^2 + 1^2,
    # and similarly for the second.  As a result,
    # Var[mixture] = 1.09.
    target = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[-1., -1], [1., 1.]],
            scale_identity_multiplier=[0.3, 0.3]))

    inverse_temperatures = 10.**tf.linspace(0., -2., 4)
    step_sizes = tf.constant([0.3, 0.6, 1.2, 2.4])
    def make_kernel_fn(target_log_prob_fn, seed):
      kernel = tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=step_sizes[make_kernel_fn.idx],
          num_leapfrog_steps=2)
      make_kernel_fn.idx += 1
      return kernel
    # TODO(b/124770732): Remove this hack.
    make_kernel_fn.idx = 0

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tf.function(target.log_prob, autograph=False),
        # Verified that test fails if inverse_temperatures = [1.]
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed(888))

    def _trace_log_accept_ratio(state, results):
      del state
      return [r.log_accept_ratio for r in results.sampled_replica_results]

    num_results = 1000
    samples, log_accept_ratios = tfp.mcmc.sample_chain(
        num_results=num_results,
        # Start at one of the modes, in order to make mode jumping necessary
        # if we want to pass test.
        current_state=np.ones(2, dtype=dtype),
        kernel=remc,
        num_burnin_steps=500,
        trace_fn=_trace_log_accept_ratio,
        parallel_iterations=1)  # For determinism.
    self.assertAllEqual((num_results, 2), samples.shape)
    log_accept_ratios = [
        tf.reduce_mean(input_tensor=tf.exp(tf.minimum(0., lar)))
        for lar in log_accept_ratios
    ]

    sample_mean = tf.reduce_mean(input_tensor=samples, axis=0)
    sample_std = tf.sqrt(
        tf.reduce_mean(
            input_tensor=tf.math.squared_difference(samples, sample_mean),
            axis=0))
    [sample_mean_, sample_std_, log_accept_ratios_] = self.evaluate(
        [sample_mean, sample_std, log_accept_ratios])
    tf.compat.v1.logging.vlog(1, 'log_accept_ratios: %s  eager: %s',
                              log_accept_ratios_, tf.executing_eagerly())

    self.assertAllClose(sample_mean_, [0., 0.], atol=0.3, rtol=0.3)
    self.assertAllClose(
        sample_std_, [np.sqrt(1.09), np.sqrt(1.09)], atol=0.1, rtol=0.1)

  def testMultipleCorrelatedStatesWithNoBatchDims(self):

    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    # Use LinearOperatorLowerTriangular to get broadcasting ability.
    linop = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(true_cov))
    num_results = 1000

    def target_log_prob(x, y):
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      xy = tf.stack([x, y], axis=-1) - true_mean
      z = linop.solvevec(xy)
      return -0.5 * tf.reduce_sum(input_tensor=z**2., axis=-1)

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=[0.5, 0.5],
          num_leapfrog_steps=5)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tf.function(target_log_prob, autograph=False),
        inverse_temperatures=[1., 0.9, 0.8],
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed(3))

    states = tfp.mcmc.sample_chain(
        num_results=num_results,
        # batch_shape = [] for each initial state
        current_state=[1., 1.],
        kernel=remc,
        num_burnin_steps=100,
        trace_fn=None,
        parallel_iterations=1)  # For determinism.
    self.assertAllEqual((num_results,), states[0].shape)
    self.assertAllEqual((num_results,), states[1].shape)

    states = tf.stack(states, axis=-1)
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(input_tensor=states, axis=0)
    x = states - sample_mean
    sample_cov = tf.matmul(x, x, transpose_a=True) / dtype(num_results)
    sample_mean_, sample_cov_ = self.evaluate([sample_mean, sample_cov])
    self.assertAllClose(true_mean, sample_mean_, atol=0.06, rtol=0.)
    self.assertAllClose(true_cov, sample_cov_, atol=0., rtol=0.2)

  def testNormalWithTwoBatchDimsAndThreeReplicas(self):
    """Sampling from the Standard Normal Distribution."""
    # Small scale and well-separated modes mean we need replica exchange to
    # work or else tests fail.
    loc = np.array(
        [
            # Use 3-D normals, ensuring batch and event sizes don't broadcast.
            [-1., -1., -1.],  # loc of first batch member
            [1., 1., 1.]  # loc of second batch member
        ],
        dtype=np.float32)
    scale_identity_multiplier = [0.5, 0.8]
    target = tfd.MultivariateNormalDiag(
        loc=loc, scale_identity_multiplier=scale_identity_multiplier)

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=0.15,
          num_leapfrog_steps=5)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=tf.function(
            lambda x: target.copy().log_prob(x), autograph=False),
        inverse_temperatures=[1., 0.9, 0.8],
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed(700))

    num_results = 500
    states = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=loc[::-1],  # Batch members start at wrong mode!
        kernel=remc,
        num_burnin_steps=50,
        trace_fn=None,
        parallel_iterations=1)  # For determinism.

    self.assertAllEqual((num_results, 2, 3), states.shape)
    states_ = self.evaluate(states)

    self.assertAllClose(loc, states_.mean(axis=0), rtol=0.2)
    self.assertAllClose(
        [[0.5**2, 0., 0.], [0., 0.5**2, 0.], [0., 0., 0.5**2]],
        np.cov(states_[:, 0, :], rowvar=False),
        atol=0.2)
    self.assertAllClose(
        [[0.8**2, 0., 0.], [0., 0.8**2, 0.], [0., 0., 0.8**2]],
        np.cov(states_[:, 1, :], rowvar=False),
        atol=0.2)

  def testMultipleCorrelatedStatesWithFourBatchDims(self):

    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    # Use LinearOperatorLowerTriangular to get broadcasting ability.
    linop = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(true_cov))
    num_results = 1000

    def target_log_prob(x, y):
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      xy = tf.stack([x, y], axis=-1) - true_mean
      z = linop.solvevec(xy)
      return -0.5 * tf.reduce_sum(input_tensor=z**2., axis=-1)

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
        seed=_set_seed(96))

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

    states_ = self.evaluate(states)

    self.assertAllClose(true_mean, states_[:, 0, :].mean(axis=0), atol=0.1)
    self.assertAllClose(true_mean, states_[:, 1, :].mean(axis=0), atol=0.1)
    self.assertAllClose(
        true_cov, np.cov(states_[:, 0, :], rowvar=False), atol=0.2)
    self.assertAllClose(
        true_cov, np.cov(states_[:, 1, :], rowvar=False), atol=0.2)

  def testInverseTemperaturesValueError(self):
    """Using invalid `inverse_temperatures`."""
    if tf.executing_eagerly(): return
    dtype = np.float32

    with self.assertRaisesRegexp(ValueError, 'not fully defined'):
      target = tfd.Normal(loc=dtype(0), scale=dtype(1))

      def make_kernel_fn(target_log_prob_fn, seed):
        return tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            seed=seed,
            step_size=1.0,
            num_leapfrog_steps=3)

      tfp.mcmc.ReplicaExchangeMC(
          target_log_prob_fn=tf.function(target.log_prob, autograph=False),
          inverse_temperatures=10.**tf.linspace(
              0., -2., tf.random.uniform([], maxval=10, dtype=tf.int32)),
          make_kernel_fn=make_kernel_fn,
          seed=_set_seed(13))


if __name__ == '__main__':
  tf.test.main()
