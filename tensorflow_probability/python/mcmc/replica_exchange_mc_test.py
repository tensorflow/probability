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

tfd = tfp.distributions
tfe = tf.contrib.eager


def _set_seed(seed):
  """Helper which uses graph seed if using TFE."""
  # TODO(b/68017812): Deprecate once TFE supports seed.
  if tf.executing_eagerly():
    tf.set_random_seed(seed)
    return None
  return seed


class DefaultExchangeProposedFnTest(tf.test.TestCase):

  def generate_exchanges(self, exchange_proposed, num_replica):
    exchanges = []
    for _ in range(1000):
      exchange_ = self.evaluate(exchange_proposed)
      self.assertEqual(len(exchange_.ravel()), len(np.unique(exchange_)))
      self.assertContainsSubset(exchange_.ravel(), list(range(num_replica)))
      exchanges.append(exchange_)
    return exchanges

  def testProbExchange0p5NumReplica2(self):
    prob_exchange = 0.5
    num_replica = 2
    fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)
    exchange_proposed = fn(num_replica, seed=_set_seed(123))

    exchanges = self.generate_exchanges(exchange_proposed, num_replica)

    # All exchanges, if proposed, will be length 1.
    self.assertAllClose(
        prob_exchange, np.mean([len(e) == 1 for e in exchanges]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange,
        np.mean([len(e) == 0 for e in exchanges]),  # pylint: disable=g-explicit-length-test
        atol=0.05)

  def testProbExchange0p5NumReplica4(self):
    with self.cached_session():
      prob_exchange = 0.5
      num_replica = 4
      fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)
      exchange_proposed = fn(num_replica, seed=_set_seed(321))

      exchanges = self.generate_exchanges(exchange_proposed, num_replica)

    # No exchanges 1 - prob_exchange of the time.
    self.assertAllClose(
        1 - prob_exchange,
        np.mean([len(e) == 0 for e in exchanges]),  # pylint: disable=g-explicit-length-test
        atol=0.05)

    # All exchanges, if proposed, will be length 0 or 1.
    self.assertAllClose(
        prob_exchange / 2, np.mean([len(e) == 1 for e in exchanges]), atol=0.05)
    self.assertAllClose(
        prob_exchange / 2, np.mean([len(e) == 2 for e in exchanges]), atol=0.05)

  def testProbExchange0p5NumReplica3(self):
    with self.cached_session():
      prob_exchange = 0.5
      num_replica = 3
      fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)
      exchange_proposed = fn(num_replica, seed=_set_seed(42))

      exchanges = self.generate_exchanges(exchange_proposed, num_replica)

    # All exchanges, if proposed, will be length 1.
    self.assertAllClose(
        prob_exchange, np.mean([len(e) == 1 for e in exchanges]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange,
        np.mean([len(e) == 0 for e in exchanges]),  # pylint: disable=g-explicit-length-test
        atol=0.05)

  def testProbExchange0p5NumReplica5(self):
    with self.cached_session():
      prob_exchange = 0.5
      num_replica = 5
      fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)
      exchange_proposed = fn(num_replica, seed=_set_seed(0))

      exchanges = self.generate_exchanges(exchange_proposed, num_replica)

    # All exchanges, if proposed, will be length 2.
    self.assertAllClose(
        prob_exchange, np.mean([len(e) == 2 for e in exchanges]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange,
        np.mean([len(e) == 0 for e in exchanges]),  # pylint: disable=g-explicit-length-test
        atol=0.05)

  def testProbExchange1p0(self):
    with self.cached_session():
      prob_exchange = 1.0
      num_replica = 15
      fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)
      exchange_proposed = fn(num_replica, seed=_set_seed(667))

      exchanges = self.generate_exchanges(exchange_proposed, num_replica)

    # All exchanges, if proposed, will be length 7.  And prob_exchange is 1.
    self.assertAllClose(
        prob_exchange, np.mean([len(e) == 7 for e in exchanges]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange,
        np.mean([len(e) == 0 for e in exchanges]),  # pylint: disable=g-explicit-length-test
        atol=0.05)

  def testProbExchange0p0(self):
    with self.cached_session():
      prob_exchange = 0.0
      num_replica = 15
      fn = tfp.mcmc.default_exchange_proposed_fn(prob_exchange)
      exchange_proposed = fn(num_replica, seed=_set_seed(665))

      exchanges = self.generate_exchanges(exchange_proposed, num_replica)

    # All exchanges, if proposed, will be length 7.  And prob_exchange is 0.
    self.assertAllClose(
        prob_exchange, np.mean([len(e) == 7 for e in exchanges]), atol=0.05)

    self.assertAllClose(
        1 - prob_exchange,
        np.mean([len(e) == 0 for e in exchanges]),  # pylint: disable=g-explicit-length-test
        atol=0.05)


class REMCTest(tf.test.TestCase):

  def _getNormalREMCSamples(self,
                            inverse_temperatures,
                            num_results=1000,
                            dtype=np.float32):
    """Sampling from standard normal with REMC."""

    target = tfd.Normal(dtype(0.), dtype(1.))

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=0.3001,
          num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target.log_prob,
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed(0))

    samples, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=target.sample(seed=_set_seed(1)),
        kernel=remc,
        num_burnin_steps=50,
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

  @tfe.run_test_in_graph_and_eager_modes()
  def testNormalOddNumReplicasLowTolerance(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(
        inverse_temperatures=[1., 0.3, 0.1, 0.03, 0.01], num_results=100)

    self.assertAllClose(samps_.mean(), 0., atol=0.3, rtol=0.1)
    self.assertAllClose(samps_.std(), 1., atol=0.3, rtol=0.1)

  @tfe.run_test_in_graph_and_eager_modes()
  def testNormalEvenNumReplicasLowTolerance(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(
        inverse_temperatures=[1., 0.9, 0.8, 0.7], num_results=100)

    self.assertAllClose(samps_.mean(), 0., atol=0.3, rtol=0.1)
    self.assertAllClose(samps_.std(), 1., atol=0.3, rtol=0.1)

  def testNormalHighTemperatureOnlyHasLargerStddev(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(
        inverse_temperatures=[0.2], num_results=5000)

    self.assertAllClose(samps_.mean(), 0., atol=0.2, rtol=0.1)
    self.assertGreater(samps_.std(), 2.)

  def testNormalLowTemperatureOnlyHasSmallerStddev(self):
    """Sampling from the Standard Normal Distribution."""
    samps_ = self._getNormalREMCSamples(inverse_temperatures=[6.0])

    self.assertAllClose(samps_.mean(), 0., atol=0.2, rtol=0.1)
    self.assertLess(samps_.std(), 0.6)

  def testRWM2DMixNormal(self):
    """Sampling from a 2-D Mixture Normal Distribution."""
    dtype = np.float32

    # By symmetry, target has mean [0, 0]
    # Therefore, Var = E[X^2] = E[E[X^2 | c]], where c is the component.
    # Now..., for the first component,
    #   E[X1^2] =  Var[X1] + Mean[X1]^2
    #           =  0.1^2 + 1^2,
    # and similarly for the second.  As a result,
    # Var[mixture] = 1.01.
    target = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
        components_distribution=tfd.MultivariateNormalDiag(
            # Mixture components are 20 standard deviations apart!
            loc=[[-1., -1], [1., 1.]],
            scale_identity_multiplier=[0.1, 0.1]))

    def make_kernel_fn(target_log_prob_fn, seed):
      return tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          seed=seed,
          step_size=0.3,
          num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target.log_prob,
        # Verified that test fails if inverse_temperatures = [1]
        inverse_temperatures=10.**tf.linspace(0., -2., 5),
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed(888))

    samples, _ = tfp.mcmc.sample_chain(
        num_results=2000,
        # Start at one of the modes, in order to make mode jumping necessary
        # if we want to pass test.
        current_state=np.ones(2, dtype=dtype),
        kernel=remc,
        num_burnin_steps=500,
        parallel_iterations=1)  # For determinism.
    self.assertAllEqual((2000, 2), samples.shape)

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_std = tf.sqrt(
        tf.reduce_mean(tf.squared_difference(samples, sample_mean), axis=0))
    [sample_mean_, sample_std_] = self.evaluate([sample_mean, sample_std])

    self.assertAllClose(sample_mean_, [0., 0.], atol=0.3, rtol=0.3)
    self.assertAllClose(
        sample_std_, [np.sqrt(1.01), np.sqrt(1.01)], atol=0.1, rtol=0.1)

  def testMultipleCorrelatedStatesWithNoBatchDims(self):

    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    # Use LinearOperatorLowerTriangular to get broadcasting ability.
    linop = tf.linalg.LinearOperatorLowerTriangular(tf.cholesky(true_cov))
    num_results = 1000

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
          step_size=[0.5, 0.5],
          num_leapfrog_steps=5)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target_log_prob,
        inverse_temperatures=[1., 0.9, 0.8],
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed(3))

    states, unused_kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        # batch_shape = [] for each initial state
        current_state=[1., 1.],
        kernel=remc,
        num_burnin_steps=100,
        parallel_iterations=1)  # For determinism.
    self.assertAllEqual((num_results,), states[0].shape)
    self.assertAllEqual((num_results,), states[1].shape)

    states = tf.stack(states, axis=-1)
    self.assertEqual(num_results, tf.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
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
        target_log_prob_fn=target.log_prob,
        inverse_temperatures=[1., 0.9, 0.8],
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed(700))

    num_results = 500
    states, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=loc[::-1],  # Batch members start at wrong mode!
        kernel=remc,
        num_burnin_steps=50,
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
    linop = tf.linalg.LinearOperatorLowerTriangular(tf.cholesky(true_cov))
    num_results = 4000

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
          step_size=[0.5, 0.5],
          num_leapfrog_steps=5)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target_log_prob,
        inverse_temperatures=[1., 0.9, 0.8],
        make_kernel_fn=make_kernel_fn,
        seed=_set_seed(96))

    states, unused_kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        # batch_shape = [4] for each initial state
        current_state=[[1.] * 4, [1.] * 4],
        kernel=remc,
        num_burnin_steps=400,
        parallel_iterations=1)  # For determinism.

    states = tf.stack(states, axis=-1)
    self.assertAllEqual((num_results, 4, 2), states.shape)

    states_ = self.evaluate(states)

    self.assertAllClose(true_mean, states_[:, 0, :].mean(axis=0), atol=0.05)
    self.assertAllClose(true_mean, states_[:, 1, :].mean(axis=0), atol=0.05)
    self.assertAllClose(
        true_cov, np.cov(states_[:, 0, :], rowvar=False), atol=0.1)
    self.assertAllClose(
        true_cov, np.cov(states_[:, 1, :], rowvar=False), atol=0.1)

  def testInverseTemperaturesValueError(self):
    """Using invalid `inverse_temperatures`."""
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
          target_log_prob_fn=target.log_prob,
          inverse_temperatures=10.**tf.linspace(
              0., -2., tf.random_uniform([], maxval=10, dtype=tf.int32)),
          make_kernel_fn=make_kernel_fn,
          seed=_set_seed(13))


if __name__ == '__main__':
  tf.test.main()
