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
"""Tests for `TransformedTransitionKernel` `TransitionKernel`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
tfb = tfp.bijectors


FakeInnerKernelResults = collections.namedtuple(
    'FakeInnerKernelResults', [])


class FakeInnerKernel(tfp.mcmc.TransitionKernel):
  """Fake Transition Kernel."""

  def __init__(self, target_log_prob_fn):
    self._parameters = dict(target_log_prob_fn=target_log_prob_fn)

  @property
  def parameters(self):
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  def one_step(self, current_state, previous_kernel_results):
    pass

  def bootstrap_results(self, init_state):
    return FakeInnerKernelResults()


class TransformedTransitionKernelTest(tf.test.TestCase):

  def setUp(self):
    self.dtype = np.float32

  def test_support_works_correctly_with_HMC(self):
    num_results = 2000
    with self.cached_session(graph=tf.Graph()) as sess:
      target = tfd.Beta(
          concentration1=self.dtype(1.),
          concentration0=self.dtype(10.))
      transformed_hmc = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=target.log_prob,
              step_size=1.64,
              num_leapfrog_steps=2,
              seed=55),
          bijector=tfb.Sigmoid())
      # Recall, tfp.mcmc.sample_chain calls
      # transformed_hmc.bootstrap_results too.
      states, kernel_results = tfp.mcmc.sample_chain(
          num_results=num_results,
          # The initial state is used by inner_kernel.bootstrap_results.
          # Note the input is *after* bijector.forward.
          current_state=self.dtype(0.25),
          kernel=transformed_hmc,
          num_burnin_steps=200,
          num_steps_between_results=1,
          parallel_iterations=1)
      self.assertEqual(num_results, states.shape[0].value)
      sample_mean = tf.reduce_mean(states, axis=0)
      sample_var = tf.reduce_mean(
          tf.squared_difference(states, sample_mean),
          axis=0)
      [
          sample_mean_,
          sample_var_,
          is_accepted_,
          true_mean_,
          true_var_,
      ] = sess.run([
          sample_mean,
          sample_var,
          kernel_results.inner_results.is_accepted,
          target.mean(),
          target.variance(),
      ])
      self.assertAllClose(true_mean_, sample_mean_,
                          atol=0.06, rtol=0.)
      self.assertAllClose(true_var_, sample_var_,
                          atol=0.01, rtol=0.1)
      self.assertNear(0.6, is_accepted_.mean(), err=0.05)

  def test_support_works_correctly_with_MALA(self):
    num_results = 2000
    with self.cached_session(graph=tf.Graph()) as sess:
      target = tfd.Beta(
          concentration1=self.dtype(1.),
          concentration0=self.dtype(10.))
      transformed_mala = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
              target_log_prob_fn=target.log_prob,
              step_size=1.,
              seed=55),
          bijector=tfb.Sigmoid())
      # Recall, tfp.mcmc.sample_chain calls
      # transformed_hmc.bootstrap_results too.
      states, _ = tfp.mcmc.sample_chain(
          num_results=num_results,
          # The initial state is used by inner_kernel.bootstrap_results.
          # Note the input is *after* bijector.forward.
          current_state=self.dtype(0.25),
          kernel=transformed_mala,
          num_burnin_steps=200,
          num_steps_between_results=1,
          parallel_iterations=1)
      self.assertEqual(num_results, states.shape[0].value)
      sample_mean = tf.reduce_mean(states, axis=0)
      sample_var = tf.reduce_mean(
          tf.squared_difference(states, sample_mean),
          axis=0)
      [
          sample_mean_,
          sample_var_,
          true_mean_,
          true_var_,
      ] = sess.run([
          sample_mean,
          sample_var,
          target.mean(),
          target.variance(),
      ])
      self.assertAllClose(true_mean_, sample_mean_,
                          atol=0.06, rtol=0.)
      self.assertAllClose(true_var_, sample_var_,
                          atol=0.01, rtol=0.1)

  def test_support_works_correctly_with_RWM(self):
    num_results = 2000
    with self.cached_session(graph=tf.Graph()) as sess:
      target = tfd.Beta(
          concentration1=self.dtype(1.),
          concentration0=self.dtype(10.))
      transformed_rwm = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=tfp.mcmc.RandomWalkMetropolis(
              target_log_prob_fn=target.log_prob,
              new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=1.5),
              seed=55),
          bijector=tfb.Sigmoid())
      # Recall, tfp.mcmc.sample_chain calls
      # transformed_hmc.bootstrap_results too.
      states, _ = tfp.mcmc.sample_chain(
          num_results=num_results,
          # The initial state is used by inner_kernel.bootstrap_results.
          # Note the input is *after* bijector.forward.
          current_state=self.dtype(0.25),
          kernel=transformed_rwm,
          num_burnin_steps=200,
          num_steps_between_results=1,
          parallel_iterations=1)
      self.assertEqual(num_results, states.shape[0].value)
      sample_mean = tf.reduce_mean(states, axis=0)
      sample_var = tf.reduce_mean(
          tf.squared_difference(states, sample_mean),
          axis=0)
      [
          sample_mean_,
          sample_var_,
          true_mean_,
          true_var_,
      ] = sess.run([
          sample_mean,
          sample_var,
          target.mean(),
          target.variance(),
      ])
      self.assertAllClose(true_mean_, sample_mean_,
                          atol=0.06, rtol=0.)
      self.assertAllClose(true_var_, sample_var_,
                          atol=0.01, rtol=0.1)

  def test_end_to_end_works_correctly(self):
    true_mean = self.dtype([0, 0])
    true_cov = self.dtype([[1, 0.5],
                           [0.5, 1]])
    num_results = 2000
    counter = collections.Counter()
    with self.cached_session(graph=tf.Graph()) as sess:
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

      transformed_hmc = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=target_log_prob,
              # Affine scaling means we have to change the step_size
              # in order to get 60% acceptance, as was done in mcmc/hmc_test.py.
              step_size=[1.23 / 0.75, 1.23 / 0.5],
              num_leapfrog_steps=2,
              seed=54),
          bijector=[
              tfb.AffineScalar(scale=0.75),
              tfb.AffineScalar(scale=0.5),
          ])
      # Recall, tfp.mcmc.sample_chain calls
      # transformed_hmc.bootstrap_results too.
      states, kernel_results = tfp.mcmc.sample_chain(
          num_results=num_results,
          # The initial state is used by inner_kernel.bootstrap_results.
          # Note the input is *after* `bijector.forward`.
          current_state=[self.dtype(-2), self.dtype(2)],
          kernel=transformed_hmc,
          num_burnin_steps=200,
          num_steps_between_results=1,
          parallel_iterations=1)
      self.assertAllEqual(dict(target_calls=2), counter)
      states = tf.stack(states, axis=-1)
      self.assertEqual(num_results, states.shape[0].value)
      sample_mean = tf.reduce_mean(states, axis=0)
      x = states - sample_mean
      sample_cov = tf.matmul(x, x, transpose_a=True) / self.dtype(num_results)
      [sample_mean_, sample_cov_, is_accepted_] = sess.run([
          sample_mean, sample_cov, kernel_results.inner_results.is_accepted])
      self.assertNear(0.6, is_accepted_.mean(), err=0.05)
      self.assertAllClose(true_mean, sample_mean_,
                          atol=0.06, rtol=0.)
      self.assertAllClose(true_cov, sample_cov_,
                          atol=0., rtol=0.1)

  def test_bootstrap_requires_xor_args(self):
    def fake_target_log_prob(x):
      return -x**2 / 2.

    transformed_fake = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=FakeInnerKernel(target_log_prob_fn=fake_target_log_prob),
        bijector=tfb.Exp())
    with self.assertRaisesWithPredicateMatch(
        ValueError, r'Must specify exactly one'):
      transformed_fake.bootstrap_results()
    with self.assertRaisesWithPredicateMatch(
        ValueError, r'Must specify exactly one'):
      transformed_fake.bootstrap_results(
          init_state=2., transformed_init_state=np.log(2.))

  def test_bootstrap_correctly_untransforms(self):
    def fake_target_log_prob(x):
      return -x**2 / 2.

    transformed_fake = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=FakeInnerKernel(target_log_prob_fn=fake_target_log_prob),
        bijector=tfb.Exp())
    with self.cached_session(graph=tf.Graph()) as sess:
      [
          automatic_pkr,
          manual_pkr,
      ] = sess.run([
          transformed_fake.bootstrap_results(2.),
          transformed_fake.bootstrap_results(transformed_init_state=[4., 5.]),
      ])
      self.assertNear(np.log(2.), automatic_pkr.transformed_state, err=1e-6)
      self.assertAllClose(
          [4., 5.], manual_pkr.transformed_state, atol=0., rtol=1e-6)


if __name__ == '__main__':
  tf.test.main()
