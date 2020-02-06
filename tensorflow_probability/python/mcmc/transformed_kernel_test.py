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
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


FakeInnerKernelResults = collections.namedtuple(
    'FakeInnerKernelResults', ['target_log_prob'])


def _maybe_seed(seed):
  if tf.executing_eagerly():
    tf.random.set_seed(seed)
    return None
  return seed


class FakeInnerKernel(tfp.mcmc.TransitionKernel):
  """Fake Transition Kernel."""

  def __init__(self, target_log_prob_fn, is_calibrated=True):
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn, is_calibrated=is_calibrated)

  @property
  def parameters(self):
    return self._parameters

  @property
  def is_calibrated(self):
    return self._parameters['is_calibrated']

  def one_step(self, current_state, previous_kernel_results):
    pass

  def bootstrap_results(self, init_state):
    return FakeInnerKernelResults(
        target_log_prob=self._parameters['target_log_prob_fn'](init_state))


@test_util.test_all_tf_execution_regimes
class TransformedTransitionKernelTest(test_util.TestCase):

  def setUp(self):
    super(TransformedTransitionKernelTest, self).setUp()
    self.dtype = np.float32

  def test_support_works_correctly_with_HMC(self):
    num_results = 2000
    target = tfd.Beta(
        concentration1=self.dtype(1.),
        concentration0=self.dtype(10.))
    transformed_hmc = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            step_size=1.64,
            num_leapfrog_steps=2,
            seed=_maybe_seed(55)),
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
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    sample_var = tf.reduce_mean(
        tf.math.squared_difference(states, sample_mean), axis=0)
    [
        sample_mean_,
        sample_var_,
        is_accepted_,
        true_mean_,
        true_var_,
    ] = self.evaluate([
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
    target = tfd.Beta(
        concentration1=self.dtype(1.),
        concentration0=self.dtype(10.))
    transformed_mala = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            step_size=1.,
            seed=_maybe_seed(55)),
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
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    sample_var = tf.reduce_mean(
        tf.math.squared_difference(states, sample_mean), axis=0)
    [
        sample_mean_,
        sample_var_,
        true_mean_,
        true_var_,
    ] = self.evaluate([
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
    target = tfd.Beta(
        concentration1=self.dtype(1.),
        concentration0=self.dtype(10.))
    transformed_rwm = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=1.5),
            seed=_maybe_seed(55)),
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
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    sample_var = tf.reduce_mean(
        tf.math.squared_difference(states, sample_mean), axis=0)
    [
        sample_mean_,
        sample_var_,
        true_mean_,
        true_var_,
    ] = self.evaluate([
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
    def target_log_prob(x, y):
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
            target_log_prob_fn=tf.function(target_log_prob, autograph=False),
            # Affine scaling means we have to change the step_size
            # in order to get 60% acceptance, as was done in mcmc/hmc_test.py.
            step_size=[1.23 / 0.75, 1.23 / 0.5],
            num_leapfrog_steps=2,
            seed=_maybe_seed(54)),
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
    states = tf.stack(states, axis=-1)
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    x = states - sample_mean
    sample_cov = tf.matmul(x, x, transpose_a=True) / self.dtype(num_results)
    [sample_mean_, sample_cov_, is_accepted_] = self.evaluate([
        sample_mean, sample_cov, kernel_results.inner_results.is_accepted])
    self.assertNear(0.6, is_accepted_.mean(), err=0.05)
    self.assertAllClose(true_mean, sample_mean_,
                        atol=0.06, rtol=0.)
    self.assertAllClose(true_cov, sample_cov_,
                        atol=0., rtol=0.16)

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
    automatic_pkr, manual_pkr = self.evaluate([
        transformed_fake.bootstrap_results(2.),
        transformed_fake.bootstrap_results(transformed_init_state=[4., 5.]),
    ])
    self.assertNear(np.log(2.), automatic_pkr.transformed_state, err=1e-6)
    self.assertAllClose(
        [4., 5.], manual_pkr.transformed_state, atol=0., rtol=1e-6)

  def test_copy_works(self):
    def fake_target_log_prob(x):
      return -x**2 / 2.

    transformed = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=FakeInnerKernel(target_log_prob_fn=fake_target_log_prob),
        bijector=tfb.AffineScalar(2.))

    transformed_copy = tfp.mcmc.TransformedTransitionKernel(
        **transformed.parameters)

    pkr, pkr_copy = self.evaluate([
        transformed.bootstrap_results(1.),
        transformed_copy.bootstrap_results(1.)
    ])

    self.assertAllClose(pkr.inner_results.target_log_prob,
                        pkr_copy.inner_results.target_log_prob)

  def test_is_calibrated(self):
    self.assertTrue(
        tfp.mcmc.TransformedTransitionKernel(
            FakeInnerKernel(lambda x: -x**2 / 2, True),
            tfb.Identity()).is_calibrated)
    self.assertFalse(
        tfp.mcmc.TransformedTransitionKernel(
            FakeInnerKernel(lambda x: -x**2 / 2, False),
            tfb.Identity()).is_calibrated)


if __name__ == '__main__':
  tf.test.main()
