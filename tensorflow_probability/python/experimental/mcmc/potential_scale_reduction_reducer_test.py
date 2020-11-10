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
"""Tests for PotentialScaleReductionReducer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


TestTransitionKernelResults = collections.namedtuple(
    'TestTransitionKernelResults', 'counter_1, counter_2')


class TestTransitionKernel(tfp.mcmc.TransitionKernel):
  """Fake deterministic Transition Kernel."""

  def __init__(self, shape=(), target_log_prob_fn=None, is_calibrated=True):
    self._is_calibrated = is_calibrated
    self._shape = shape
    # for composition purposes
    self.parameters = dict(
        target_log_prob_fn=target_log_prob_fn)

  def one_step(self, current_state, previous_kernel_results, seed=None):
    return (current_state + tf.ones(self._shape),
            TestTransitionKernelResults(
                counter_1=previous_kernel_results.counter_1 + 1,
                counter_2=previous_kernel_results.counter_2 + 2))

  def bootstrap_results(self, current_state):
    return TestTransitionKernelResults(
        counter_1=tf.zeros(()),
        counter_2=tf.zeros(()))

  @property
  def is_calibrated(self):
    return self._is_calibrated


@test_util.test_all_tf_execution_regimes
class PotentialScaleReductionReducerTest(test_util.TestCase):

  def test_simple_operation(self):
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=1,
    )
    state = rhat_reducer.initialize(tf.zeros(5,))
    chain_state = np.arange(20, dtype=np.float32).reshape((4, 5))
    for sample in chain_state:
      state = rhat_reducer.one_step(sample, state)
    rhat = rhat_reducer.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=chain_state,
        independent_chain_ndims=1,
    )
    rhat, true_rhat = self.evaluate([rhat, true_rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_non_scalar_sample(self):
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=1,
    )
    state = rhat_reducer.initialize(tf.zeros((5, 3)))
    chain_state = np.arange(60, dtype=np.float32).reshape((4, 5, 3))
    for sample in chain_state:
      state = rhat_reducer.one_step(sample, state)
    rhat = rhat_reducer.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=chain_state,
        independent_chain_ndims=1,
    )
    rhat, true_rhat = self.evaluate([rhat, true_rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_independent_chain_ndims(self):
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=2,
    )
    state = rhat_reducer.initialize(tf.zeros((2, 5, 3)))
    chain_state = np.arange(120, dtype=np.float32).reshape((4, 2, 5, 3))
    for sample in chain_state:
      state = rhat_reducer.one_step(sample, state)
    rhat = rhat_reducer.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=chain_state,
        independent_chain_ndims=2,
    )
    rhat, true_rhat = self.evaluate([rhat, true_rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_int_samples(self):
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=1,
    )
    state = rhat_reducer.initialize(tf.zeros((5, 3), dtype=tf.int64))
    chain_state = np.arange(60).reshape((4, 5, 3))
    for sample in chain_state:
      state = rhat_reducer.one_step(sample, state)
    rhat = rhat_reducer.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=chain_state,
        independent_chain_ndims=1,
    )
    self.assertEqual(tf.float64, rhat.dtype)
    rhat, true_rhat = self.evaluate([rhat, true_rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_in_with_reductions(self):
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=1,
    )
    fake_kernel = TestTransitionKernel(shape=(5,))
    reduced_kernel = tfp.experimental.mcmc.WithReductions(
        inner_kernel=fake_kernel,
        reducer=rhat_reducer,
    )
    chain_state = tf.zeros(5,)
    pkr = reduced_kernel.bootstrap_results(chain_state)
    for _ in range(2):
      chain_state, pkr = reduced_kernel.one_step(
          chain_state, pkr)
    rhat = self.evaluate(
        rhat_reducer.finalize(pkr.reduction_results))
    self.assertEqual(0.5, rhat)

  def test_iid_normal_passes(self):
    n_samples = 500
    # two scalar chains taken from iid Normal(0, 1)
    rng = test_util.test_np_rng()
    iid_normal_samples = rng.randn(n_samples, 2)
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=1,
    )
    state = rhat_reducer.initialize(iid_normal_samples[0])
    for sample in iid_normal_samples:
      state = rhat_reducer.one_step(sample, state)
    rhat = self.evaluate(rhat_reducer.finalize(state))
    self.assertAllEqual((), rhat.shape)
    self.assertAllClose(1., rhat, rtol=0.02)

  def test_offset_normal_fails(self):
    n_samples = 500
    # three 4-variate chains taken from Normal(0, 1) that have been
    # shifted. Since every chain is shifted, they are not the same, and the
    # test should fail.
    offset = np.array([1., -1., 2.]).reshape(3, 1)
    rng = test_util.test_np_rng()
    offset_samples = rng.randn(n_samples, 3, 4) + offset
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=1,
    )
    state = rhat_reducer.initialize(offset_samples[0])
    for sample in offset_samples:
      state = rhat_reducer.one_step(sample, state)
    rhat = self.evaluate(rhat_reducer.finalize(state))
    self.assertAllEqual((4,), rhat.shape)
    self.assertAllEqual(np.ones_like(rhat).astype(bool), rhat > 1.2)

  def test_with_hmc(self):
    target_dist = tfp.distributions.Normal(loc=0., scale=1.)
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_dist.log_prob,
        num_leapfrog_steps=27,
        step_size=1/3)
    reduced_stats, _, _ = tfp.experimental.mcmc.sample_fold(
        num_steps=50,
        current_state=tf.zeros((2,)),
        kernel=hmc_kernel,
        reducer=[
            tfp.experimental.mcmc.TracingReducer(),
            tfp.experimental.mcmc.PotentialScaleReductionReducer()
        ]
    )
    rhat = reduced_stats[1]
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=reduced_stats[0][0],
        independent_chain_ndims=1,
    )
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_multiple_latent_state(self):
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=1,
    )
    state = rhat_reducer.initialize([tf.zeros(5,), tf.zeros((2, 5))])
    chain_state = np.arange(20, dtype=np.float32).reshape((4, 5))
    second_chain_state = np.arange(40, dtype=np.float32).reshape((4, 2, 5))
    for latent in zip(chain_state, second_chain_state):
      state = rhat_reducer.one_step(latent, state)
    rhat = rhat_reducer.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=[chain_state, second_chain_state],
        independent_chain_ndims=1,
    )
    rhat, true_rhat = self.evaluate([rhat, true_rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)


if __name__ == '__main__':
  tf.test.main()
