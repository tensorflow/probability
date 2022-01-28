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

# Dependency imports

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc.internal import test_fixtures
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class PotentialScaleReductionReducerTest(test_util.TestCase):

  def test_int_samples(self):
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=1)
    state = rhat_reducer.initialize(tf.zeros((5, 3), dtype=tf.int64))
    chain_state = np.arange(60).reshape((4, 5, 3))
    for sample in chain_state:
      state = rhat_reducer.one_step(sample, state)
    rhat = rhat_reducer.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=chain_state,
        independent_chain_ndims=1)
    self.assertEqual(tf.float64, rhat.dtype)
    rhat, true_rhat = self.evaluate([rhat, true_rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_iid_normal_passes(self):
    n_samples = 500
    # five scalar chains taken from iid Normal(0, 1)
    rng = test_util.test_np_rng()
    iid_normal_samples = rng.randn(n_samples, 5)
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=1)
    rhat = self.evaluate(test_fixtures.reduce(rhat_reducer, iid_normal_samples))
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
        independent_chain_ndims=1)
    rhat = self.evaluate(test_fixtures.reduce(rhat_reducer, offset_samples))
    self.assertAllEqual((4,), rhat.shape)
    self.assertAllGreater(rhat, 1.2)

  @test_util.numpy_disable_gradient_test
  def test_with_hmc(self):
    target_dist = tfp.distributions.Normal(loc=0., scale=1.)
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_dist.log_prob,
        num_leapfrog_steps=27,
        step_size=0.33)
    reduced_stats, _, _ = tfp.experimental.mcmc.sample_fold(
        num_steps=50,
        current_state=tf.zeros((2,)),
        kernel=hmc_kernel,
        reducer=[
            tfp.experimental.mcmc.TracingReducer(size=50),
            tfp.experimental.mcmc.PotentialScaleReductionReducer()
        ],
        seed=test_util.test_seed())
    rhat = reduced_stats[1]
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=reduced_stats[0][0],
        independent_chain_ndims=1)
    true_rhat, rhat = self.evaluate([true_rhat, rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)

  def test_multiple_latent_states_and_independent_chain_ndims(self):
    rng = test_util.test_np_rng()
    rhat_reducer = tfp.experimental.mcmc.PotentialScaleReductionReducer(
        independent_chain_ndims=2)
    state = rhat_reducer.initialize([tf.zeros((2, 5, 3)), tf.zeros((7, 2, 8))])
    chain_state = rng.randn(4, 2, 5, 3)
    second_chain_state = rng.randn(4, 7, 2, 8)
    for latent in zip(chain_state, second_chain_state):
      state = rhat_reducer.one_step(latent, state)
    rhat = rhat_reducer.finalize(state)
    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains_states=[chain_state, second_chain_state],
        independent_chain_ndims=2)
    rhat, true_rhat = self.evaluate([rhat, true_rhat])
    self.assertAllClose(true_rhat, rhat, rtol=1e-6)


if __name__ == '__main__':
  test_util.main()
