# Copyright 2021 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the _License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for thermodynamic integrals."""

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.mcmc import thermodynamic_integrals
from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc import diagnostic
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import replica_exchange_mc
from tensorflow_probability.python.mcmc import sample


def make_inverse_temperatures(
    n_replica,
    min_nonzero_value,
    zero_final_value=True,
    dtype=np.float32,
):
  beta = 10**(np.linspace(0., np.log10(min_nonzero_value), num=n_replica))
  if zero_final_value:
    beta[-1] = 0.
  return beta.astype(dtype)


@test_util.test_graph_and_eager_modes
class REMCThermodynamicIntegralsTest(test_util.TestCase):

  @test_util.numpy_disable_gradient_test('HMC')
  def testAllIntegralsNoBatchDim(self):
    self.checkAllIntegrals(
        prior_scale=1.0,
        likelihood_scale=0.5,
        inverse_temperatures=make_inverse_temperatures(
            n_replica=30, min_nonzero_value=0.002),
        iid_chain_ndims=0,
    )

  @test_util.numpy_disable_gradient_test('HMC')
  def testAllIntegralsOneBatchDimNoIIDChainDims(self):
    n_batch = 2

    self.checkAllIntegrals(
        prior_scale=[1.0] * n_batch,
        likelihood_scale=np.linspace(0.25, 0.85, n_batch, dtype=np.float32),
        inverse_temperatures=make_inverse_temperatures(
            n_replica=30, min_nonzero_value=0.001, zero_final_value=True),
        iid_chain_ndims=0,
    )

  @test_util.numpy_disable_gradient_test('HMC')
  def testAllIntegralsOneBatchDimOneIIDChainDims(self):
    n_batch = 3

    self.checkAllIntegrals(
        prior_scale=[1.0] * n_batch,
        likelihood_scale=np.linspace(0.25, 0.85, n_batch, dtype=np.float32),
        inverse_temperatures=make_inverse_temperatures(
            n_replica=30, min_nonzero_value=0.001, zero_final_value=True),
        iid_chain_ndims=1,
    )

  def checkAllIntegrals(self, prior_scale, likelihood_scale,
                        inverse_temperatures, iid_chain_ndims):
    prior_scale = tf.convert_to_tensor(prior_scale, name='prior_scale')
    likelihood_scale = tf.convert_to_tensor(
        likelihood_scale, name='likelihood_scale')

    # Create (normalized) prior and likelihood. Their product of course is not
    # normalized. In particular, there is a number `normalizing_const` such that
    #   posterior(z) = prior.prob(x) * likelihood.prob(x) / normalizing_const
    # is normalized.
    prior = normal.Normal(0., prior_scale)
    likelihood = normal.Normal(0., likelihood_scale)
    posterior = normal.Normal(0.,
                              (prior_scale**-2 + likelihood_scale**-2)**(-0.5))

    # Get a good step size, custom for every replica/batch member.
    bcast_inv_temperatures = bu.left_justified_expand_dims_to(
        inverse_temperatures,
        # Broadcast over replicas.
        1 +
        # Broadcast over chains.
        iid_chain_ndims +
        # Broadcast over batch dims.
        tf.rank(likelihood_scale)
    )
    tempered_posteriors = normal.Normal(
        0.,
        # One tempered posterior for every inverse_temperature.
        (prior_scale**-2 + bcast_inv_temperatures * likelihood_scale**-2
        )**(-0.5))
    step_size = 0.71234 * tempered_posteriors.stddev()

    num_leapfrog_steps = tf.cast(
        tf.math.ceil(1.567 / tf.reduce_min(step_size)), tf.int32)

    def make_kernel_fn(target_log_prob_fn):
      return hmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          step_size=step_size,
          num_leapfrog_steps=num_leapfrog_steps,
      )

    remc = replica_exchange_mc.ReplicaExchangeMC(
        target_log_prob_fn=None,
        untempered_log_prob_fn=prior.log_prob,
        tempered_log_prob_fn=likelihood.log_prob,
        inverse_temperatures=inverse_temperatures,
        state_includes_replicas=False,
        make_kernel_fn=make_kernel_fn,
        swap_proposal_fn=replica_exchange_mc.even_odd_swap_proposal_fn(1.),
    )

    def trace_fn(state, results):  # pylint: disable=unused-argument
      return {
          'replica_log_accept_ratio':
              results.post_swap_replica_results.log_accept_ratio,
          'is_swap_accepted_adjacent':
              results.is_swap_accepted_adjacent,
          'is_swap_proposed_adjacent':
              results.is_swap_proposed_adjacent,
          'potential_energy':
              results.potential_energy,
      }

    if tf.executing_eagerly():
      num_results = 100
    else:
      num_results = 1000
    num_burnin_steps = num_results // 10

    n_samples_per_chain = 2
    initial_sample_shape = [n_samples_per_chain] * iid_chain_ndims

    unused_replica_states, trace = self.evaluate(
        sample.sample_chain(
            num_results=num_results,
            # Start at one of the modes, in order to make mode jumping necessary
            # if we want to pass test.
            current_state=prior.sample(
                initial_sample_shape, seed=test_util.test_seed()),
            kernel=remc,
            num_burnin_steps=num_burnin_steps,
            trace_fn=trace_fn,
            seed=test_util.test_seed()))

    # Tolerance depends on samples * replicas * number of (iid) chains.
    # ess.shape = [n_replica, ...]
    # We will sum over batch dims, then take min over replica.
    ess = diagnostic.effective_sample_size(trace['potential_energy'])
    if iid_chain_ndims:
      ess = tf.reduce_sum(ess, axis=tf.range(1, 1 + iid_chain_ndims))
    min_ess = self.evaluate(tf.reduce_min(ess))

    n_combined_results = min_ess * inverse_temperatures.shape[0]

    # Make sure sampling worked well enough, for every replica/chain.
    conditional_swap_prob = (
        np.sum(trace['is_swap_accepted_adjacent'], axis=0) /
        np.sum(trace['is_swap_proposed_adjacent'], axis=0))
    self.assertAllGreater(conditional_swap_prob, 0.5)

    replica_mean_accept_prob = np.mean(
        np.exp(np.minimum(0, trace['replica_log_accept_ratio'])), axis=0)
    self.assertAllGreater(replica_mean_accept_prob, 0.5)

    integrals = self.evaluate(
        thermodynamic_integrals.remc_thermodynamic_integrals(
            inverse_temperatures,
            trace['potential_energy'],
            iid_chain_ndims=iid_chain_ndims,
        ))

    self.assertAllEqual(posterior.batch_shape,
                        integrals.log_normalizing_constant_ratio.shape)
    actual_log_normalizing_const = self.evaluate(
        # Use arbitrary point, 0, to find the constant.
        prior.log_prob(0.) + likelihood.log_prob(0.) - posterior.log_prob(0.))
    self.assertAllClose(
        integrals.log_normalizing_constant_ratio,
        actual_log_normalizing_const,
        rtol=10 / np.sqrt(n_combined_results))

    self.assertAllEqual(posterior.batch_shape,
                        integrals.cross_entropy_difference.shape)

    def cross_entropy(dist):
      z = dist.sample(50000, seed=test_util.test_seed())
      return tf.reduce_mean(likelihood.log_prob(z), axis=0)

    iid_cross_entropy_difference = self.evaluate(
        cross_entropy(posterior) - cross_entropy(prior))
    self.assertAllClose(
        integrals.cross_entropy_difference,
        iid_cross_entropy_difference,
        rtol=30 / np.sqrt(n_combined_results))


if __name__ == '__main__':
  test_util.main()
