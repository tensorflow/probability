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
"""Tests for prefabs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from jax import random as jax_random
from jax.config import config as jax_config
import tensorflow.compat.v2 as real_tf

from discussion import fun_mcmc
from discussion.fun_mcmc import backend
from tensorflow_probability.python.internal import test_util as tfp_test_util

tf = backend.tf
util = backend.util
tfp = backend.tfp

real_tf.enable_v2_behavior()
jax_config.update('jax_enable_x64', True)


def _test_seed():
  return tfp_test_util.test_seed() % (2**32 - 1)


class PrefabTestTensorFlow32(tfp_test_util.TestCase):

  _is_on_jax = False

  def setUp(self):
    super(PrefabTestTensorFlow32, self).setUp()
    backend.set_backend(backend.TENSORFLOW, backend.MANUAL_TRANSFORMS)

  def _make_seed(self, seed):
    return seed

  @property
  def _dtype(self):
    return tf.float32

  def _constant(self, value):
    return tf.constant(value, self._dtype)

  def testAdaptiveHMC(self):
    num_chains = 16
    num_steps = 4000
    num_warmup_steps = num_steps // 2
    num_adapt_steps = int(0.8 * num_warmup_steps)

    # Setup the model and state constraints.
    model = tfp.distributions.JointDistributionSequential([
        tfp.distributions.Normal(loc=self._constant(0.), scale=1.),
        tfp.distributions.Independent(
            tfp.distributions.LogNormal(
                loc=self._constant([1., 1.]), scale=0.5), 1),
    ])
    bijector = [tfp.bijectors.Identity(), tfp.bijectors.Exp()]
    transform_fn = fun_mcmc.util_tfp.bijector_to_transform_fn(
        bijector, model.dtype, batch_ndims=1)

    def target_log_prob_fn(*x):
      return model.log_prob(x), ()

    # Start out at zeros (in the unconstrained space).
    state, _ = transform_fn(*[
        tf.zeros([num_chains] + list(e), dtype=self._dtype)
        for e in model.event_shape
    ])

    reparam_log_prob_fn, reparam_state = fun_mcmc.reparameterize_potential_fn(
        target_log_prob_fn, transform_fn, state)

    # Define the kernel.
    def kernel(adaptive_hmc_state, seed):
      if not self._is_on_jax:
        hmc_seed = _test_seed()
      else:
        hmc_seed, seed = util.split_seed(seed, 2)

      adaptive_hmc_state, adaptive_hmc_extra = (
          fun_mcmc.prefab.adaptive_hamiltonian_monte_carlo_step(
              adaptive_hmc_state,
              target_log_prob_fn=reparam_log_prob_fn,
              num_adaptation_steps=num_adapt_steps,
              seed=hmc_seed))

      return (adaptive_hmc_state,
              seed), (adaptive_hmc_extra.state, adaptive_hmc_extra.is_accepted,
                      adaptive_hmc_extra.step_size)

    if not self._is_on_jax:
      seed = _test_seed()
    else:
      seed = self._make_seed(_test_seed())

    # Subtle: Unlike TF, JAX needs a data dependency from the inputs to outputs
    # for the jit to do anything.
    _, (state_chain, is_accepted_chain,
        _) = tf.function(lambda reparam_state, seed: fun_mcmc.trace(  # pylint: disable=g-long-lambda
            state=(fun_mcmc.prefab.adaptive_hamiltonian_monte_carlo_init(
                reparam_state, reparam_log_prob_fn), seed),
            fn=kernel,
            num_steps=num_steps))(reparam_state, seed)

    # Discard the warmup samples.
    state_chain = [s[num_warmup_steps:] for s in state_chain]
    is_accepted_chain = is_accepted_chain[num_warmup_steps:]

    accept_rate = tf.reduce_mean(tf.cast(is_accepted_chain, tf.float32))
    rhat = tfp.mcmc.potential_scale_reduction(state_chain)
    sample_mean = [tf.reduce_mean(s, axis=[0, 1]) for s in state_chain]
    sample_var = [tf.math.reduce_variance(s, axis=[0, 1]) for s in state_chain]

    self.assertAllAssertsNested(lambda rhat: self.assertAllLess(rhat, 1.1),
                                rhat)
    self.assertAllClose(0.8, accept_rate, atol=0.05)
    self.assertAllClose(model.mean(), sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(model.variance(), sample_var, rtol=0.1, atol=0.1)


class PrefabTestJAX32(PrefabTestTensorFlow32):

  _is_on_jax = True

  def setUp(self):
    super(PrefabTestJAX32, self).setUp()
    backend.set_backend(backend.JAX, backend.MANUAL_TRANSFORMS)

  def _make_seed(self, seed):
    return jax_random.PRNGKey(seed)


class PrefabTestTensorFlow64(PrefabTestTensorFlow32):

  @property
  def _dtype(self):
    return tf.float64


class PrefabTestJAX64(PrefabTestJAX32):

  @property
  def _dtype(self):
    return tf.float64


if __name__ == '__main__':
  real_tf.test.main()
