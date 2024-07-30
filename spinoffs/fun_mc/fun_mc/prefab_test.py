# Copyright 2021 The TensorFlow Probability Authors.
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

import functools
import os

# Dependency imports

import jax as real_jax
from jax import config as jax_config
import numpy as np
import tensorflow.compat.v2 as real_tf
from tensorflow_probability.python.internal import test_util as tfp_test_util
from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc
from fun_mc import prefab
from fun_mc import test_util
from fun_mc import util_tfp

jax = backend.jax
jnp = backend.jnp
tfp = backend.tfp
util = backend.util
tfd = tfp.distributions

real_tf.enable_v2_behavior()
jax_config.update('jax_enable_x64', True)

BACKEND = None  # Rewritten by backends/rewrite.py.

if BACKEND == 'backend_jax':
  os.environ['XLA_FLAGS'] = (
      f'{os.environ.get("XLA_FLAGS", "")} '
      '--xla_force_host_platform_device_count=4'
  )


def _test_seed():
  return tfp_test_util.test_seed() % (2**32 - 1)


class PrefabTest(tfp_test_util.TestCase):
  _is_on_jax = BACKEND == 'backend_jax'

  def _make_seed(self, seed):
    if self._is_on_jax:
      return real_jax.random.PRNGKey(seed)
    else:
      return util.make_tensor_seed([seed, 0])

  @property
  def _dtype(self):
    raise NotImplementedError()

  def _constant(self, value):
    return jnp.array(value, self._dtype)

  def testAdaptiveHMC(self):
    num_chains = 16
    num_steps = 4000
    num_warmup_steps = num_steps // 2
    num_adapt_steps = int(0.8 * num_warmup_steps)

    # Setup the model and state constraints.
    model = tfp.distributions.JointDistributionSequential([
        tfp.distributions.Normal(loc=self._constant(0.0), scale=1.0),
        tfp.distributions.Independent(
            tfp.distributions.LogNormal(
                loc=self._constant([1.0, 1.0]), scale=0.5
            ),
            1,
        ),
    ])
    bijector = [tfp.bijectors.Identity(), tfp.bijectors.Exp()]
    transform_fn = util_tfp.bijector_to_transform_fn(
        bijector, model.dtype, batch_ndims=1
    )

    def target_log_prob_fn(*x):
      return model.log_prob(x), ()

    # Start out at zeros (in the unconstrained space).
    state, _ = transform_fn(
        *[
            jnp.zeros([num_chains] + list(e), dtype=self._dtype)
            for e in model.event_shape
        ]
    )

    reparam_log_prob_fn, reparam_state = fun_mc.reparameterize_potential_fn(
        target_log_prob_fn, transform_fn, state
    )

    # Define the kernel.
    def kernel(adaptive_hmc_state, seed):
      hmc_seed, seed = util.split_seed(seed, 2)

      adaptive_hmc_state, adaptive_hmc_extra = (
          prefab.adaptive_hamiltonian_monte_carlo_step(
              adaptive_hmc_state,
              target_log_prob_fn=reparam_log_prob_fn,
              num_adaptation_steps=num_adapt_steps,
              seed=hmc_seed,
          )
      )

      return (adaptive_hmc_state, seed), (
          adaptive_hmc_extra.state,
          adaptive_hmc_extra.is_accepted,
          adaptive_hmc_extra.step_size,
      )

    seed = self._make_seed(_test_seed())

    _, (state_chain, is_accepted_chain, _) = jax.jit(
        lambda reparam_state, seed: fun_mc.trace(  # pylint: disable=g-long-lambda
            state=(
                prefab.adaptive_hamiltonian_monte_carlo_init(
                    reparam_state, reparam_log_prob_fn
                ),
                seed,
            ),
            fn=kernel,
            num_steps=num_steps,
        )
    )(reparam_state, seed)

    # Discard the warmup samples.
    state_chain = [s[num_warmup_steps:] for s in state_chain]
    is_accepted_chain = is_accepted_chain[num_warmup_steps:]

    accept_rate = jnp.mean(jnp.asarray(is_accepted_chain, jnp.float32))
    rhat = tfp.mcmc.potential_scale_reduction(state_chain)
    sample_mean = [jnp.mean(s, axis=[0, 1]) for s in state_chain]
    sample_var = [jnp.var(s, axis=[0, 1]) for s in state_chain]

    self.assertAllAssertsNested(
        lambda rhat: self.assertAllLess(rhat, 1.1), rhat
    )
    self.assertAllClose(0.8, accept_rate, atol=0.05)
    self.assertAllClose(model.mean(), sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(model.variance(), sample_var, rtol=0.1, atol=0.1)

  def testInteractiveTrace(self):
    def kernel(x):
      return x + 1, x

    counter = [0]

    def progress_bar_fn(iterable):
      for _ in iterable:
        counter[0] += 1
        yield

    x_fin, x_trace = prefab.interactive_trace(
        0.0, kernel, num_steps=5, progress_bar_fn=progress_bar_fn
    )

    self.assertAllClose(5, x_fin)
    self.assertAllClose(np.arange(5), x_trace)
    self.assertEqual(5, counter[0])

  def testStepSizeAdaptation(self):
    def log_accept_ratio_fn(step_size):
      return -(step_size**2)

    def kernel(ssa_state, seed):
      normal_seed, seed = util.split_seed(seed, 2)
      log_accept_ratio = log_accept_ratio_fn(
          ssa_state.step_size()
      ) + 0.01 * util.random_normal([4], self._dtype, normal_seed)
      ssa_state, ssa_extra = prefab.step_size_adaptation_step(
          ssa_state, log_accept_ratio, num_adaptation_steps=100
      )
      return (ssa_state, seed), (
          ssa_extra.accept_prob,
          ssa_state.step_size(),
          ssa_state.step_size(num_adaptation_steps=100),
      )

    seed = self._make_seed(_test_seed())

    _, (p_accept, step_size, rms_step_size) = fun_mc.trace(
        (prefab.step_size_adaptation_init(jnp.array(0.1, self._dtype)), seed),
        kernel,
        200,
    )

    self.assertAllClose(0.8, p_accept[100], atol=0.1)
    self.assertAllClose(step_size[100], step_size[150])
    self.assertAllClose(rms_step_size[100], rms_step_size[150])

  def testInteractiveIterationAxis1(self):
    def kernel(x):
      return x + 1, x

    state, trace = prefab.interactive_trace(
        0.0,
        lambda x: fun_mc.trace(x, kernel, 5),
        20,
        iteration_axis=1,
        progress_bar_fn=None,
    )

    self.assertAllClose(100.0, state)
    self.assertEqual([100], list(trace.shape))
    self.assertAllClose(99.0, trace[-1])

  def testInteractiveIterationAxis2(self):
    def kernel(x):
      return x + 1, x

    def inner(x):
      state, trace = fun_mc.trace(x, kernel, 5)
      trace = jnp.transpose(trace, [1, 0])
      return state, trace

    state, trace = prefab.interactive_trace(
        jnp.zeros(2), inner, 20, iteration_axis=2, progress_bar_fn=None
    )

    self.assertAllClose([100.0, 100.0], state)
    self.assertEqual([2, 100], list(trace.shape))
    self.assertAllClose([99.0, 99.0], trace[:, -1])

  def testPHMC(self):
    step_size = self._constant(0.2)
    num_steps = 2000
    num_leapfrog_steps = 10
    state = jnp.ones([16, 2], dtype=self._dtype)

    base_mean = self._constant([2.0, 3.0])
    base_scale = self._constant([2.0, 0.5])

    def target_log_prob_fn(x):
      return -jnp.sum(0.5 * jnp.square((x - base_mean) / base_scale), -1), ()

    def kernel(phmc_state, seed):
      phmc_seed, seed = util.split_seed(seed, 2)
      phmc_state, _ = prefab.persistent_hamiltonian_monte_carlo_step(
          phmc_state,
          step_size=step_size,
          num_integrator_steps=num_leapfrog_steps,
          target_log_prob_fn=target_log_prob_fn,
          noise_fraction=self._constant(0.5),
          mh_drift=self._constant(0.127),
          seed=phmc_seed,
      )
      return (phmc_state, seed), phmc_state.state

    seed = self._make_seed(_test_seed())

    _, chain = jax.jit(
        lambda state, seed: fun_mc.trace(  # pylint: disable=g-long-lambda
            state=(
                prefab.persistent_hamiltonian_monte_carlo_init(
                    state, target_log_prob_fn
                ),
                seed,
            ),
            fn=kernel,
            num_steps=num_steps,
        )
    )(state, seed)
    # Discard the warmup samples.
    chain = chain[1000:]

    sample_mean = jnp.mean(chain, axis=[0, 1])
    sample_var = jnp.var(chain, axis=[0, 1])

    true_samples = (
        util.random_normal(shape=[4096, 2], dtype=self._dtype, seed=seed)
        * base_scale
        + base_mean
    )

    true_mean = jnp.mean(true_samples, axis=0)
    true_var = jnp.var(true_samples, axis=0)

    self.assertAllClose(true_mean, sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(true_var, sample_var, rtol=0.1, atol=0.1)

  def testPHMCWithLogUniform(self):
    def target_log_prob_fn(x):
      return -(x**2), ()

    phmc_state = prefab.persistent_hamiltonian_monte_carlo_init(
        self._constant(0.0), target_log_prob_fn
    )
    seed = self._make_seed(_test_seed())
    _, accepted_phmc_extra = prefab.persistent_hamiltonian_monte_carlo_step(
        phmc_state,
        target_log_prob_fn,
        step_size=self._constant(0.1),
        num_integrator_steps=1,
        noise_fraction=self._constant(0.5),
        mh_drift=self._constant(0.1),
        log_uniform=jnp.log(self._constant(0.0)),
        seed=seed,
    )
    _, rejected_phmc_extra = prefab.persistent_hamiltonian_monte_carlo_step(
        phmc_state,
        target_log_prob_fn,
        step_size=self._constant(0.1),
        num_integrator_steps=1,
        noise_fraction=self._constant(0.5),
        mh_drift=self._constant(0.1),
        log_uniform=0.0,
        seed=seed,
    )

    self.assertTrue(accepted_phmc_extra.is_accepted)
    self.assertFalse(rejected_phmc_extra.is_accepted)

  def testPHMCNamedAxis(self):
    if BACKEND != 'backend_jax':
      self.skipTest('JAX-only')

    state = {
        'sharded': jnp.zeros([4, 1024], self._dtype),
        'shared': jnp.zeros([1024], self._dtype),
    }
    in_axes = {
        'sharded': 0,
        'shared': None,
    }
    named_axis = {
        'sharded': 'named_axis',
        'shared': None,
    }

    def target_log_prob_fn(sharded, shared):
      return (
          -(
              backend.distribute_lib.psum(jnp.square(sharded), 'named_axis')
              + jnp.square(shared)
          ),
          (),
      )

    @functools.partial(
        real_jax.pmap, in_axes=(in_axes, None), axis_name='named_axis'
    )
    def kernel(state, seed):
      phmc_state = prefab.persistent_hamiltonian_monte_carlo_init(
          state, target_log_prob_fn=target_log_prob_fn
      )
      phmc_state, phmc_extra = prefab.persistent_hamiltonian_monte_carlo_step(
          phmc_state,
          step_size=self._constant(0.2),
          num_integrator_steps=4,
          noise_fraction=0.5,
          mh_drift=0.1,
          target_log_prob_fn=target_log_prob_fn,
          named_axis=named_axis,
          seed=seed,
      )
      return phmc_state, phmc_extra

    seed = self._make_seed(_test_seed())
    phmc_state, phmc_extra = kernel(state, seed)
    self.assertAllClose(
        phmc_state.state['shared'][0], phmc_state.state['shared'][1]
    )
    self.assertTrue(
        np.any(
            np.abs(
                phmc_state.state['sharded'][0] - phmc_state.state['sharded'][1]
            )
            > 1e-3
        )
    )
    self.assertAllClose(phmc_extra.is_accepted[0], phmc_extra.is_accepted[1])
    self.assertAllClose(
        phmc_extra.log_accept_ratio[0], phmc_extra.log_accept_ratio[1]
    )


@test_util.multi_backend_test(globals(), 'prefab_test')
class PrefabTest32(PrefabTest):

  @property
  def _dtype(self):
    return jnp.float32


@test_util.multi_backend_test(globals(), 'prefab_test')
class PrefabTest64(PrefabTest):

  @property
  def _dtype(self):
    return jnp.float64


del PrefabTest

if __name__ == '__main__':
  tfp_test_util.main()
