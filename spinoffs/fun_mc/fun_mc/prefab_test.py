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

# Dependency imports

from jax.config import config as jax_config
import numpy as np
import tensorflow.compat.v2 as real_tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc
from fun_mc import prefab
from fun_mc import test_util
from fun_mc import util_tfp

tf = backend.tf
tfp = backend.tfp
util = backend.util

real_tf.enable_v2_behavior()
jax_config.update('jax_enable_x64', True)


BACKEND = None  # Rewritten by backends/rewrite.py.


def _test_seed():
  return tfp_test_util.test_seed() % (2**32 - 1)


class PrefabTest(tfp_test_util.TestCase):

  def _make_seed(self, seed):
    return util.make_tensor_seed([seed, 0])

  @property
  def _dtype(self):
    raise NotImplementedError()

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
    transform_fn = util_tfp.bijector_to_transform_fn(
        bijector, model.dtype, batch_ndims=1)

    def target_log_prob_fn(*x):
      return model.log_prob(x), ()

    # Start out at zeros (in the unconstrained space).
    state, _ = transform_fn(*[
        tf.zeros([num_chains] + list(e), dtype=self._dtype)
        for e in model.event_shape
    ])

    reparam_log_prob_fn, reparam_state = fun_mc.reparameterize_potential_fn(
        target_log_prob_fn, transform_fn, state)

    # Define the kernel.
    def kernel(adaptive_hmc_state, seed):
      hmc_seed, seed = util.split_seed(seed, 2)

      adaptive_hmc_state, adaptive_hmc_extra = (
          prefab.adaptive_hamiltonian_monte_carlo_step(
              adaptive_hmc_state,
              target_log_prob_fn=reparam_log_prob_fn,
              num_adaptation_steps=num_adapt_steps,
              seed=hmc_seed))

      return (adaptive_hmc_state,
              seed), (adaptive_hmc_extra.state, adaptive_hmc_extra.is_accepted,
                      adaptive_hmc_extra.step_size)

    seed = self._make_seed(_test_seed())

    # Subtle: Unlike TF, JAX needs a data dependency from the inputs to outputs
    # for the jit to do anything.
    _, (state_chain, is_accepted_chain,
        _) = tf.function(lambda reparam_state, seed: fun_mc.trace(  # pylint: disable=g-long-lambda
            state=(prefab.adaptive_hamiltonian_monte_carlo_init(
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

  def testInteractiveTrace(self):

    def kernel(x):
      return x + 1, x

    counter = [0]

    def progress_bar_fn(iterable):
      for _ in iterable:
        counter[0] += 1
        yield

    x_fin, x_trace = prefab.interactive_trace(
        0., kernel, num_steps=5, progress_bar_fn=progress_bar_fn)

    self.assertAllClose(5, x_fin)
    self.assertAllClose(np.arange(5), x_trace)
    self.assertEqual(5, counter[0])

  def testStepSizeAdaptation(self):

    def log_accept_ratio_fn(step_size):
      return -step_size**2

    def kernel(ssa_state, seed):
      normal_seed, seed = util.split_seed(seed, 2)
      log_accept_ratio = (
          log_accept_ratio_fn(ssa_state.step_size()) +
          0.01 * util.random_normal([4], self._dtype, normal_seed))
      ssa_state, ssa_extra = prefab.step_size_adaptation_step(
          ssa_state, log_accept_ratio, num_adaptation_steps=100)
      return (ssa_state, seed), (ssa_extra.accept_prob, ssa_state.step_size(),
                                 ssa_state.step_size(num_adaptation_steps=100))

    seed = self._make_seed(_test_seed())

    _, (p_accept, step_size, rms_step_size) = fun_mc.trace(
        (prefab.step_size_adaptation_init(tf.constant(0.1, self._dtype)), seed),
        kernel, 200)

    self.assertAllClose(0.8, p_accept[100], atol=0.1)
    self.assertAllClose(step_size[100], step_size[150])
    self.assertAllClose(rms_step_size[100], rms_step_size[150])

  def testInteractiveIterationAxis1(self):
    def kernel(x):
      return x + 1, x

    state, trace = prefab.interactive_trace(
        0.,
        lambda x: fun_mc.trace(x, kernel, 5),
        20,
        iteration_axis=1,
        progress_bar_fn=None)

    self.assertAllClose(100., state)
    self.assertEqual([100], list(trace.shape))
    self.assertAllClose(99., trace[-1])

  def testInteractiveIterationAxis2(self):
    def kernel(x):
      return x + 1, x

    def inner(x):
      state, trace = fun_mc.trace(x, kernel, 5)
      trace = tf.transpose(trace, [1, 0])
      return state, trace

    state, trace = prefab.interactive_trace(
        tf.zeros(2),
        inner,
        20,
        iteration_axis=2,
        progress_bar_fn=None)

    self.assertAllClose([100., 100.], state)
    self.assertEqual([2, 100], list(trace.shape))
    self.assertAllClose([99., 99.], trace[:, -1])


@test_util.multi_backend_test(globals(), 'prefab_test')
class PrefabTest32(PrefabTest):

  @property
  def _dtype(self):
    return tf.float32


@test_util.multi_backend_test(globals(), 'prefab_test')
class PrefabTest64(PrefabTest):

  @property
  def _dtype(self):
    return tf.float64


del PrefabTest

if __name__ == '__main__':
  real_tf.test.main()
