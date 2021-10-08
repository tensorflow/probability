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

import jax
from jax.config import config as jax_config
import tensorflow.compat.v2 as real_tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc
from fun_mc import sga_hmc
from fun_mc import test_util

tf = backend.tf
tfp = backend.tfp
util = backend.util
tfd = tfp.distributions
distribute_lib = backend.distribute_lib
Root = tfd.JointDistributionCoroutine.Root


real_tf.enable_v2_behavior()
jax_config.update('jax_enable_x64', True)

BACKEND = None  # Rewritten by backends/rewrite.py.

if BACKEND == 'backend_jax':
  os.environ['XLA_FLAGS'] = (f'{os.environ.get("XLA_FLAGS", "")} '
                             '--xla_force_host_platform_device_count=4')


def _test_seed():
  return tfp_test_util.test_seed() % (2**32 - 1)


class SGAHMCTest(tfp_test_util.TestCase):

  def _make_seed(self, seed):
    return util.make_tensor_seed([seed, 0])

  @property
  def _dtype(self):
    raise NotImplementedError()

  def _constant(self, value):
    return tf.constant(value, self._dtype)

  def testHMCWithStateGrads(self):
    trajectory_length = 1.
    epsilon = 1e-3

    seed = self._make_seed(_test_seed())

    def hmc_step(trajectory_length, axis_name=()):

      @tfp.experimental.distribute.JointDistributionCoroutine
      def model():
        z = yield Root(tfd.Normal(0., 1))
        yield tfp.experimental.distribute.Sharded(
            tfd.Sample(tfd.Normal(z, 1.), 8), axis_name)

      @tfp.experimental.distribute.JointDistributionCoroutine
      def momentum_dist():
        yield Root(tfd.Normal(0., 2))
        yield Root(
            tfp.experimental.distribute.Sharded(
                tfd.Sample(tfd.Normal(0., 3.), 8), axis_name))

      def target_log_prob_fn(x):
        return model.log_prob(x), ()

      def kinetic_energy_fn(m):
        return -momentum_dist.log_prob(m), ()

      def momentum_sample_fn(seed):
        return momentum_dist.sample(2, seed=seed)

      state = model.sample(2, seed=seed)
      hmc_state = fun_mc.hamiltonian_monte_carlo_init(state, target_log_prob_fn)
      hmc_state, hmc_extra = (
          sga_hmc.hamiltonian_monte_carlo_with_state_grads_step(
              hmc_state,
              trajectory_length=trajectory_length,
              scalar_step_size=epsilon,
              step_size_scale=util.map_tree(lambda x: 1. + tf.abs(x), state),
              target_log_prob_fn=target_log_prob_fn,
              seed=seed,
              kinetic_energy_fn=kinetic_energy_fn,
              momentum_sample_fn=momentum_sample_fn,
              named_axis=model.experimental_shard_axis_names))

      def sum_state(x, axis_name):
        res = tf.reduce_sum(x**2)
        if axis_name:
          res = backend.distribute_lib.psum(res, axis_name)
        return res

      sum_sq = util.map_tree_up_to(hmc_extra.proposed_state, sum_state,
                                   hmc_extra.proposed_state,
                                   model.experimental_shard_axis_names)
      sum_sq = sum(util.flatten_tree(sum_sq))
      return sum_sq, ()

    def finite_diff_grad(f, epsilon, x):
      return (fun_mc.call_potential_fn(f, util.map_tree(
          lambda x: x + epsilon, x))[0] - fun_mc.call_potential_fn(
              f, util.map_tree(lambda x: x - epsilon, x))[0]) / (2 * epsilon)

    f = tf.function(hmc_step)
    auto_diff = util.value_and_grad(f, trajectory_length)[2]
    finite_diff = finite_diff_grad(f, epsilon, trajectory_length)

    self.assertAllClose(auto_diff, finite_diff, rtol=0.01)

    if BACKEND == 'backend_jax':

      @functools.partial(jax.pmap, axis_name='i')
      def run(_):
        f = tf.function(lambda trajectory_length: hmc_step(  # pylint: disable=g-long-lambda
            trajectory_length, axis_name='i'))
        auto_diff = util.value_and_grad(f, trajectory_length)[2]
        finite_diff = finite_diff_grad(f, epsilon, trajectory_length)
        return auto_diff, finite_diff

      auto_diff, finite_diff = run(tf.ones(4))
      self.assertAllClose(auto_diff, finite_diff, rtol=0.01)


@test_util.multi_backend_test(globals(), 'sga_hmc_test')
class SGAHMCTest32(SGAHMCTest):

  @property
  def _dtype(self):
    return tf.float32


@test_util.multi_backend_test(globals(), 'sga_hmc_test')
class SGAHMCTest64(SGAHMCTest):

  @property
  def _dtype(self):
    return tf.float64


del SGAHMCTest

if __name__ == '__main__':
  tfp_test_util.main()
