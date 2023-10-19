# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for malt."""

import functools
import os

# Dependency imports

import jax
from jax import config as jax_config
import numpy as np
import tensorflow.compat.v2 as real_tf

from tensorflow_probability.python.internal import test_util as tfp_test_util
from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc
from fun_mc import malt
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


class MALTTest(tfp_test_util.TestCase):

  _is_on_jax = BACKEND == 'backend_jax'

  def _make_seed(self, seed):
    if self._is_on_jax:
      return jax.random.PRNGKey(seed)
    else:
      return util.make_tensor_seed([seed, 0])

  @property
  def _dtype(self):
    raise NotImplementedError()

  def _constant(self, value):
    return tf.constant(value, self._dtype)

  def testPreconditionedMALT(self):
    step_size = self._constant(0.2)
    num_steps = 2000
    num_leapfrog_steps = 10
    damping = 0.5
    state = tf.ones([16, 2], dtype=self._dtype)

    base_mean = self._constant([1., 0])
    base_cov = self._constant([[1, 0.5], [0.5, 1]])

    bijector = tfp.bijectors.Softplus()
    base_dist = tfp.distributions.MultivariateNormalFullCovariance(
        loc=base_mean, covariance_matrix=base_cov)
    target_dist = bijector(base_dist)

    def orig_target_log_prob_fn(x):
      return target_dist.log_prob(x), ()

    target_log_prob_fn, state = fun_mc.transform_log_prob_fn(
        orig_target_log_prob_fn, bijector, state)

    # pylint: disable=g-long-lambda
    def kernel(malt_state, seed):
      malt_seed, seed = util.split_seed(seed, 2)
      malt_state, _ = malt.metropolis_adjusted_langevin_trajectories_step(
          malt_state,
          step_size=step_size,
          damping=damping,
          num_integrator_steps=num_leapfrog_steps,
          target_log_prob_fn=target_log_prob_fn,
          seed=malt_seed)
      return (malt_state, seed), malt_state.state_extra[0]

    seed = self._make_seed(_test_seed())

    # Subtle: Unlike TF, JAX needs a data dependency from the inputs to outputs
    # for the jit to do anything.
    _, chain = tf.function(lambda state, seed: fun_mc.trace(  # pylint: disable=g-long-lambda
        state=(malt.metropolis_adjusted_langevin_trajectories_init(
            state, target_log_prob_fn), seed),
        fn=kernel,
        num_steps=num_steps))(state, seed)
    # Discard the warmup samples.
    chain = chain[1000:]

    sample_mean = tf.reduce_mean(chain, axis=[0, 1])
    sample_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    true_samples = target_dist.sample(4096, seed=self._make_seed(_test_seed()))

    true_mean = tf.reduce_mean(true_samples, axis=0)
    true_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    self.assertAllClose(true_mean, sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(true_cov, sample_cov, rtol=0.1, atol=0.1)

  def testMALTNamedAxis(self):
    if BACKEND != 'backend_jax':
      self.skipTest('JAX-only')

    state = {
        'sharded': tf.zeros([4, 1024], self._dtype),
        'shared': tf.zeros([1024], self._dtype),
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
      return -(backend.distribute_lib.psum(tf.square(sharded), 'named_axis') +
               tf.square(shared)), ()

    @functools.partial(
        jax.pmap, in_axes=(in_axes, None), axis_name='named_axis')
    def kernel(state, seed):
      malt_state = malt.metropolis_adjusted_langevin_trajectories_init(
          state, target_log_prob_fn=target_log_prob_fn)
      (malt_state,
       malt_extra) = malt.metropolis_adjusted_langevin_trajectories_step(
           malt_state,
           damping=self._constant(0.5),
           step_size=self._constant(0.2),
           num_integrator_steps=4,
           target_log_prob_fn=target_log_prob_fn,
           named_axis=named_axis,
           seed=seed)
      return malt_state, malt_extra

    seed = self._make_seed(_test_seed())
    malt_state, malt_extra = kernel(state, seed)
    self.assertAllClose(malt_state.state['shared'][0],
                        malt_state.state['shared'][1])
    self.assertTrue(
        np.any(
            np.abs(malt_state.state['sharded'][0] -
                   malt_state.state['sharded'][1]) > 1e-3))
    self.assertAllClose(malt_extra.is_accepted[0], malt_extra.is_accepted[1])
    self.assertAllClose(malt_extra.log_accept_ratio[0],
                        malt_extra.log_accept_ratio[1])


@test_util.multi_backend_test(globals(), 'malt_test')
class MALTTest32(MALTTest):

  @property
  def _dtype(self):
    return tf.float32


@test_util.multi_backend_test(globals(), 'malt_test')
class MALTTest64(MALTTest):

  @property
  def _dtype(self):
    return tf.float64


del MALTTest

if __name__ == '__main__':
  tfp_test_util.main()
