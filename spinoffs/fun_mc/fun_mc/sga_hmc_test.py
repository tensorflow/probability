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

from absl.testing import parameterized
import jax
from jax import config as jax_config
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

  @parameterized.named_parameters(
      dict(
          testcase_name='_chees',
          criterion_fn=sga_hmc.chees_criterion,
          with_trajectory=False,
          with_state_mean=False,
      ),
      dict(
          testcase_name='_chees_per_grad',
          criterion_fn=sga_hmc.chees_per_grad_criterion,
          with_trajectory=True,
          with_state_mean=False,
      ),
      dict(
          testcase_name='_chees_with_state_mean',
          criterion_fn=sga_hmc.chees_criterion,
          with_trajectory=False,
          with_state_mean=True,
      ),
      dict(
          testcase_name='_chees_per_grad_with_state_mean',
          criterion_fn=sga_hmc.chees_per_grad_criterion,
          with_trajectory=True,
          with_state_mean=True,
      ),
  )
  def testCriterion(self, criterion_fn, with_trajectory, with_state_mean):
    """Evaluates the criterion in sharded and non-sharded contexts."""
    seed = self._make_seed(_test_seed())
    seeds = util.split_seed(seed, 4)
    previous_state = {
        'global':
            self._constant(util.random_normal([2, 3], self._dtype, seeds[0])),
        'local':
            self._constant(
                util.random_normal([2, 2, 3], self._dtype, seeds[1]))
    }
    named_axis = util.map_tree(lambda _: [], previous_state)
    chain_named_axis = []
    trajectory_length = self._constant(0.5)
    accept_prob = self._constant([0.1, 0.5])
    state_mean = {
        'global':
            self._constant(util.random_normal([3], self._dtype, seeds[2])),
        'local':
            self._constant(util.random_normal([2, 3], self._dtype, seeds[3]))
    }

    def eval_criterion(trajectory_length, previous_state, accept_prob,
                       chain_named_axis, state_mean, named_axis):
      extra_kwargs = {}
      if with_trajectory:
        extra_kwargs.update(trajectory_length=trajectory_length)
      if with_state_mean:
        extra_kwargs.update(state_mean=state_mean, state_mean_weight=0.5)

      def proposed_state_part(previous_state, named_axis):
        if BACKEND == 'backend_jax':
          part_trajectory_length = distribute_lib.pbroadcast(
              trajectory_length, [chain_named_axis, named_axis])
        else:
          part_trajectory_length = trajectory_length
        return (part_trajectory_length + 1) * previous_state

      proposed_state = util.map_tree_up_to(previous_state, proposed_state_part,
                                           previous_state, named_axis)

      return criterion_fn(
          previous_state=previous_state,
          proposed_state=proposed_state,
          accept_prob=accept_prob,
          named_axis=named_axis,
          chain_named_axis=chain_named_axis,
          **extra_kwargs)

    value, _, grad = fun_mc.call_potential_fn_with_grads(
        functools.partial(
            eval_criterion,
            previous_state=previous_state,
            chain_named_axis=chain_named_axis,
            accept_prob=accept_prob,
            state_mean=state_mean,
            named_axis=named_axis), trajectory_length)

    self.assertEqual(self._dtype, value.dtype)
    self.assertEqual(self._dtype, grad.dtype)
    self.assertAllGreater(tf.abs(grad), 0.)

    if BACKEND == 'backend_jax':

      named_axis = {
          'global': [],
          'local': 'local',
      }
      chain_named_axis = 'chain'
      in_axes = {
          'global': None,
          'local': 0,
      }

      @functools.partial(jax.pmap, axis_name='chain')
      def run_chain(previous_state, accept_prob):

        @functools.partial(
            jax.pmap, axis_name='local', in_axes=(in_axes, in_axes))
        def run_state(previous_state, state_mean):
          value, _, grad = fun_mc.call_potential_fn_with_grads(
              functools.partial(
                  eval_criterion,
                  previous_state=previous_state,
                  chain_named_axis=chain_named_axis,
                  accept_prob=accept_prob,
                  state_mean=state_mean,
                  named_axis=named_axis), trajectory_length)
          return value, grad

        return run_state(previous_state, state_mean)

      sharded_value, sharded_grad = run_chain(previous_state, accept_prob)
      self.assertAllClose(value, sharded_value[0, 0])
      self.assertAllClose(grad, sharded_grad[0, 0])

  def testSGAHMC(self):

    @tfd.JointDistributionCoroutine
    def model():
      x = yield Root(tfd.Normal(self._constant(0.), 1.))
      yield tfd.Sample(tfd.Normal(x, 1.), 2)

    def target_log_prob_fn(x):
      return model.log_prob(x), ()

    @tf.function
    def kernel(sga_hmc_state, step, seed):
      adapt = step < num_adapt_steps
      seed, hmc_seed = util.split_seed(seed, 2)

      sga_hmc_state, sga_hmc_extra = sga_hmc.stochastic_gradient_ascent_hmc_step(
          sga_hmc_state,
          scalar_step_size=self._constant(0.1),
          step_size_scale=self._constant(1.),
          target_log_prob_fn=target_log_prob_fn,
          criterion_fn=sga_hmc.chees_criterion,
          adapt=adapt,
          seed=hmc_seed,
      )

      return (sga_hmc_state, step + 1, seed
             ), sga_hmc_extra.trajectory_length_params.mean_trajectory_length()

    init_trajectory_length = self._constant(0.1)
    num_adapt_steps = 10
    _, trajectory_length = fun_mc.trace(
        (sga_hmc.stochastic_gradient_ascent_hmc_init(
            util.map_tree_up_to(
                model.dtype, lambda dtype, shape: tf.zeros(  # pylint: disable=g-long-lambda
                    (16,) + tuple(shape), dtype), model.dtype,
                model.event_shape),
            target_log_prob_fn,
            init_trajectory_length=init_trajectory_length), 0,
         self._make_seed(_test_seed())), kernel, num_adapt_steps + 2)

    # We expect it to increase as part of adaptation.
    self.assertAllGreater(trajectory_length[-1], init_trajectory_length)
    # After adaptation is done, the trajectory length should remain constant.
    self.assertAllClose(trajectory_length[-1], trajectory_length[-2])


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
