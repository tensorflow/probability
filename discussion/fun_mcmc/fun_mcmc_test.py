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
"""Tests for fun_mcmc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

# Dependency imports

from absl.testing import parameterized
from jax import random as jax_random
import numpy as np
import tensorflow.compat.v2 as real_tf
import tensorflow_probability as tfp

from discussion import fun_mcmc
from discussion.fun_mcmc import backend
from tensorflow_probability.python.internal import test_util as tfp_test_util

tf = backend.tf
tfb = tfp.bijectors
tfd = tfp.distributions
util = backend.util

real_tf.enable_v2_behavior()


def _test_seed():
  return tfp_test_util.test_seed() % (2**32 - 1)


def _no_compile(fn):
  return fn


def _fwd_mclachlan_optimal_4th_order_step(*args, **kwargs):
  return fun_mcmc.mclachlan_optimal_4th_order_step(
      *args, forward=True, **kwargs)


def _rev_mclachlan_optimal_4th_order_step(*args, **kwargs):
  return fun_mcmc.mclachlan_optimal_4th_order_step(
      *args, forward=False, **kwargs)


def _skip_on_jax(fn):

  @functools.wraps(fn)
  def _wrapper(self, *args, **kwargs):
    if not self._is_on_jax:
      return fn(self, *args, **kwargs)

  return _wrapper


def _gen_cov(data, axis):
  """This computes a generalized covariance, supporting batch and reduction.

  This computes a batch of covariances from data, with the relevant dimensions
  are determined as follows:

  - Dimensions specified by `axis` are reduced over.
  - The final non-reduced over dimension is taken as the event dimension.
  - The remaining dimensions are batch dimensions.

  Args:
    data: An NDArray.
    axis: An integer or a tuple of integers.

  Returns:
    cov: A batch of covariances.
  """
  axis = tuple(util.flatten_tree(axis))
  shape = tuple(data.shape)
  rank = len(shape)
  centered_data = data - np.mean(data, axis, keepdims=True)
  symbols = 'abcdefg'
  # Destination is missing the axes we reduce over.
  dest = []
  last_unaggregated_src_dim = None
  last_unaggregated_dest_dim = None
  for i in range(rank):
    if i not in axis:
      dest.append(symbols[i])
      last_unaggregated_src_dim = i
      last_unaggregated_dest_dim = len(dest) - 1
  source_1 = list(symbols[:rank])
  source_1[last_unaggregated_src_dim] = 'x'
  source_2 = list(symbols[:rank])
  source_2[last_unaggregated_src_dim] = 'y'
  dest = dest[:last_unaggregated_dest_dim] + [
      'x', 'y'
  ] + dest[last_unaggregated_dest_dim + 1:]
  formula = '{source_1},{source_2}->{dest}'.format(
      source_1=''.join(source_1),
      source_2=''.join(source_2),
      dest=''.join(dest))
  cov = (
      np.einsum(formula, centered_data, centered_data) /
      np.prod(np.array(shape)[np.array(axis)]))
  return cov


class GenCovTest(real_tf.test.TestCase):

  def testGenCov(self):
    x = np.arange(10).reshape(5, 2)
    true_cov = np.cov(x, rowvar=False, bias=True)
    self.assertAllClose(true_cov, _gen_cov(x, 0))

    true_cov = np.cov(x, rowvar=True, bias=True)
    self.assertAllClose(true_cov, _gen_cov(x, 1))


class FunMCMCTestTensorFlow(real_tf.test.TestCase, parameterized.TestCase):

  _is_on_jax = False

  def setUp(self):
    super(FunMCMCTestTensorFlow, self).setUp()
    backend.set_backend(backend.TENSORFLOW, backend.MANUAL_TRANSFORMS)

  def _make_seed(self, seed):
    return seed

  def testTraceSingle(self):

    def fun(x):
      return x + 1., 2 * x

    x, e_trace = fun_mcmc.trace(
        state=0., fn=fun, num_steps=5, trace_fn=lambda _, xp1: xp1)

    self.assertAllEqual(5., x)
    self.assertAllEqual([0., 2., 4., 6., 8.], e_trace)

  def testTraceNested(self):

    def fun(x, y):
      return (x + 1., y + 2.), ()

    (x, y), (x_trace, y_trace) = fun_mcmc.trace(
        state=(0., 0.), fn=fun, num_steps=5, trace_fn=lambda xy, _: xy)

    self.assertAllEqual(5., x)
    self.assertAllEqual(10., y)
    self.assertAllEqual([1., 2., 3., 4., 5.], x_trace)
    self.assertAllEqual([2., 4., 6., 8., 10.], y_trace)

  def testTraceTrace(self):

    def fun(x):
      return fun_mcmc.trace(x, lambda x: (x + 1., x + 1.), 2, trace_mask=False)

    x, trace = fun_mcmc.trace(0., fun, 2)
    self.assertAllEqual(4., x)
    self.assertAllEqual([2., 4.], trace)

  def testTraceDynamic(self):

    @tf.function
    def trace_n(num_steps):
      return fun_mcmc.trace(0, lambda x: (x + 1, ()), num_steps)[0]

    x = trace_n(5)
    self.assertAllEqual(5, x)

  def testTraceMask(self):

    def fun(x):
      return x + 1, (2 * x, 3 * x)

    x, (trace_1, trace_2) = fun_mcmc.trace(
        state=0, fn=fun, num_steps=3, trace_mask=(True, False))

    self.assertAllEqual(3, x)
    self.assertAllEqual([0, 2, 4], trace_1)
    self.assertAllEqual(6, trace_2)

    x, (trace_1, trace_2) = fun_mcmc.trace(
        state=0, fn=fun, num_steps=3, trace_mask=False)

    self.assertAllEqual(3, x)
    self.assertAllEqual(4, trace_1)
    self.assertAllEqual(6, trace_2)

  def testCallFn(self):
    sum_fn = lambda *args: sum(args)

    self.assertEqual(1, fun_mcmc.call_fn(sum_fn, 1))
    self.assertEqual(3, fun_mcmc.call_fn(sum_fn, (1, 2)))

  def testCallFnDict(self):
    sum_fn = lambda a, b: a + b

    self.assertEqual(3, fun_mcmc.call_fn(sum_fn, [1, 2]))
    self.assertEqual(3, fun_mcmc.call_fn(sum_fn, {'a': 1, 'b': 2}))

  def testBroadcastStructure(self):
    struct = fun_mcmc.maybe_broadcast_structure(1, [1, 2])
    self.assertEqual([1, 1], struct)

    struct = fun_mcmc.maybe_broadcast_structure([3, 4], [1, 2])
    self.assertEqual([3, 4], struct)

  def testCallPotentialFn(self):

    def potential(x):
      return x, ()

    x, extra = fun_mcmc.call_potential_fn(potential, 0.)

    self.assertEqual(0., x)
    self.assertEqual((), extra)

  def testCallPotentialFnMissingExtra(self):

    def potential(x):
      return x

    with self.assertRaisesRegexp(TypeError, 'A common solution is to adjust'):
      fun_mcmc.call_potential_fn(potential, 0.)

  def testCallTransitionOperator(self):

    def kernel(x, y):
      del y
      return [x, [1]], ()

    [x, [y]], extra = fun_mcmc.call_transition_operator(kernel, [0., None])
    self.assertEqual(0., x)
    self.assertEqual(1, y)
    self.assertEqual((), extra)

  def testCallTransitionOperatorMissingExtra(self):

    def potential(x):
      return x

    with self.assertRaisesRegexp(TypeError, 'A common solution is to adjust'):
      fun_mcmc.call_transition_operator(potential, 0.)

  def testCallTransitionOperatorBadArgs(self):

    def potential(x, y, z):
      del z
      return (x, y), ()

    with self.assertRaisesRegexp(TypeError, 'The structure of `new_args=`'):
      fun_mcmc.call_transition_operator(potential, (1, 2, 3))

  @_skip_on_jax  # No JAX bijectors.
  def testTransformLogProbFn(self):

    def log_prob_fn(x, y):
      return tfd.Normal(0., 1.).log_prob(x) + tfd.Normal(1., 1.).log_prob(y), ()

    bijectors = [tfb.AffineScalar(scale=2.), tfb.AffineScalar(scale=3.)]

    (transformed_log_prob_fn,
     transformed_init_state) = fun_mcmc.transform_log_prob_fn(
         log_prob_fn, bijectors, [2., 3.])

    self.assertIsInstance(transformed_init_state, list)
    self.assertAllClose([1., 1.], transformed_init_state)
    tlp, (orig_space, _) = transformed_log_prob_fn(1., 1.)
    lp = log_prob_fn(2., 3.)[0] + sum(
        b.forward_log_det_jacobian(1., event_ndims=0) for b in bijectors)

    self.assertAllClose([2., 3.], orig_space)
    self.assertAllClose(lp, tlp)

  @_skip_on_jax  # No JAX bijectors.
  def testTransformLogProbFnKwargs(self):

    def log_prob_fn(x, y):
      return tfd.Normal(0., 1.).log_prob(x) + tfd.Normal(1., 1.).log_prob(y), ()

    bijectors = {
        'x': tfb.AffineScalar(scale=2.),
        'y': tfb.AffineScalar(scale=3.)
    }

    (transformed_log_prob_fn,
     transformed_init_state) = fun_mcmc.transform_log_prob_fn(
         log_prob_fn, bijectors, {
             'x': 2.,
             'y': 3.
         })

    self.assertIsInstance(transformed_init_state, dict)
    self.assertAllClose({'x': 1., 'y': 1.}, transformed_init_state)

    tlp, (orig_space, _) = transformed_log_prob_fn(x=1., y=1.)
    lp = log_prob_fn(
        x=2., y=3.)[0] + sum(
            b.forward_log_det_jacobian(1., event_ndims=0)
            for b in bijectors.values())

    self.assertAllClose({'x': 2., 'y': 3.}, orig_space)
    self.assertAllClose(lp, tlp)

  # The +1's here are because we initialize the `state_grads` at 1, which
  # require an extra call to `target_log_prob_fn`.
  @parameterized.parameters(
      (fun_mcmc.leapfrog_step, 1 + 1),
      (fun_mcmc.ruth4_step, 3 + 1),
      (fun_mcmc.blanes_3_stage_step, 3 + 1),
      (_fwd_mclachlan_optimal_4th_order_step, 4 + 1, 9),
      (_rev_mclachlan_optimal_4th_order_step, 4 + 1, 9),
  )
  def testIntegratorStep(self, method, num_tlp_calls, num_tlp_calls_jax=None):

    tlp_call_counter = [0]

    def target_log_prob_fn(q):
      tlp_call_counter[0] += 1
      return -q**2, 1.

    def kinetic_energy_fn(p):
      return tf.abs(p)**3., 2.

    state = 1.
    _, _, state_grads = fun_mcmc.call_potential_fn_with_grads(
        target_log_prob_fn,
        state,
    )

    state, extras = method(
        integrator_step_state=fun_mcmc.IntegratorStepState(
            state=state, state_grads=state_grads, momentum=2.),
        step_size=0.1,
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn)

    if num_tlp_calls_jax is not None and self._is_on_jax:
      num_tlp_calls = num_tlp_calls_jax
    self.assertEqual(num_tlp_calls, tlp_call_counter[0])
    self.assertEqual(1., extras.state_extra)
    self.assertEqual(2., extras.kinetic_energy_extra)

    initial_hamiltonian = -target_log_prob_fn(1.)[0] + kinetic_energy_fn(2.)[0]
    fin_hamiltonian = -target_log_prob_fn(state.state)[0] + kinetic_energy_fn(
        state.momentum)[0]

    self.assertAllClose(fin_hamiltonian, initial_hamiltonian, atol=0.2)

  @parameterized.parameters(
      fun_mcmc.leapfrog_step,
      fun_mcmc.ruth4_step,
      fun_mcmc.blanes_3_stage_step,
  )
  def testIntegratorStepReversible(self, method):

    def target_log_prob_fn(q):
      return -q**2, []

    def kinetic_energy_fn(p):
      return p**2., []

    seed = self._make_seed(_test_seed())

    state = 1.
    _, _, state_grads = fun_mcmc.call_potential_fn_with_grads(
        target_log_prob_fn,
        state,
    )

    state_fwd, _ = method(
        integrator_step_state=fun_mcmc.IntegratorStepState(
            state=state,
            state_grads=state_grads,
            momentum=util.random_normal([], tf.float32, seed)),
        step_size=0.1,
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn)

    state_rev, _ = method(
        integrator_step_state=state_fwd._replace(momentum=-state_fwd.momentum),
        step_size=0.1,
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn)

    self.assertAllClose(state, state_rev.state, atol=1e-6)

  def testMclachlanIntegratorStepReversible(self):

    def target_log_prob_fn(q):
      return -q**2, []

    def kinetic_energy_fn(p):
      return p**2., []

    seed = self._make_seed(_test_seed())

    state = 1.
    _, _, state_grads = fun_mcmc.call_potential_fn_with_grads(
        target_log_prob_fn,
        state,
    )

    state_fwd, _ = _fwd_mclachlan_optimal_4th_order_step(
        integrator_step_state=fun_mcmc.IntegratorStepState(
            state=state,
            state_grads=state_grads,
            momentum=util.random_normal([], tf.float32, seed)),
        step_size=0.1,
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn)

    state_rev, _ = _rev_mclachlan_optimal_4th_order_step(
        integrator_step_state=state_fwd._replace(momentum=-state_fwd.momentum),
        step_size=0.1,
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn)

    self.assertAllClose(state, state_rev.state, atol=1e-6)

  def testMetropolisHastingsStep(self):
    seed = self._make_seed(_test_seed())

    accepted, mh_extra = fun_mcmc.metropolis_hastings_step(
        current_state=0., proposed_state=1., energy_change=-np.inf, seed=seed)
    self.assertAllEqual(1., accepted)
    self.assertAllEqual(True, mh_extra.is_accepted)

    accepted, mh_extra = fun_mcmc.metropolis_hastings_step(
        current_state=0., proposed_state=1., energy_change=np.inf, seed=seed)
    self.assertAllEqual(0., accepted)
    self.assertAllEqual(False, mh_extra.is_accepted)

    accepted, mh_extra = fun_mcmc.metropolis_hastings_step(
        current_state=0., proposed_state=1., energy_change=np.nan, seed=seed)
    self.assertAllEqual(0., accepted)
    self.assertAllEqual(False, mh_extra.is_accepted)

    accepted, mh_extra = fun_mcmc.metropolis_hastings_step(
        current_state=0., proposed_state=1., energy_change=np.nan, seed=seed)
    self.assertAllEqual(0., accepted)
    self.assertAllEqual(False, mh_extra.is_accepted)

    accepted, mh_extra = fun_mcmc.metropolis_hastings_step(
        current_state=0.,
        proposed_state=1.,
        log_uniform=-10.,
        energy_change=-np.log(0.5),
        seed=seed)
    self.assertAllEqual(1., accepted)
    self.assertAllEqual(True, mh_extra.is_accepted)

    accepted, mh_extra = fun_mcmc.metropolis_hastings_step(
        current_state=0.,
        proposed_state=1.,
        log_uniform=0.,
        energy_change=-np.log(0.5),
        seed=seed)
    self.assertAllEqual(0., accepted)
    self.assertAllEqual(False, mh_extra.is_accepted)

    accepted, _ = fun_mcmc.metropolis_hastings_step(
        current_state=tf.zeros(1000),
        proposed_state=tf.ones(1000),
        energy_change=-tf.math.log(0.5 * tf.ones(1000)),
        seed=seed)
    self.assertAllClose(0.5, tf.reduce_mean(accepted), rtol=0.1)

  def testMetropolisHastingsStepStructure(self):
    struct_type = collections.namedtuple('Struct', 'a, b')

    current = struct_type([1, 2], (3, [4, [0, 0]]))
    proposed = struct_type([5, 6], (7, [8, [0, 0]]))

    accepted, mh_extra = fun_mcmc.metropolis_hastings_step(
        current_state=current,
        proposed_state=proposed,
        energy_change=-np.inf,
        seed=self._make_seed(_test_seed()))
    self.assertAllEqual(True, mh_extra.is_accepted)
    self.assertAllEqual(
        util.flatten_tree(proposed), util.flatten_tree(accepted))

  def testBasicHMC(self):
    step_size = 0.2
    num_steps = 2000
    num_leapfrog_steps = 10
    state = tf.ones([16, 2])

    base_mean = tf.constant([2., 3.])
    base_scale = tf.constant([2., 0.5])

    def target_log_prob_fn(x):
      return -tf.reduce_sum(0.5 * tf.square(
          (x - base_mean) / base_scale), -1), ()

    def kernel(hmc_state, seed):
      if not self._is_on_jax:
        hmc_seed = _test_seed()
      else:
        hmc_seed, seed = util.split_seed(seed, 2)
      hmc_state, hmc_extra = fun_mcmc.hamiltonian_monte_carlo(
          hmc_state,
          step_size=step_size,
          num_integrator_steps=num_leapfrog_steps,
          target_log_prob_fn=target_log_prob_fn,
          seed=hmc_seed)
      return (hmc_state, seed), hmc_extra

    if not self._is_on_jax:
      seed = _test_seed()
    else:
      seed = self._make_seed(_test_seed())

    # Subtle: Unlike TF, JAX needs a data dependency from the inputs to outputs
    # for the jit to do anything.
    _, chain = tf.function(lambda state, seed: fun_mcmc.trace(  # pylint: disable=g-long-lambda
        state=(fun_mcmc.hamiltonian_monte_carlo_init(state, target_log_prob_fn),
               seed),
        fn=kernel,
        num_steps=num_steps,
        trace_fn=lambda state, extra: state[0].state))(state, seed)
    # Discard the warmup samples.
    chain = chain[1000:]

    sample_mean = tf.reduce_mean(chain, axis=[0, 1])
    sample_var = tf.math.reduce_variance(chain, axis=[0, 1])

    true_samples = util.random_normal(
        shape=[4096, 2], dtype=tf.float32, seed=seed) * base_scale + base_mean

    true_mean = tf.reduce_mean(true_samples, axis=0)
    true_var = tf.math.reduce_variance(true_samples, axis=0)

    self.assertAllClose(true_mean, sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(true_var, sample_var, rtol=0.1, atol=0.1)

  @_skip_on_jax  # No JAX bijectors.
  def testPreconditionedHMC(self):
    step_size = 0.2
    num_steps = 2000
    num_leapfrog_steps = 10
    state = tf.ones([16, 2])

    base_mean = [1., 0]
    base_cov = [[1, 0.5], [0.5, 1]]

    bijector = tfb.Softplus()
    base_dist = tfd.MultivariateNormalFullCovariance(
        loc=base_mean, covariance_matrix=base_cov)
    target_dist = bijector(base_dist)

    def orig_target_log_prob_fn(x):
      return target_dist.log_prob(x), ()

    target_log_prob_fn, state = fun_mcmc.transform_log_prob_fn(
        orig_target_log_prob_fn, bijector, state)

    # pylint: disable=g-long-lambda
    kernel = tf.function(lambda state: fun_mcmc.hamiltonian_monte_carlo(
        state,
        step_size=step_size,
        num_integrator_steps=num_leapfrog_steps,
        target_log_prob_fn=target_log_prob_fn,
        seed=_test_seed()))

    _, chain = fun_mcmc.trace(
        state=fun_mcmc.hamiltonian_monte_carlo_init(state, target_log_prob_fn),
        fn=kernel,
        num_steps=num_steps,
        trace_fn=lambda state, extra: state.state_extra[0])
    # Discard the warmup samples.
    chain = chain[1000:]

    sample_mean = tf.reduce_mean(chain, axis=[0, 1])
    sample_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    true_samples = target_dist.sample(4096, seed=_test_seed())

    true_mean = tf.reduce_mean(true_samples, axis=0)
    true_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    self.assertAllClose(true_mean, sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(true_cov, sample_cov, rtol=0.1, atol=0.1)

  @parameterized.parameters((tf.function, 1), (_no_compile, 2))
  @_skip_on_jax  # `trace` doesn't have an efficient path in JAX yet.
  def testHMCCountTargetLogProb(self, compile_fn, expected_count):

    counter = [0]

    @compile_fn
    def target_log_prob_fn(x):
      counter[0] += 1
      return -tf.square(x), []

    # pylint: disable=g-long-lambda
    @tf.function
    def trace():
      kernel = lambda state: fun_mcmc.hamiltonian_monte_carlo(
          state,
          step_size=0.1,
          num_integrator_steps=3,
          target_log_prob_fn=target_log_prob_fn,
          seed=_test_seed())

      fun_mcmc.trace(
          state=fun_mcmc.hamiltonian_monte_carlo_init(
              tf.zeros([1]), target_log_prob_fn),
          fn=kernel,
          num_steps=4,
          trace_fn=lambda *args: ())

    trace()

    self.assertEqual(expected_count, counter[0])

  @_skip_on_jax  # `trace` doesn't have an efficient path in JAX yet.
  def testHMCCountTargetLogProbEfficient(self):

    counter = [0]

    def target_log_prob_fn(x):
      counter[0] += 1
      return -tf.square(x), []

    @tf.function
    def trace():
      # pylint: disable=g-long-lambda
      kernel = lambda state: fun_mcmc.hamiltonian_monte_carlo(
          state,
          step_size=0.1,
          num_integrator_steps=3,
          target_log_prob_fn=target_log_prob_fn,
          seed=self._make_seed(_test_seed()))

      fun_mcmc.trace(
          state=fun_mcmc.hamiltonian_monte_carlo_init(
              state=tf.zeros([1]), target_log_prob_fn=target_log_prob_fn),
          fn=kernel,
          num_steps=4,
          trace_fn=lambda *args: ())

    trace()

    self.assertEqual(2, counter[0])

  @_skip_on_jax  # No JAX bijectors.
  def testAdaptiveStepSize(self):
    step_size = 0.2
    num_steps = 2000
    num_adapt_steps = 1000
    num_leapfrog_steps = 10
    state = tf.ones([16, 2])

    base_mean = [1., 0]
    base_cov = [[1, 0.5], [0.5, 1]]

    @tf.function
    def computation(state):
      bijector = tfb.Softplus()
      base_dist = tfd.MultivariateNormalFullCovariance(
          loc=base_mean, covariance_matrix=base_cov)
      target_dist = bijector(base_dist)

      def orig_target_log_prob_fn(x):
        return target_dist.log_prob(x), ()

      target_log_prob_fn, state = fun_mcmc.transform_log_prob_fn(
          orig_target_log_prob_fn, bijector, state)

      def kernel(hmc_state, step_size_state, step):
        hmc_state, hmc_extra = fun_mcmc.hamiltonian_monte_carlo(
            hmc_state,
            step_size=tf.exp(step_size_state.state),
            num_integrator_steps=num_leapfrog_steps,
            target_log_prob_fn=target_log_prob_fn)

        rate = tf.compat.v1.train.polynomial_decay(
            0.01,
            global_step=step,
            power=0.5,
            decay_steps=num_adapt_steps,
            end_learning_rate=0.)
        mean_p_accept = tf.reduce_mean(
            tf.exp(tf.minimum(0., hmc_extra.log_accept_ratio)))

        loss_fn = fun_mcmc.make_surrogate_loss_fn(
            lambda _: (0.9 - mean_p_accept, ()))
        step_size_state, _ = fun_mcmc.adam_step(
            step_size_state, loss_fn, learning_rate=rate)

        return ((hmc_state, step_size_state, step + 1),
                (hmc_state.state_extra[0], hmc_extra.log_accept_ratio))

      _, (chain, log_accept_ratio_trace) = fun_mcmc.trace(
          state=(fun_mcmc.hamiltonian_monte_carlo_init(state,
                                                       target_log_prob_fn),
                 fun_mcmc.adam_init(tf.math.log(step_size)), 0),
          fn=kernel,
          num_steps=num_adapt_steps + num_steps,
      )
      true_samples = target_dist.sample(4096, seed=_test_seed())
      return chain, log_accept_ratio_trace, true_samples

    chain, log_accept_ratio_trace, true_samples = computation(state)

    log_accept_ratio_trace = log_accept_ratio_trace[num_adapt_steps:]
    chain = chain[num_adapt_steps:]

    sample_mean = tf.reduce_mean(chain, axis=[0, 1])
    sample_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    true_mean = tf.reduce_mean(true_samples, axis=0)
    true_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    self.assertAllClose(true_mean, sample_mean, rtol=0.05, atol=0.05)
    self.assertAllClose(true_cov, sample_cov, rtol=0.05, atol=0.05)
    self.assertAllClose(
        tf.reduce_mean(tf.exp(tf.minimum(0., log_accept_ratio_trace))),
        0.9,
        rtol=0.1)

  def testSignAdaptation(self):
    new_control = fun_mcmc.sign_adaptation(
        control=1., output=0.5, set_point=1., adaptation_rate=0.1)
    self.assertAllClose(new_control, 1. / 1.1)

    new_control = fun_mcmc.sign_adaptation(
        control=1., output=0.5, set_point=0., adaptation_rate=0.1)
    self.assertAllClose(new_control, 1. * 1.1)

  def testRaggedIntegrator(self):

    def target_log_prob_fn(q):
      return -q**2, q

    def kinetic_energy_fn(p):
      return tf.abs(p)**3., p

    integrator_fn = lambda state, num_steps: fun_mcmc.hamiltonian_integrator(  # pylint: disable=g-long-lambda
        state,
        num_steps=num_steps,
        integrator_step_fn=lambda state: fun_mcmc.leapfrog_step(  # pylint: disable=g-long-lambda
            state,
            step_size=0.1,
            target_log_prob_fn=target_log_prob_fn,
            kinetic_energy_fn=kinetic_energy_fn),
        kinetic_energy_fn=kinetic_energy_fn,
        integrator_trace_fn=lambda state, extra: (state, extra))

    state = tf.zeros([2])
    momentum = tf.ones([2])
    target_log_prob, _, state_grads = fun_mcmc.call_potential_fn_with_grads(
        target_log_prob_fn, state)

    start_state = fun_mcmc.IntegratorState(
        target_log_prob=target_log_prob,
        momentum=momentum,
        state=state,
        state_grads=state_grads,
        state_extra=state,
    )

    state_1 = integrator_fn(start_state, 1)
    state_2 = integrator_fn(start_state, 2)
    state_1_2 = integrator_fn(start_state, [1, 2])

    # Make sure integrators actually integrated to different points.
    self.assertFalse(np.all(state_1[0].state == state_2[0].state))

    # Ragged integration should be consistent with the non-ragged equivalent.
    def get_batch(state, idx):
      # For the integrator trace, we'll grab the final value.
      return util.map_tree(
          lambda x: x[idx] if len(x.shape) == 1 else x[-1, idx], state)

    self.assertAllClose(get_batch(state_1, 0), get_batch(state_1_2, 0))
    self.assertAllClose(get_batch(state_2, 0), get_batch(state_1_2, 1))

    # Ragged traces should be equal up to the number of steps for the batch
    # element.
    def get_slice(state, num, idx):
      return util.map_tree(lambda x: x[:num, idx], state[1].integrator_trace)

    self.assertAllClose(get_slice(state_1, 1, 0), get_slice(state_1_2, 1, 0))
    self.assertAllClose(get_slice(state_2, 2, 0), get_slice(state_1_2, 2, 1))

  def testAdam(self):

    def loss_fn(x, y):
      return tf.square(x - 1.) + tf.square(y - 2.), []

    _, [(x, y), loss] = fun_mcmc.trace(
        fun_mcmc.adam_init([tf.zeros([]), tf.zeros([])]),
        lambda adam_state: fun_mcmc.adam_step(  # pylint: disable=g-long-lambda
            adam_state,
            loss_fn,
            learning_rate=0.01),
        num_steps=1000,
        trace_fn=lambda state, extra: [state.state, extra.loss])

    self.assertAllClose(1., x[-1], atol=1e-3)
    self.assertAllClose(2., y[-1], atol=1e-3)
    self.assertAllClose(0., loss[-1], atol=1e-3)

  def testGradientDescent(self):

    def loss_fn(x, y):
      return tf.square(x - 1.) + tf.square(y - 2.), []

    _, [(x, y), loss] = fun_mcmc.trace(
        fun_mcmc.GradientDescentState([tf.zeros([]), tf.zeros([])]),
        lambda gd_state: fun_mcmc.gradient_descent_step(  # pylint: disable=g-long-lambda
            gd_state,
            loss_fn,
            learning_rate=0.01),
        num_steps=1000,
        trace_fn=lambda state, extra: [state.state, extra.loss])

    self.assertAllClose(1., x[-1], atol=1e-3)
    self.assertAllClose(2., y[-1], atol=1e-3)
    self.assertAllClose(0., loss[-1], atol=1e-3)

  def testSimpleDualAverages(self):

    def loss_fn(x, y):
      return tf.square(x - 1.) + tf.square(y - 2.), []

    def kernel(sda_state, rms_state):
      sda_state, _ = fun_mcmc.simple_dual_averages_step(sda_state, loss_fn, 1.)
      rms_state, _ = fun_mcmc.running_mean_step(rms_state, sda_state.state)
      return (sda_state, rms_state), rms_state.mean

    _, (x, y) = fun_mcmc.trace(
        (
            fun_mcmc.simple_dual_averages_init([tf.zeros([]),
                                                tf.zeros([])]),
            fun_mcmc.running_mean_init([[], []], [tf.float32, tf.float32]),
        ),
        kernel,
        num_steps=1000,
    )

    self.assertAllClose(1., x[-1], atol=1e-1)
    self.assertAllClose(2., y[-1], atol=1e-1)

  def testRandomWalkMetropolis(self):
    num_steps = 1000
    state = tf.ones([16], dtype=tf.int32)
    target_logits = tf.constant([1., 2., 3., 4.]) + 2.
    proposal_logits = tf.constant([4., 3., 2., 1.]) + 2.

    def target_log_prob_fn(x):
      return tf.gather(target_logits, x), ()

    def proposal_fn(x, seed):
      current_logits = tf.gather(proposal_logits, x)
      proposal = util.random_categorical(proposal_logits[tf.newaxis],
                                         x.shape[0], seed)[0]
      proposed_logits = tf.gather(proposal_logits, proposal)
      return tf.cast(proposal, x.dtype), ((), proposed_logits - current_logits)

    def kernel(rwm_state, seed):
      if not self._is_on_jax:
        rwm_seed = _test_seed()
      else:
        rwm_seed, seed = util.split_seed(seed, 2)
      rwm_state, rwm_extra = fun_mcmc.random_walk_metropolis(
          rwm_state,
          target_log_prob_fn=target_log_prob_fn,
          proposal_fn=proposal_fn,
          seed=rwm_seed)
      return (rwm_state, seed), rwm_extra

    if not self._is_on_jax:
      seed = _test_seed()
    else:
      seed = self._make_seed(_test_seed())

    # Subtle: Unlike TF, JAX needs a data dependency from the inputs to outputs
    # for the jit to do anything.
    _, chain = tf.function(lambda state, seed: fun_mcmc.trace(  # pylint: disable=g-long-lambda
        state=(fun_mcmc.random_walk_metropolis_init(state, target_log_prob_fn),
               seed),
        fn=kernel,
        num_steps=num_steps,
        trace_fn=lambda state, extra: state[0].state))(state, seed)
    # Discard the warmup samples.
    chain = chain[500:]

    sample_mean = tf.reduce_mean(tf.one_hot(chain, 4), axis=[0, 1])
    self.assertAllClose(tf.nn.softmax(target_logits), sample_mean, atol=0.11)

  @parameterized.named_parameters(
      ('Basic', (10, 3), None),
      ('Batched', (10, 4, 3), None),
      ('Aggregated0', (10, 4, 3), 0),
      ('Aggregated1', (10, 4, 3), 1),
      ('Aggregated01', (10, 4, 3), (0, 1)),
      ('Aggregated02', (10, 4, 5, 3), (0, 2)),
  )
  def testRunningMean(self, shape, aggregation):
    rng = np.random.RandomState(_test_seed())
    data = tf.convert_to_tensor(rng.randn(*shape))

    def kernel(rms, idx):
      rms, _ = fun_mcmc.running_mean_step(rms, data[idx], axis=aggregation)
      return (rms, idx + 1), ()

    true_aggregation = (0,) + (() if aggregation is None else tuple(
        [a + 1 for a in util.flatten_tree(aggregation)]))
    true_mean = np.mean(data, true_aggregation)

    (rms, _), _ = fun_mcmc.trace(
        state=(fun_mcmc.running_mean_init(true_mean.shape, data.dtype), 0),
        fn=kernel,
        num_steps=len(data),
        trace_fn=lambda *args: ())

    self.assertAllClose(true_mean, rms.mean)

  def testRunningMeanMaxPoints(self):
    window_size = 100
    rng = np.random.RandomState(_test_seed())
    data = tf.convert_to_tensor(
        np.concatenate(
            [rng.randn(window_size), 1. + 2. * rng.randn(window_size * 10)],
            axis=0))

    def kernel(rms, idx):
      rms, _ = fun_mcmc.running_mean_step(
          rms, data[idx], window_size=window_size)
      return (rms, idx + 1), rms.mean

    _, mean = fun_mcmc.trace(
        state=(fun_mcmc.running_mean_init([], data.dtype), 0),
        fn=kernel,
        num_steps=len(data),
    )
    # Up to window_size, we compute the running mean exactly.
    self.assertAllClose(np.mean(data[:window_size]), mean[window_size - 1])
    # After window_size, we're doing exponential moving average, and pick up the
    # mean after the change in the distribution. Since the moving average is
    # computed only over ~window_size points, this test is rather noisy.
    self.assertAllClose(1., mean[-1], atol=0.2)

  @parameterized.named_parameters(
      ('Basic', (10, 3), None),
      ('Batched', (10, 4, 3), None),
      ('Aggregated0', (10, 4, 3), 0),
      ('Aggregated1', (10, 4, 3), 1),
      ('Aggregated01', (10, 4, 3), (0, 1)),
      ('Aggregated02', (10, 4, 5, 3), (0, 2)),
  )
  def testRunningVariance(self, shape, aggregation):
    rng = np.random.RandomState(_test_seed())
    data = tf.convert_to_tensor(rng.randn(*shape))

    true_aggregation = (0,) + (() if aggregation is None else tuple(
        [a + 1 for a in util.flatten_tree(aggregation)]))
    true_mean = np.mean(data, true_aggregation)
    true_var = np.var(data, true_aggregation)

    def kernel(rvs, idx):
      rvs, _ = fun_mcmc.running_variance_step(rvs, data[idx], axis=aggregation)
      return (rvs, idx + 1), ()

    (rvs, _), _ = fun_mcmc.trace(
        state=(fun_mcmc.running_variance_init(true_mean.shape,
                                              data[0].dtype), 0),
        fn=kernel,
        num_steps=len(data),
        trace_fn=lambda *args: ())
    self.assertAllClose(true_mean, rvs.mean)
    self.assertAllClose(true_var, rvs.variance)

  def testRunningVarianceMaxPoints(self):
    window_size = 100
    rng = np.random.RandomState(_test_seed())
    data = tf.convert_to_tensor(
        np.concatenate(
            [rng.randn(window_size), 1. + 2. * rng.randn(window_size * 10)],
            axis=0))

    def kernel(rvs, idx):
      rvs, _ = fun_mcmc.running_variance_step(
          rvs, data[idx], window_size=window_size)
      return (rvs, idx + 1), (rvs.mean, rvs.variance)

    _, (mean, var) = fun_mcmc.trace(
        state=(fun_mcmc.running_variance_init([], data.dtype), 0),
        fn=kernel,
        num_steps=len(data),
    )
    # Up to window_size, we compute the running mean/variance exactly.
    self.assertAllClose(np.mean(data[:window_size]), mean[window_size - 1])
    self.assertAllClose(np.var(data[:window_size]), var[window_size - 1])
    # After window_size, we're doing exponential moving average, and pick up the
    # mean/variance after the change in the distribution. Since the moving
    # average is computed only over ~window_size points, this test is rather
    # noisy.
    self.assertAllClose(1., mean[-1], atol=0.2)
    self.assertAllClose(4., var[-1], atol=0.8)

  @parameterized.named_parameters(
      ('Basic', (10, 3), None),
      ('Batched', (10, 4, 3), None),
      ('Aggregated0', (10, 4, 3), 0),
      ('Aggregated01', (10, 4, 5, 3), (0, 1)),
  )
  def testRunningCovariance(self, shape, aggregation):
    rng = np.random.RandomState(_test_seed())
    data = tf.convert_to_tensor(rng.randn(*shape))

    true_aggregation = (0,) + (() if aggregation is None else tuple(
        [a + 1 for a in util.flatten_tree(aggregation)]))
    true_mean = np.mean(data, true_aggregation)
    true_cov = _gen_cov(data, true_aggregation)

    def kernel(rcs, idx):
      rcs, _ = fun_mcmc.running_covariance_step(
          rcs, data[idx], axis=aggregation)
      return (rcs, idx + 1), ()

    (rcs, _), _ = fun_mcmc.trace(
        state=(fun_mcmc.running_covariance_init(true_mean.shape,
                                                data[0].dtype), 0),
        fn=kernel,
        num_steps=len(data),
        trace_fn=lambda *args: ())
    self.assertAllClose(true_mean, rcs.mean)
    self.assertAllClose(true_cov, rcs.covariance)

  def testRunningCovarianceMaxPoints(self):
    window_size = 100
    rng = np.random.RandomState(_test_seed())
    data = tf.convert_to_tensor(
        np.concatenate(
            [
                rng.randn(window_size, 2),
                np.array([1., 2.]) +
                np.array([2., 3.]) * rng.randn(window_size * 10, 2)
            ],
            axis=0,
        ))

    def kernel(rvs, idx):
      rvs, _ = fun_mcmc.running_covariance_step(
          rvs, data[idx], window_size=window_size)
      return (rvs, idx + 1), (rvs.mean, rvs.covariance)

    _, (mean, cov) = fun_mcmc.trace(
        state=(fun_mcmc.running_covariance_init([2], data.dtype), 0),
        fn=kernel,
        num_steps=len(data),
    )
    # Up to window_size, we compute the running mean/variance exactly.
    self.assertAllClose(
        np.mean(data[:window_size], axis=0), mean[window_size - 1])
    self.assertAllClose(
        _gen_cov(data[:window_size], axis=0), cov[window_size - 1])
    # After window_size, we're doing exponential moving average, and pick up the
    # mean/variance after the change in the distribution. Since the moving
    # average is computed only over ~window_size points, this test is rather
    # noisy.
    self.assertAllClose(np.array([1., 2.]), mean[-1], atol=0.2)
    self.assertAllClose(np.array([[4., 0.], [0., 9.]]), cov[-1], atol=1.)

  @parameterized.named_parameters(
      ('BasicScalar', (10, 20), 1),
      ('BatchedScalar', (10, 5, 20), 2),
      ('BasicVector', (10, 5, 20), 1),
      ('BatchedVector', (10, 5, 20, 7), 2),
  )
  def testPotentialScaleReduction(self, chain_shape, independent_chain_ndims):
    rng = np.random.RandomState(_test_seed())
    chain_means = rng.randn(*((1,) + chain_shape[1:])).astype(np.float32)
    chains = 0.4 * rng.randn(*chain_shape).astype(np.float32) + chain_means

    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains, independent_chain_ndims=independent_chain_ndims)

    chains = tf.convert_to_tensor(chains)
    psrs, _ = fun_mcmc.trace(
        state=fun_mcmc.potential_scale_reduction_init(chain_shape[1:],
                                                      tf.float32),
        fn=lambda psrs: fun_mcmc.potential_scale_reduction_step(  # pylint: disable=g-long-lambda
            psrs, chains[psrs.num_points]),
        num_steps=chain_shape[0],
        trace_fn=lambda *_: ())

    running_rhat = fun_mcmc.potential_scale_reduction_extract(
        psrs, independent_chain_ndims=independent_chain_ndims)
    self.assertAllClose(true_rhat, running_rhat)

  @parameterized.named_parameters(
      ('Basic', (), None, None),
      ('Batched1', (2,), 1, None),
      ('Batched2', (3, 2), 1, None),
      ('Aggregated0', (3, 2), 1, 0),
      ('Aggregated01', (3, 4, 2), 1, (0, 1)),
  )
  def testRunningApproximateAutoCovariance(self, state_shape, event_ndims,
                                           aggregation):
    # We'll use HMC as the source of our chain.
    # While HMC is being sampled, we also compute the running autocovariance.
    step_size = 0.2
    num_steps = 1000
    num_leapfrog_steps = 10
    max_lags = 300

    state = tf.constant(np.zeros(state_shape).astype(np.float32))

    def target_log_prob_fn(x):
      lp = -0.5 * tf.square(x)
      if event_ndims is None:
        return lp, ()
      else:
        return tf.reduce_sum(lp, -1), ()

    def kernel(hmc_state, raac_state, seed):
      if not self._is_on_jax:
        hmc_seed = _test_seed()
      else:
        hmc_seed, seed = util.split_seed(seed, 2)
      hmc_state, hmc_extra = fun_mcmc.hamiltonian_monte_carlo(
          hmc_state,
          step_size=step_size,
          num_integrator_steps=num_leapfrog_steps,
          target_log_prob_fn=target_log_prob_fn,
          seed=hmc_seed)
      raac_state, _ = fun_mcmc.running_approximate_auto_covariance_step(
          raac_state, hmc_state.state, axis=aggregation)
      return (hmc_state, raac_state, seed), hmc_extra

    if not self._is_on_jax:
      seed = _test_seed()
    else:
      seed = self._make_seed(_test_seed())

    # Subtle: Unlike TF, JAX needs a data dependency from the inputs to outputs
    # for the jit to do anything.
    (_, raac_state, _), chain = tf.function(lambda state, seed: fun_mcmc.trace(  # pylint: disable=g-long-lambda
        state=(
            fun_mcmc.hamiltonian_monte_carlo_init(state, target_log_prob_fn),
            fun_mcmc.running_approximate_auto_covariance_init(
                max_lags=max_lags,
                state_shape=state_shape,
                dtype=state.dtype,
                axis=aggregation),
            seed,
        ),
        fn=kernel,
        num_steps=num_steps,
        trace_fn=lambda state, extra: state[0].state))(state, seed)

    true_aggregation = (0,) + (() if aggregation is None else tuple(
        [a + 1 for a in util.flatten_tree(aggregation)]))
    true_variance = np.array(
        tf.math.reduce_variance(np.array(chain), true_aggregation))
    true_autocov = np.array(
        tfp.stats.auto_correlation(np.array(chain), axis=0, max_lags=max_lags))
    if aggregation is not None:
      true_autocov = tf.reduce_mean(
          true_autocov, [a + 1 for a in util.flatten_tree(aggregation)])

    self.assertAllClose(true_variance, raac_state.auto_covariance[0], 1e-5)
    self.assertAllClose(
        true_autocov,
        raac_state.auto_covariance / raac_state.auto_covariance[0],
        atol=0.1)

  @parameterized.named_parameters(
      ('Positional1', 0.),
      ('Positional2', (0., 1.)),
      ('Named1', {'a': 0.}),
      ('Named2', {'a': 0., 'b': 1.}),
  )
  def testSurrogateLossFn(self, state):
    def grad_fn(*args, **kwargs):
      # This is uglier than user code due to the parameterized test...
      new_state = util.unflatten_tree(state, util.flatten_tree((args, kwargs)))
      return util.map_tree(lambda x: x + 1., new_state), new_state
    loss_fn = fun_mcmc.make_surrogate_loss_fn(grad_fn)

    # Mutate the state to make sure we didn't capture anything.
    state = util.map_tree(lambda x: x + 1., state)
    ret, extra, grads = fun_mcmc.call_potential_fn_with_grads(loss_fn, state)
    # The default is 0.
    self.assertAllClose(0., ret)
    # The gradients of the surrogate loss are state + 1.
    self.assertAllClose(util.map_tree(lambda x: x + 1., state), grads)
    self.assertAllClose(state, extra)

  def testSurrogateLossFnDecorator(self):
    @fun_mcmc.make_surrogate_loss_fn(loss_value=1.)
    def loss_fn(_):
      return 3., 2.

    ret, extra, grads = fun_mcmc.call_potential_fn_with_grads(loss_fn, 0.)
    self.assertAllClose(1., ret)
    self.assertAllClose(2., extra)
    self.assertAllClose(3., grads)

  @parameterized.named_parameters(
      ('Probability', True),
      ('Loss', False),
  )
  def testReparameterizeFn(self, track_volume):

    def potential_fn(x, y):
      return -x**2 + -y**2, ()

    def transport_map_fn(x, y):
      return [2 * x, 3 * y], ((), tf.math.log(2.) + tf.math.log(3.))

    def inverse_map_fn(x, y):
      return [x / 2, y / 3], ((), -tf.math.log(2.) - tf.math.log(3.))

    transport_map_fn.inverse = inverse_map_fn

    (transformed_potential_fn,
     transformed_init_state) = fun_mcmc.reparameterize_potential_fn(
         potential_fn, transport_map_fn, [2., 3.], track_volume=track_volume)

    self.assertIsInstance(transformed_init_state, list)
    self.assertAllClose([1., 1.], transformed_init_state)
    transformed_potential, (orig_space, _, _) = transformed_potential_fn(1., 1.)
    potential = potential_fn(2., 3.)[0]
    if track_volume:
      potential += tf.math.log(2.) + tf.math.log(3.)

    self.assertAllClose([2., 3.], orig_space)
    self.assertAllClose(potential, transformed_potential)


class FunMCMCTestJAX(FunMCMCTestTensorFlow):

  _is_on_jax = True

  def setUp(self):
    super(FunMCMCTestJAX, self).setUp()
    backend.set_backend(backend.JAX, backend.MANUAL_TRANSFORMS)

  def _make_seed(self, seed):
    return jax_random.PRNGKey(seed)


if __name__ == '__main__':
  real_tf.test.main()
