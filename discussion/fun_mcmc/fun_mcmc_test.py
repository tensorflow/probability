# Copyright 2018 The TensorFlow Probability Authors.
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

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from discussion import fun_mcmc
from tensorflow_probability.python.internal import test_util as tfp_test_util

tfb = tfp.bijectors
tfd = tfp.distributions

tf.enable_v2_behavior()


def _no_compile(fn):
  return fn


class FunMCMCTest(tf.test.TestCase, parameterized.TestCase):

  def testTraceSingle(self):
    def fun(x):
      if x is None:
        x = 0.
      return x + 1., 2 * x

    x, e_trace = fun_mcmc.trace(
        state=None, fn=fun, num_steps=5, trace_fn=lambda _, xp1: xp1)

    self.assertAllEqual(5., x.numpy())
    self.assertAllEqual([0., 2., 4., 6., 8.], e_trace.numpy())

  def testTraceNested(self):
    def fun(x, y):
      if x is None:
        x = 0.
      return (x + 1., y + 2.), ()

    (x, y), (x_trace, y_trace) = fun_mcmc.trace(
        state=(None, 0.), fn=fun, num_steps=5, trace_fn=lambda xy, _: xy)

    self.assertAllEqual(5., x)
    self.assertAllEqual(10., y)
    self.assertAllEqual([1., 2., 3., 4., 5.], x_trace)
    self.assertAllEqual([2., 4., 6., 8., 10.], y_trace)

  def testTraceTrace(self):
    def fun(x):
      return fun_mcmc.trace(x, lambda x: (x + 1., ()), 2, lambda *args: ())

    x, _ = fun_mcmc.trace(0., fun, 2, lambda *args: ())
    self.assertAllEqual(4., x)

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
    self.assertIsInstance(orig_space, list)
    lp = log_prob_fn(2., 3.)[0] + sum(
        b.forward_log_det_jacobian(1., event_ndims=0) for b in bijectors)

    self.assertAllClose([2., 3.], orig_space)
    self.assertAllClose(lp, tlp)

  @parameterized.parameters(
      fun_mcmc.leapfrog_step,
      fun_mcmc.ruth4_step,
      fun_mcmc.blanes_3_stage_step,
      fun_mcmc.blanes_4_stage_step,
  )
  def testIntegratorStep(self, method):

    def target_log_prob_fn(q):
      return -q**2, 1.

    def kinetic_energy_fn(p):
      return tf.abs(p)**3., 2.

    state, extras = method(
        integrator_step_state=fun_mcmc.IntegratorStepState(
            state=1., state_grads=None, momentum=2.),
        step_size=0.1,
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn)

    self.assertEqual(1., extras.state_extra)
    self.assertEqual(2., extras.kinetic_energy_extra)

    initial_hamiltonian = -target_log_prob_fn(1.)[0] + kinetic_energy_fn(2.)[0]
    fin_hamiltonian = -target_log_prob_fn(state.state)[0] + kinetic_energy_fn(
        state.momentum)[0]

    self.assertAllClose(fin_hamiltonian, initial_hamiltonian, atol=0.2)

  def testMetropolisHastingsStep(self):
    accepted, is_accepted, _ = fun_mcmc.metropolis_hastings_step(
        current_state=0., proposed_state=1., energy_change=-np.inf)
    self.assertAllEqual(1., accepted)
    self.assertAllEqual(True, is_accepted)

    accepted, is_accepted, _ = fun_mcmc.metropolis_hastings_step(
        current_state=0., proposed_state=1., energy_change=np.inf)
    self.assertAllEqual(0., accepted)
    self.assertAllEqual(False, is_accepted)

    accepted, is_accepted, _ = fun_mcmc.metropolis_hastings_step(
        current_state=0., proposed_state=1., energy_change=np.nan)
    self.assertAllEqual(0., accepted)
    self.assertAllEqual(False, is_accepted)

    accepted, is_accepted, _ = fun_mcmc.metropolis_hastings_step(
        current_state=None, proposed_state=1., energy_change=np.nan)
    self.assertAllEqual(1., accepted)
    self.assertAllEqual(False, is_accepted)

    accepted, is_accepted, _ = fun_mcmc.metropolis_hastings_step(
        current_state=None,
        proposed_state=1.,
        log_uniform=-10.,
        energy_change=-np.log(0.5))
    self.assertAllEqual(1., accepted)
    self.assertAllEqual(True, is_accepted)

    accepted, is_accepted, _ = fun_mcmc.metropolis_hastings_step(
        current_state=None,
        proposed_state=1.,
        log_uniform=0.,
        energy_change=-np.log(0.5))
    self.assertAllEqual(1., accepted)
    self.assertAllEqual(False, is_accepted)

    accepted, _, _ = fun_mcmc.metropolis_hastings_step(
        current_state=tf.zeros(1000),
        proposed_state=tf.ones(1000),
        energy_change=-tf.math.log(0.5 * tf.ones(1000)),
        seed=tfp_test_util.test_seed())
    self.assertAllClose(0.5, tf.reduce_mean(input_tensor=accepted), rtol=0.1)

  def testMetropolisHastingsStepStructure(self):
    struct_type = collections.namedtuple('Struct', 'a, b')

    current = struct_type([1, 2], (3, [4, [0, 0]]))
    proposed = struct_type([5, 6], (7, [8, [0, 0]]))

    accepted, is_accepted, _ = fun_mcmc.metropolis_hastings_step(
        current_state=current, proposed_state=proposed, energy_change=-np.inf)
    self.assertAllEqual(True, is_accepted)
    self.assertAllEqual(tf.nest.flatten(proposed), tf.nest.flatten(accepted))

  def testBasicHMC(self):
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
        seed=tfp_test_util.test_seed()))

    _, chain = fun_mcmc.trace(
        state=fun_mcmc.HamiltonianMonteCarloState(state),
        fn=kernel,
        num_steps=num_steps,
        trace_fn=lambda state, extra: state.state_extra[0])

    sample_mean = tf.reduce_mean(input_tensor=chain, axis=[0, 1])
    sample_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    true_samples = target_dist.sample(4096, seed=tfp_test_util.test_seed())

    true_mean = tf.reduce_mean(input_tensor=true_samples, axis=0)
    true_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    self.assertAllClose(true_mean, sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(true_cov, sample_cov, rtol=0.1, atol=0.1)

  @parameterized.parameters(
      (tf.function, 1),
      (_no_compile, 3)
  )
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
          seed=tfp_test_util.test_seed())

      fun_mcmc.trace(
          state=fun_mcmc.HamiltonianMonteCarloState(tf.zeros([1])),
          fn=kernel,
          num_steps=4,
          trace_fn=lambda *args: ())
    trace()

    self.assertEqual(expected_count, counter[0])

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
          seed=tfp_test_util.test_seed())

      fun_mcmc.trace(
          state=fun_mcmc.hamiltonian_monte_carlo_init(
              state=tf.zeros([1]),
              target_log_prob_fn=target_log_prob_fn
          ),
          fn=kernel,
          num_steps=4,
          trace_fn=lambda *args: ())

    trace()

    self.assertEqual(2, counter[0])

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

      def kernel(hmc_state, step_size, step):
        hmc_state, hmc_extra = fun_mcmc.hamiltonian_monte_carlo(
            hmc_state,
            step_size=step_size,
            num_integrator_steps=num_leapfrog_steps,
            target_log_prob_fn=target_log_prob_fn)

        rate = tf.compat.v1.train.polynomial_decay(
            0.01,
            global_step=step,
            power=0.5,
            decay_steps=num_adapt_steps,
            end_learning_rate=0.)
        mean_p_accept = tf.reduce_mean(
            input_tensor=tf.exp(tf.minimum(0., hmc_extra.log_accept_ratio)))
        step_size = fun_mcmc.sign_adaptation(
            step_size,
            output=mean_p_accept,
            set_point=0.9,
            adaptation_rate=rate)

        return (hmc_state, step_size, step + 1), hmc_extra

      _, (chain, log_accept_ratio_trace) = fun_mcmc.trace(
          (fun_mcmc.HamiltonianMonteCarloState(state), step_size, 0),
          kernel,
          num_adapt_steps + num_steps,
          trace_fn=lambda state, extra: (state[0].state_extra[0], extra.
                                         log_accept_ratio))
      true_samples = target_dist.sample(4096, seed=tfp_test_util.test_seed())
      return chain, log_accept_ratio_trace, true_samples

    chain, log_accept_ratio_trace, true_samples = computation(state)

    log_accept_ratio_trace = log_accept_ratio_trace[num_adapt_steps:]
    chain = chain[num_adapt_steps:]

    sample_mean = tf.reduce_mean(input_tensor=chain, axis=[0, 1])
    sample_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    true_mean = tf.reduce_mean(input_tensor=true_samples, axis=0)
    true_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    self.assertAllClose(true_mean, sample_mean, rtol=0.05, atol=0.05)
    self.assertAllClose(true_cov, sample_cov, rtol=0.05, atol=0.05)
    self.assertAllClose(
        tf.reduce_mean(
            input_tensor=tf.exp(tf.minimum(0., log_accept_ratio_trace))),
        0.9,
        rtol=0.1)

  def testSignAdaptation(self):
    new_control = fun_mcmc.sign_adaptation(
        control=1., output=0.5, set_point=1., adaptation_rate=0.1)
    self.assertAllClose(new_control, 1. / 1.1)

    new_control = fun_mcmc.sign_adaptation(
        control=1., output=0.5, set_point=0., adaptation_rate=0.1)
    self.assertAllClose(new_control, 1. * 1.1)

  def testWrapTransitionKernel(self):

    class TestKernel(tfp.mcmc.TransitionKernel):

      def one_step(self, current_state, previous_kernel_results):
        return [x + 1 for x in current_state], previous_kernel_results + 1

      def bootstrap_results(self, current_state):
        return sum(current_state)

      def is_calibrated(self):
        return True

    def kernel(state, pkr):
      return fun_mcmc.transition_kernel_wrapper(state, pkr, TestKernel())

    (final_state, final_kr), _ = fun_mcmc.trace(({
        'x': 0.,
        'y': 1.
    }, None), kernel, 2, trace_fn=lambda *args: ())
    self.assertAllEqual({'x': 2., 'y': 3.}, self.evaluate(final_state))
    self.assertAllEqual(1. + 2., self.evaluate(final_kr))


if __name__ == '__main__':
  tf.test.main()
