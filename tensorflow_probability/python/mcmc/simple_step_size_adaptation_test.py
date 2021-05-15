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
"""Tests for SimpleStepSizeAdaptation kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


_RATE = 1.01

FakeMHKernelResults = collections.namedtuple(
    'FakeMHKernelResults', 'accepted_results, log_accept_ratio')


class FakeMHKernel(tfp.mcmc.TransitionKernel):

  def __init__(self,
               inner_kernel,
               log_accept_ratio,
               store_parameters_in_results=False):
    self.inner_kernel = inner_kernel
    self.parameters = dict(
        inner_kernel=inner_kernel,
        log_accept_ratio=log_accept_ratio,
        store_parameters_in_results=store_parameters_in_results,
    )

  def one_step(self, current_state, previous_kernel_results, seed=None):
    new_state, new_accepted_results = self.parameters['inner_kernel'].one_step(
        current_state, previous_kernel_results.accepted_results, seed=seed)
    return new_state, previous_kernel_results._replace(
        accepted_results=new_accepted_results)

  def bootstrap_results(self, current_state):
    return FakeMHKernelResults(
        accepted_results=self.parameters['inner_kernel'].bootstrap_results(
            current_state),
        log_accept_ratio=tf.convert_to_tensor(
            self.parameters['log_accept_ratio']),
    )

  @property
  def is_calibrated(self):
    return True

  @property
  def experimental_shard_axis_names(self):
    return self.inner_kernel.experimental_shard_axis_names

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(
        inner_kernel=self.inner_kernel.experimental_with_shard_axes(
            shard_axis_names))


FakeSteppedKernelResults = collections.namedtuple('FakeSteppedKernelResults',
                                                  'step_size')


class FakeSteppedKernel(tfp.mcmc.TransitionKernel):

  def __init__(self, step_size, store_parameters_in_results=False,
               experimental_shard_axis_names=None):
    self.parameters = dict(
        step_size=step_size,
        store_parameters_in_results=store_parameters_in_results,
        experimental_shard_axis_names=experimental_shard_axis_names)

  def one_step(self, current_state, previous_kernel_results, seed=None):
    return current_state, previous_kernel_results

  def bootstrap_results(self, current_state):
    return FakeSteppedKernelResults(
        step_size=tf.nest.map_structure(tf.convert_to_tensor,
                                        self.parameters['step_size']))

  @property
  def experimental_shard_axis_names(self):
    return self.parameters['experimental_shard_axis_names']

  def experimental_with_shard_axes(self, shard_axes):
    return self.copy(experimental_shard_axis_names=shard_axes)

  @property
  def is_calibrated(self):
    return False


FakeWrapperKernelResults = collections.namedtuple('FakeWrapperKernelResults',
                                                  'inner_results')


class FakeWrapperKernel(tfp.mcmc.TransitionKernel):

  def __init__(self, inner_kernel):
    self.parameters = dict(inner_kernel=inner_kernel)

  @property
  def inner_kernel(self):
    return self.parameters['inner_kernel']

  def one_step(self, current_state, previous_kernel_results, seed=None):
    new_state, new_inner_results = self.inner_kernel.one_step(
        current_state, previous_kernel_results.inner_results, seed=seed)
    return new_state, previous_kernel_results._replace(
        inner_results=new_inner_results)

  def bootstrap_results(self, current_state):
    return FakeWrapperKernelResults(
        inner_results=self.inner_kernel.bootstrap_results(current_state))

  @property
  def is_calibrated(self):
    return self.inner_kernel.is_calibrated


@test_util.test_all_tf_execution_regimes
class SimpleStepSizeAdaptationTest(test_util.TestCase):

  def testTurnOnStoreParametersInKernelResults(self):
    kernel = FakeWrapperKernel(FakeSteppedKernel(step_size=0.5))
    self.assertFalse(
        kernel.inner_kernel.parameters['store_parameters_in_results'])
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        adaptation_rate=_RATE - 1. - 1.,
        validate_args=True)
    self.assertTrue(kernel.inner_kernel.inner_kernel
                    .parameters['store_parameters_in_results'])

  def testStepSizeIncreases(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=0.1),
        # Mean is just over the target_accept_prob.
        log_accept_ratio=tf.stack(
            [tf.math.log(0.74),
             tf.math.log(0.76),
             tf.math.log(0.76)]))
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        adaptation_rate=_RATE - 1.,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(tf.zeros(3))
    for _ in range(2):
      _, kernel_results = kernel.one_step(tf.zeros(3), kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    self.assertAllClose(0.1 * _RATE, step_size)

  def testStepSizeDecreases(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=0.1),
        # Mean is just under the target_accept_prob.
        # Log-accept ratio over 0. gets clipped to 0.
        log_accept_ratio=tf.stack(
            [tf.math.log(0.24),
             tf.math.log(1.),
             tf.math.log(100.)]))
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        adaptation_rate=_RATE - 1.,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(tf.zeros(3))
    for _ in range(2):
      _, kernel_results = kernel.one_step(tf.zeros(3), kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    self.assertAllClose(0.1 / _RATE, step_size)

  def testAdaptationSteps(self):
    kernel = FakeMHKernel(FakeSteppedKernel(step_size=0.1), log_accept_ratio=0.)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=2,
        adaptation_rate=_RATE - 1.,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(0.)
    step_kernel_results = []
    for _ in range(4):
      _, kernel_results = kernel.one_step(0., kernel_results)
      step_kernel_results.append(kernel_results)

    step_sizes = self.evaluate([
        kernel_results.inner_results.accepted_results.step_size
        for kernel_results in step_kernel_results
    ])

    self.assertAllClose(0.1, step_sizes[0])
    self.assertAllClose(0.1 * _RATE, step_sizes[1])
    self.assertAllClose(0.1 * _RATE**2, step_sizes[2])
    self.assertAllClose(0.1 * _RATE**2, step_sizes[3])

  def testAdaptiveAdaptation(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=0.1),
        # Just over the target_accept_prob (set to 0.5 below).
        log_accept_ratio=tf.stack(
            [tf.math.log(0.49),
             tf.math.log(0.49),
             tf.math.log(0.51)]))
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        adaptation_rate=_RATE - 1.,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(tf.zeros(3))
    kernel_results = kernel_results._replace(
        target_accept_prob=0.5, adaptation_rate=0.1)
    for _ in range(2):
      _, kernel_results = kernel.one_step(tf.zeros(3), kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size)

    self.assertAllClose(0.1 / 1.1, step_size)

  def testListStep(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=tf.constant([0.1, 0.2, 0.3])),
        log_accept_ratio=tf.stack(
            [tf.math.log(0.74),
             tf.math.log(0.76),
             tf.math.log(0.76)]))
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        adaptation_rate=_RATE - 1.,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(tf.zeros(3))
    for _ in range(2):
      _, kernel_results = kernel.one_step(tf.zeros(3), kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    self.assertAllClose([0.1 / _RATE, 0.2 * _RATE, 0.3 * _RATE], step_size)

  def testWrapped(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=0.1),
        # Just over the target_accept_prob.
        log_accept_ratio=tf.math.log(0.76))
    kernel = FakeWrapperKernel(kernel)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        adaptation_rate=_RATE - 1.,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(0.)
    for _ in range(2):
      _, kernel_results = kernel.one_step(0., kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.inner_results.accepted_results.step_size)

    self.assertAllClose(0.1 * _RATE, step_size)

  def testChainLogProbScalarTarget(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=tf.constant([0.1, 0.2])),
        log_accept_ratio=tf.stack([tf.math.log(0.74),
                                   tf.math.log(0.76)]))
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        log_accept_prob_getter_fn=(
            lambda pkr: tf.minimum(0., pkr.log_accept_ratio)),
        adaptation_rate=_RATE - 1.,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(tf.zeros(2))
    for _ in range(2):
      _, kernel_results = kernel.one_step(tf.zeros(2), kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    self.assertAllClose([0.1 / _RATE, 0.2 * _RATE], step_size)

  def testChainLogProbChainTarget(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=tf.constant([0.1, 0.2])),
        log_accept_ratio=tf.stack([tf.math.log(0.74),
                                   tf.math.log(0.76)]))
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        log_accept_prob_getter_fn=(
            lambda pkr: tf.minimum(0., pkr.log_accept_ratio)),
        validate_args=True,
        adaptation_rate=_RATE - 1.,
        target_accept_prob=tf.stack([0.7, 0.8]))

    kernel_results = kernel.bootstrap_results(tf.zeros(2))
    for _ in range(2):
      _, kernel_results = kernel.one_step(tf.zeros(2), kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    self.assertAllClose([0.1 * _RATE, 0.2 / _RATE], step_size)

  @parameterized.parameters((-1., r'`target_accept_prob` must be > 0.'),
                            (0., r'`target_accept_prob` must be > 0.'),
                            (0.999, None),
                            (1., r'`target_accept_prob` must be < 1.'))
  def testTargetAcceptanceProbChecks(self, target_accept_prob, message):

    def _impl():
      kernel = FakeMHKernel(
          FakeSteppedKernel(step_size=1.), log_accept_ratio=0.)
      kernel = tfp.mcmc.SimpleStepSizeAdaptation(
          kernel,
          num_adaptation_steps=1,
          target_accept_prob=target_accept_prob,
          validate_args=True)
      self.evaluate(kernel.bootstrap_results(tf.zeros(2)))

    if message:
      with self.assertRaisesOpError(message):
        _impl()
    else:
      _impl()

  def testIsCalibrated(self):
    test_kernel = collections.namedtuple('TestKernel', 'is_calibrated')
    self.assertTrue(
        tfp.mcmc.SimpleStepSizeAdaptation(test_kernel(True), 1).is_calibrated)
    self.assertFalse(
        tfp.mcmc.SimpleStepSizeAdaptation(test_kernel(False), 1).is_calibrated)

  def testCustomReduceFn(self):
    log_accept_ratio = tf.constant(
        [np.log(0.1), np.log(1.)])
    state = [
        tf.zeros([2]),
    ]

    old_step_size = 1.
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=old_step_size),
        log_accept_ratio=log_accept_ratio)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        adaptation_rate=tf.constant(_RATE - 1., dtype=tf.float64),
        reduce_fn=tf.reduce_max,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(state)
    for _ in range(2):
      _, kernel_results = kernel.one_step(state, kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    # If reduce_fn was left at default, this would have decreased.
    self.assertAllClose(old_step_size * _RATE, step_size)


# Reduce test weight by not running the (slow) `eager_no_tf_function` regime.
@test_util.test_graph_and_eager_modes()
class SimpleStepSizeAdaptationExampleTest(test_util.TestCase):

  @test_util.numpy_disable_gradient_test('HMC')
  def test_example(self):
    target_log_prob_fn = tfd.Normal(loc=0., scale=1.).log_prob
    num_burnin_steps = 500
    num_results = 500
    num_chains = 64
    step_size = 0.1

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=2,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))

    @tf.function(autograph=False)
    def do_sampling():
      _, log_accept_ratio = tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          current_state=tf.zeros(num_chains),
          kernel=kernel,
          trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio,
          seed=test_util.test_seed())
      return log_accept_ratio

    log_accept_ratio = do_sampling()
    p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(
        tf.minimum(log_accept_ratio, 0.)))

    self.assertAllClose(0.75, self.evaluate(p_accept), atol=0.15)


@test_util.test_all_tf_execution_regimes
class SimpleStepSizeAdaptationStaticBroadcastingTest(test_util.TestCase):
  use_static_shape = True

  @parameterized.parameters(
      (np.float64(1.), np.float64(1. / _RATE)),
      ([np.float64(1.), np.ones([3, 1])], [
          np.float64(1. / _RATE),
          np.array([[1. / _RATE], [1. * _RATE], [1. / _RATE]])
      ]),
      ([np.float64(1.), np.ones([2, 3, 1])], [
          np.float64(1. / _RATE),
          np.array([[[1. / _RATE], [1. * _RATE], [1. / _RATE]],
                    [[1. * _RATE], [1. * _RATE], [1. / _RATE]]])
      ]),
      ([np.float64(1.), np.ones([2, 1, 1])], [
          np.float64(1. / _RATE),
          np.array([[[1. / _RATE]], [[1. * _RATE]]])
      ]),
      ([np.float64(1.), np.ones([1, 3, 1])], [
          np.float64(1. / _RATE),
          np.array([[[1. / _RATE], [1. * _RATE], [1. / _RATE]]])
      ]),
      ([np.float64(1.), np.ones([1, 1, 1])], [
          np.float64(1. / _RATE),
          np.array([[[1. / _RATE]]])
      ]),
      ([np.float64(1.), np.ones([1, 1])], [
          np.float64(1. / _RATE),
          np.array([[1. / _RATE]])
      ]),
      ([np.float64(1.), np.ones([1])], [
          np.float64(1. / _RATE),
          np.array([1. / _RATE])
      ]),
  )
  def testBroadcasting(self, old_step_size, new_step_size):
    log_accept_ratio = tf.constant(
        [[np.log(0.73), np.log(0.76), np.log(0.73)],
         [np.log(0.77), np.log(0.77), np.log(0.73)]],
        dtype=tf.float64)
    log_accept_ratio = tf1.placeholder_with_default(
        log_accept_ratio,
        shape=log_accept_ratio.shape if self.use_static_shape else None)
    state = [
        tf.zeros([2, 3], dtype=tf.float64),
        tf.zeros([2, 3, 4], dtype=tf.float64)
    ]

    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=old_step_size),
        log_accept_ratio=log_accept_ratio)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        adaptation_rate=tf.constant(_RATE - 1., dtype=tf.float64),
        validate_args=True)

    kernel_results = kernel.bootstrap_results(state)
    for _ in range(2):
      _, kernel_results = kernel.one_step(state, kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    self.assertAllClose(new_step_size, step_size)

  def testShouldPropagateShardAxisNames(self):
    test_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        FakeMHKernel(FakeSteppedKernel(step_size=1.), log_accept_ratio=0.),
        num_adaptation_steps=1)
    self.assertIsNone(test_kernel.experimental_shard_axis_names)
    sharded_test_kernel = test_kernel.experimental_with_shard_axes(['foo'])
    self.assertListEqual(
        sharded_test_kernel.experimental_shard_axis_names, ['foo'])
    sharded_inner_kernel = sharded_test_kernel.inner_kernel
    self.assertListEqual(
        sharded_inner_kernel.experimental_shard_axis_names, ['foo'])
    self.assertListEqual(
        sharded_inner_kernel.inner_kernel.experimental_shard_axis_names,
        ['foo'])


@test_util.test_all_tf_execution_regimes
class SimpleStepSizeAdaptationDynamicBroadcastingTest(
    SimpleStepSizeAdaptationStaticBroadcastingTest):
  use_static_shape = False


if __name__ == '__main__':
  tf.test.main()
