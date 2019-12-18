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
"""Tests for DualAveragingStepSizeAdaptation kernel."""

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


_INITIAL_T = 10.0
_EXPLORATION_SHRINKAGE = 0.05

# Hardcoded expected step after two steps. Calculate with
# tf.math.exp(tf.math.log(10.) - (err - 0.75) / ((10. + 1.) * 0.05))
_UPDATE_M05 = 9.131008  # err = -0.05
_UPDATE_M02 = 9.642897  # err = -0.02
_UPDATE_M01 = 9.819825  # err = -0.01
_UPDATE_0 = 10.  # err = 0
_UPDATE_01 = 10.183481  # err = +0.01


def _set_seed(seed):
  """Helper which uses graph seed if using eager."""
  # TODO(b/68017812): Deprecate once eager correctly supports seed.
  if tf.executing_eagerly():
    return None
  return seed


FakeMHKernelResults = collections.namedtuple(
    'FakeMHKernelResults', 'accepted_results, log_accept_ratio')


class FakeMHKernel(tfp.mcmc.TransitionKernel):

  def __init__(self,
               inner_kernel,
               log_accept_ratio,
               store_parameters_in_results=False):
    self.parameters = dict(
        inner_kernel=inner_kernel,
        log_accept_ratio=log_accept_ratio,
        store_parameters_in_results=store_parameters_in_results,
    )

  def one_step(self, current_state, previous_kernel_results):
    new_state, new_accepted_results = self.parameters['inner_kernel'].one_step(
        current_state, previous_kernel_results.accepted_results)
    return new_state, previous_kernel_results._replace(
        accepted_results=new_accepted_results)

  def bootstrap_results(self, current_state):
    return FakeMHKernelResults(
        accepted_results=self.parameters['inner_kernel'].bootstrap_results(
            current_state),
        log_accept_ratio=tf.convert_to_tensor(
            value=self.parameters['log_accept_ratio']),
    )

  @property
  def is_calibrated(self):
    return True


FakeSteppedKernelResults = collections.namedtuple('FakeSteppedKernelResults',
                                                  'step_size')


class FakeSteppedKernel(tfp.mcmc.TransitionKernel):

  def __init__(self, step_size, store_parameters_in_results=False):
    self.parameters = dict(
        step_size=step_size,
        store_parameters_in_results=store_parameters_in_results)

  def one_step(self, current_state, previous_kernel_results):
    return current_state, previous_kernel_results

  def bootstrap_results(self, current_state):
    return FakeSteppedKernelResults(
        step_size=tf.nest.map_structure(tf.convert_to_tensor,
                                        self.parameters['step_size']))

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

  def one_step(self, current_state, previous_kernel_results):
    new_state, new_inner_results = self.inner_kernel.one_step(
        current_state, previous_kernel_results.inner_results)
    return new_state, previous_kernel_results._replace(
        inner_results=new_inner_results)

  def bootstrap_results(self, current_state):
    return FakeWrapperKernelResults(
        inner_results=self.inner_kernel.bootstrap_results(current_state))

  @property
  def is_calibrated(self):
    return self.inner_kernel.is_calibrated


@test_util.test_all_tf_execution_regimes
class DualAveragingStepSizeAdaptationTest(test_util.TestCase):

  def testTurnOnStoreParametersInKernelResults(self):
    kernel = FakeWrapperKernel(FakeSteppedKernel(step_size=0.5))
    self.assertFalse(
        kernel.inner_kernel.parameters['store_parameters_in_results'])
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        validate_args=True)
    self.assertTrue(kernel.inner_kernel.inner_kernel
                    .parameters['store_parameters_in_results'])

  def testListStep(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=tf.constant([0.1, 0.2, 0.3])),
        log_accept_ratio=tf.stack(
            [tf.math.log(0.74),
             tf.math.log(0.76),
             tf.math.log(0.76)]))
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(tf.zeros(3))
    for _ in range(2):
      _, kernel_results = kernel.one_step(tf.zeros(3), kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    expected = tf.math.exp(
        tf.math.log(10. * tf.constant([0.1, 0.2, 0.3])) -
        tf.constant([0.01, -0.01, -0.01]) / (
            (_INITIAL_T + 1.) * _EXPLORATION_SHRINKAGE))
    self.assertAllClose(expected, step_size)

  def testWrapped(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=0.1),
        # Just over the target_accept_prob.
        log_accept_ratio=tf.math.log(0.76))
    kernel = FakeWrapperKernel(kernel)
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        validate_args=True)

    init_state = tf.constant(0.)
    kernel_results = kernel.bootstrap_results(init_state)
    for _ in range(2):
      _, kernel_results = kernel.one_step(init_state, kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.inner_results.accepted_results.step_size)

    expected = tf.math.exp(
        tf.math.log(10. * 0.1) -
        -0.01 / ((_INITIAL_T + 1.) * _EXPLORATION_SHRINKAGE))
    self.assertAllClose(expected, step_size)

  def testRecoversFromNaNAcceptProb(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=0.1),
        log_accept_ratio=tf.convert_to_tensor(np.nan))
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        validate_args=True)

    init_state = tf.constant(0.)
    kernel_results = kernel.bootstrap_results(init_state)
    for _ in range(2):
      _, kernel_results = kernel.one_step(init_state, kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size)

    self.assertTrue(np.isfinite(step_size))

  def testChainLogProbScalarTarget(self):
    init_step = tf.constant([0.1, 0.2])
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=init_step),
        log_accept_ratio=tf.stack([tf.math.log(0.74),
                                   tf.math.log(0.76)]))
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        log_accept_prob_getter_fn=(
            lambda pkr: tf.minimum(0., pkr.log_accept_ratio)),
        validate_args=True)

    kernel_results = kernel.bootstrap_results(tf.zeros(2))
    for _ in range(2):
      _, kernel_results = kernel.one_step(tf.zeros(2), kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    expected = tf.math.exp(
        tf.math.log(10. * init_step) -
        tf.constant([0.01, -0.01]) / (
            (_INITIAL_T + 1.) * _EXPLORATION_SHRINKAGE))
    self.assertAllClose(expected, step_size)

  def testChainLogProbChainTarget(self):
    init_step = tf.constant([0.1, 0.2])
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=init_step),
        log_accept_ratio=tf.stack([tf.math.log(0.74),
                                   tf.math.log(0.76)]))
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        log_accept_prob_getter_fn=(
            lambda pkr: tf.minimum(0., pkr.log_accept_ratio)),
        validate_args=True,
        target_accept_prob=tf.stack([0.7, 0.8]))

    kernel_results = kernel.bootstrap_results(tf.zeros(2))
    for _ in range(2):
      _, kernel_results = kernel.one_step(tf.zeros(2), kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    expected = tf.math.exp(
        tf.math.log(10. * init_step) -
        tf.constant([-0.04, 0.04]) / (
            (_INITIAL_T + 1.) * _EXPLORATION_SHRINKAGE))
    self.assertAllClose(expected, step_size)

  @parameterized.parameters((-1., '`target_accept_prob` must be > 0.'),
                            (0., '`target_accept_prob` must be > 0.'),
                            (0.999, None),
                            (1., '`target_accept_prob` must be < 1.'))
  def testTargetAcceptanceProbChecks(self, target_accept_prob, message):

    def _impl():
      kernel = FakeMHKernel(
          FakeSteppedKernel(step_size=1.), log_accept_ratio=0.)
      kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
          kernel,
          num_adaptation_steps=1,
          target_accept_prob=target_accept_prob,
          validate_args=True)
      self.evaluate(kernel.bootstrap_results(tf.zeros(2)))

    if message:
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, message):
        _impl()
    else:
      _impl()

  def testExample(self):
    tf.random.set_seed(test_util.test_seed())
    target_dist = tfd.JointDistributionSequential([
        tfd.Normal(0., 1.5),
        tfd.Independent(
            tfd.Normal(tf.zeros([2, 5], dtype=tf.float32), 5.),
            reinterpreted_batch_ndims=2),
    ])
    num_burnin_steps = 500
    num_results = 500
    num_chains = 64

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda *args: target_dist.log_prob(args),
        num_leapfrog_steps=2,
        step_size=target_dist.stddev(),
        seed=_set_seed(test_util.test_seed()))
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))

    _, log_accept_ratio = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=target_dist.sample(num_chains),
        kernel=kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio)

    p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(
        tf.minimum(log_accept_ratio, 0.)))

    self.assertAllClose(0.75, self.evaluate(p_accept), atol=0.15)

  def testIsCalibrated(self):
    test_kernel = collections.namedtuple('TestKernel', 'is_calibrated')
    self.assertTrue(
        tfp.mcmc.DualAveragingStepSizeAdaptation(test_kernel(True),
                                                 1).is_calibrated)
    self.assertFalse(
        tfp.mcmc.DualAveragingStepSizeAdaptation(test_kernel(False),
                                                 1).is_calibrated)


@test_util.test_all_tf_execution_regimes
class DualAveragingStepSizeAdaptationStaticBroadcastingTest(test_util.TestCase):
  use_static_shape = True

  @parameterized.parameters(
      (np.float64(1.), _UPDATE_M01),
      ([np.float64(1.), np.ones([3, 1])], [
          _UPDATE_M01,
          np.array([[_UPDATE_M02],
                    [_UPDATE_01],
                    [_UPDATE_M02]])
      ]),
      ([np.float64(1.), np.ones([2, 3, 1])], [
          _UPDATE_M01,
          np.array([[[_UPDATE_M05], [_UPDATE_01], [_UPDATE_M02]],
                    [[_UPDATE_01], [_UPDATE_01], [_UPDATE_M02]]])
      ]),
      ([np.float64(1.), np.ones([2, 1, 1])], [
          _UPDATE_M01,
          np.array([[[_UPDATE_M02]], [[_UPDATE_0]]])
      ]),
      ([np.float64(1.), np.ones([1, 3, 1])], [
          _UPDATE_M01,
          np.array([[[_UPDATE_M02],
                     [_UPDATE_01],
                     [_UPDATE_M02]]])
      ]),
      ([np.float64(1.), np.ones([1, 1, 1])], [
          _UPDATE_M01,
          np.array([[[_UPDATE_M01]]])
      ]),
      ([np.float64(1.), np.ones([1, 1])], [
          _UPDATE_M01,
          np.array([[_UPDATE_M01]])
      ]),
      ([np.float64(1.), np.ones([1])], [
          _UPDATE_M01,
          np.array([_UPDATE_M01])
      ]),
  )
  def testBroadcasting(self, old_step_size, new_step_size):
    log_accept_ratio = tf.constant(
        np.log([[0.70, 0.76, 0.73],
                [0.76, 0.76, 0.73]]),
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
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        target_accept_prob=0.75,
        num_adaptation_steps=1,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(state)
    for _ in range(2):
      _, kernel_results = kernel.one_step(state, kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size)

    self.assertAllClose(new_step_size, step_size)


@test_util.test_all_tf_execution_regimes
class DualAveragingStepSizeAdaptationDynamicBroadcastingTest(
    DualAveragingStepSizeAdaptationStaticBroadcastingTest):
  use_static_shape = False


class TfFunctionTest(test_util.TestCase):

  def testDtypeIssue(self):
    # Test issue https://github.com/tensorflow/probability/issues/543
    # There were some stray, implicit, float64 typed values cropping up, but
    # only when one_step was executed in a tf.function context. The fix was to
    # use the correct dtype in those spots; this test verifies the fix.
    normal_2d = tfd.MultivariateNormalDiag([0., 0.], [1., 1.])

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        normal_2d.log_prob, step_size=np.float32(1e-3), num_leapfrog_steps=3)
    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=100)

    init = tf.constant([0.0, 0.0])
    extra = adaptive_kernel.bootstrap_results(init)
    tf.function(lambda: adaptive_kernel.one_step(init, extra))()


if __name__ == '__main__':
  tf.test.main()
