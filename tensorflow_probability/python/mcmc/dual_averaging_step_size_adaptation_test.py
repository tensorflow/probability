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

import collections
import functools

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.experimental.mcmc import preconditioned_nuts
from tensorflow_probability.python.experimental.mcmc import sharded
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.mcmc import dual_averaging_step_size_adaptation as duassa
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import kernel as kernel_lib
from tensorflow_probability.python.mcmc import nuts
from tensorflow_probability.python.mcmc import sample
from tensorflow_probability.python.mcmc import simple_step_size_adaptation as ssa


JAX_MODE = False
NUMPY_MODE = False
TF_MODE = not (JAX_MODE or NUMPY_MODE)

_INITIAL_T = 10.0
_EXPLORATION_SHRINKAGE = 0.05

# Hardcoded expected step after two steps. Calculate with
# tf.math.exp(tf.math.log(10.) - (err - 0.75) / ((10. + 1.) * 0.05))
_UPDATE_M05 = 9.131008  # err = -0.05
_UPDATE_M02 = 9.642897  # err = -0.02
_UPDATE_M01 = 9.819825  # err = -0.01
_UPDATE_0 = 10.  # err = 0
_UPDATE_01 = 10.183481  # err = +0.01


FakeMHKernelResults = collections.namedtuple(
    'FakeMHKernelResults', 'accepted_results, log_accept_ratio')


class FakeMHKernel(kernel_lib.TransitionKernel):

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
            value=self.parameters['log_accept_ratio']),
    )

  @property
  def experimental_shard_axis_names(self):
    return self.inner_kernel.experimental_shard_axis_names

  def experimental_with_shard_axes(self, shard_axes):
    return self.copy(
        inner_kernel=self.inner_kernel.experimental_with_shard_axes(shard_axes))

  @property
  def is_calibrated(self):
    return True


FakeSteppedKernelResults = collections.namedtuple('FakeSteppedKernelResults',
                                                  'step_size')


class FakeSteppedKernel(kernel_lib.TransitionKernel):

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


class FakeWrapperKernel(kernel_lib.TransitionKernel):

  def __init__(self, inner_kernel):
    self.parameters = dict(inner_kernel=inner_kernel)

  @property
  def inner_kernel(self):
    return self.parameters['inner_kernel']

  def one_step(self, current_state, previous_kernel_results, seed=None):
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
    kernel = duassa.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=1, validate_args=True)
    self.assertTrue(kernel.inner_kernel.inner_kernel
                    .parameters['store_parameters_in_results'])

  def testListStep(self):
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=tf.constant([0.1, 0.2, 0.3])),
        log_accept_ratio=tf.stack(
            [tf.math.log(0.74),
             tf.math.log(0.76),
             tf.math.log(0.76)]))
    kernel = duassa.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=1, validate_args=True)

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
    kernel = duassa.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=1, validate_args=True)

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
    kernel = duassa.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=1, validate_args=True)

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
    kernel = duassa.DualAveragingStepSizeAdaptation(
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
    kernel = duassa.DualAveragingStepSizeAdaptation(
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

  @parameterized.parameters((-1., r'`target_accept_prob` must be > 0.'),
                            (0., r'`target_accept_prob` must be > 0.'),
                            (0.999, None),
                            (1., r'`target_accept_prob` must be < 1.'))
  def testTargetAcceptanceProbChecks(self, target_accept_prob, message):

    def _impl():
      kernel = FakeMHKernel(
          FakeSteppedKernel(step_size=1.), log_accept_ratio=0.)
      kernel = duassa.DualAveragingStepSizeAdaptation(
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

  def testExample(self):
    target_dist = jds.JointDistributionSequential([
        normal.Normal(0., 1.5),
        independent.Independent(
            normal.Normal(tf.zeros([2, 5], dtype=tf.float32), 5.),
            reinterpreted_batch_ndims=2),
    ])
    num_burnin_steps = 500
    num_results = 500
    num_chains = 64

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda *args: target_dist.log_prob(args),
        num_leapfrog_steps=2,
        step_size=target_dist.stddev())
    kernel = duassa.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(num_burnin_steps * 0.8),
        # Cast to int32.  Not necessary for operation since we cast internally
        # to a float type.  This is done to check that we are able to pass in
        # integer types (since they are the natural type for this).
        step_count_smoothing=tf.cast(10, tf.int32))

    seed_stream = test_util.test_seed_stream()
    _, log_accept_ratio = sample.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=target_dist.sample(num_chains, seed=seed_stream()),
        kernel=kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.log_accept_ratio,
        seed=seed_stream())

    p_accept = tf.reduce_mean(tf.math.exp(tf.minimum(log_accept_ratio, 0.)))

    self.assertAllClose(0.75, self.evaluate(p_accept), atol=0.15)

  def testShrinkageTargetDefaultsTo10xStepSize(self):
    target_dist = normal.Normal(0., 1.)

    # Choose an initial_step_size that is too big.  We will make it even bigger
    # during the initial adaptation steps by using carefully selected
    # shrinkage parameters.
    initial_step_size = 2.5
    expected_final_step_size = 1.5

    hmc_kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_dist.log_prob,
        # Small num_leapfrog_steps, to ensure stability even though we're doing
        # extreme stuff with the step size.
        num_leapfrog_steps=3,
        step_size=initial_step_size)
    kernel = duassa.DualAveragingStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps=500,
        step_count_smoothing=5.,
        target_accept_prob=0.75,
        shrinkage_target=None,  # Default
        # Huge exploration_shrinkage moves us close to the shrinkage_target.
        exploration_shrinkage=1.,  # Default is 0.05, so this is huge.
    )

    def trace_fn(_, pkr):
      return (pkr.log_shrinkage_target, pkr.inner_results.log_accept_ratio,
              ssa.hmc_like_step_size_getter_fn(pkr))

    stream = test_util.test_seed_stream()
    _, (log_shrinkage_target, log_accept_ratio,
        step_size) = sample.sample_chain(
            num_results=500,
            num_burnin_steps=0,
            current_state=target_dist.sample(64, seed=stream()),
            kernel=kernel,
            trace_fn=trace_fn,
            seed=stream(),
        )

    log_shrinkage_target, log_accept_ratio, step_size = self.evaluate((
        log_shrinkage_target, log_accept_ratio, step_size))

    self.assertAllClose(
        10 * initial_step_size * np.ones_like(log_shrinkage_target),
        np.exp(log_shrinkage_target))

    # Verify that we adapted as desired.
    p_accept = np.mean(np.exp(np.minimum(log_accept_ratio[-250:], 0.)))
    self.assertAllClose(0.75, p_accept, atol=0.15)

    self.assertAllClose(expected_final_step_size, step_size[-1], atol=0.5)

    # We start out at the initial_step_size.
    self.assertAllClose(initial_step_size, step_size[0])

    # The default shrinkage_target = 10 x initial_step_size, so our
    # first few step sizes will be large.
    self.assertAllGreater(step_size[1], 8 * initial_step_size)
    self.assertAllGreater(step_size[2], 5 * initial_step_size)
    self.assertAllGreater(step_size[3], 3 * initial_step_size)

  def testShrinkageTargetSetVeryLowMeansIntialStepSizeIsSmall(self):
    target_dist = normal.Normal(0., 1.)

    expected_final_step_size = 1.5
    initial_step_size = expected_final_step_size
    shrinkage_target_kwarg = 0.1

    hmc_kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_dist.log_prob,
        # Small num_leapfrog_steps, to ensure stability even though we're doing
        # extreme stuff with the step size.
        num_leapfrog_steps=3,
        step_size=initial_step_size)
    kernel = duassa.DualAveragingStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps=500,
        step_count_smoothing=15.,
        target_accept_prob=0.75,
        shrinkage_target=shrinkage_target_kwarg,
        # Huge exploration_shrinkage moves us close to the shrinkage_target.
        exploration_shrinkage=1.,
    )

    def trace_fn(_, pkr):
      return (pkr.log_shrinkage_target, pkr.inner_results.log_accept_ratio,
              ssa.hmc_like_step_size_getter_fn(pkr))

    stream = test_util.test_seed_stream()
    _, (log_shrinkage_target, log_accept_ratio, step_size) = (
        sample.sample_chain(
            num_results=500,
            num_burnin_steps=0,
            current_state=target_dist.sample(64, seed=stream()),
            kernel=kernel,
            trace_fn=trace_fn,
            seed=stream(),
        ))

    log_shrinkage_target, log_accept_ratio, step_size = self.evaluate((
        log_shrinkage_target, log_accept_ratio, step_size))

    self.assertAllClose(
        shrinkage_target_kwarg * np.ones_like(log_shrinkage_target),
        np.exp(log_shrinkage_target))

    # Verify that we adapted as desired.
    p_accept = np.mean(np.exp(np.minimum(log_accept_ratio[-250:], 0.)))
    self.assertAllClose(0.75, p_accept, atol=0.15)

    self.assertAllClose(expected_final_step_size, step_size[-1], atol=0.5)

    # We start out at the initial_step_size.
    self.assertAllClose(initial_step_size, step_size[0])

    # step_size stays close to shrinkage_target for a bit, even though we
    # eventually drift away to expected_final_step_size.
    self.assertAllClose(step_size[1], shrinkage_target_kwarg, rtol=0.15)
    self.assertAllClose(step_size[2], shrinkage_target_kwarg, rtol=0.15)
    self.assertAllClose(step_size[3], shrinkage_target_kwarg, rtol=0.15)

  def testShrinkageTargetPartsAndLowShrinkageTarget(self):
    def log_prob_fn(x, y):
      # X ~ Normal(0, 1),  Y ~ Normal(0, 2^2)
      return - 0.5 * (tf.reduce_sum(x**2) + tf.reduce_sum((y / 2)**2))

    # Empirically determined.
    expected_final_step_size = [0.5, 1.0]

    # Arbirary, but different than the final step size.
    initial_step_size = [s * 2 for s in expected_final_step_size]

    # Something very small
    # Note we set the initial size of the second component to 2x the first.
    # Why? Because, the step sizes will adjust together, and so finish in the
    # same ratio they started with...so they better start in the expected final
    # ratio.
    shrinkage_target_kwarg = [0.1, 0.2]

    hmc_kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        # Small num_leapfrog_steps, to ensure stability even though we're doing
        # extreme stuff with the step size.
        num_leapfrog_steps=3,
        step_size=initial_step_size)
    kernel = duassa.DualAveragingStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps=500,
        step_count_smoothing=15.,
        target_accept_prob=0.75,
        shrinkage_target=shrinkage_target_kwarg,
        # Huge exploration_shrinkage moves us close to the shrinkage_target.
        exploration_shrinkage=1.,
    )

    def trace_fn(_, pkr):
      return (pkr.log_shrinkage_target, pkr.inner_results.log_accept_ratio,
              ssa.hmc_like_step_size_getter_fn(pkr))

    stream = test_util.test_seed_stream()
    _, (log_shrinkage_target, log_accept_ratio, step_size) = (
        sample.sample_chain(
            num_results=500,
            num_burnin_steps=0,
            current_state=[
                tf.random.normal((64,), seed=stream()),
                tf.random.normal((64,), seed=stream())
            ],
            kernel=kernel,
            trace_fn=trace_fn,
            seed=stream(),
        ))

    log_shrinkage_target, log_accept_ratio, step_size = self.evaluate((
        log_shrinkage_target, log_accept_ratio, step_size))

    # Verify that we adapted as desired.
    p_accept = np.mean(np.exp(np.minimum(log_accept_ratio[-250:], 0.)))
    self.assertAllClose(0.75, p_accept, atol=0.15)

    for i in range(len(shrinkage_target_kwarg)):
      self.assertAllClose(
          shrinkage_target_kwarg[i] * np.ones_like(log_shrinkage_target[i]),
          np.exp(log_shrinkage_target[i]))

      self.assertAllClose(
          expected_final_step_size[i], step_size[i][-1], atol=0.5)

      # We start out at the initial_step_size.
      self.assertAllClose(initial_step_size[i], step_size[i][0])

      # step_size stays close to shrinkage_target for a bit, even though we
      # eventually drift away to expected_final_step_size.
      self.assertAllClose(step_size[i][1], shrinkage_target_kwarg[i], rtol=0.15)
      self.assertAllClose(step_size[i][2], shrinkage_target_kwarg[i], rtol=0.15)
      self.assertAllClose(step_size[i][3], shrinkage_target_kwarg[i], rtol=0.15)

  def testIsCalibrated(self):
    test_kernel = collections.namedtuple('TestKernel', 'is_calibrated')
    self.assertTrue(
        duassa.DualAveragingStepSizeAdaptation(test_kernel(True),
                                               1).is_calibrated)
    self.assertFalse(
        duassa.DualAveragingStepSizeAdaptation(test_kernel(False),
                                               1).is_calibrated)

  def testCustomReduceFn(self):
    log_accept_ratio = tf.constant(
        [np.log(0.73), np.log(0.76)],
        dtype=tf.float32)
    state = [
        tf.zeros([2]),
    ]

    old_step_size = 1.
    kernel = FakeMHKernel(
        FakeSteppedKernel(step_size=old_step_size),
        log_accept_ratio=log_accept_ratio)
    kernel = duassa.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=1,
        reduce_fn=tf.reduce_max,
        validate_args=True)

    kernel_results = kernel.bootstrap_results(state)
    for _ in range(2):
      _, kernel_results = kernel.one_step(state, kernel_results)

    step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size,)

    # If reduce_fn was left at default, this would have decreased.
    self.assertAllClose(_UPDATE_01, step_size)

  def testShouldPropagateShardAxisNames(self):
    test_kernel = duassa.DualAveragingStepSizeAdaptation(
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
    kernel = duassa.DualAveragingStepSizeAdaptation(
        kernel,
        target_accept_prob=0.75,
        num_adaptation_steps=1,
        validate_args=True)

    seed_stream = test_util.test_seed_stream()
    kernel_results = kernel.bootstrap_results(state)
    for _ in range(2):
      _, kernel_results = kernel.one_step(state, kernel_results,
                                          seed=seed_stream())

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
    normal_2d = mvn_diag.MultivariateNormalDiag([0., 0.], [1., 1.])

    kernel = hmc.HamiltonianMonteCarlo(
        normal_2d.log_prob, step_size=np.float32(1e-3), num_leapfrog_steps=3)
    adaptive_kernel = duassa.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=100)

    seed_stream = test_util.test_seed_stream()
    init = tf.constant([0.0, 0.0])
    extra = adaptive_kernel.bootstrap_results(init)
    tf.function(
        lambda: adaptive_kernel.one_step(init, extra, seed=seed_stream()))()

  @parameterized.named_parameters(
      dict(testcase_name='_nuts', preconditioned=False),
      dict(testcase_name='_pnuts', preconditioned=True))
  def test_nuts_f64(self, preconditioned):
    target_log_prob_fn = lambda x: -x**2
    if preconditioned:
      nuts_kernel = preconditioned_nuts.PreconditionedNoUTurnSampler
    else:
      nuts_kernel = nuts.NoUTurnSampler
    kernel = nuts_kernel(target_log_prob_fn, step_size=1.)
    kernel = duassa.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=50)
    init = tf.constant(0., dtype=tf.float64)
    extra = kernel.bootstrap_results(init)
    seed_stream = test_util.test_seed_stream()
    tf.function(lambda: kernel.one_step(init, extra, seed=seed_stream()))()


def reduce_mean(x, experimental_named_axis=None, **kwargs):
  return distribute_lib.reduce_mean(x, named_axis=experimental_named_axis,
                                    **kwargs)


@test_util.test_all_tf_execution_regimes
class DistributedDualAveragingStepSizeAdaptationTest(
    distribute_test_lib.DistributedTest):

  @parameterized.named_parameters(
      ('mean', reduce_mean),
      ('logmeanexp',
       functools.partial(
           generic.reduce_logmeanexp, experimental_allow_all_gather=True)))
  @test_util.numpy_disable_test_missing_functionality(
      'NumPy backend does not support distributed computation.')
  def test_kernel_can_shard_chains_across_devices(self, reduce_fn):
    if (tf.executing_eagerly() and TF_MODE):
      self.skipTest('Not supported in Eager.')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) +
              sample_dist_lib.Sample(normal.Normal(a, 1.), 4).log_prob(b))

    def run(seed, log_accept_ratio):
      kernel = hmc.UncalibratedHamiltonianMonteCarlo(
          target_log_prob, step_size=1e-2, num_leapfrog_steps=2)
      kernel = FakeMHKernel(kernel, log_accept_ratio)
      sharded_kernel = sharded.Sharded(
          duassa.DualAveragingStepSizeAdaptation(
              kernel,
              10,
              reduce_fn=reduce_fn,
              target_accept_prob=0.5,
              experimental_reduce_chain_axis_names=self.axis_name),
          self.axis_name)
      init_seed, sample_seed = samplers.split_seed(seed)
      state_seeds = samplers.split_seed(init_seed)
      state = [
          samplers.normal(seed=state_seeds[0], shape=[]),
          samplers.normal(seed=state_seeds[1], shape=[4])
      ]
      kr = sharded_kernel.bootstrap_results(state)
      _, kr = sharded_kernel.one_step(state, kr, seed=sample_seed)
      return kr.new_step_size, kr.error_sum

    seeds = self.shard_values(
        tf.stack(
            samplers.split_seed(samplers.zeros_seed(),
                                distribute_test_lib.NUM_DEVICES)), 0)
    log_accept_ratios = self.shard_values(tf.convert_to_tensor([
        -3., -2., -1., 0.
    ]))

    step_size, error_sum = self.evaluate(
        self.per_replica_to_tensor(self.strategy_run(
            run, args=(seeds, log_accept_ratios), axis_name=self.axis_name), 0))

    true_error_sum = 0.5 - tf.math.exp(
        reduce_fn(self.per_replica_to_tensor(log_accept_ratios)))
    for i in range(distribute_test_lib.NUM_DEVICES):
      self.assertAllClose(error_sum[0][i], true_error_sum)
      self.assertAllClose(step_size[0], step_size[i])


if __name__ == '__main__':
  test_util.main()
