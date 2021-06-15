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
"""Tests for gradient_based_trajectory_length_adaptation."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions

JAX_MODE = False


@test_util.test_graph_and_eager_modes
class GradientBasedTrajectoryLengthAdaptationTestGeneric(
    test_util.TestCase, parameterized.TestCase):

  def testForbiddenTransformedKernel(self):
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -x**2, step_size=0.1, num_leapfrog_steps=1)
    kernel = tfp.mcmc.TransformedTransitionKernel(kernel, tfb.Identity())
    with self.assertRaisesRegex(
        ValueError,
        'The inner kernel cannot contain a `TransformedTransitionKernel`'):
      kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
          kernel, num_adaptation_steps=100)

  def testNestedStepSizeError(self):
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -x**2,
        step_size=[0.1],
        num_leapfrog_steps=1)
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel, num_adaptation_steps=100)
    with self.assertRaisesRegex(ValueError, 'Step size must be a scalar'):
      kernel.bootstrap_results([1.])

  @parameterized.named_parameters(('StaticShape', True),
                                  ('DynamicShape', False))
  def testNonScalarStepSizeError(self, use_static_shape):
    step_size = tf1.placeholder_with_default(
        [0.1, 0.2], shape=[2] if use_static_shape else None)

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -x**2,
        step_size=step_size,
        num_leapfrog_steps=1)
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel, num_adaptation_steps=100, validate_args=True)
    with self.assertRaisesRegex(Exception, 'Step size must be a scalar'):
      self.evaluate(kernel.bootstrap_results(tf.constant(1.)))

  @parameterized.named_parameters(('StaticShape', True),
                                  ('DynamicShape', False))
  def testChEESTooFewChains(self, use_static_shape):
    state = tf1.placeholder_with_default(
        [[0.1, 0.2]], shape=[1, 2] if use_static_shape else None)
    accept_prob = tf1.placeholder_with_default(
        [1.], shape=[1] if use_static_shape else None)
    with self.assertRaisesRegex(Exception,
                                'chees_criterion requires at least 2 chains'):
      self.evaluate(
          tfp.experimental.mcmc.chees_criterion(
              state, state, accept_prob, validate_args=True))

  @parameterized.named_parameters(('StaticShape', True),
                                  ('DynamicShape', False))
  def testChEESNoBatchDims(self, use_static_shape):
    state = tf1.placeholder_with_default(
        [[0.1, 0.2]], shape=[1, 2] if use_static_shape else None)
    accept_prob = tf1.placeholder_with_default(
        1., shape=[] if use_static_shape else None)
    with self.assertRaisesRegex(Exception,
                                'chees_criterion requires at least 2 chains'):
      self.evaluate(
          tfp.experimental.mcmc.chees_criterion(
              state, state, accept_prob, validate_args=True))


class _GradientBasedTrajectoryLengthAdaptationTest(test_util.TestCase):

  def testDocstringExample(self):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Too slow for TF Eager.')

    target = tfd.JointDistributionSequential([
        tfd.Normal(0., tf.constant(20., dtype=self.dtype)),
        tfd.HalfNormal(tf.constant(10., dtype=self.dtype)),
    ])

    def target_log_prob_fn(*x):
      return tf.cast(target.log_prob(x), self.dtype)

    num_burnin_steps = 1000
    num_adaptation_steps = int(num_burnin_steps * 0.8)
    num_results = 500
    num_chains = 16
    step_size = 0.1

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=1,
    )
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel,
        num_adaptation_steps=num_adaptation_steps,
        validate_args=True)
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=num_adaptation_steps)
    kernel = tfp.mcmc.TransformedTransitionKernel(
        kernel, [tfb.Identity(), tfb.Exp()])

    def trace_fn(_, pkr):
      return (
          pkr.inner_results.inner_results.inner_results.accepted_results
          .step_size,
          pkr.inner_results.inner_results.max_trajectory_length,
          pkr.inner_results.inner_results.inner_results.log_accept_ratio,
      )

    # The chain will be stepped for num_results + num_burnin_steps, adapting for
    # the first num_adaptation_steps.
    chain, [step_size, max_trajectory_length, log_accept_ratio] = (
        tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=[
                tf.ones(num_chains, dtype=self.dtype),
                tf.ones(num_chains, dtype=self.dtype)
            ],
            kernel=kernel,
            trace_fn=trace_fn,
            seed=test_util.test_seed(sampler_type='stateless')))

    p_accept = tf.math.exp(
        tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
    mean_step_size = tf.reduce_mean(step_size)
    mean_max_trajectory_length = tf.reduce_mean(max_trajectory_length)

    self.assertAllClose(0.75, p_accept, atol=0.1)
    self.assertAllClose(0.52, mean_step_size, atol=0.2)
    self.assertAllClose(46., mean_max_trajectory_length, atol=15)
    self.assertAllClose(
        target.mean(), [tf.reduce_mean(x, axis=[0, 1]) for x in chain],
        atol=1.5)
    self.assertAllClose(
        target.variance(),
        [tf.math.reduce_variance(x, axis=[0, 1]) for x in chain],
        rtol=0.2)

  def testScalarState(self):

    def target_log_prob_fn(x):
      return -x**2 / 2

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.1,
        num_leapfrog_steps=1,
    )
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel,
        num_adaptation_steps=5,
        adaptation_rate=1.,
        validate_args=True)

    state = tf.zeros([64], self.dtype)
    init_kernel_results = kernel.bootstrap_results(state)
    init_kernel_results, (_, final_kernel_results) = self.evaluate([
        init_kernel_results,
        kernel.one_step(
            state,
            init_kernel_results,
            seed=test_util.test_seed(sampler_type='stateless'))
    ])

    # We expect it to move it a little bit.
    self.assertGreater(
        np.abs(init_kernel_results.max_trajectory_length -
               final_kernel_results.max_trajectory_length), 0.0005)

  def testTensorState(self):

    def target_log_prob_fn(x):
      return -tf.reduce_mean(x**2, [-1, -2]) / 2

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.1,
        num_leapfrog_steps=1,
    )
    kernel = (
        tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
            kernel,
            num_adaptation_steps=5,
            adaptation_rate=1.,
            validate_args=True))

    state = tf.zeros([64, 2, 3], self.dtype)
    init_kernel_results = kernel.bootstrap_results(state)
    init_kernel_results, (_, final_kernel_results) = self.evaluate([
        init_kernel_results,
        kernel.one_step(
            state,
            init_kernel_results,
            seed=test_util.test_seed(sampler_type='stateless'))
    ])

    # We expect it to move it a little bit.
    self.assertGreater(
        np.abs(init_kernel_results.max_trajectory_length -
               final_kernel_results.max_trajectory_length), 0.0005)

  def testListState(self):

    def target_log_prob_fn(x, y):
      return -x**2 / 2 - y**2 / 2

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.1,
        num_leapfrog_steps=1,
    )
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel,
        num_adaptation_steps=5,
        adaptation_rate=1.,
        validate_args=True)

    state = [tf.zeros([64], self.dtype), tf.zeros([64], self.dtype)]
    init_kernel_results = kernel.bootstrap_results(state)
    init_kernel_results, (_, final_kernel_results) = self.evaluate([
        init_kernel_results,
        kernel.one_step(
            state,
            init_kernel_results,
            seed=test_util.test_seed(sampler_type='stateless'))
    ])

    # We expect it to move it a little bit.
    self.assertGreater(
        np.abs(init_kernel_results.max_trajectory_length -
               final_kernel_results.max_trajectory_length), 0.0005)

  def testNumAdaptationSteps(self):

    def target_log_prob_fn(x):
      return -x**2

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.1,
        num_leapfrog_steps=1,
    )
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel,
        num_adaptation_steps=1,
        adaptation_rate=1.,
        validate_args=True)

    state = tf.zeros([64], self.dtype)
    seed = test_util.test_seed(sampler_type='stateless')
    step_0_kernel_results = kernel.bootstrap_results(state)
    state, step_1_kernel_results = kernel.one_step(
        state, step_0_kernel_results, seed=seed)
    _, step_2_kernel_results = kernel.one_step(
        state, step_1_kernel_results, seed=seed)

    (step_0_kernel_results, step_1_kernel_results,
     step_2_kernel_results) = self.evaluate([
         step_0_kernel_results,
         step_1_kernel_results,
         step_2_kernel_results,
     ])

    # The intention of num_adaptation_steps is that we should adapt for 1 step
    # and then hold the hyperparameters constant.
    self.assertGreater(
        np.abs(step_0_kernel_results.max_trajectory_length -
               step_1_kernel_results.max_trajectory_length), 0.005)
    self.assertAllClose(step_1_kernel_results.max_trajectory_length,
                        step_2_kernel_results.max_trajectory_length)

  def testChEESStateEquivalence(self):
    # ChEES criterion should not care about the exact arrangement of state
    # parts.
    previous_state = np.random.randn(4, 6).astype(self.dtype)
    new_state = np.random.randn(4, 6).astype(self.dtype)
    accept_prob = np.random.uniform(size=(4,)).astype(self.dtype)

    matrix_previous_state = previous_state.reshape([4, 3, 2])
    matrix_new_state = new_state.reshape([4, 3, 2])

    list_previous_state = [previous_state[:, :2], previous_state[:, 2:]]
    list_new_state = [new_state[:, :2], new_state[:, 2:]]

    chees = tfp.experimental.mcmc.chees_criterion(
        previous_state, new_state, accept_prob)
    matrix_chees = tfp.experimental.mcmc.chees_criterion(
        matrix_previous_state, matrix_new_state, accept_prob)
    list_chees = tfp.experimental.mcmc.chees_criterion(
        list_previous_state, list_new_state, accept_prob)

    self.assertAllEqual([4], chees.shape)
    self.assertAllClose(chees, matrix_chees)
    self.assertAllClose(chees, list_chees)


class GradientBasedTrajectoryLengthAdaptationTestFloat32(
    _GradientBasedTrajectoryLengthAdaptationTest):
  dtype = np.float32


class GradientBasedTrajectoryLengthAdaptationTestFloat64(
    _GradientBasedTrajectoryLengthAdaptationTest):
  dtype = np.float64


@test_util.test_all_tf_execution_regimes
class DistributedGBTLATest(distribute_test_lib.DistributedTest):

  def test_gbtla_kernel_tracks_axis_names(self):
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(tfd.Normal(0, 1).log_prob,
                                                  step_size=1.9,
                                                  num_leapfrog_steps=2)
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        inner_kernel, 1)
    self.assertIsNone(kernel.experimental_shard_axis_names)
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        inner_kernel, 1, experimental_shard_axis_names=['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        inner_kernel, 1).experimental_with_shard_axes(['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])

  def test_gbtla_kernel_computes_same_criterion_info_with_sharded_state(self):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (
          tfd.Normal(0., 1.).log_prob(a)
          + distribute_lib.psum(tfd.Normal(
              distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b), 'foo'))

    kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob,
                                            step_size=1e-2,
                                            num_leapfrog_steps=2)
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel, 10)
    sharded_kernel = kernel.experimental_with_shard_axes([None, ['foo']])

    def run(seed):
      init_seed, sample_seed = samplers.split_seed(seed)
      state_seeds = samplers.split_seed(init_seed)
      state = [
          samplers.normal(seed=state_seeds[0], shape=[5]),
          samplers.normal(seed=state_seeds[1], shape=[5])
      ]
      kr = sharded_kernel.bootstrap_results(state)
      _, kr = sharded_kernel.one_step(state, kr, seed=sample_seed)
      return (
          kr.criterion,
          kr.averaged_sq_grad,
          kr.averaged_max_trajectory_length
      )

    criterion, avg_sq_grad, avg_max_tl = self.evaluate(
        self.per_replica_to_tensor(self.strategy_run(
            run, args=(samplers.zeros_seed(),), in_axes=None, axis_name='foo'),
                                   0))

    for i in range(distribute_test_lib.NUM_DEVICES):
      self.assertAllClose(criterion[0], criterion[i])
      self.assertAllClose(avg_sq_grad[0], avg_sq_grad[i])
      self.assertAllClose(avg_max_tl[0], avg_max_tl[i])

  def test_gbtla_kernel_can_shard_chains_across_devices(self):

    def target_log_prob(a, b):
      return (
          tfd.Normal(0., 1.).log_prob(a)
          + tfd.Sample(tfd.Normal(a, 1.), 4).log_prob(b))

    kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob,
                                            step_size=1e-2,
                                            num_leapfrog_steps=2)
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel, 10)
    sharded_kernel = kernel.experimental_with_chain_axes(self.axis_name)

    def run(seed):
      init_seed, sample_seed = samplers.split_seed(seed)
      state_seeds = samplers.split_seed(init_seed)
      state = [
          samplers.normal(seed=state_seeds[0], shape=[]),
          samplers.normal(seed=state_seeds[1], shape=[4])
      ]
      kr = sharded_kernel.bootstrap_results(state)
      _, kr = sharded_kernel.one_step(state, kr, seed=sample_seed)
      return (
          kr.averaged_sq_grad,
          kr.averaged_max_trajectory_length
      )

    seeds = self.shard_values(tf.stack(tfp.random.split_seed(
        samplers.zeros_seed(), distribute_test_lib.NUM_DEVICES)), 0)

    avg_sq_grad, avg_max_tl = self.evaluate(
        self.per_replica_to_tensor(self.strategy_run(
            run, args=(seeds,), axis_name=self.axis_name), 0))

    for i in range(distribute_test_lib.NUM_DEVICES):
      self.assertAllClose(avg_sq_grad[0], avg_sq_grad[i])
      self.assertAllClose(avg_max_tl[0], avg_max_tl[i])


del _GradientBasedTrajectoryLengthAdaptationTest

if __name__ == '__main__':
  tf.test.main()
