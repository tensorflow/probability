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
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.distributions import half_normal
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.experimental.mcmc import gradient_based_trajectory_length_adaptation as gbtla
from tensorflow_probability.python.experimental.mcmc import preconditioned_hmc as phmc
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.mcmc import dual_averaging_step_size_adaptation as dassa
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import sample
from tensorflow_probability.python.mcmc import transformed_kernel

JAX_MODE = False


def snaper_criterion_dummy_direction(previous_state, *args, **kwargs):
  # Technically direction should be normalized, but omitting the normalization
  # term only rescales the criterion so we're fine.
  return gbtla.snaper_criterion(
      previous_state,
      *args,
      direction=tf.nest.map_structure(tf.ones_like, previous_state),
      **kwargs,
  )


def snaper_criterion_2d_direction(previous_state, *args, **kwargs):
  return gbtla.snaper_criterion(
      previous_state,
      *args,
      direction=tf.constant([0., 1.], previous_state.dtype),
      **kwargs,
  )


@test_util.test_graph_and_eager_modes
class GradientBasedTrajectoryLengthAdaptationTestGeneric(
    test_util.TestCase, parameterized.TestCase):

  def testForbiddenTransformedKernel(self):
    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -x**2, step_size=0.1, num_leapfrog_steps=1)
    kernel = transformed_kernel.TransformedTransitionKernel(
        kernel, identity.Identity())
    with self.assertRaisesRegex(
        ValueError,
        'The inner kernel cannot contain a `TransformedTransitionKernel`'):
      kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
          kernel, num_adaptation_steps=100)

  def testNestedStepSizeError(self):
    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -x**2,
        step_size=[0.1],
        num_leapfrog_steps=1)
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel, num_adaptation_steps=100)
    with self.assertRaisesRegex(ValueError, 'Step size must be a scalar'):
      kernel.bootstrap_results([1.])

  @parameterized.named_parameters(('StaticShape', True),
                                  ('DynamicShape', False))
  def testNonScalarStepSizeError(self, use_static_shape):
    step_size = tf1.placeholder_with_default(
        [0.1, 0.2], shape=[2] if use_static_shape else None)

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -x**2,
        step_size=step_size,
        num_leapfrog_steps=1)
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel, num_adaptation_steps=100, validate_args=True)
    with self.assertRaisesRegex(Exception, 'Step size must be a scalar'):
      self.evaluate(kernel.bootstrap_results(tf.constant(1.)))

  @parameterized.named_parameters(
      ('ChEESStaticShape', True, gbtla.chees_criterion),
      ('ChEESDynamicShape', False, gbtla.chees_criterion),
      ('SNAPERStaticShape', True, snaper_criterion_dummy_direction),
      ('SNAPERDynamicShape', False, snaper_criterion_dummy_direction),
  )
  def testTooFewChains(self, use_static_shape, criterion_fn):
    state = tf1.placeholder_with_default(
        [[0.1, 0.2]], shape=[1, 2] if use_static_shape else None)
    accept_prob = tf1.placeholder_with_default(
        [1.], shape=[1] if use_static_shape else None)
    with self.assertRaisesRegex(Exception,
                                'chees_criterion requires at least 2 chains'):
      self.evaluate(
          gbtla.chees_criterion(
              state, state, accept_prob, 1., validate_args=True))

  @parameterized.named_parameters(
      ('ChEESStaticShape', True, gbtla.chees_criterion),
      ('ChEESDynamicShape', False, gbtla.chees_criterion),
      ('SNAPERStaticShape', True, snaper_criterion_dummy_direction),
      ('SNAPERDynamicShape', False, snaper_criterion_dummy_direction),
  )
  def testNoBatchDims(self, use_static_shape, criterion_fn):
    state = tf1.placeholder_with_default(
        [[0.1, 0.2]], shape=[1, 2] if use_static_shape else None)
    accept_prob = tf1.placeholder_with_default(
        1., shape=[] if use_static_shape else None)
    with self.assertRaisesRegex(Exception, 'requires at least 2 chains'):
      self.evaluate(
          criterion_fn(state, state, accept_prob, 1., validate_args=True))


class _GradientBasedTrajectoryLengthAdaptationTest(test_util.TestCase):

  def testDocstringExample(self):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Too slow for TF Eager.')

    target = jds.JointDistributionSequential([
        normal.Normal(0., tf.constant(20., dtype=self.dtype)),
        half_normal.HalfNormal(tf.constant(10., dtype=self.dtype)),
    ])

    def target_log_prob_fn(*x):
      return tf.cast(target.log_prob(x), self.dtype)

    num_burnin_steps = 1000
    num_adaptation_steps = int(num_burnin_steps * 0.8)
    num_results = 500
    num_chains = 16
    step_size = 0.1

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=1,
    )
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel, num_adaptation_steps=num_adaptation_steps, validate_args=True)
    kernel = dassa.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=num_adaptation_steps)
    kernel = transformed_kernel.TransformedTransitionKernel(
        kernel, [identity.Identity(), exp.Exp()])

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
        sample.sample_chain(
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
        generic.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
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

  def testStateMeanSNAPER(self):
    state = np.array([[0.1, 0.2]], self.dtype)
    accept_prob = np.ones([], self.dtype)
    # This doesn't fail because state_mean is provided externally.
    self.evaluate(
        gbtla.snaper_criterion(
            state,
            state,
            accept_prob,
            2.,
            direction=tf.ones_like(state),
            state_mean=state,
            state_mean_weight=0.1,
        ))

  @parameterized.named_parameters(
      ('ChEES', gbtla.chees_criterion),
      ('SNAPER', snaper_criterion_dummy_direction),
  )
  def testScalarState(self, criterion_fn):

    def target_log_prob_fn(x):
      return -x**2 / 2

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.1,
        num_leapfrog_steps=1,
    )
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel,
        num_adaptation_steps=5,
        adaptation_rate=1.,
        criterion_fn=criterion_fn,
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

  @parameterized.named_parameters(
      ('ChEES', gbtla.chees_criterion),
      ('SNAPER', snaper_criterion_dummy_direction),
  )
  def testTensorState(self, criterion_fn):

    def target_log_prob_fn(x):
      return -tf.reduce_mean(x**2, [-1, -2]) / 2

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.1,
        num_leapfrog_steps=1,
    )
    kernel = (
        gbtla.GradientBasedTrajectoryLengthAdaptation(
            kernel,
            num_adaptation_steps=5,
            adaptation_rate=1.,
            criterion_fn=criterion_fn,
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

  @parameterized.named_parameters(
      ('ChEES', gbtla.chees_criterion),
      ('SNAPER', snaper_criterion_dummy_direction),
  )
  def testListState(self, criterion_fn):

    def target_log_prob_fn(x, y):
      return -x**2 / 2 - y**2 / 2

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.1,
        num_leapfrog_steps=1,
    )
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel,
        num_adaptation_steps=5,
        adaptation_rate=1.,
        criterion_fn=criterion_fn,
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

  @parameterized.named_parameters(
      ('ChEES', gbtla.chees_rate_criterion),
      ('SNAPER', snaper_criterion_2d_direction),
  )
  def testAdaptation(self, criterion_fn):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Too slow for TF Eager.')

    target = independent.Independent(
        normal.Normal(0., tf.constant([1., 10.], self.dtype)), 1)

    num_burnin_steps = 1000
    num_adaptation_steps = int(num_burnin_steps * 0.8)
    num_results = 500
    num_chains = 16
    step_size = 0.1

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target.log_prob,
        step_size=step_size,
        num_leapfrog_steps=1,
    )
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel,
        num_adaptation_steps=num_adaptation_steps,
        criterion_fn=criterion_fn,
        validate_args=True)
    kernel = dassa.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=num_adaptation_steps)

    def trace_fn(_, pkr):
      return (
          pkr.inner_results.inner_results.accepted_results
          .step_size,
          pkr.inner_results.max_trajectory_length,
          pkr.inner_results.inner_results.log_accept_ratio,
      )

    # The chain will be stepped for num_results + num_burnin_steps, adapting for
    # the first num_adaptation_steps.
    chain, [step_size, max_trajectory_length, log_accept_ratio] = (
        sample.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=tf.zeros([num_chains, 2], dtype=self.dtype),
            kernel=kernel,
            trace_fn=trace_fn,
            seed=test_util.test_seed(sampler_type='stateless')))

    p_accept = tf.math.exp(
        generic.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
    mean_step_size = tf.reduce_mean(step_size)
    mean_max_trajectory_length = tf.reduce_mean(max_trajectory_length)

    self.assertAllClose(0.75, p_accept, atol=0.1)
    self.assertAllClose(1.5, mean_step_size, atol=0.2)
    # Both SNAPER and ChEES-rate find roughly the same trajectory length for
    # this target.
    self.assertAllClose(15., mean_max_trajectory_length, rtol=0.3)
    self.assertAllClose(
        target.mean(), tf.reduce_mean(chain, axis=[0, 1]),
        atol=1.)
    self.assertAllClose(
        target.variance(),
        tf.math.reduce_variance(chain, axis=[0, 1]),
        rtol=0.1)

  def testPreconditionedHMC(self):
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Too slow for TF Eager.')

    target = independent.Independent(
        normal.Normal(0., tf.constant([1., 10.], self.dtype)), 1)

    num_burnin_steps = 1000
    num_adaptation_steps = int(num_burnin_steps * 0.8)
    num_results = 500
    num_chains = 16
    step_size = 0.1

    kernel = phmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=target.log_prob,
        step_size=step_size,
        num_leapfrog_steps=1,
        momentum_distribution=independent.Independent(
            normal.Normal(0., tf.constant([1., 1. / 10.], self.dtype)), 1),
    )
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel, num_adaptation_steps=num_adaptation_steps, validate_args=True)
    kernel = dassa.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=num_adaptation_steps)

    def trace_fn(_, pkr):
      return (
          pkr.inner_results.inner_results.accepted_results
          .step_size,
          pkr.inner_results.max_trajectory_length,
          pkr.inner_results.inner_results.log_accept_ratio,
      )

    # The chain will be stepped for num_results + num_burnin_steps, adapting for
    # the first num_adaptation_steps.
    chain, [step_size, max_trajectory_length, log_accept_ratio] = (
        sample.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=tf.zeros([num_chains, 2], dtype=self.dtype),
            kernel=kernel,
            trace_fn=trace_fn,
            seed=test_util.test_seed(sampler_type='stateless')))

    p_accept = tf.math.exp(
        generic.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
    mean_step_size = tf.reduce_mean(step_size)
    mean_max_trajectory_length = tf.reduce_mean(max_trajectory_length)

    self.assertAllClose(0.75, p_accept, atol=0.1)
    self.assertAllClose(1.2, mean_step_size, atol=0.2)
    self.assertAllClose(1.5, mean_max_trajectory_length, rtol=0.25)
    self.assertAllClose(
        target.mean(), tf.reduce_mean(chain, axis=[0, 1]),
        atol=0.3)
    self.assertAllClose(
        target.variance(),
        tf.math.reduce_variance(chain, axis=[0, 1]),
        rtol=0.1)

  def testNumAdaptationSteps(self):

    def target_log_prob_fn(x):
      return -x**2

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.1,
        num_leapfrog_steps=1,
    )
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel, num_adaptation_steps=1, adaptation_rate=1., validate_args=True)

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

  @parameterized.named_parameters(
      ('ChEES', gbtla.chees_criterion),
      ('ChEESR', gbtla.chees_rate_criterion),
      ('SNAPER', snaper_criterion_dummy_direction),
  )
  def testCriterionStateEquivalence(self, criterion_fn):
    # Criteria should not care about the exact arrangement of state parts.
    previous_state = np.random.randn(4, 6).astype(self.dtype)
    new_state = np.random.randn(4, 6).astype(self.dtype)
    accept_prob = np.random.uniform(size=(4,)).astype(self.dtype)

    matrix_previous_state = previous_state.reshape([4, 3, 2])
    matrix_new_state = new_state.reshape([4, 3, 2])

    list_previous_state = [previous_state[:, :2], previous_state[:, 2:]]
    list_new_state = [new_state[:, :2], new_state[:, 2:]]

    criterion = criterion_fn(
        previous_state, new_state, accept_prob, 1.)
    matrix_criterion = criterion_fn(
        matrix_previous_state, matrix_new_state, accept_prob, 1.)
    list_criterion = criterion_fn(
        list_previous_state, list_new_state, accept_prob, 1.)

    self.assertAllEqual([4], criterion.shape)
    self.assertAllClose(criterion, matrix_criterion)
    self.assertAllClose(criterion, list_criterion)


class GradientBasedTrajectoryLengthAdaptationTestFloat32(
    _GradientBasedTrajectoryLengthAdaptationTest):
  dtype = np.float32


class GradientBasedTrajectoryLengthAdaptationTestFloat64(
    _GradientBasedTrajectoryLengthAdaptationTest):
  dtype = np.float64


@test_util.test_all_tf_execution_regimes
class DistributedGBTLATest(distribute_test_lib.DistributedTest):

  def test_gbtla_kernel_tracks_axis_names(self):
    inner_kernel = hmc.HamiltonianMonteCarlo(
        normal.Normal(0, 1).log_prob, step_size=1.9, num_leapfrog_steps=2)
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(inner_kernel, 1)
    self.assertIsNone(kernel.experimental_shard_axis_names)
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        inner_kernel, 1, experimental_shard_axis_names=['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        inner_kernel, 1).experimental_with_shard_axes(['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])

  @parameterized.named_parameters(
      ('ChEES', gbtla.chees_criterion),
      ('ChEESR', gbtla.chees_rate_criterion),
      ('SNAPER', snaper_criterion_dummy_direction),
  )
  def test_gbtla_kernel_computes_same_criterion_info_with_sharded_state(
      self,
      criterion_fn,
  ):

    if not JAX_MODE:
      self.skipTest('Test in TF runs into `merge_call` error: see b/178944108')

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) + distribute_lib.psum(
          normal.Normal(distribute_lib.pbroadcast(a, 'foo'), 1.).log_prob(b),
          'foo'))

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob, step_size=1e-2, num_leapfrog_steps=2)
    kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
        kernel, 10, criterion_fn=criterion_fn)
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

  @parameterized.named_parameters(
      ('ChEES', gbtla.chees_criterion),
      ('ChEESR', gbtla.chees_rate_criterion),
      ('SNAPER', snaper_criterion_dummy_direction),
  )
  def test_gbtla_kernel_can_shard_chains_across_devices(self, criterion_fn):

    def target_log_prob(a, b):
      return (normal.Normal(0., 1.).log_prob(a) +
              sample_dist_lib.Sample(normal.Normal(a, 1.), 4).log_prob(b))

    kernel = hmc.HamiltonianMonteCarlo(
        target_log_prob, step_size=1e-2, num_leapfrog_steps=2)
    sharded_kernel = (
        gbtla.GradientBasedTrajectoryLengthAdaptation(
            kernel,
            10,
            experimental_reduce_chain_axis_names=self.axis_name,
            criterion_fn=criterion_fn))

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

    seeds = self.shard_values(
        tf.stack(
            samplers.split_seed(samplers.zeros_seed(),
                                distribute_test_lib.NUM_DEVICES)), 0)

    avg_sq_grad, avg_max_tl = self.evaluate(
        self.per_replica_to_tensor(self.strategy_run(
            run, args=(seeds,), axis_name=self.axis_name), 0))

    for i in range(distribute_test_lib.NUM_DEVICES):
      self.assertAllClose(avg_sq_grad[0], avg_sq_grad[i])
      self.assertAllClose(avg_max_tl[0], avg_max_tl[i])

  @parameterized.named_parameters(
      ('ChEES', gbtla.chees_rate_criterion),
      ('SNAPER', snaper_criterion_2d_direction),
  )
  def test_adaptation(self, criterion_fn):
    # Compare this to testAdaptation. There we don't use SPMD, but should
    # get the same hyperparameters.

    if not JAX_MODE:
      self.skipTest('TF does not have pmax implemented.')

    target = independent.Independent(
        normal.Normal(0., tf.constant([1., 10.])), 1)

    def run(seed):
      num_burnin_steps = 1000
      num_adaptation_steps = int(num_burnin_steps * 0.8)
      num_results = 500
      num_chains = 16 // distribute_test_lib.NUM_DEVICES
      step_size = 0.1

      kernel = hmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target.log_prob,
          step_size=step_size,
          num_leapfrog_steps=1,
      )
      kernel = gbtla.GradientBasedTrajectoryLengthAdaptation(
          kernel,
          num_adaptation_steps=num_adaptation_steps,
          criterion_fn=criterion_fn,
          experimental_reduce_chain_axis_names=self.axis_name,
          validate_args=True)
      kernel = dassa.DualAveragingStepSizeAdaptation(
          kernel,
          num_adaptation_steps=num_adaptation_steps,
          experimental_reduce_chain_axis_names=self.axis_name)

      def trace_fn(_, pkr):
        return (
            pkr.inner_results.inner_results.accepted_results
            .step_size,
            pkr.inner_results.max_trajectory_length,
            pkr.inner_results.inner_results.log_accept_ratio,
        )

      # The chain will be stepped for num_results + num_burnin_steps, adapting
      # for the first num_adaptation_steps.
      chain, [step_size, max_trajectory_length, log_accept_ratio] = (
          sample.sample_chain(
              num_results=num_results,
              num_burnin_steps=num_burnin_steps,
              current_state=tf.zeros([num_chains, 2]),
              kernel=kernel,
              trace_fn=trace_fn,
              seed=seed))

      p_accept = tf.math.exp(
          generic.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
      mean_step_size = tf.reduce_mean(step_size)
      mean_max_trajectory_length = tf.reduce_mean(max_trajectory_length)
      mean = tf.reduce_mean(chain, axis=[0, 1])
      var = tf.reduce_variance(chain, axis=[0, 1])

      return mean, var, p_accept, mean_step_size, mean_max_trajectory_length

    seeds = self.shard_values(
        tf.stack(
            samplers.split_seed(samplers.zeros_seed(),
                                distribute_test_lib.NUM_DEVICES)), 0)

    (mean, var, p_accept, mean_step_size, mean_max_trajectory_length) = (
        self.evaluate(
            self.per_replica_to_tensor(
                self.strategy_run(run, args=(seeds,), axis_name=self.axis_name),
                0,
            )))

    self.assertAllClose(0.75, p_accept.mean(), atol=0.1)
    # Both ChEES-rate and SNAPER learn roughly the same trajectory length.
    self.assertAllClose(1.5, mean_step_size[0], atol=0.2)
    self.assertAllClose(15., mean_max_trajectory_length[0], rtol=0.3)
    self.assertAllClose(
        target.mean(), mean.mean(0),
        atol=1.)
    self.assertAllClose(
        target.variance(),
        var.mean(0) + mean.var(0),
        rtol=0.1)


del _GradientBasedTrajectoryLengthAdaptationTest

if __name__ == '__main__':
  test_util.main()
