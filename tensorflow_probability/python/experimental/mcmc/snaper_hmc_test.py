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
"""Tests for SNAPER HMC."""

from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import distribute_test_lib
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import unnest

tfb = tfp.bijectors
tfd = tfp.distributions

JAX_MODE = False


class _SNAPERHMCTest(test_util.TestCase, parameterized.TestCase):
  dtype = np.float32

  def testEndToEndAdaptation(self):
    """End-to-end adaptation."""
    if tf.executing_eagerly() and not JAX_MODE:
      self.skipTest('Too slow for TF Eager.')

    num_dims = 8
    num_burnin_steps = 1000
    num_adaptation_steps = int(num_burnin_steps * 0.8)
    num_results = 500
    num_chains = 64
    step_size = 1e-2
    num_mala_steps = 100

    eigenvalues = np.exp(np.linspace(0., 3., num_dims))
    q, r = np.linalg.qr(np.random.randn(num_dims, num_dims))
    q *= np.sign(np.diag(r))
    covariance = (q * eigenvalues).dot(q.T).astype(self.dtype)

    _, eigs = np.linalg.eigh(covariance)
    principal_component = eigs[:, -1]

    gaussian = tfd.MultivariateNormalTriL(
        loc=tf.zeros(num_dims, self.dtype),
        scale_tril=tf.linalg.cholesky(covariance),
    )

    kernel = tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo(
        gaussian.log_prob,
        step_size=step_size,
        num_adaptation_steps=num_adaptation_steps,
        num_mala_steps=num_mala_steps,
    )
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=num_adaptation_steps)

    def trace_fn(_, pkr):
      return {
          'step_size':
              unnest.get_innermost(pkr, 'step_size'),
          'mean_trajectory_length':
              unnest.get_innermost(pkr, 'max_trajectory_length') / 2.,
          'principal_component':
              unnest.get_innermost(pkr, 'ema_principal_component'),
          'variance':
              unnest.get_innermost(pkr, 'ema_variance'),
          'num_leapfrog_steps':
              unnest.get_innermost(pkr, 'num_leapfrog_steps'),
      }

    init_x = tf.zeros([num_chains, num_dims], self.dtype)

    chain, trace = self.evaluate(
        tf.function(
            lambda seed: tfp.mcmc.sample_chain(  # pylint: disable=g-long-lambda
                num_results=num_burnin_steps + num_results,
                num_burnin_steps=0,
                current_state=init_x,
                kernel=kernel,
                trace_fn=trace_fn,
                seed=seed),
            autograph=False)(test_util.test_seed(sampler_type='stateless')))

    # Dtype propagation.
    self.assertEqual(self.dtype, chain.dtype)
    self.assertEqual(self.dtype, trace['step_size'].dtype)
    self.assertEqual(self.dtype, trace['mean_trajectory_length'].dtype)
    self.assertEqual(self.dtype, trace['variance'].dtype)
    self.assertEqual(self.dtype, trace['principal_component'].dtype)

    # Adaptation results.
    self.assertAllClose(1.75, trace['step_size'][-1], rtol=0.2)
    self.assertAllClose(4., trace['mean_trajectory_length'][-1], atol=1.)
    self.assertAllClose(np.diag(covariance), trace['variance'][-1], rtol=0.2)
    self.assertAllClose(
        principal_component / np.sign(principal_component[0]),
        trace['principal_component'][-1] /
        np.sign(trace['principal_component'][-1][0]),
        atol=0.2,
        rtol=0.2,
    )

    # Adaptation invariants.
    self.assertAllClose(
        np.ones(num_mala_steps, dtype=np.int32),
        trace['num_leapfrog_steps'][:num_mala_steps])
    self.assertAllClose(trace['step_size'][num_adaptation_steps],
                        trace['step_size'][-1])
    self.assertAllClose(trace['mean_trajectory_length'][num_adaptation_steps],
                        trace['mean_trajectory_length'][-1])
    self.assertAllClose(trace['variance'][num_adaptation_steps],
                        trace['variance'][-1])
    self.assertAllClose(trace['principal_component'][num_adaptation_steps],
                        trace['principal_component'][-1])

  @parameterized.named_parameters(
      ('Scalar', np.zeros((2,)), 1),
      ('Vector', np.zeros((2, 3)), 1),
      ('Tensor', np.zeros((2, 3, 4)), 1),
      ('List', [np.zeros((2,)), np.zeros((2,))], 1),
      ('Dict', {
          'x': np.zeros((2,)),
          'y': np.zeros((2,))
      }, 1),
      ('ScalarMultiBatch', np.zeros((2, 3)), 2),
  )
  def testStateStructure(self, init_x, batch_ndims):
    """Tests that one_step preserves structure/shape/dtype."""
    kernel = tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo(
        lambda *x: sum([  # pylint: disable=g-long-lambda
            tf.reduce_sum(part, list(range(batch_ndims, len(part.shape))))
            for part in x
        ]),
        step_size=0.1,
        num_adaptation_steps=2,
        num_mala_steps=100,
    )
    init_x = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(x, self.dtype), init_x)
    fin_x, kr = kernel.one_step(
        init_x,
        kernel.bootstrap_results(init_x),
        seed=test_util.test_seed(sampler_type='stateless'))

    self.assertAllAssertsNested(lambda x, y: self.assertEqual(x.dtype, y.dtype),
                                init_x, fin_x)
    self.assertAllAssertsNested(
        lambda x, y: self.assertAllClose(tf.shape(x), tf.shape(y)), init_x,
        fin_x)
    self.assertAllAssertsNested(
        lambda x, y: self.assertAllClose(  # pylint: disable=g-long-lambda
            tf.shape(x),
            tf.shape(y)[batch_ndims:]),
        kr.ema_principal_component,
        init_x)
    self.assertAllAssertsNested(
        lambda x, y: self.assertAllClose(  # pylint: disable=g-long-lambda
            tf.shape(x),
            tf.shape(y)[batch_ndims:]),
        kr.ema_variance,
        init_x)

  @test_util.jax_disable_test_missing_functionality('No stateful PRNGs')
  def testStatefulSeed(self):
    kernel = tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo(
        tfd.Normal(0., 1.).log_prob, step_size=0.1, num_adaptation_steps=1)
    _, kr = tfp.mcmc.sample_chain(
        current_state=tf.zeros(2),
        kernel=kernel,
        num_results=1,
        # seed is intentionally not specified so that the kernel gets a None
        # seed.
        trace_fn=lambda _, kr: kr)
    self.assertEqual([1, 2], list(kr.seed.shape))


@test_util.test_graph_and_eager_modes
class SNAPERHMCTestFloat32(_SNAPERHMCTest):
  dtype = np.float32


@test_util.test_graph_and_eager_modes
class SNAPERHMCTestFloat64(_SNAPERHMCTest):
  dtype = np.float64


del _SNAPERHMCTest


@test_util.test_graph_and_eager_modes
class DistributedSNAPERHMCTest(distribute_test_lib.DistributedTest):

  def testAxisNameTracking(self):
    kernel = tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo(
        tfd.Normal(0., 1.).log_prob, step_size=0.1, num_adaptation_steps=1)
    self.assertIsNone(kernel.experimental_shard_axis_names)
    kernel = kernel.experimental_with_shard_axes(['a'])
    self.assertListEqual(kernel.experimental_shard_axis_names, ['a'])

  def testShardedChainAxes(self):
    """Compare to testDocstring above. This shards the independent chains."""
    if not JAX_MODE:
      self.skipTest('b/181800108')

    num_dims = 8
    num_burnin_steps = 1000
    num_adaptation_steps = int(num_burnin_steps * 0.8)
    num_results = 500
    num_chains = 64
    step_size = 1e-2
    num_mala_steps = 100

    eigenvalues = np.exp(np.linspace(0., 3., num_dims))
    q, r = np.linalg.qr(np.random.randn(num_dims, num_dims))
    q *= np.sign(np.diag(r))
    covariance = (q * eigenvalues).dot(q.T).astype(np.float32)

    _, eigs = np.linalg.eigh(covariance)
    principal_component = eigs[:, -1]

    gaussian = tfd.MultivariateNormalTriL(
        loc=tf.zeros(num_dims),
        scale_tril=tf.linalg.cholesky(covariance),
    )

    kernel = tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo(
        gaussian.log_prob,
        step_size=step_size,
        num_adaptation_steps=num_adaptation_steps,
        num_mala_steps=num_mala_steps,
        experimental_reduce_chain_axis_names=self.axis_name,
    )
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        num_adaptation_steps=num_adaptation_steps,
        experimental_reduce_chain_axis_names=self.axis_name,
    )
    kernel = tfp.experimental.mcmc.Sharded(
        kernel, chain_axis_names=self.axis_name)

    def trace_fn(_, pkr):
      return {
          'step_size':
              unnest.get_innermost(pkr, 'step_size'),
          'mean_trajectory_length':
              unnest.get_innermost(pkr, 'max_trajectory_length') / 2.,
          'principal_component':
              unnest.get_innermost(pkr, 'ema_principal_component'),
          'variance':
              unnest.get_innermost(pkr, 'ema_variance'),
          'num_leapfrog_steps':
              unnest.get_innermost(pkr, 'num_leapfrog_steps'),
      }

    seed = test_util.test_seed(sampler_type='stateless')
    init_x = self.shard_values(
        tf.zeros([
            distribute_test_lib.NUM_DEVICES,
            num_chains // distribute_test_lib.NUM_DEVICES, num_dims
        ]))

    @tf.function(autograph=False)
    def run(init_x):
      return tfp.mcmc.sample_chain(
          num_results=num_burnin_steps + num_results,
          num_burnin_steps=0,
          current_state=init_x,
          kernel=kernel,
          trace_fn=trace_fn,
          seed=seed)

    _, trace = self.evaluate(
        self.per_replica_to_tensor(
            self.strategy_run(
                run,
                args=(
                    init_x,
                ),
                axis_name=self.axis_name,
            )))

    # Adaptation results.
    self.assertAllClose(1.75, trace['step_size'][0, -1], rtol=0.2)
    self.assertAllClose(4., trace['mean_trajectory_length'][0, -1], atol=1.1)
    self.assertAllClose(np.diag(covariance), trace['variance'][0, -1], rtol=0.2)
    self.assertAllClose(
        principal_component / np.sign(principal_component[0]),
        trace['principal_component'][0, -1] /
        np.sign(trace['principal_component'][0, -1][0]),
        atol=0.2,
        rtol=0.2,
    )

    # Shard consistency.
    self.assertAllClose(trace['step_size'][0], trace['step_size'][1])
    self.assertAllClose(trace['mean_trajectory_length'][0],
                        trace['mean_trajectory_length'][1])
    self.assertAllClose(trace['variance'][0], trace['variance'][1])
    self.assertAllClose(trace['principal_component'][0],
                        trace['principal_component'][1])

  def testShardedState(self):

    if not JAX_MODE:
      self.skipTest('b/181800108')

    num_burnin_steps = 1000
    num_adaptation_steps = int(num_burnin_steps * 0.8)
    num_results = 500
    num_chains = 64
    step_size = 1e-2
    num_mala_steps = 100

    def trace_fn(_, pkr):
      return {
          'step_size':
              unnest.get_innermost(pkr, 'step_size'),
          'mean_trajectory_length':
              unnest.get_innermost(pkr, 'max_trajectory_length') / 2.,
          'principal_component':
              unnest.get_innermost(pkr, 'ema_principal_component'),
          'variance':
              unnest.get_innermost(pkr, 'ema_variance'),
          'num_leapfrog_steps':
              unnest.get_innermost(pkr, 'num_leapfrog_steps'),
      }

    init_x = ([
        self.shard_values(
            tf.zeros((distribute_test_lib.NUM_DEVICES, num_chains)))
    ] * 2)
    local_scale = self.shard_values(
        1. + tf.one_hot(0, distribute_test_lib.NUM_DEVICES))

    @tf.function(autograph=False)
    def run(init_x, local_scale):

      @tfp.experimental.distribute.JointDistributionCoroutine
      def model():
        yield tfd.Normal(0., 1.)
        yield tfp.experimental.distribute.Sharded(
            tfd.Normal(0., local_scale), shard_axis_name=self.axis_name)

      kernel = tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo(
          model.log_prob,
          step_size=step_size,
          num_adaptation_steps=num_adaptation_steps,
          num_mala_steps=num_mala_steps,
          experimental_shard_axis_names=list(
              model.experimental_shard_axis_names),
      )
      kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
          kernel,
          num_adaptation_steps=num_adaptation_steps,
      )

      return tfp.mcmc.sample_chain(
          num_results=num_burnin_steps + num_results,
          num_burnin_steps=0,
          current_state=init_x,
          kernel=kernel,
          trace_fn=trace_fn,
          seed=test_util.test_seed(sampler_type='stateless'))

    _, trace = self.evaluate(
        self.per_replica_to_tensor(
            self.strategy_run(
                run,
                args=(init_x, local_scale),
                axis_name=self.axis_name,
            )))

    self.assertAllClose(0., trace['principal_component'][0][0, -1], atol=0.1)
    expected_local_principal_component = np.zeros(
        distribute_test_lib.NUM_DEVICES)
    expected_local_principal_component[0] = 1.
    self.assertAllClose(
        expected_local_principal_component,
        trace['principal_component'][1][:, -1],
        atol=0.1)

    self.assertAllClose(1., trace['variance'][0][0, -1], atol=0.1)
    expected_local_variance = np.ones(distribute_test_lib.NUM_DEVICES)
    expected_local_variance[0] = 4.
    self.assertAllClose(
        expected_local_variance, trace['variance'][1][:, -1], rtol=0.2)

    # Shard consistency.
    self.assertAllClose(trace['step_size'][0], trace['step_size'][1])
    self.assertAllClose(trace['mean_trajectory_length'][0],
                        trace['mean_trajectory_length'][1])


@test_util.test_graph_and_eager_modes
class GenericSNAPERHMCTest(test_util.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('StaticShape', True),
      ('DynamicShape', False),
  )
  def testTooFewChains(self, use_static_shape):
    state = tf.constant([[0.1, 0.2]])

    def tlp_fn(x):
      return tf1.placeholder_with_default(
          tf.reduce_sum(x, -1), shape=[1] if use_static_shape else [None])

    kernel = tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo(
        tlp_fn,
        step_size=0.1,
        num_adaptation_steps=2,
        num_mala_steps=100,
        validate_args=True,
    )

    with self.assertRaisesRegex(Exception,
                                'SNAPERHMC requires at least 2 chains'):
      self.evaluate(
          unnest.get_innermost(
              kernel.bootstrap_results(state), 'target_log_prob'))

  def testUnknownBatchNdims(self):
    if tf.executing_eagerly() or JAX_MODE:
      self.skipTest('Dynamic shape test.')

    state = tf.constant([[0.1, 0.2]])

    def tlp_fn(x):
      # Note how this is shape None rather than [None].
      return tf1.placeholder_with_default(tf.reduce_sum(x, -1), shape=None)

    kernel = tfp.experimental.mcmc.SNAPERHamiltonianMonteCarlo(
        tlp_fn,
        step_size=0.1,
        num_adaptation_steps=2,
        num_mala_steps=100,
        validate_args=True,
    )

    with self.assertRaisesRegex(
        Exception, 'SNAPERHMC currently requires a statically known rank of '
        'the target log probability'):
      kernel.bootstrap_results(state)


if __name__ == '__main__':
  test_util.main()
