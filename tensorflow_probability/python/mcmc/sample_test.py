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
"""Tests for MCMC drivers (e.g., `sample_chain`)."""

import collections
import warnings

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import pad
from tensorflow_probability.python.distributions import beta as beta_lib
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import multinomial
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental.mcmc import diagonal_mass_matrix_adaptation as dmma
from tensorflow_probability.python.experimental.mcmc import preconditioned_hmc
from tensorflow_probability.python.experimental.stats import sample_stats
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import kernel as kernel_lib
from tensorflow_probability.python.mcmc import nuts
from tensorflow_probability.python.mcmc import random_walk_metropolis
from tensorflow_probability.python.mcmc import simple_step_size_adaptation as ssa
from tensorflow_probability.python.mcmc import transformed_kernel
from tensorflow_probability.python.mcmc.sample import sample_chain
from tensorflow_probability.python.random.random_ops import rayleigh


JAX_MODE = False
NUMPY_MODE = False

TestTransitionKernelResults = collections.namedtuple(
    'TestTransitionKernelResults', 'counter_1, counter_2')


class TestTransitionKernel(kernel_lib.TransitionKernel):
  """Fake deterministic `TransitionKernel` for testing purposes."""

  def __init__(self, is_calibrated=True, accepts_seed=True):
    self._is_calibrated = is_calibrated
    self._accepts_seed = accepts_seed

  def one_step(self, current_state, previous_kernel_results, seed=None):
    if seed is not None and not self._accepts_seed:
      raise TypeError('seed arg not accepted')
    return current_state + 1, TestTransitionKernelResults(
        counter_1=previous_kernel_results.counter_1 + 1,
        counter_2=previous_kernel_results.counter_2 + 2)

  def bootstrap_results(self, current_state):
    return TestTransitionKernelResults(counter_1=0, counter_2=0)

  @property
  def is_calibrated(self):
    return self._is_calibrated


class RandomTransitionKernel(kernel_lib.TransitionKernel):
  """Fake `TransitionKernel` that randomly assigns the next state.

  Regardless of the current state, the `one_step` method will always
  randomly sample from a Rayleigh Distribution.
  """

  def __init__(self, is_calibrated=True, accepts_seed=True):
    self._is_calibrated = is_calibrated
    self._accepts_seed = accepts_seed

  def one_step(self, current_state, previous_kernel_results, seed=None):
    if seed is not None and not self._accepts_seed:
      raise TypeError('seed arg not accepted')
    random_next_state = rayleigh(current_state.shape, seed=seed)
    return random_next_state, previous_kernel_results

  @property
  def is_calibrated(self):
    return self._is_calibrated


@test_util.test_all_tf_execution_regimes
class SampleChainTest(test_util.TestCase):

  def setUp(self):
    self._shape_param = 5.
    self._rate_param = 10.

    super(SampleChainTest, self).setUp()

  @test_util.numpy_disable_gradient_test('HMC')
  def testChainWorksCorrelatedMultivariate(self):

    # Disabling this test in eager mode as it is very slow.
    # Other tests check eager behavior, so we can safely disable this test.
    if tf.executing_eagerly() and not JAX_MODE:
      return

    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5],
                      [0.5, 1]])
    true_cov_chol = np.linalg.cholesky(true_cov)
    num_results = 3000
    counter = collections.Counter()
    def target_log_prob(x, y):
      counter['target_calls'] += 1
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      z = tf.stack([x, y], axis=-1) - true_mean
      z = tf.linalg.triangular_solve(true_cov_chol, z[..., tf.newaxis])[..., 0]
      return -0.5 * tf.reduce_sum(z**2., axis=-1)

    states = sample_chain(
        num_results=num_results,
        current_state=[dtype(-2), dtype(2)],
        kernel=hmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            step_size=[0.5, 0.5],
            num_leapfrog_steps=2),
        num_burnin_steps=200,
        num_steps_between_results=1,
        trace_fn=None,
        seed=test_util.test_seed())
    self.assertAllEqual(dict(target_calls=2), counter)
    states = tf.stack(states, axis=-1)
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    x = states - sample_mean
    sample_cov = tf.matmul(x, x, transpose_a=True) / dtype(num_results)
    sample_mean_, sample_cov_ = self.evaluate([sample_mean, sample_cov])
    self.assertAllClose(true_mean, sample_mean_,
                        atol=0.1, rtol=0.)
    self.assertAllClose(true_cov, sample_cov_,
                        atol=0., rtol=0.175)

  def testBasicOperation(self):
    kernel = TestTransitionKernel()
    samples, kernel_results = sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        seed=test_util.test_seed())

    self.assertAllClose(
        [2], tensorshape_util.as_list(samples.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_2.shape))

    samples, kernel_results = self.evaluate([samples, kernel_results])
    self.assertAllClose([1, 2], samples)
    self.assertAllClose([1, 2], kernel_results.counter_1)
    self.assertAllClose([2, 4], kernel_results.counter_2)

  def testBurnin(self):
    kernel = TestTransitionKernel()
    samples, kernel_results = sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        num_burnin_steps=1,
        seed=test_util.test_seed())

    self.assertAllClose([2], tensorshape_util.as_list(samples.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_2.shape))

    samples, kernel_results = self.evaluate([samples, kernel_results])
    self.assertAllClose([2, 3], samples)
    self.assertAllClose([2, 3], kernel_results.counter_1)
    self.assertAllClose([4, 6], kernel_results.counter_2)

  def testThinning(self):
    kernel = TestTransitionKernel()
    samples, kernel_results = sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        num_steps_between_results=2,
        seed=test_util.test_seed())

    self.assertAllClose([2], tensorshape_util.as_list(samples.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(kernel_results.counter_2.shape))

    samples, kernel_results = self.evaluate([samples, kernel_results])
    self.assertAllClose([1, 4], samples)
    self.assertAllClose([1, 4], kernel_results.counter_1)
    self.assertAllClose([2, 8], kernel_results.counter_2)

  def testDefaultTraceNamedTuple(self):
    kernel = TestTransitionKernel()
    res = sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        seed=test_util.test_seed())

    self.assertAllClose([2], tensorshape_util.as_list(res.all_states.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(res.trace.counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(res.trace.counter_2.shape))

    res = self.evaluate(res)
    self.assertAllClose([1, 2], res.all_states)
    self.assertAllClose([1, 2], res.trace.counter_1)
    self.assertAllClose([2, 4], res.trace.counter_2)

  def testNoTraceFn(self):
    kernel = TestTransitionKernel()
    samples = sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        trace_fn=None,
        seed=test_util.test_seed())

    self.assertAllClose([2], tensorshape_util.as_list(samples.shape))

    samples = self.evaluate(samples)
    self.assertAllClose([1, 2], samples)

  def testCustomTrace(self):
    kernel = TestTransitionKernel()
    res = sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        trace_fn=lambda *args: args,
        seed=test_util.test_seed())

    self.assertAllClose([2], tensorshape_util.as_list(res.all_states.shape))
    self.assertAllClose([2], tensorshape_util.as_list(res.trace[0].shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(res.trace[1].counter_1.shape))
    self.assertAllClose(
        [2], tensorshape_util.as_list(res.trace[1].counter_2.shape))

    res = self.evaluate(res)
    self.assertAllClose([1, 2], res.all_states)
    self.assertAllClose([1, 2], res.trace[0])
    self.assertAllClose([1, 2], res.trace[1].counter_1)
    self.assertAllClose([2, 4], res.trace[1].counter_2)

  def testCheckpointing(self):
    kernel = TestTransitionKernel()
    res = sample_chain(
        num_results=2,
        current_state=0,
        kernel=kernel,
        trace_fn=None,
        return_final_kernel_results=True,
        seed=test_util.test_seed())

    self.assertAllClose([2], tensorshape_util.as_list(res.all_states.shape))
    self.assertEqual((), res.trace)
    self.assertAllClose(
        [], tensorshape_util.as_list(res.final_kernel_results.counter_1.shape))
    self.assertAllClose(
        [], tensorshape_util.as_list(res.final_kernel_results.counter_2.shape))

    res = self.evaluate(res)
    self.assertAllClose([1, 2], res.all_states)
    self.assertAllClose(2, res.final_kernel_results.counter_1)
    self.assertAllClose(4, res.final_kernel_results.counter_2)

  def testWarningsDefault(self):
    with warnings.catch_warnings(record=True) as triggered:
      kernel = TestTransitionKernel()
      sample_chain(
          num_results=2,
          current_state=0,
          kernel=kernel,
          seed=test_util.test_seed())
    self.assertTrue(
        any('Tracing all kernel results by default is deprecated' in str(
            warning.message) for warning in triggered))

  def testNoWarningsExplicit(self):
    with warnings.catch_warnings(record=True) as triggered:
      kernel = TestTransitionKernel()
      sample_chain(
          num_results=2,
          current_state=0,
          kernel=kernel,
          trace_fn=lambda current_state, kernel_results: kernel_results,
          seed=test_util.test_seed())
    self.assertFalse(
        any('Tracing all kernel results by default is deprecated' in str(
            warning.message) for warning in triggered))

  def testIsCalibrated(self):
    with warnings.catch_warnings(record=True) as triggered:
      kernel = TestTransitionKernel(False)
      sample_chain(
          num_results=2,
          current_state=0,
          kernel=kernel,
          trace_fn=lambda current_state, kernel_results: kernel_results,
          seed=test_util.test_seed())
    self.assertTrue(
        any('supplied `TransitionKernel` is not calibrated.' in str(
            warning.message) for warning in triggered))

  @test_util.jax_disable_test_missing_functionality('no tf.TensorSpec')
  @test_util.numpy_disable_test_missing_functionality('no tf.TensorSpec')
  def testReproduceBug159550941(self):
    # Reproduction for b/159550941.
    input_signature = [tf.TensorSpec([], tf.int32)]

    @tf.function(input_signature=input_signature)
    def sample(chains):
      initial_state = tf.zeros([chains, 1])
      def log_prob(x):
        return tf.reduce_sum(normal.Normal(0, 1).log_prob(x), -1)

      kernel = hmc.HamiltonianMonteCarlo(
          target_log_prob_fn=log_prob, num_leapfrog_steps=3, step_size=1e-3)
      return sample_chain(
          num_results=5,
          num_burnin_steps=4,
          current_state=initial_state,
          kernel=kernel,
          trace_fn=None)

    # Checking that shape inference doesn't fail.
    sample(2)

  def testSeedReproducibility(self):
    first_fake_kernel = RandomTransitionKernel()
    second_fake_kernel = RandomTransitionKernel()
    seed = samplers.sanitize_seed(test_util.test_seed())
    first_final_state = sample_chain(
        num_results=5,
        current_state=0.,
        kernel=first_fake_kernel,
        seed=seed,
    )
    second_final_state = sample_chain(
        num_results=5,
        current_state=1.,  # difference should be irrelevant
        kernel=second_fake_kernel,
        seed=seed,
    )
    first_final_state, second_final_state = self.evaluate([
        first_final_state, second_final_state
    ])
    self.assertAllCloseNested(
        first_final_state, second_final_state, rtol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='RWM_tuple',
          kernel_from_log_prob=random_walk_metropolis.RandomWalkMetropolis,
          sample_dtype=(tf.float32,) * 4),
      dict(
          testcase_name='RWM_namedtuple',
          kernel_from_log_prob=random_walk_metropolis.RandomWalkMetropolis),
      dict(
          testcase_name='HMC_tuple',
          kernel_from_log_prob=lambda lp_fn: hmc.HamiltonianMonteCarlo(  # pylint: disable=g-long-lambda
              lp_fn,
              step_size=0.1,
              num_leapfrog_steps=10),
          skip='HMC requires gradients' if NUMPY_MODE else '',
          sample_dtype=(tf.float32,) * 4),
      dict(
          testcase_name='HMC_namedtuple',
          kernel_from_log_prob=lambda lp_fn: hmc.HamiltonianMonteCarlo(  # pylint: disable=g-long-lambda
              lp_fn,
              step_size=0.1,
              num_leapfrog_steps=10),
          skip='HMC requires gradients' if NUMPY_MODE else ''),
      dict(
          testcase_name='NUTS_tuple',
          kernel_from_log_prob=lambda lp_fn: nuts.NoUTurnSampler(  # pylint: disable=g-long-lambda
              lp_fn, step_size=0.1),
          skip='NUTS requires gradients' if NUMPY_MODE else '',
          sample_dtype=(tf.float32,) * 4),
      dict(
          testcase_name='NUTS_namedtuple',
          kernel_from_log_prob=lambda lp_fn: nuts.NoUTurnSampler(  # pylint: disable=g-long-lambda
              lp_fn, step_size=0.1),
          skip='NUTS requires gradients' if NUMPY_MODE else ''))
  def testStructuredState(self, kernel_from_log_prob, skip='',
                          **model_kwargs):
    if skip:
      self.skipTest(skip)
    seed_stream = test_util.test_seed_stream()

    n = 300
    p = 50
    x = tf.random.normal([n, p], seed=seed_stream())

    def beta_proportion(mu, kappa):
      return beta_lib.Beta(
          concentration0=mu * kappa, concentration1=(1 - mu) * kappa)

    root = jdc.JointDistributionCoroutine.Root
    def model_coroutine():
      beta = yield root(
          sample_dist_lib.Sample(normal.Normal(0, 1), [p], name='beta'))
      alpha = yield root(normal.Normal(0, 1, name='alpha'))
      kappa = yield root(gamma.Gamma(1, 1, name='kappa'))
      mu = tf.math.sigmoid(alpha[..., tf.newaxis] +
                           tf.einsum('...p,np->...n', beta, x))
      yield independent.Independent(
          beta_proportion(mu, kappa[..., tf.newaxis]),
          reinterpreted_batch_ndims=1,
          name='prob')

    model = jdc.JointDistributionCoroutine(model_coroutine, **model_kwargs)
    probs = model.sample(seed=seed_stream())[-1]
    pinned = model.experimental_pin(prob=probs)

    kernel = kernel_from_log_prob(pinned.unnormalized_log_prob)
    nburnin = 5
    if not isinstance(kernel, random_walk_metropolis.RandomWalkMetropolis):
      kernel = ssa.SimpleStepSizeAdaptation(
          kernel, num_adaptation_steps=nburnin // 2)
    kernel = transformed_kernel.TransformedTransitionKernel(
        kernel, pinned.experimental_default_event_space_bijector())
    nchains = 4

    @tf.function
    def sample():
      return sample_chain(
          1,
          current_state=pinned.sample_unpinned(nchains, seed=seed_stream()),
          kernel=kernel,
          num_burnin_steps=nburnin,
          trace_fn=None,
          seed=seed_stream())
    self.evaluate(sample())

  @test_util.jax_disable_test_missing_functionality('PHMC b/175107050')
  @test_util.numpy_disable_gradient_test('HMC')
  def testStructuredState2(self):

    @jdab.JointDistributionCoroutineAutoBatched
    def model():
      mu = yield sample_dist_lib.Sample(normal.Normal(0, 1), [65], name='mu')
      sigma = yield sample_dist_lib.Sample(
          exponential.Exponential(1.), [65], name='sigma')
      beta = yield sample_dist_lib.Sample(
          normal.Normal(
              loc=tf.gather(mu, tf.range(436) % 65, axis=-1),
              scale=tf.gather(sigma, tf.range(436) % 65, axis=-1)),
          4,
          name='beta')
      _ = yield multinomial.Multinomial(
          total_count=100., logits=pad.Pad([[0, 1]])(beta), name='y')

    stream = test_util.test_seed_stream()
    pinned = model.experimental_pin(y=model.sample(seed=stream()).y)
    struct = pinned.dtype
    stddevs = struct._make([
        tf.fill([65], .1), tf.fill([65], 1.), tf.fill([436, 4], 10.)])
    momentum_dist = jdab.JointDistributionNamedAutoBatched(
        struct._make(normal.Normal(0, 1 / std) for std in stddevs))
    kernel = preconditioned_hmc.PreconditionedHamiltonianMonteCarlo(
        pinned.unnormalized_log_prob,
        step_size=.1,
        num_leapfrog_steps=10,
        momentum_distribution=momentum_dist)
    bijector = pinned.experimental_default_event_space_bijector()
    kernel = transformed_kernel.TransformedTransitionKernel(kernel, bijector)
    pullback_shape = bijector.inverse_event_shape(pinned.event_shape)
    kernel = dmma.DiagonalMassMatrixAdaptation(
        kernel,
        initial_running_variance=struct._make(
            sample_stats.RunningVariance.from_shape(t) for t in pullback_shape))
    state = bijector(
        struct._make(
            uniform.Uniform(-2., 2.).sample(shp)
            for shp in bijector.inverse_event_shape(pinned.event_shape)))
    self.evaluate(
        sample_chain(3, current_state=state, kernel=kernel,
                     seed=stream()).all_states)


if __name__ == '__main__':
  test_util.main()
