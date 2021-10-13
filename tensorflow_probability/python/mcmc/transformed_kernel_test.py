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
"""Tests for `TransformedTransitionKernel` `TransitionKernel`."""

import collections

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


FakeInnerKernelResults = collections.namedtuple(
    'FakeInnerKernelResults', ['target_log_prob', 'step_size'])


def _maybe_seed(seed):
  if tf.executing_eagerly():
    tf.random.set_seed(seed)
    return None
  return seed


def make_transform_then_adapt_kernel(bijector):
  trans_kernel = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=FakeInnerKernel(target_log_prob_fn=fake_target_log_prob),
      bijector=bijector)
  return tfp.mcmc.SimpleStepSizeAdaptation(
      inner_kernel=trans_kernel,
      num_adaptation_steps=9)


def make_adapt_then_transform_kernel(bijector):
  step_adaptation_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
      inner_kernel=FakeInnerKernel(target_log_prob_fn=fake_target_log_prob),
      num_adaptation_steps=9)
  return tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=step_adaptation_kernel,
      bijector=bijector)


def fake_target_log_prob(x):
  return -x**2 / 2.


class FakeInnerKernel(tfp.mcmc.TransitionKernel):
  """Fake Transition Kernel."""

  def __init__(self, target_log_prob_fn, is_calibrated=True, step_size=10):
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn, is_calibrated=is_calibrated,
        step_size=step_size)

  @property
  def parameters(self):
    return self._parameters

  @property
  def is_calibrated(self):
    return self._parameters['is_calibrated']

  def one_step(self, current_state, previous_kernel_results):
    pass

  def bootstrap_results(self, init_state):
    return FakeInnerKernelResults(
        target_log_prob=self._parameters['target_log_prob_fn'](init_state),
        step_size=tf.nest.map_structure(tf.convert_to_tensor,
                                        self.parameters['step_size']))


@test_util.test_all_tf_execution_regimes
class TransformedTransitionKernelTest(test_util.TestCase):

  def setUp(self):
    super(TransformedTransitionKernelTest, self).setUp()
    self.dtype = np.float32

  @test_util.numpy_disable_gradient_test('HMC')
  def test_support_works_correctly_with_hmc(self):
    num_results = 500
    target = tfd.Beta(
        concentration1=self.dtype(1.),
        concentration0=self.dtype(10.))
    transformed_hmc = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            step_size=1.64,
            num_leapfrog_steps=2),
        bijector=tfb.Sigmoid())
    # Recall, tfp.mcmc.sample_chain calls
    # transformed_hmc.bootstrap_results too.
    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        # The initial state is used by inner_kernel.bootstrap_results.
        # Note the input is *after* bijector.forward.
        current_state=self.dtype(0.25),
        kernel=transformed_hmc,
        num_burnin_steps=200,
        num_steps_between_results=1,
        seed=test_util.test_seed())
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    sample_var = tf.reduce_mean(
        tf.math.squared_difference(states, sample_mean), axis=0)
    [
        sample_mean_,
        sample_var_,
        is_accepted_,
        true_mean_,
        true_var_,
    ] = self.evaluate([
        sample_mean,
        sample_var,
        kernel_results.inner_results.is_accepted,
        target.mean(),
        target.variance(),
    ])
    self.assertAllClose(true_mean_, sample_mean_,
                        atol=0.15, rtol=0.)
    self.assertAllClose(true_var_, sample_var_,
                        atol=0.03, rtol=0.2)
    self.assertNear(0.6, is_accepted_.mean(), err=0.15)

  @test_util.numpy_disable_gradient_test('Langevin')
  def test_support_works_correctly_with_mala(self):
    num_results = 500
    target = tfd.Beta(
        concentration1=self.dtype(1.),
        concentration0=self.dtype(10.))
    transformed_mala = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            step_size=1.),
        bijector=tfb.Sigmoid())
    # Recall, tfp.mcmc.sample_chain calls
    # transformed_hmc.bootstrap_results too.
    states = tfp.mcmc.sample_chain(
        num_results=num_results,
        # The initial state is used by inner_kernel.bootstrap_results.
        # Note the input is *after* bijector.forward.
        current_state=self.dtype(0.25),
        kernel=transformed_mala,
        num_burnin_steps=200,
        num_steps_between_results=1,
        trace_fn=None,
        seed=test_util.test_seed())
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    sample_var = tf.reduce_mean(
        tf.math.squared_difference(states, sample_mean), axis=0)
    [
        sample_mean_,
        sample_var_,
        true_mean_,
        true_var_,
    ] = self.evaluate([
        sample_mean,
        sample_var,
        target.mean(),
        target.variance(),
    ])
    self.assertAllClose(true_mean_, sample_mean_,
                        atol=0.15, rtol=0.)
    self.assertAllClose(true_var_, sample_var_,
                        atol=0.03, rtol=0.2)

  def test_support_works_correctly_with_rwm(self):
    num_results = 500
    target = tfd.Beta(
        concentration1=self.dtype(1.),
        concentration0=self.dtype(10.))
    transformed_rwm = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=tf.function(target.log_prob, autograph=False),
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=1.5)),
        bijector=tfb.Sigmoid())
    # Recall, tfp.mcmc.sample_chain calls
    # transformed_hmc.bootstrap_results too.
    states = tfp.mcmc.sample_chain(
        num_results=num_results,
        # The initial state is used by inner_kernel.bootstrap_results.
        # Note the input is *after* bijector.forward.
        current_state=self.dtype(0.25),
        kernel=transformed_rwm,
        num_burnin_steps=200,
        num_steps_between_results=1,
        trace_fn=None,
        seed=test_util.test_seed())
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    sample_var = tf.reduce_mean(
        tf.math.squared_difference(states, sample_mean), axis=0)
    [
        sample_mean_,
        sample_var_,
        true_mean_,
        true_var_,
    ] = self.evaluate([
        sample_mean,
        sample_var,
        target.mean(),
        target.variance(),
    ])
    self.assertAllClose(true_mean_, sample_mean_,
                        atol=0.15, rtol=0.)
    self.assertAllClose(true_var_, sample_var_,
                        atol=0.03, rtol=0.2)

  @test_util.numpy_disable_gradient_test('HMC')
  def test_end_to_end_works_correctly(self):
    true_mean = self.dtype([0, 0])
    true_cov = self.dtype([[1, 0.5],
                           [0.5, 1]])
    num_results = 500
    def target_log_prob(x, y):
      # Corresponds to unnormalized MVN.
      # z = matmul(inv(chol(true_cov)), [x, y] - true_mean)
      z = tf.stack([x, y], axis=-1) - true_mean
      z = tf.squeeze(
          tf.linalg.triangular_solve(
              np.linalg.cholesky(true_cov),
              z[..., tf.newaxis]),
          axis=-1)
      return -0.5 * tf.reduce_sum(z**2., axis=-1)

    transformed_hmc = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=tf.function(target_log_prob, autograph=False),
            # Affine scaling means we have to change the step_size
            # in order to get 60% acceptance, as was done in mcmc/hmc_test.py.
            step_size=[1.23 / 0.75, 1.23 / 0.5],
            num_leapfrog_steps=2),
        bijector=[
            tfb.Scale(scale=0.75),
            tfb.Scale(scale=0.5),
        ])
    # Recall, tfp.mcmc.sample_chain calls
    # transformed_hmc.bootstrap_results too.
    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        # The initial state is used by inner_kernel.bootstrap_results.
        # Note the input is *after* `bijector.forward`.
        current_state=[self.dtype(-2), self.dtype(2)],
        kernel=transformed_hmc,
        num_burnin_steps=200,
        num_steps_between_results=1,
        seed=test_util.test_seed())
    states = tf.stack(states, axis=-1)
    self.assertEqual(num_results, tf.compat.dimension_value(states.shape[0]))
    sample_mean = tf.reduce_mean(states, axis=0)
    x = states - sample_mean
    sample_cov = tf.matmul(x, x, transpose_a=True) / self.dtype(num_results)
    [sample_mean_, sample_cov_, is_accepted_] = self.evaluate([
        sample_mean, sample_cov, kernel_results.inner_results.is_accepted])
    self.assertAllClose(0.6, is_accepted_.mean(), atol=0.15, rtol=0.)
    self.assertAllClose(sample_mean_, true_mean, atol=0.2, rtol=0.)
    self.assertAllClose(sample_cov_, true_cov, atol=0., rtol=0.4)

  def test_bootstrap_requires_xor_args(self):
    transformed_fake = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=FakeInnerKernel(target_log_prob_fn=fake_target_log_prob),
        bijector=tfb.Exp())
    with self.assertRaisesWithPredicateMatch(
        ValueError, r'Must specify exactly one'):
      transformed_fake.bootstrap_results()
    with self.assertRaisesWithPredicateMatch(
        ValueError, r'Must specify exactly one'):
      transformed_fake.bootstrap_results(
          init_state=2., transformed_init_state=np.log(2.))

  def test_bootstrap_correctly_untransforms(self):
    transformed_fake = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=FakeInnerKernel(target_log_prob_fn=fake_target_log_prob),
        bijector=tfb.Exp())
    automatic_pkr, manual_pkr = self.evaluate([
        transformed_fake.bootstrap_results(2.),
        transformed_fake.bootstrap_results(transformed_init_state=[4., 5.]),
    ])
    self.assertNear(np.log(2.), automatic_pkr.transformed_state, err=1e-6)
    self.assertAllClose(
        [4., 5.], manual_pkr.transformed_state, atol=0., rtol=1e-6)

  def test_copy_works(self):
    transformed = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=FakeInnerKernel(target_log_prob_fn=fake_target_log_prob),
        bijector=tfb.Scale(2.))

    transformed_copy = tfp.mcmc.TransformedTransitionKernel(
        **transformed.parameters)

    pkr, pkr_copy = self.evaluate([
        transformed.bootstrap_results(1.),
        transformed_copy.bootstrap_results(1.)
    ])

    self.assertAllClose(pkr.inner_results.target_log_prob,
                        pkr_copy.inner_results.target_log_prob)

  def test_is_calibrated(self):
    self.assertTrue(
        tfp.mcmc.TransformedTransitionKernel(
            FakeInnerKernel(lambda x: -x**2 / 2, True),
            tfb.Identity()).is_calibrated)
    self.assertFalse(
        tfp.mcmc.TransformedTransitionKernel(
            FakeInnerKernel(lambda x: -x**2 / 2, False),
            tfb.Identity()).is_calibrated)

  def test_bijector_valid_transform_then_adapt(self):
    new_kernel = make_transform_then_adapt_kernel(tfb.Exp())
    pkr_one, pkr_two = self.evaluate([
        new_kernel.bootstrap_results(2.),
        new_kernel.bootstrap_results(9.),
    ])
    self.assertNear(np.log(2.),
                    pkr_one.inner_results.transformed_state,
                    err=1e-6)
    self.assertNear(np.log(9.),
                    pkr_two.inner_results.transformed_state,
                    err=1e-6)

  def test_bijector_valid_adapt_then_transform(self):
    new_kernel = make_adapt_then_transform_kernel(tfb.Exp())
    pkr_one, pkr_two = self.evaluate([
        new_kernel.bootstrap_results(2.),
        new_kernel.bootstrap_results(9.),
    ])
    self.assertNear(np.log(2.), pkr_one.transformed_state, err=1e-6)
    self.assertNear(np.log(9.), pkr_two.transformed_state, err=1e-6)

  @test_util.numpy_disable_gradient_test('HMC')
  def test_step_size_changed(self):
    target_dist = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 10.])
    # `hmc_kernel`'s step size is far from optimal
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_dist.log_prob,
        num_leapfrog_steps=27,
        step_size=10)
    step_adaptation_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        adaptation_rate=0.8,
        num_adaptation_steps=9)
    trans_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=step_adaptation_kernel,
        bijector=tfb.Exp()
    )
    kernel_results = trans_kernel.inner_kernel.bootstrap_results(tf.zeros(2))
    stream = test_util.test_seed_stream()
    for _ in range(2):
      _, kernel_results = trans_kernel.inner_kernel.one_step(tf.zeros(2),
                                                             kernel_results,
                                                             seed=stream())
    adapted_step_size = self.evaluate(
        kernel_results.inner_results.accepted_results.step_size)
    self.assertLess(adapted_step_size, 7)

  def test_deeply_nested(self):
    step_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=FakeInnerKernel(target_log_prob_fn=fake_target_log_prob),
        num_adaptation_steps=9)
    double_step_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=step_kernel,
        num_adaptation_steps=9)
    trans_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=double_step_kernel,
        bijector=tfb.Exp())
    pkr_one, pkr_two = self.evaluate([
        trans_kernel.bootstrap_results(2.),
        trans_kernel.bootstrap_results(9.),
    ])
    self.assertNear(np.log(2.),
                    pkr_one.transformed_state,
                    err=1e-6)
    self.assertNear(np.log(9.),
                    pkr_two.transformed_state,
                    err=1e-6)

  @test_util.numpy_disable_gradient_test('HMC')
  def test_nested_transform(self):
    target_dist = tfd.Normal(loc=0., scale=1.)
    b1 = tfb.Scale(0.5)
    b2 = tfb.Exp()
    chain = tfb.Chain([b2, b1])  # applies bijectors right to left (b1 then b2).
    inner_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_dist.log_prob,
            num_leapfrog_steps=27,
            step_size=10),
        bijector=b1)
    outer_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=inner_kernel,
        bijector=b2)
    chain_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_dist.log_prob,
            num_leapfrog_steps=27,
            step_size=10),
        bijector=chain)
    outer_pkr_one, outer_pkr_two = self.evaluate([
        outer_kernel.bootstrap_results(2.),
        outer_kernel.bootstrap_results(9.),
    ])

    # the outermost kernel only applies the outermost bijector
    self.assertNear(np.log(2.), outer_pkr_one.transformed_state, err=1e-6)
    self.assertNear(np.log(9.), outer_pkr_two.transformed_state, err=1e-6)

    chain_pkr_one, chain_pkr_two = self.evaluate([
        chain_kernel.bootstrap_results(2.),
        chain_kernel.bootstrap_results(9.),
    ])

    # all bijectors are applied to the inner kernel, from innermost to outermost
    # this behavior is completely analogous to a bijector Chain
    self.assertNear(chain_pkr_one.transformed_state,
                    outer_pkr_one.inner_results.transformed_state,
                    err=1e-6)
    self.assertEqual(chain_pkr_one.inner_results.accepted_results,
                     outer_pkr_one.inner_results.inner_results.accepted_results)
    self.assertNear(chain_pkr_two.transformed_state,
                    outer_pkr_two.inner_results.transformed_state,
                    err=1e-6)
    self.assertEqual(chain_pkr_two.inner_results.accepted_results,
                     outer_pkr_two.inner_results.inner_results.accepted_results)

    seed = test_util.test_seed(sampler_type='stateless')
    outer_results_one, outer_results_two = self.evaluate([
        outer_kernel.one_step(2., outer_pkr_one, seed=seed),
        outer_kernel.one_step(9., outer_pkr_two, seed=seed)
    ])
    chain_results_one, chain_results_two = self.evaluate([
        chain_kernel.one_step(2., chain_pkr_one, seed=seed),
        chain_kernel.one_step(9., chain_pkr_two, seed=seed)
    ])
    self.assertNear(chain_results_one[0],
                    outer_results_one[0],
                    err=1e-6)
    self.assertNear(chain_results_two[0],
                    outer_results_two[0],
                    err=1e-6)

  @test_util.numpy_disable_gradient_test('HMC')
  def test_multipart_bijector(self):
    seed_stream = test_util.test_seed_stream()

    prior = tfd.JointDistributionSequential([
        tfd.Gamma(1., 1.),
        lambda scale: tfd.Uniform(0., scale),
        lambda concentration: tfd.CholeskyLKJ(4, concentration),
    ], validate_args=True)
    likelihood = lambda corr: tfd.MultivariateNormalTriL(scale_tril=corr)
    obs = self.evaluate(
        likelihood(
            prior.sample(seed=seed_stream())[-1]).sample(seed=seed_stream()))

    bij = prior.experimental_default_event_space_bijector()

    def target_log_prob(scale, conc, corr):
      return prior.log_prob(scale, conc, corr) + likelihood(corr).log_prob(obs)
    kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob,
                                            num_leapfrog_steps=3, step_size=.5)
    kernel = tfp.mcmc.TransformedTransitionKernel(kernel, bij)

    init = self.evaluate(
        tuple(tf.random.uniform(s, -2., 2., seed=seed_stream())
              for s in bij.inverse_event_shape(prior.event_shape)))
    state = bij.forward(init)
    kr = kernel.bootstrap_results(state)
    next_state, next_kr = kernel.one_step(state, kr, seed=seed_stream())
    self.evaluate((state, kr, next_state, next_kr))
    expected = (target_log_prob(*state) -
                bij.inverse_log_det_jacobian(state, [0, 0, 2]))
    actual = kernel._inner_kernel.target_log_prob_fn(*init)  # pylint: disable=protected-access
    self.assertAllClose(expected, actual)


if __name__ == '__main__':
  test_util.main()
