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
"""Tests for MCMC driver, `sample_sequential_monte_carlo`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import os

# Dependency imports
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions.internal import statistical_testing as st
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import compute_hmc_step_size
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import gen_make_hmc_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import gen_make_transform_hmc_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import make_rwmh_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import simple_heuristic_tuning
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


def make_test_nuts_kernel_fn(target_log_prob_fn,
                             init_state,
                             scalings,
                             seed=None):
  """Set up a function to generate nuts kernel for testing."""
  max_tree_depth = 3

  state_std = [
      tf.math.reduce_std(x, axis=0, keepdims=True)
      for x in init_state
  ]
  step_size = compute_hmc_step_size(scalings, state_std, max_tree_depth**2)
  return tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn=target_log_prob_fn,
      step_size=step_size,
      max_tree_depth=max_tree_depth,
      seed=seed)


@test_util.test_all_tf_execution_regimes
class SampleSequentialMonteCarloTest(test_util.TestCase):

  def testCorrectStepSizeTransformedkernel(self):
    scalings = .1
    bijector = tfb.Sigmoid()
    prior = tfd.Beta(.1, .1)
    likelihood = tfd.Beta(5., 5.)
    init_state = [tf.clip_by_value(prior.sample(10000), 1e-5, 1.-1e-5)]
    make_transform_kernel_fn = gen_make_transform_hmc_kernel_fn(
        [bijector], num_leapfrog_steps=1)

    kernel = make_transform_kernel_fn(likelihood.log_prob,
                                      init_state,
                                      scalings=scalings)
    step_size, expected_step_size = self.evaluate([
        tf.squeeze(kernel.inner_kernel.step_size),
        scalings * tf.math.reduce_std(bijector.inverse(init_state))
    ])
    self.assertAllGreater(step_size, 0.)
    self.assertAllEqual(step_size, expected_step_size)

  @parameterized.named_parameters(
      ('RWMH', make_rwmh_kernel_fn, 0.45),
      ('HMC', gen_make_hmc_kernel_fn(5), 0.651),
      ('NUTS', make_test_nuts_kernel_fn, 0.8),
  )
  def testMixtureTargetLogProb(self, make_kernel_fn, optimal_accept):
    seed = test_util.test_seed()
    # Generate a 2 component Gaussian Mixture in 3 dimension
    nd = 3
    w = 0.1
    mixture_weight = tf.constant([w, 1. - w], tf.float64)
    mu = np.ones(nd) * .5
    component_loc = tf.cast(np.asarray([mu, -mu]), tf.float64)

    proposal = tfd.Sample(tfd.Normal(tf.constant(0., tf.float64), 10.),
                          sample_shape=nd)
    init_state = proposal.sample(5000, seed=seed)

    likelihood_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=mixture_weight),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=component_loc,
            scale_identity_multiplier=[.1, .2]))

    # Uniform prior
    init_log_prob = tf.zeros_like(proposal.log_prob(init_state))

    [
        n_stage, final_state, _
    ] = tfp.experimental.mcmc.sample_sequential_monte_carlo(
        lambda x: init_log_prob,
        likelihood_dist.log_prob,
        init_state,
        make_kernel_fn=make_kernel_fn,
        tuning_fn=functools.partial(simple_heuristic_tuning,
                                    optimal_accept=optimal_accept),
        max_num_steps=50,
        parallel_iterations=1,
        seed=None if tf.executing_eagerly() else seed)

    assert_cdf_equal_sample = st.assert_true_cdf_equal_by_dkwm_two_sample(
        final_state, likelihood_dist.sample(5000, seed=seed), 1e-5)

    n_stage, _ = self.evaluate((n_stage, assert_cdf_equal_sample))
    self.assertLess(n_stage, 15)

  def testMixtureMultiBatch(self):
    seed = test_util.test_seed()
    # Generate 3 copies (batches) of 2 component Gaussian Mixture in 2 dimension
    nd = 2
    n_batch = 3
    w = tf.constant([0.1, .25, .5], tf.float64)
    mixture_weight = tf.transpose(tf.stack([w, 1. - w]))
    mu = np.ones(nd) * .5
    loc = tf.cast(np.asarray([mu, -mu]), tf.float64)
    component_loc = tf.repeat(loc[tf.newaxis, ...], n_batch, axis=0)

    likelihood_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=mixture_weight),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=component_loc,
            scale_identity_multiplier=[.1, .2]))

    proposal = tfd.Sample(tfd.Normal(tf.constant(0., tf.float64), 10.),
                          sample_shape=nd)
    init_state = proposal.sample([5000, n_batch], seed=seed)
    log_prob_fn = likelihood_dist.log_prob
    print(log_prob_fn(init_state).shape)

    # Uniform prior
    init_log_prob = tf.zeros_like(log_prob_fn(init_state))

    [
        n_stage, final_state, _
    ] = tfp.experimental.mcmc.sample_sequential_monte_carlo(
        lambda x: init_log_prob,
        log_prob_fn,
        init_state,
        make_kernel_fn=make_test_nuts_kernel_fn,
        tuning_fn=functools.partial(simple_heuristic_tuning,
                                    optimal_accept=0.8),
        max_num_steps=50,
        parallel_iterations=1,
        seed=None if tf.executing_eagerly() else seed)

    assert_cdf_equal_sample = st.assert_true_cdf_equal_by_dkwm_two_sample(
        final_state, likelihood_dist.sample(5000, seed=seed), 1e-5)

    n_stage, _ = self.evaluate((n_stage, assert_cdf_equal_sample))
    self.assertLess(n_stage, 15)

  def testSampleEndtoEndXLA(self):
    """An end-to-end test of sampling using SMC."""
    if tf.executing_eagerly() or tf.config.experimental_functions_run_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    seed = test_util.test_seed()
    dtype = tf.float32
    # Set up data.
    predictors = np.asarray([
        201., 244., 47., 287., 203., 58., 210., 202., 198., 158., 165., 201.,
        157., 131., 166., 160., 186., 125., 218., 146.
    ])
    obs = np.asarray([
        592., 401., 583., 402., 495., 173., 479., 504., 510., 416., 393., 442.,
        317., 311., 400., 337., 423., 334., 533., 344.
    ])
    y_sigma = np.asarray([
        61., 25., 38., 15., 21., 15., 27., 14., 30., 16., 14., 25., 52., 16.,
        34., 31., 42., 26., 16., 22.
    ])
    y_sigma = tf.cast(y_sigma / (2 * obs.std(axis=0)), dtype)
    obs = tf.cast((obs - obs.mean(axis=0)) / (2 * obs.std(axis=0)), dtype)
    predictors = tf.cast(
        (predictors - predictors.mean(axis=0)) / (2 * predictors.std(axis=0)),
        dtype)

    hyper_mean = tf.cast(0, dtype)
    hyper_scale = tf.cast(10, dtype)
    # Generate model prior_log_prob_fn and likelihood_log_prob_fn.
    prior_jd = tfd.JointDistributionSequential([
        tfd.Normal(loc=hyper_mean, scale=hyper_scale),
        tfd.Normal(loc=hyper_mean, scale=hyper_scale),
        tfd.Normal(loc=hyper_mean, scale=hyper_scale),
        tfd.HalfNormal(scale=tf.cast(1., dtype)),
        tfd.Uniform(low=tf.cast(0, dtype), high=.5),
    ], validate_args=True)

    def likelihood_log_prob_fn(b0, b1, mu_out, sigma_out, weight):
      return tfd.Independent(
          tfd.Mixture(
              tfd.Categorical(
                  probs=tf.stack([
                      tf.repeat(1 - weight[..., tf.newaxis], 20, axis=-1),
                      tf.repeat(weight[..., tf.newaxis], 20, axis=-1)
                  ], -1)), [
                      tfd.Normal(
                          loc=b0[..., tf.newaxis] +
                          b1[..., tf.newaxis] * predictors,
                          scale=y_sigma),
                      tfd.Normal(
                          loc=mu_out[..., tf.newaxis],
                          scale=y_sigma + sigma_out[..., tf.newaxis])
                  ]), 1).log_prob(obs)

    unconstraining_bijectors = [
        tfb.Identity(),
        tfb.Identity(),
        tfb.Identity(),
        tfb.Exp(),
        tfb.Sigmoid(tf.constant(0., dtype), .5),
    ]
    make_transform_hmc_kernel_fn = gen_make_transform_hmc_kernel_fn(
        unconstraining_bijectors, num_leapfrog_steps=10)

    @tf.function(autograph=False, experimental_compile=True)
    def run_smc():
      # Ensure we're really in graph mode.
      assert hasattr(tf.constant([]), 'graph')

      return tfp.experimental.mcmc.sample_sequential_monte_carlo(
          prior_jd.log_prob,
          likelihood_log_prob_fn,
          prior_jd.sample([1000, 5], seed=seed),
          make_kernel_fn=make_transform_hmc_kernel_fn,
          tuning_fn=functools.partial(simple_heuristic_tuning,
                                      optimal_accept=.6),
          min_num_steps=10,
          parallel_iterations=1,
          seed=seed)

    n_stage, (b0, b1, mu_out, sigma_out, weight), _ = run_smc()

    (
        n_stage, b0, b1, mu_out, sigma_out, weight
    ) = self.evaluate((n_stage, b0, b1, mu_out, sigma_out, weight))

    self.assertTrue(n_stage, 15)

    # Compare the SMC posterior with the result from a calibrated HMC.
    self.assertAllClose(tf.reduce_mean(b0), 0.016, atol=0.005, rtol=0.005)
    self.assertAllClose(tf.reduce_mean(b1), 1.245, atol=0.005, rtol=0.035)
    self.assertAllClose(tf.reduce_mean(weight), 0.27, atol=0.02, rtol=0.02)
    self.assertAllClose(tf.reduce_mean(mu_out), 0.13, atol=0.2, rtol=0.2)
    self.assertAllClose(tf.reduce_mean(sigma_out), 0.46, atol=0.5, rtol=0.5)

    self.assertAllClose(tf.math.reduce_std(b0), 0.031, atol=0.015, rtol=0.3)
    self.assertAllClose(tf.math.reduce_std(b1), 0.068, atol=0.1, rtol=0.1)
    self.assertAllClose(tf.math.reduce_std(weight), 0.1, atol=0.1, rtol=0.1)


if __name__ == '__main__':
  # TODO(b/130689556) XLA's default fastmath config makes tests unstable.
  os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                             ' --xla_cpu_fast_math_honor_nans=true'
                             ' --xla_cpu_fast_math_honor_infs=true')
  tf.test.main()
