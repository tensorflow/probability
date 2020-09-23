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
"""Tests for tensorflow_probability.python.experimental.mcmc.preconditioned_hmc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions
tfd_e = tfp.experimental.distributions


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters(
    dict(testcase_name='_default', use_default=True),
    dict(testcase_name='_explicit', use_default=False))
class PreconditionedHMCTest(test_util.TestCase):

  def test_diag(self, use_default):
    mvn = tfd.MultivariateNormalDiag(
        loc=[1., 2., 3.], scale_diag=[0.1, 1., 10.])

    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = tfd_e.MultivariateNormalInverseScaleLinearOperator(
          tf.zeros(3), inverse_scale=mvn.scale)
    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=0.3,
        num_leapfrog_steps=10)
    draws = tfp.mcmc.sample_chain(
        110,
        tf.zeros(3),
        kernel=hmc_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(draws[-100:])

    if not use_default:
      self.assertAllClose(ess, 100 * tf.ones(3))

  def test_tril(self, use_default):
    cov = 0.9 * tf.ones([3, 3]) + 0.1 * tf.eye(3)
    scale = tf.linalg.cholesky(cov)
    mv_tril = tfd.MultivariateNormalTriL(loc=[1., 2., 3.],
                                         scale_tril=scale)

    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = tfd_e.MultivariateNormalInverseScaleLinearOperator(
          tf.zeros(3), inverse_scale=tf.linalg.LinearOperatorFullMatrix(cov))
    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mv_tril.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=0.2,
        num_leapfrog_steps=10)
    draws = tfp.mcmc.sample_chain(
        120,
        tf.zeros(3),
        kernel=hmc_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(draws[-100:])
    if not use_default:
      self.assertAllEqual(ess, 100 * tf.ones(3))

  def test_transform(self, use_default):
    mvn = tfd.MultivariateNormalDiag(loc=[1., 2., 3.], scale_diag=[1., 1., 1.])
    diag_covariance = tf.constant([0.1, 1., 10.])

    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = tfd_e.MultivariateNormalInverseScaleLinearOperator(
          tf.zeros(3),
          inverse_scale=tf.linalg.LinearOperatorDiag(
              tf.math.sqrt(diag_covariance)))
    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=0.3,
        num_leapfrog_steps=10)

    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
        hmc_kernel, bijector=tfb.Scale(tf.math.rsqrt(diag_covariance)))

    draws = tfp.mcmc.sample_chain(
        110,
        tf.zeros(3),
        kernel=transformed_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(draws[-100:])
    if not use_default:
      self.assertAllClose(ess, 100 * tf.ones(3))

  def test_multi_state_part(self, use_default):
    mvn = tfd.JointDistributionSequential([
        tfd.Normal(1., 0.1),
        tfd.Normal(2., 1.),
        tfd.Independent(tfd.Normal(3 * tf.ones([2, 3, 4]), 10.), 3)
    ])

    if use_default:
      momentum_distribution = None
    else:
      reshape_to_scalar = tfp.bijectors.Reshape(event_shape_out=[])
      reshape_to_234 = tfp.bijectors.Reshape(event_shape_out=[2, 3, 4])
      momentum_distribution = tfd.JointDistributionSequential([
          reshape_to_scalar(
              tfd_e.MultivariateNormalInverseScaleLinearOperator(
                  0., tf.linalg.LinearOperatorDiag([0.1]))),
          reshape_to_scalar(
              tfd_e.MultivariateNormalInverseScaleLinearOperator(
                  0., tf.linalg.LinearOperatorDiag([1.]))),
          reshape_to_234(
              tfd_e.MultivariateNormalInverseScaleLinearOperator(
                  0., tf.linalg.LinearOperatorDiag(tf.fill([24], 10.))))
      ])
    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=0.3,
        num_leapfrog_steps=10)

    draws = tfp.mcmc.sample_chain(
        100, [0., 0., tf.zeros((2, 3, 4))],
        kernel=hmc_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(draws)
    if not use_default:
      self.assertAllClose(
          self.evaluate(ess),
          [tf.constant(100.),
           tf.constant(100.), 100. * tf.ones((2, 3, 4))])

  def test_batched_state(self, use_default):
    mvn = tfd.MultivariateNormalDiag(
        loc=[1., 2., 3.], scale_diag=[0.1, 1., 10.])
    batch_shape = [2, 4]
    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = tfd_e.MultivariateNormalInverseScaleLinearOperator(
          tf.zeros((2, 4, 3)), inverse_scale=mvn.scale)

    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=0.3,
        num_leapfrog_steps=10)

    draws = tfp.mcmc.sample_chain(
        110,
        tf.zeros(batch_shape + [3]),
        kernel=hmc_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(draws[10:], cross_chain_dims=[1, 2])
    if not use_default:
      self.assertAllClose(self.evaluate(ess), 100 * 2. * 4. * tf.ones(3))

  def test_batches(self, use_default):
    mvn = tfd.JointDistributionSequential(
        [tfd.Normal(1., 0.1),
         tfd.Normal(2., 1.),
         tfd.Normal(3., 10.)])
    n_chains = 10
    if use_default:
      momentum_distribution = None
    else:
      reshape_to_scalar = tfp.bijectors.Reshape(event_shape_out=[])
      momentum_distribution = tfd.JointDistributionSequential([
          reshape_to_scalar(
              tfd_e.MultivariateNormalInverseScaleLinearOperator(
                  0., tf.linalg.LinearOperatorDiag(tf.fill([n_chains, 1],
                                                           0.1)))),
          reshape_to_scalar(
              tfd_e.MultivariateNormalInverseScaleLinearOperator(
                  0., tf.linalg.LinearOperatorDiag(tf.fill([n_chains, 1],
                                                           1.)))),
          reshape_to_scalar(
              tfd_e.MultivariateNormalInverseScaleLinearOperator(
                  0., tf.linalg.LinearOperatorDiag(tf.fill([n_chains, 1],
                                                           10.)))),
      ])

    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=0.3,
        num_leapfrog_steps=10)

    draws = tfp.mcmc.sample_chain(
        100, [tf.zeros([n_chains]) for _ in range(3)],
        kernel=hmc_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(
        draws, cross_chain_dims=[1 for _ in draws])
    if not use_default:
      self.assertAllClose(self.evaluate(ess), 100 * n_chains * tf.ones(3))


if __name__ == '__main__':
  tf.test.main()
