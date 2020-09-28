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
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions
tfd_e = tfp.experimental.distributions


@test_util.test_graph_mode_only
class PreconditionedHMCPropertyTest(test_util.TestCase):
  """More careful tests that preconditioning is actually working."""

  def _calculate_expected_step_size(self, posterior_scales, target_accept):
    """Calculate the (asymptotic) expectation of the step size.

    The posterior should be a multivariate Gaussian, and the approximation is
    appropriate in high dimensions. For details, see [1], equations (3.1, 3.2).

    Args:
      posterior_scales: Tensor with the square roots of the eigenvalues of the
        covariance matrix.
      target_accept: Float of the acceptance rate.

    Returns:
     step_size: Float of approximate step size to achieve the target acceptance
       rate.

    #### References

    [1]: Langmore, Ian, Michael Dikovsky, Scott Geraedts, Peter Norgaard, and
         Rob Von Behren. 2019. “A Condition Number for Hamiltonian Monte Carlo."
         http://arxiv.org/abs/1905.09813.
    """
    inv_nu = tf.reduce_sum((1. / posterior_scales) ** 4, axis=-1)  ** -0.25
    step_size = (inv_nu *
                 (2**1.75) *
                 tf.sqrt(tfd.Normal(0., 1.).quantile(1 - target_accept / 2.)))
    return step_size

  def _run_hmc(self, scale_diag, target_accept, precondition):
    dims = ps.shape(scale_diag)[0]
    loc = tf.range(0., dims)
    mvn = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

    if precondition:
      momentum_distribution = tfd.MultivariateNormalDiag(
          loc=tf.zeros(dims),
          scale_diag=1. / scale_diag)
    else:
      momentum_distribution = None

    ndraws = 1000
    hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=mvn.log_prob,
            momentum_distribution=momentum_distribution,
            step_size=0.6,
            num_leapfrog_steps=10),
        num_adaptation_steps=ndraws - 1,
        target_accept_prob=target_accept)

    @tf.function
    def do_sample():
      _, step = tfp.mcmc.sample_chain(
          ndraws,
          tf.zeros(dims),
          kernel=hmc_kernel,
          trace_fn=lambda _, pkr: pkr.inner_results.accepted_results.step_size)
      return step[-1]

    return do_sample()

  def test_adapt_step_size(self):
    """Test that step size adaptation finds the theoretical optimal step size.

    See _caclulate_expected_step_size for formula details, but roughly, for a
    high dimensional Gaussian posterior, we can calculate the approximate step
    size to achieve a given target accept rate. For such a posterior,
    `PreconditionedHMC` mimics the dynamics of sampling from an isotropic
    standard normal distribution, and so should adapt to the step size where
    the scales are all ones.

    In the examples below, `preconditioned_expected_step` is around 0.6, and
    `unpreconditioned_expected_step` is around 0.00002, so there is
    significantly different behavior when conditioning.
    """
    dims = 100
    target_accept = 0.75
    scale_diag = tf.linspace(1e-5, 1., dims)
    preconditioned_step_size = self._run_hmc(scale_diag, target_accept,
                                             precondition=True)
    unpreconditioned_step_size = self._run_hmc(scale_diag, target_accept,
                                               precondition=False)
    preconditioned_expected_step = self._calculate_expected_step_size(
        tf.ones(dims), target_accept)
    unpreconditioned_expected_step = self._calculate_expected_step_size(
        scale_diag, target_accept)

    self.assertAllClose(
        preconditioned_step_size, preconditioned_expected_step, atol=0.1)
    self.assertAllClose(
        unpreconditioned_step_size, unpreconditioned_expected_step, atol=0.001)


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters(
    dict(testcase_name='_default', use_default=True),
    dict(testcase_name='_explicit', use_default=False))
class PreconditionedHMCTest(test_util.TestCase):

  def test_diag(self, use_default):
    """Test that a diagonal multivariate normal can be effectively sampled from.

    Note that the effective sample size is expected to be exactly 100: this is
    because the step size is tuned well enough that a single HMC step takes
    a point to nearly the antipodal point, which causes a negative lag 1
    autocorrelation, and the effective sample size calculation cuts off when
    the autocorrelation drops below zero.

    Args:
      use_default: bool, whether to use a custom momentum distribution, or
        the default.
    """
    mvn = tfd.MultivariateNormalDiag(
        loc=[1., 2., 3.], scale_diag=[0.1, 1., 10.])

    if use_default:
      momentum_distribution = None
      step_size = 0.1
    else:
      momentum_distribution = tfd_e.MultivariateNormalInverseScaleLinearOperator(
          tf.zeros(3), inverse_scale=mvn.scale)
      step_size = 0.3
    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=step_size,
        num_leapfrog_steps=10)
    draws = tfp.mcmc.sample_chain(
        110,
        tf.zeros(3),
        kernel=hmc_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(draws[-100:],
                                         filter_threshold=0,
                                         filter_beyond_positive_pairs=False)

    if not use_default:
      self.assertAllClose(ess, tf.fill([3], 100.))
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 50.)

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
    ess = tfp.mcmc.effective_sample_size(draws[-100:],
                                         filter_threshold=0,
                                         filter_beyond_positive_pairs=False)

    if not use_default:
      self.assertAllClose(ess, tf.fill([3], 100.))
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 50.)

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
    ess = tfp.mcmc.effective_sample_size(draws[-100:],
                                         filter_threshold=0,
                                         filter_beyond_positive_pairs=False)

    if not use_default:
      self.assertAllClose(ess, tf.fill([3], 100.))
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 50.)

  def test_multi_state_part(self, use_default):
    mvn = tfd.JointDistributionSequential([
        tfd.Normal(1., 0.1),
        tfd.Normal(2., 1.),
        tfd.Independent(tfd.Normal(3 * tf.ones([2, 3, 4]), 10.), 3)
    ])

    if use_default:
      momentum_distribution = None
      step_size = 0.1
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
      step_size = 0.3
    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=step_size,
        num_leapfrog_steps=10)

    draws = tfp.mcmc.sample_chain(
        100, [0., 0., tf.zeros((2, 3, 4))],
        kernel=hmc_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(draws,
                                         filter_threshold=0,
                                         filter_beyond_positive_pairs=False)
    if not use_default:
      self.assertAllClose(
          self.evaluate(ess),
          [tf.constant(100.),
           tf.constant(100.), 100. * tf.ones((2, 3, 4))])
    else:
      self.assertLess(
          self.evaluate(
              tf.reduce_min(tf.nest.map_structure(tf.reduce_min, ess))),
          50.)

  def test_batched_state(self, use_default):
    mvn = tfd.MultivariateNormalDiag(
        loc=[1., 2., 3.], scale_diag=[0.1, 1., 10.])
    batch_shape = [2, 4]
    if use_default:
      momentum_distribution = None
      step_size = 0.1
    else:
      momentum_distribution = tfd_e.MultivariateNormalInverseScaleLinearOperator(
          tf.zeros((2, 4, 3)), inverse_scale=mvn.scale)
      step_size = 0.3

    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=step_size,
        num_leapfrog_steps=10)

    draws = tfp.mcmc.sample_chain(
        110,
        tf.zeros(batch_shape + [3]),
        kernel=hmc_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(draws[10:], cross_chain_dims=[1, 2],
                                         filter_threshold=0,
                                         filter_beyond_positive_pairs=False)
    if not use_default:
      self.assertAllClose(self.evaluate(ess), 100 * 2. * 4. * tf.ones(3))
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 50.)

  def test_batches(self, use_default):
    mvn = tfd.JointDistributionSequential(
        [tfd.Normal(1., 0.1),
         tfd.Normal(2., 1.),
         tfd.Normal(3., 10.)])
    n_chains = 10
    if use_default:
      momentum_distribution = None
      step_size = 0.1
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
      step_size = 0.3

    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=step_size,
        num_leapfrog_steps=10)

    draws = tfp.mcmc.sample_chain(
        100, [tf.zeros([n_chains]) for _ in range(3)],
        kernel=hmc_kernel,
        seed=test_util.test_seed(),
        trace_fn=None)
    ess = tfp.mcmc.effective_sample_size(
        draws, cross_chain_dims=[1 for _ in draws],
        filter_threshold=0, filter_beyond_positive_pairs=False)
    if not use_default:
      self.assertAllClose(self.evaluate(ess), 100 * n_chains * tf.ones(3))
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 50.)


if __name__ == '__main__':
  tf.test.main()
