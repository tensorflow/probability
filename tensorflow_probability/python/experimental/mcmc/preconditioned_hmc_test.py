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
"""Tests for preconditioned_hmc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions
tfde = tfp.experimental.distributions


# Allowed type of preconditioning schemes to use.
# See code for details.
PRECONDITION_SCHEMES = {
    'direct', 'precision_factor', 'sqrtm', 'scale',
    # `None` ==> No preconditioner. This is different than a "bad"
    # preconditioner. We will be able to check asymptotics with "None".
    'no_preconditioner',
}


RunHMCResults = collections.namedtuple('RunHMCResults', [
    'draws',
    'step_size',
    'final_step_size',
    'asymptotic_step_size',
    'accept_prob',
    'mean_accept_prob',
    'min_ess',
    'sample_mean',
    'sample_cov',
    'sample_var',
    'mean_atol',
    'cov_atol',
    'var_rtol',
])


def _make_composite_tensor(dist):
  """Wrapper to make distributions of linear operators composite."""
  if dist is None:
    return dist
  composite_dist = tfp.experimental.auto_composite_tensor(dist.__class__,
                                                          omit_kwargs='name')
  p = dist.parameters

  for k in p:
    if isinstance(p[k], tfp.distributions.Distribution):
      p[k] = _make_composite_tensor(p[k])
    elif isinstance(p[k], tf.linalg.LinearOperator):
      composite_linop = tfp.experimental.auto_composite_tensor(p[k].__class__)
      p[k] = composite_linop(**p[k].parameters)
  ac_dist = composite_dist(**p)
  return ac_dist


@test_util.test_graph_and_eager_modes
class PreconditionedHMCCorrectnessTest(test_util.TestCase):
  """More careful tests that sampling/preconditioning is actually working."""

  def _calculate_asymptotic_step_size(self, scales, prob_accept):
    """Calculate the (asymptotic) expected step size for given scales/P[accept].

    The distribution should be a multivariate Gaussian, and the approximation is
    appropriate in high dimensions when the spectrum is polynomially decreasing.
    For details, see [1], equations (3.1, 3.2).

    Args:
      scales: Tensor with the square roots of the eigenvalues of the
        covariance matrix.
      prob_accept: Average acceptance probability.

    Returns:
     step_size: Float of approximate step size to achieve the target acceptance
       rate.

    #### References

    [1]: Langmore, Ian, Michael Dikovsky, Scott Geraedts, Peter Norgaard, and
         Rob Von Behren. 2019. “A Condition Number for Hamiltonian Monte Carlo."
         http://arxiv.org/abs/1905.09813.
    """
    inv_nu = tf.reduce_sum((1. / scales) ** 4, axis=-1)  ** -0.25
    step_size = (inv_nu *
                 (2**1.75) *
                 tf.sqrt(tfd.Normal(0., 1.).quantile(1 - prob_accept / 2.)))
    return step_size

  def _run_hmc_with_step_size(
      self,
      target_mvn,
      precondition_scheme,
      target_accept=0.75,
      num_results=2000,
      num_adaptation_steps=20,
  ):
    """Run HMC with step_size adaptation, and return RunHMCResults."""
    assert precondition_scheme in PRECONDITION_SCHEMES

    dims = target_mvn.event_shape[0]
    target_cov = target_mvn.covariance()

    cov_linop = tf.linalg.LinearOperatorFullMatrix(
        target_cov,
        is_self_adjoint=True,
        is_positive_definite=True)

    if precondition_scheme == 'no_preconditioner':
      momentum_distribution = None
      # Internal to the sampler, these scales are being used (implicitly).
      internal_scales = tf.sqrt(tf.linalg.eigvalsh(target_cov))
    elif precondition_scheme == 'direct':
      momentum_distribution = tfd.MultivariateNormalLinearOperator(
          # The covariance of momentum is inv(covariance of position), and we
          # parameterize distributions by a square root of the covariance.
          scale=cov_linop.inverse().cholesky(),
      )
      # Internal to the sampler, these scales are being used (implicitly).
      internal_scales = tf.ones(dims)
    elif precondition_scheme == 'precision_factor':
      momentum_distribution = tfde.MultivariateNormalPrecisionFactorLinearOperator(
          # The precision of momentum is the covariance of position.
          # The "factor" is the cholesky factor.
          precision_factor=cov_linop.cholesky(),
      )
      # Internal to the sampler, these scales are being used (implicitly).
      internal_scales = tf.ones(dims)
    elif precondition_scheme == 'sqrtm':
      momentum_distribution = tfde.MultivariateNormalPrecisionFactorLinearOperator(
          # The symmetric square root is a perfectly valid "factor".
          precision_factor=tf.linalg.LinearOperatorFullMatrix(
              tf.linalg.sqrtm(target_cov)),
      )
      # Internal to the sampler, these scales are being used (implicitly).
      internal_scales = tf.ones(dims)
    elif precondition_scheme == 'scale':
      momentum_distribution = tfde.MultivariateNormalPrecisionFactorLinearOperator(
          # Nothing wrong with using "scale", since the scale should be the
          # same as cov_linop.cholesky().
          precision_factor=target_mvn.scale,
      )
      # Internal to the sampler, these scales are being used (implicitly).
      internal_scales = tf.ones(dims)
    else:
      raise RuntimeError(
          'Unhandled precondition_scheme: {}'.format(precondition_scheme))
    momentum_distribution = _make_composite_tensor(momentum_distribution)

    # Asyptotic step size, assuming P[accept] = target_accept.
    expected_step = self._calculate_asymptotic_step_size(
        scales=internal_scales,
        prob_accept=target_accept,
    )

    # Initialize step size to something close to the expected required step
    # size. This helps reduce the need for a long burn-in. Don't use the
    # expected step size exactly, since that would be cheating.
    initial_step_size = expected_step / 2.345

    # Set num_leapfrog_steps so that we get decent ESS.
    max_internal_scale = tf.reduce_max(internal_scales)
    num_leapfrog_steps = tf.minimum(
        tf.cast(
            tf.math.ceil(1.5 * max_internal_scale / expected_step),
            dtype=tf.int32), 30)

    hmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=target_mvn.log_prob,
            momentum_distribution=momentum_distribution,
            step_size=initial_step_size,
            num_leapfrog_steps=num_leapfrog_steps),
        num_adaptation_steps=num_adaptation_steps,
        target_accept_prob=target_accept)

    def trace_fn(_, pkr):
      results = pkr.inner_results
      return {
          'accept_prob':
              tf.exp(tf.minimum(0., results.log_accept_ratio)),
          'step_size':
              results.accepted_results.step_size,
      }

    @tf.function
    def do_run_run_run():
      """Do a run, return RunHMCResults."""
      states, trace = tfp.mcmc.sample_chain(
          num_results,
          current_state=tf.identity(target_mvn.sample(seed=0)),
          kernel=hmc_kernel,
          num_burnin_steps=num_adaptation_steps,
          seed=test_util.test_seed(),
          trace_fn=trace_fn)

      # If we had some number of chain dimensions, we would change sample_axis.
      sample_axis = 0

      sample_cov = tfp.stats.covariance(states, sample_axis=sample_axis)
      max_variance = tf.reduce_max(tf.linalg.diag_part(sample_cov))
      max_stddev = tf.sqrt(max_variance)
      min_ess = tf.reduce_min(tfp.mcmc.effective_sample_size(states))
      mean_accept_prob = tf.reduce_mean(trace['accept_prob'])

      # Asymptotic step size given that P[accept] = mean_accept_prob.
      asymptotic_step_size = self._calculate_asymptotic_step_size(
          scales=internal_scales,
          prob_accept=mean_accept_prob,
      )

      return RunHMCResults(
          draws=states,
          step_size=trace['step_size'],
          final_step_size=trace['step_size'][-1],
          asymptotic_step_size=asymptotic_step_size,
          accept_prob=trace['accept_prob'],
          mean_accept_prob=mean_accept_prob,
          min_ess=tf.reduce_min(tfp.mcmc.effective_sample_size(states)),
          sample_mean=tf.reduce_mean(states, axis=sample_axis),
          sample_cov=sample_cov,
          sample_var=tf.linalg.diag_part(sample_cov),

          # Standard error in variance estimation is related to standard
          # deviation of variance estimates. For a Normal, this is just Sqrt(2)
          # times variance divided by sqrt sample size (or so my old notes say).
          # So a relative tolerance is useful.
          # Add in a factor of 5 as a buffer.
          var_rtol=5 * tf.sqrt(2.) / tf.sqrt(min_ess),

          # For covariance matrix estimates, there can be terms that have
          # expectation = 0 (e.g. off diagonal entries). So the above doesn't
          # hold. So use an atol.
          cov_atol=5 * max_variance / tf.sqrt(min_ess),

          # Standard error in mean estimation is stddev divided by sqrt
          # sample size. This is an absolute tolerance.
          # Add in a factor of 5 as a buffer.
          mean_atol=5 * max_stddev / tf.sqrt(min_ess),
      )

    # Evaluate now, to ensure that states/accept_prob/etc... all match up with
    # the same graph evaluation. This is a gotcha about TFP MCMC in graph mode.
    return self.evaluate(do_run_run_run())

  def _check_correctness_of_moments_and_preconditioning(
      self,
      target_mvn,
      num_results,
      precondition_scheme,
      check_step_size_asymptotics=True,
      asymptotic_step_size_rtol=0.2,
  ):
    """Test that step size adaptation finds the theoretical optimal step size.

    See _caclulate_expected_step_size for formula details, but roughly, for a
    high dimensional Gaussian posterior, we can calculate the approximate step
    size to achieve a given target accept rate. For such a posterior,
    `PreconditionedHMC` mimics the dynamics of sampling from an isotropic
    standard normal distribution, and so should adapt to the step size where
    the scales are all ones.

    In the example below, `expected_step` is around 0.00002, so there is
    significantly different behavior when conditioning.

    Args:
      target_mvn: Multivariate normal instance to sample from.
      num_results: Number of samples to collect (post burn-in).
      precondition_scheme: String telling how to do preconditioning.
        Should be in PRECONDITION_SCHEMES.
      check_step_size_asymptotics: Boolean telling whether to check that the
        step size and P[accept] match up with expected values. This checks
        that the "internal/implicit" sampling distribution is as expected. E.g.
        when preconditioning, we expect the internal distribution to be a
        standard Normal. When not preconditioning we expect it to be the target.
      asymptotic_step_size_rtol: rtol for the asymptotic step size test.
        The "nastier" spectra (with a small number of tiny eigenvalues) often
        require larger tolerance.  About 10% rtol is what we can expect.
        20% is the default for safety.  When a "bad preconditioner" is used,
        these two are off by 100% or more (but no guarantee, since luck may
        prevail).

    Returns:
      RunHMCResults
    """
    results = self._run_hmc_with_step_size(
        target_mvn, precondition_scheme=precondition_scheme)

    if check_step_size_asymptotics:
      self.assertAllClose(
          results.final_step_size,
          results.asymptotic_step_size,
          rtol=asymptotic_step_size_rtol)

    self.assertAllClose(
        results.sample_mean, target_mvn.mean(), atol=results.mean_atol)
    self.assertAllClose(
        results.sample_var, target_mvn.variance(), rtol=results.var_rtol)
    self.assertAllClose(
        results.sample_cov, target_mvn.covariance(), atol=results.cov_atol)

    return results

  @parameterized.named_parameters(
      dict(testcase_name='_' + str(scheme), precondition_scheme=scheme)
      for scheme in PRECONDITION_SCHEMES)
  def test_correctness_with_2d_mvn_tril(self, precondition_scheme):
    # Low dimensional test to help people who want to step through and debug.
    target_mvn = tfd.MultivariateNormalTriL(
        loc=tf.constant([0., 0.]),
        scale_tril=[[1., 0.], [0.5, 2.]],
    )
    self._check_correctness_of_moments_and_preconditioning(
        target_mvn,
        # Lots of results, to test tight tolerance.
        # We're using a small dims here, so this isn't a big deal.
        num_results=5000,
        precondition_scheme=precondition_scheme,
        # We're in such low dimensions that we don't expect asymptotics to work.
        check_step_size_asymptotics=False)

  @parameterized.named_parameters(
      dict(testcase_name='_' + str(scheme), precondition_scheme=scheme)
      for scheme in PRECONDITION_SCHEMES)
  def test_correctness_with_200d_mvn_tril(self, precondition_scheme):
    # This is an almost complete check of the Gaussian case.
    dims = 200
    scale_wishart = tfd.WishartLinearOperator(
        # Important that df is just slightly bigger than dims. This makes the
        # scale_wishart ill condtioned. The result is that tests fail if we do
        # not handle transposes correctly.
        df=1.1 * dims,
        scale=tf.linalg.LinearOperatorIdentity(dims),
        input_output_cholesky=True,
        name='wishart_for_samples',
    )

    # evaluate right here to avoid working with a random target_mvn in graph
    # mode....that would cause issues, since we read off expected statistics
    # from looking at the mvn properties, so it would be bad if these properties
    # changed with every graph eval.
    scale_tril = self.evaluate(scale_wishart.sample(seed=test_util.test_seed()))

    target_mvn = tfd.MultivariateNormalTriL(
        # Non-trivial "loc" ensures we do not rely on being centered at 0.
        loc=tf.range(0., dims),
        scale_tril=scale_tril,
    )

    self._check_correctness_of_moments_and_preconditioning(
        target_mvn,
        # Lots of results, to test tight tolerance.
        num_results=3000,
        precondition_scheme=precondition_scheme,
        asymptotic_step_size_rtol=(
            0.5 if precondition_scheme == 'no_preconditioner' else 0.25),
    )


@test_util.test_all_tf_execution_regimes
@parameterized.named_parameters(
    dict(testcase_name='_default', use_default=True),
    dict(testcase_name='_explicit', use_default=False))
class PreconditionedHMCTest(test_util.TestCase):

  def test_f64(self, use_default):
    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = tfp.experimental.as_composite(
          tfd.Normal(0., tf.constant(.5, dtype=tf.float64)))
    kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        lambda x: -x**2, step_size=.5, num_leapfrog_steps=2,
        momentum_distribution=momentum_distribution)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(kernel, num_adaptation_steps=3)
    self.evaluate(tfp.mcmc.sample_chain(
        1, kernel=kernel, current_state=tf.ones([], tf.float64),
        num_burnin_steps=5, trace_fn=None))

  # TODO(b/175787154): Enable this test
  def DISABLED_test_f64_multichain(self, use_default):
    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = tfp.experimental.as_composite(
          tfd.Normal(0., tf.constant(.5, dtype=tf.float64)))
    kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        lambda x: -x**2, step_size=.5, num_leapfrog_steps=2,
        momentum_distribution=momentum_distribution)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(kernel, num_adaptation_steps=3)
    nchains = 7
    self.evaluate(tfp.mcmc.sample_chain(
        1, kernel=kernel, current_state=tf.ones([nchains], tf.float64),
        num_burnin_steps=5, trace_fn=None))

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
      momentum_distribution = tfde.MultivariateNormalPrecisionFactorLinearOperator(
          precision_factor=mvn.scale,
      )
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
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 100.)

  def test_tril(self, use_default):
    if tf.executing_eagerly():
      self.skipTest('b/169882656 Too many warnings are issued in eager logs')
    cov = 0.9 * tf.ones([3, 3]) + 0.1 * tf.eye(3)
    scale = tf.linalg.cholesky(cov)
    mv_tril = tfd.MultivariateNormalTriL(loc=[1., 2., 3.],
                                         scale_tril=scale)

    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = tfde.MultivariateNormalPrecisionFactorLinearOperator(
          # TODO(b/170015229) Don't use the covariance as inverse scale,
          # it is the wrong preconditioner.
          precision_factor=tf.linalg.LinearOperatorFullMatrix(cov),
      )
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

    # TODO(b/170015229): These and other tests like it, which assert ess is
    # greater than some number, were all passing, even though the preconditioner
    # was the wrong one. Why is that? A guess is that since there are *many*
    # ways to have larger ess, these tests don't really test correctness.
    # Perhaps remove all tests like these.
    if not use_default:
      self.assertAllClose(ess, tf.fill([3], 100.))
    else:
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 100.)

  def test_transform(self, use_default):
    mvn = tfd.MultivariateNormalDiag(loc=[1., 2., 3.], scale_diag=[1., 1., 1.])
    diag_variance = tf.constant([0.1, 1., 10.])

    if use_default:
      momentum_distribution = None
    else:
      momentum_distribution = tfde.MultivariateNormalPrecisionFactorLinearOperator(
          precision_factor=tf.linalg.LinearOperatorDiag(
              tf.math.sqrt(diag_variance)))
    hmc_kernel = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=mvn.log_prob,
        momentum_distribution=momentum_distribution,
        step_size=0.3,
        num_leapfrog_steps=10)

    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
        hmc_kernel, bijector=tfb.Scale(tf.math.rsqrt(diag_variance)))

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
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 100.)

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
              tfde.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag([0.1]))),
          reshape_to_scalar(
              tfde.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag([1.]))),
          reshape_to_234(
              tfde.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag(
                      tf.fill([24], 10.))))
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
      momentum_distribution = tfde.MultivariateNormalPrecisionFactorLinearOperator(
          tf.zeros((2, 4, 3)), precision_factor=mvn.scale)
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
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 100.)

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
              tfde.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag(
                      tf.fill([n_chains, 1], 0.1)))),
          reshape_to_scalar(
              tfde.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag(
                      tf.fill([n_chains, 1], 1.)))),
          reshape_to_scalar(
              tfde.MultivariateNormalPrecisionFactorLinearOperator(
                  precision_factor=tf.linalg.LinearOperatorDiag(
                      tf.fill([n_chains, 1], 10.)))),
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
      self.assertLess(self.evaluate(tf.reduce_min(ess)), 100.)


if __name__ == '__main__':
  tf.test.main()
