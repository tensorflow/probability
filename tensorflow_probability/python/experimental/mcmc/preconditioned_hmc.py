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
"""Hamiltonian Monte Carlo, a gradient-based MCMC algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    'PreconditionedHamiltonianMonteCarlo',
]


class UncalibratedPreconditionedHamiltonianMonteCarloKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'UncalibratedPreconditionedHamiltonianMonteCarloKernelResults',
        hmc.UncalibratedHamiltonianMonteCarloKernelResults._fields +
        ('momentum_distribution',))):
  """Internal state and diagnostics for Uncalibrated HMC."""
  __slots__ = ()


class PreconditionedHamiltonianMonteCarlo(hmc.HamiltonianMonteCarlo):
  """Hamiltonian Monte Carlo, with given momentum distribution.

  See `tfp.mcmc.HamiltonianMonteCarlo` for details on HMC.

  HMC produces samples much more efficiently if properly preconditioned. This
  can be done by choosing a momentum distribution with covariance equal to
  the inverse of the state's covariance.

  #### Examples:

  ##### Simple chain with warm-up.

  In this example we sample from a non-isotropic distribution, and show that
  we may sample efficiently with HMC by pre-conditioning.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfed = tfp.experimental.distributions

  # Suppose we have a target log prob fn, as well as an estimate of its
  # covariance.
  log_prob_fn = ...
  cov_estimate = ...

  # We want the mass matrix to be the *inverse* of the covariance estimate,
  # so we can use the symmetric square root:
  momentum_distribution = (
      tfed.MultivariateNormalPrecisionFactorLinearOperator(
          precision_factor=tf.linalg.LinearOperatorLowerTriangular(
              tf.linalg.cholesky(cov_estimate),
          ),
          precision=tf.linalg.LinearOperatorFullMatrix(cov_estimate),
  )

  # Run standard HMC below
  num_burnin_steps = 100
  num_results = 1000
  adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
      tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        momentum_distribution=momentum_distribution,
        step_size=0.3,
        num_leapfrog_steps=10),
      num_adaptation_steps=int(num_burnin_steps * 0.8))

  @tf.function
  def run_chain_and_compute_ess():
    draws = tfp.mcmc.sample_chain(
        num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=tf.zeros(3),  # 3 chains.
        kernel=adaptive_hmc,
        trace_fn=None)
    return tfp.mcmc.effective_sample_size(draws, cross_chain_dims=1)

  run_chain_and_compute_ess()  # Something close to 3 x 1000.
  ```

  ##### Estimate parameters of a more complicated distribution.

  This demonstrates using multiple state parts, and reshaping a
  `tfde.MultivariateNormalPrecisionFactorLinearOperator`
  to use with a scalar or a non-square shape (in this case, `[2, 3, 4]`).

  ```python
  mvn = tfd.JointDistributionSequential([
      tfd.Normal(0., 0.1),
      tfd.Normal(0., 10.),
      tfd.Independent(tfd.Normal(tf.fill([2, 3, 4], 3.), 10.),
                      reinterpreted_batch_ndims=3)])

  reshape_to_scalar = tfp.bijectors.Reshape(event_shape_out=[])
  reshape_to_234 = tfp.bijectors.Reshape(event_shape_out=[2, 3, 4])
  momentum_distribution = tfd.JointDistributionSequential([
      tfd.Normal(0., 10.),
      tfd.Normal(0., 0.1),
      reshape_to_234(
          tfde.MultivariateNormalPrecisionFactorLinearOperator(
              0., tf.linalg.LinearOperatorDiag(tf.fill([24], 10.))))
  ])
  num_burnin_steps = 100
  num_results = 1000
  adaptive_hmc = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
    target_log_prob_fn=mvn.log_prob,
    momentum_distribution=momentum_distribution,
    step_size=0.3,
    num_leapfrog_steps=10)

  @tf.function
  def run_chain_and_compute_ess():
    draws = tfp.mcmc.sample_chain(
        num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=mvn.sample(),
        kernel=adaptive_hmc,
        trace_fn=None)
    return tfp.mcmc.effective_sample_size(draws)

  run_chain_and_compute_ess()  # [1000, 1000, 1000 * tf.ones([2, 3, 4])]
  ```
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               momentum_distribution=None,
               state_gradients_are_stopped=False,
               step_size_update_fn=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
        for. Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
      momentum_distribution: A `tfp.distributions.Distribution` instance to draw
        momentum from. Defaults to isotropic normal distributions.
      state_gradients_are_stopped: Python `bool` indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      step_size_update_fn: Python `callable` taking current `step_size`
        (typically a `tf.Variable`) and `kernel_results` (typically
        `collections.namedtuple`) and returns updated step_size (`Tensor`s).
        Default value: `None` (i.e., do not update `step_size` automatically).
      store_parameters_in_results: If `True`, then `step_size`,
        `momentum_distribution`, and `num_leapfrog_steps` are written to and
        read from eponymous fields in the kernel results objects returned from
        `one_step` and `bootstrap_results`. This allows wrapper kernels to
        adjust those parameters on the fly. In case this is `True`, the
        `momentum_distribution` must be a `CompositeTensor`. See
        `tfp.experimental.as_composite` and `tfp.experimental.auto_composite`.
        This is incompatible with `step_size_update_fn`, which must be set to
        `None`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    if step_size_update_fn and store_parameters_in_results:
      raise ValueError('It is invalid to simultaneously specify '
                       '`step_size_update_fn` and set '
                       '`store_parameters_in_results` to `True`.')
    self._impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UncalibratedPreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            momentum_distribution=momentum_distribution,
            name=name or 'hmc_kernel',
            store_parameters_in_results=store_parameters_in_results))
    self._parameters = self._impl.inner_kernel.parameters.copy()
    self._parameters.pop('seed', None)  # TODO(b/159636942): Remove this line.
    self._parameters['step_size_update_fn'] = step_size_update_fn


class UncalibratedPreconditionedHamiltonianMonteCarlo(
    hmc.UncalibratedHamiltonianMonteCarlo):
  """Runs one step of Uncalibrated Hamiltonian Monte Carlo.

  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use
  `PreconditionedHamiltonianMonteCarlo(...)` or
  `MetropolisHastings(UncalibratedPreconditionedHamiltonianMonteCarlo(...))`.

  For more details on `UncalibratedPreconditionedHamiltonianMonteCarlo`, see
  `PreconditionedHamiltonianMonteCarlo`.
  """

  @mcmc_util.set_doc(hmc.UncalibratedHamiltonianMonteCarlo.__init__)
  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               momentum_distribution=None,
               state_gradients_are_stopped=False,
               store_parameters_in_results=False,
               name=None):
    super(UncalibratedPreconditionedHamiltonianMonteCarlo, self).__init__(
        target_log_prob_fn,
        step_size,
        num_leapfrog_steps,
        state_gradients_are_stopped=state_gradients_are_stopped,
        store_parameters_in_results=store_parameters_in_results,
        name=name)
    self._parameters['momentum_distribution'] = momentum_distribution
    self._parameters.pop('seed', None)  # TODO(b/159636942): Remove this line.

  @property
  def momentum_distribution(self):
    return self._parameters['momentum_distribution']

  @mcmc_util.set_doc(hmc.HamiltonianMonteCarlo.one_step.__doc__)
  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(mcmc_util.make_name(self.name, 'phmc', 'one_step')):
      if self._store_parameters_in_results:
        step_size = previous_kernel_results.step_size
        num_leapfrog_steps = previous_kernel_results.num_leapfrog_steps
        momentum_distribution = previous_kernel_results.momentum_distribution
      else:
        step_size = self.step_size
        num_leapfrog_steps = self.num_leapfrog_steps
        momentum_distribution = self.momentum_distribution

      [
          current_state_parts,
          step_sizes,
          momentum_distribution,
          current_target_log_prob,
          current_target_log_prob_grad_parts,
      ] = _prepare_args(
          self.target_log_prob_fn,
          current_state,
          step_size,
          momentum_distribution,
          previous_kernel_results.target_log_prob,
          previous_kernel_results.grads_target_log_prob,
          maybe_expand=True,
          state_gradients_are_stopped=self.state_gradients_are_stopped)

      seed = samplers.sanitize_seed(seed)
      current_momentum_parts = momentum_distribution.sample(seed=seed)
      momentum_log_prob = getattr(momentum_distribution,
                                  '_log_prob_unnormalized',
                                  momentum_distribution.log_prob)
      kinetic_energy_fn = lambda *args: -momentum_log_prob(*args)

      # Let the integrator handle the case where no momentum distribution
      # is provided
      if self.momentum_distribution is None:
        leapfrog_kinetic_energy_fn = None
      else:
        leapfrog_kinetic_energy_fn = kinetic_energy_fn

      integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
          self.target_log_prob_fn, step_sizes, num_leapfrog_steps)

      [
          next_momentum_parts,
          next_state_parts,
          next_target_log_prob,
          next_target_log_prob_grad_parts,
      ] = integrator(
          current_momentum_parts,
          current_state_parts,
          target=current_target_log_prob,
          target_grad_parts=current_target_log_prob_grad_parts,
          kinetic_energy_fn=leapfrog_kinetic_energy_fn)
      if self.state_gradients_are_stopped:
        next_state_parts = [tf.stop_gradient(x) for x in next_state_parts]

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]
      new_kernel_results = previous_kernel_results._replace(
          log_acceptance_correction=_compute_log_acceptance_correction(
              kinetic_energy_fn, current_momentum_parts,
              next_momentum_parts),
          target_log_prob=next_target_log_prob,
          grads_target_log_prob=next_target_log_prob_grad_parts,
          initial_momentum=current_momentum_parts,
          final_momentum=next_momentum_parts,
          seed=seed,
      )

      return maybe_flatten(next_state_parts), new_kernel_results

  @mcmc_util.set_doc(hmc.HamiltonianMonteCarlo.bootstrap_results.__doc__)
  def bootstrap_results(self, init_state):
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'phmc', 'bootstrap_results')):
      result = super(UncalibratedPreconditionedHamiltonianMonteCarlo,
                     self).bootstrap_results(init_state)
      result = UncalibratedPreconditionedHamiltonianMonteCarloKernelResults(
          **result._asdict(),  # pylint: disable=protected-access
          momentum_distribution=[])

      if self._store_parameters_in_results:
        result = result._replace(
            momentum_distribution=[] if self.momentum_distribution is None else
            self.momentum_distribution)
    return result


def _compute_log_acceptance_correction(kinetic_energy_fn,
                                       current_momentums,
                                       proposed_momentums,
                                       name=None):
  """Helper to `kernel` which computes the log acceptance-correction.

  A sufficient but not necessary condition for the existence of a stationary
  distribution, `p(x)`, is "detailed balance", i.e.:

  ```none
  p(x'|x) p(x) = p(x|x') p(x')
  ```

  In the Metropolis-Hastings algorithm, a state is proposed according to
  `g(x'|x)` and accepted according to `a(x'|x)`, hence
  `p(x'|x) = g(x'|x) a(x'|x)`.

  Inserting this into the detailed balance equation implies:

  ```none
      g(x'|x) a(x'|x) p(x) = g(x|x') a(x|x') p(x')
  ==> a(x'|x) / a(x|x') = p(x') / p(x) [g(x|x') / g(x'|x)]    (*)
  ```

  One definition of `a(x'|x)` which satisfies (*) is:

  ```none
  a(x'|x) = min(1, p(x') / p(x) [g(x|x') / g(x'|x)])
  ```

  (To see that this satisfies (*), notice that under this definition only at
  most one `a(x'|x)` and `a(x|x') can be other than one.)

  We call the bracketed term the "acceptance correction".

  In the case of UncalibratedHMC, the log acceptance-correction is not the log
  proposal-ratio. UncalibratedHMC augments the state-space with momentum, z.
  Given a probability density of `m(z)` for momentums, the chain eventually
  converges to:

  ```none
  p([x, z]) propto= target_prob(x) m(z)
  ```

  Relating this back to Metropolis-Hastings parlance, for HMC we have:

  ```none
  p([x, z]) propto= target_prob(x) m(z)
  g([x, z] | [x', z']) = g([x', z'] | [x, z])
  ```

  In other words, the MH bracketed term is `1`. However, because we desire to
  use a general MH framework, we can place the momentum probability ratio inside
  the metropolis-correction factor thus getting an acceptance probability:

  ```none
                       target_prob(x')
  accept_prob(x'|x) = -----------------  [m(z') / m(z)]
                       target_prob(x)
  ```
  (Note: we actually need to handle the kinetic energy change at each leapfrog
  step, but this is the idea.)

  For consistency, we compute this correction in log space, using the kinetic
  energy function, `K(z)`, which is the negative log probability of the momentum
  distribution. So the log acceptance probability is

  ```none
  log(correction) = log(m(z')) - log(m(z))
                  = K(z) - K(z')
  ```

  Note that this is equality, since the normalization constants on `m` cancel
  out.


  Args:
    kinetic_energy_fn: Python callable that can evaluate the kinetic energy
      of the given momentum. This is typically the negative log probability of
      the distribution over the momentum.
    current_momentums: (List of) `Tensor`s representing the value(s) of the
      current momentum(s) of the state (parts).
    proposed_momentums: (List of) `Tensor`s representing the value(s) of the
      proposed momentum(s) of the state (parts).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'compute_log_acceptance_correction').

  Returns:
    log_acceptance_correction: `Tensor` representing the `log`
      acceptance-correction.  (See docstring for mathematical definition.)
  """
  with tf.name_scope(name or 'compute_log_acceptance_correction'):
    current_kinetic = kinetic_energy_fn(current_momentums)
    proposed_kinetic = kinetic_energy_fn(proposed_momentums)
    return mcmc_util.safe_sum([current_kinetic, -proposed_kinetic])


def _prepare_args(target_log_prob_fn,
                  state,
                  step_size,
                  momentum_distribution,
                  target_log_prob=None,
                  grads_target_log_prob=None,
                  maybe_expand=False,
                  state_gradients_are_stopped=False):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
  if state_gradients_are_stopped:
    state_parts = [tf.stop_gradient(x) for x in state_parts]
  target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      target_log_prob_fn, state_parts, target_log_prob, grads_target_log_prob)
  step_sizes, _ = mcmc_util.prepare_state_parts(
      step_size, dtype=target_log_prob.dtype, name='step_size')

  # Default momentum distribution is None, but if `store_parameters_in_results`
  # is true, then `momentum_distribution` defaults to an empty list
  if momentum_distribution is None or isinstance(momentum_distribution, list):
    batch_rank = ps.rank(target_log_prob)
    def _batched_isotropic_normal_like(state_part):
      event_ndims = ps.rank(state_part) - batch_rank
      return independent.Independent(
          normal.Normal(ps.zeros_like(state_part, tf.float32), 1.),
          reinterpreted_batch_ndims=event_ndims)

    momentum_distribution = jds.JointDistributionSequential(
        [_batched_isotropic_normal_like(state_part)
         for state_part in state_parts])

  # The momentum will get "maybe listified" to zip with the state parts,
  # and this step makes sure that the momentum distribution will have the
  # same "maybe listified" underlying shape.
  if not mcmc_util.is_list_like(momentum_distribution.dtype):
    momentum_distribution = jds.JointDistributionSequential(
        [momentum_distribution])

  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  def maybe_flatten(x):
    return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
  return [
      maybe_flatten(state_parts),
      maybe_flatten(step_sizes),
      momentum_distribution,
      target_log_prob,
      grads_target_log_prob,
  ]
