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
"""Gradient-based trajectory length adaptation kernel."""

import collections
import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import simple_step_size_adaptation
from tensorflow_probability.python.mcmc import transformed_kernel
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'chees_criterion',
    'chees_rate_criterion',
    'GradientBasedTrajectoryLengthAdaptation',
    'GradientBasedTrajectoryLengthAdaptationResults',
    'snaper_criterion',
]

MAX_HALTON_SEQUENCE_BITS = 10  # Generates up to 1024 unique trajectory lengths.


class GradientBasedTrajectoryLengthAdaptationResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('GradientBasedTrajectoryLengthAdaptationResults', [
        'inner_results',
        'max_trajectory_length',
        'step',
        'adaptation_rate',
        'jitter_amount',
        'averaged_sq_grad',
        'averaged_sq_grad_adaptation_rate',
        'averaged_max_trajectory_length',
        'criterion',
        'seed',
    ])):
  """Internal state of GradientBasedTrajectoryLengthAdaptation.

  Attributes:
    inner_results: Results of the inner kernel.
    max_trajectory_length: Floating point scalar `Tensor`. Maximum HMC
      trajectory length.
    step: Int32 scalar `Tensor`. The number of steps this kernel has taken.
      Increases by 1 for every call to `one_step`.
    adaptation_rate: Floating point scalar `Tensor`. How rapidly to adapt the
      trajectory length.
    jitter_amount: Floating point scalar `Tensor`. How much to jitter the
      trajectory on the next step. The trajectory length is sampled from `[(1 -
      jitter_amount) * max_trajectory_length, max_trajectory_length]`.
    averaged_sq_grad: Floating point scalar `Tensor`. Moving average of squared
      criterion gradients.
    averaged_sq_grad_adaptation_rate: Floating point scalar `Tensor`. How
      rapidly to adapt the running average squared gradient. This is `1 -
      beta_2` from Adam.
    averaged_max_trajectory_length: Floating point scalar `Tensor`. Moving
      average of the maximum of trajectory length. This is used after the burnin
      period.
    criterion: Floating point `Tensor` with shape `[C0, ..., Cb]` with `b > 0`.
      The value of the criterion returned by the `criterion_fn` corresponding to
      each Markov chain.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details. The random seed
      used by the kernel in the previous step.
  """
  __slots__ = ()


def _map_structure_up_to_with_axes(structure,
                                   fn,
                                   *args,
                                   experimental_shard_axis_names=None):
  if experimental_shard_axis_names is None:
    return nest.map_structure_up_to(structure, fn, *args)
  return nest.map_structure_up_to(structure, fn, *args,
                                  experimental_shard_axis_names)


def _reduce_with_axes(index_op, name_op, x, axis_idx=None, axis_names=None):
  return name_op(index_op(x, axis_idx), axis_names)


_reduce_sum_with_axes = functools.partial(_reduce_with_axes, tf.reduce_sum,
                                          distribute_lib.psum)
_reduce_mean_with_axes = functools.partial(_reduce_with_axes, tf.reduce_mean,
                                           distribute_lib.pmean)


def _estimate_empirical_mean(x, accept_prob, safe, reduce_chain_axis_names):
  """Estimates the empirical mean of x."""
  batch_ndims = ps.rank(accept_prob)
  batch_axes = ps.range(batch_ndims, dtype=tf.int32)

  if safe:
    # Note that we don't do a monte carlo average of the accepted chain
    # position, but rather try to get an estimate of the underlying dynamics.
    # This is done by only looking at proposed states where the integration
    # error is low.
    # TODO(mhoffman): Needs more experimentation.
    expanded_accept_prob = bu.left_justified_expand_dims_like(accept_prob, x)

    # accept_prob is zero when x is NaN, but we still want to sanitize such
    # values.
    x_safe = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    # If all accept_prob's are zero, the x_center will have a nonsense value,
    # but we'll discard the resultant gradients later on, so it's fine.
    x_mean = _reduce_sum_with_axes(
        expanded_accept_prob * x_safe, batch_axes, reduce_chain_axis_names
    ) / (
        _reduce_sum_with_axes(
            expanded_accept_prob, batch_axes, reduce_chain_axis_names
        )
        + 1e-20
    )
  else:
    x_mean = _reduce_mean_with_axes(x, batch_axes, reduce_chain_axis_names)
  # The empirical mean here is a stand-in for the true mean, so we drop the
  # gradient that flows through this term.
  return tf.stop_gradient(x_mean)


def hmc_like_num_leapfrog_steps_getter_fn(kernel_results):
  """Getter for `num_leapfrog_steps` so it can be inspected."""
  return unnest.get_innermost(kernel_results, 'num_leapfrog_steps')


def hmc_like_num_leapfrog_steps_setter_fn(kernel_results,
                                          new_num_leapfrog_steps):
  """Setter for `num_leapfrog_steps` so it can be adapted."""
  return unnest.replace_innermost(
      kernel_results, num_leapfrog_steps=new_num_leapfrog_steps)


def _hmc_like_velocity_getter_fn(kernel_results, momentum_name):
  """Getter for a velocity so it can be inspected."""
  momentum = unnest.get_innermost(kernel_results, momentum_name)
  proposed_state = unnest.get_innermost(kernel_results, 'proposed_state')

  momentum_distribution = unnest.get_innermost(
      kernel_results, 'momentum_distribution', default=None)
  if momentum_distribution is None:
    velocity = momentum
  else:
    momentum_log_prob = getattr(momentum_distribution, '_log_prob_unnormalized',
                                momentum_distribution.log_prob)
    kinetic_energy_fn = lambda *args: -momentum_log_prob(*args)
    _, velocity = mcmc_util.maybe_call_fn_and_grads(
        kinetic_energy_fn, momentum)
  # proposed_velocity has the wrong structure when state is a scalar.
  return tf.nest.pack_sequence_as(proposed_state,
                                  tf.nest.flatten(velocity))


hmc_like_proposed_velocity_getter_fn = functools.partial(
    _hmc_like_velocity_getter_fn, momentum_name='final_momentum'
)
hmc_like_initial_velocity_getter_fn = functools.partial(
    _hmc_like_velocity_getter_fn, momentum_name='initial_momentum'
)


def hmc_like_proposed_state_getter_fn(kernel_results):
  """Getter for `proposed_state` so it can be inspected."""
  return unnest.get_innermost(kernel_results, 'proposed_state')


def hmc_like_step_size_getter_fn(kernel_results):
  # This is here due to the circular imports.
  return simple_step_size_adaptation.hmc_like_step_size_getter_fn(
      kernel_results)


def hmc_like_log_accept_prob_getter_fn(kernel_results):
  # This is here due to the circular imports.
  return simple_step_size_adaptation.hmc_like_log_accept_prob_getter_fn(
      kernel_results)


def chees_criterion(previous_state,
                    proposed_state,
                    accept_prob,
                    trajectory_length,
                    forward=True,
                    validate_args=False,
                    experimental_shard_axis_names=None,
                    experimental_reduce_chain_axis_names=None):
  """The ChEES criterion from [1].

  ChEES stands for Change in the Estimator of the Expected Square.

  ```None
  ChEES = 1/4 E[(||x' - E[x]||**2 - ||x - E[x]||**2)**2],
  ```

  where `x` is the previous chain state, `x'` is the next chain state, and
  `||.||` is the L2 norm. Both expectations are with respect to the chain's
  stationary distribution. In practice, the inner expectation is replaced by the
  empirical mean across chains, so computing this criterion requires that at
  least 2 chains are present. The outer expectation is computed by the caller
  (e.g. in the `GradientBasedTrajectoryLengthAdaptation` kernel).

  This can be thought of as the standard expected squared jump distance (ESJD)
  criterion, except that the jump distance is computed in the space of centered
  squared L2 norms.

  Unlike ChEES, regular ESJD is maximized by perfectly anticorrelated proposals,
  which can give excellent mean estimates but terrible variance estimates;
  maximizing ChEES should give good estimates across a wider range of types of
  posterior expectations.

  Args:
    previous_state: (Possibly nested) floating point `Tensor`. The previous
      state of the HMC chain.
    proposed_state: (Possibly nested) floating point `Tensor`. The proposed
      state of the HMC chain.
    accept_prob: Floating `Tensor`. Probability of acceping the proposed state.
    trajectory_length: Floating `Tensor`. Mean trajectory length (not used in
      this criterion).
    forward: Whether accept_prob refers to the proposed_state (True) or the
      previous_state (False).
    validate_args: Whether to perform non-static argument validation.
    experimental_shard_axis_names: A structure of string names indicating how
      members of the state are sharded.
    experimental_reduce_chain_axis_names: A string or list of string names
      indicating which named chain axes to reduce over when computing the
      criterion.

  Returns:
    chees: The value of the ChEES criterion.

  Raises:
    ValueError: If `accept_prob` indicates that there are fewer than 2 chains.

  #### References

  [1]: Hoffman, M., Radul, A., & Sountsov, P. (2020). An Adaptive MCMC Scheme
       for Setting Trajectory Lengths in Hamiltonian Monte Carlo.
       <https://proceedings.mlr.press/v130/hoffman21a>

  """
  del trajectory_length
  batch_ndims = ps.rank(accept_prob)
  reduce_chain_axis_names = distribute_lib.canonicalize_named_axis(
      experimental_reduce_chain_axis_names)

  accept_prob = _check_at_least_two_chains(
      accept_prob,
      reduce_chain_axis_names=reduce_chain_axis_names,
      validate_args=validate_args,
      message='chees_criterion requires at least 2 chains.',
  )

  def _center_previous_state(x):
    return x - _estimate_empirical_mean(
        x,
        accept_prob=accept_prob,
        safe=not forward,
        reduce_chain_axis_names=reduce_chain_axis_names,
    )

  def _center_proposed_state(x):
    return x - _estimate_empirical_mean(
        x,
        accept_prob=accept_prob,
        safe=forward,
        reduce_chain_axis_names=reduce_chain_axis_names,
    )

  def _sum_event_part(x, shard_axes=None):
    event_axes = ps.range(batch_ndims, ps.rank(x))
    return distribute_lib.psum(tf.reduce_sum(x, axis=event_axes), shard_axes)

  def _sum_event(x):
    event_parts = _map_structure_up_to_with_axes(
        x,
        _sum_event_part,
        x,
        experimental_shard_axis_names=experimental_shard_axis_names)
    return sum(tf.nest.flatten(event_parts))

  def _square(x):
    return tf.nest.map_structure(tf.square, x)

  def _sub(x, y):
    return tf.nest.map_structure(lambda x, y: x - y, x, y)

  previous_state = tf.nest.map_structure(_center_previous_state, previous_state)
  proposed_state = tf.nest.map_structure(_center_proposed_state, proposed_state)
  chees = 0.25 * tf.square(
      _sum_event(_sub(_square(proposed_state), _square(previous_state))))
  return chees


def chees_rate_criterion(previous_state,
                         proposed_state,
                         accept_prob,
                         trajectory_length,
                         forward=True,
                         validate_args=False,
                         experimental_shard_axis_names=None,
                         experimental_reduce_chain_axis_names=None):
  """ChEES rate criterion.

  This is just like `chees_criterion`, but normalized by the trajectory
  length:
  ```none
  ChEES rate = 1/4 E[(||x' - E[x]||**2 - ||x - E[x]||**2)**2 /
    trajectory_length]
  ```

  Args:
    previous_state: (Possibly nested) floating point `Tensor`. The previous
      state of the HMC chain.
    proposed_state: (Possibly nested) floating point `Tensor`. The proposed
      state of the HMC chain.
    accept_prob: Floating `Tensor`. Probability of acceping the proposed state.
    trajectory_length: Floating `Tensor`. Trajectory length.
    forward: Whether accept_prob refers to the proposed_state (True) or the
      previous_state (False).
    validate_args: Whether to perform non-static argument validation.
    experimental_shard_axis_names: A structure of string names indicating how
      members of the state are sharded.
    experimental_reduce_chain_axis_names: A string or list of string names
      indicating which named chain axes to reduce over when computing the
      criterion.

  Returns:
    chees_rate: The value of the ChEES rate criterion.
  """
  return chees_criterion(
      previous_state=previous_state,
      proposed_state=proposed_state,
      accept_prob=accept_prob,
      trajectory_length=trajectory_length,
      forward=forward,
      validate_args=validate_args,
      experimental_shard_axis_names=experimental_shard_axis_names,
      experimental_reduce_chain_axis_names=experimental_reduce_chain_axis_names,
  ) / trajectory_length


def snaper_criterion(previous_state,
                     proposed_state,
                     accept_prob,
                     trajectory_length,
                     direction,
                     state_mean=None,
                     state_mean_weight=0.,
                     forward=True,
                     validate_args=False,
                     experimental_shard_axis_names=None,
                     experimental_reduce_chain_axis_names=None):
  """The SNAPER criterion from [1].

  SNAPER stands for Squared Norm Along Principal component ESJD Rate:

  ```None
  SNAPER = E[(((x' - E[x'])^T p)**2 - ((x' - E[x])^T p)**2)**2 /
             trajectory_length],
  ```

  where `x` is the previous chain state, `x'` is the next chain state, and `p`
  is a unit vector (the `direction` argument). Both expectations are with
  respect to the chain's stationary distribution. In practice, the inner
  expectation is replaced by the empirical mean across chains, so computing this
  criterion requires that at least 2 chains are present unless `state_mean` and
  `state_mean_weight` are set. The outer expectation is computed by the caller
  (e.g. in the `GradientBasedTrajectoryLengthAdaptation` kernel).

  This can be thought of as the standard expected squared jump distance (ESJD)
  criterion, except that the jump distance is computed in the space of squared
  projections onto a vector.

  The `direction` vector is typically chosen to be an approximation to the first
  principal component of the state covariance matrix.

  `state_mean` and `state_mean_weight` can be used to supplement the empirical
  means as follows:

  ```None
  E[x] ≈ (1 - state_mean_weight) * x.mean() + state_mean_weight * state_mean.
  ```

  Args:
    previous_state: (Possibly nested) floating point `Tensor`. The previous
      state of the HMC chain.
    proposed_state: (Possibly nested) floating point `Tensor`. The proposed
      state of the HMC chain.
    accept_prob: Floating `Tensor`. Probability of acceping the proposed state.
    trajectory_length: Floating `Tensor`. Mean trajectory length (not used in
      this criterion).
    direction: (Possibly nested) floating point `Tensor`. A unit vector onto
      which the centered state should be projected before computing ESJD.
      Typically this chosen to be an approximation to the first principal
      component of the state covariance matrix.
    state_mean: Optional (Possibly nested) floating point `Tensor`. The
      estimated state mean.
    state_mean_weight: Floating point `Tensor`. The weight of the `state_mean`.
    forward: Whether accept_prob refers to the proposed_state (True) or the
      previous_state (False).
    validate_args: Whether to perform non-static argument validation.
    experimental_shard_axis_names: A structure of string names indicating how
      members of the state are sharded.
    experimental_reduce_chain_axis_names: A string or list of string names
      indicating which named chain axes to reduce over when computing the
      criterion.

  Returns:
    snaper: The value of the SNAPER criterion.

  #### References

  [1]: Sountsov, P. & Hoffman, M. (2021). Focusing on Difficult Directions for
       Learning HMC Trajectory Lengths. <https://arxiv.org/abs/2110.11576>

  """
  batch_ndims = ps.rank(accept_prob)
  reduce_chain_axis_names = distribute_lib.canonicalize_named_axis(
      experimental_reduce_chain_axis_names)

  if state_mean is None:
    state_mean = tf.nest.map_structure(lambda _: None, previous_state)

    accept_prob = _check_at_least_two_chains(
        accept_prob,
        reduce_chain_axis_names=reduce_chain_axis_names,
        validate_args=validate_args,
        message=(
            'snaper_criterion requires at least 2 chains when `state_mean` is'
            ' `None`'
        ),
    )

  def _mix_in_state_mean(empirical_mean, state_mean):
    if state_mean is None:
      return empirical_mean
    else:
      return ((1. - state_mean_weight) * empirical_mean +
              state_mean_weight * state_mean)

  def _center_previous_state(x, x_mean):
    emp_x_mean = _estimate_empirical_mean(
        x,
        accept_prob=accept_prob,
        safe=not forward,
        reduce_chain_axis_names=reduce_chain_axis_names,
    )
    x_mean = _mix_in_state_mean(emp_x_mean, x_mean)
    return x - x_mean

  def _center_proposed_state(x, x_mean):
    emp_x_mean = _estimate_empirical_mean(
        x,
        accept_prob=accept_prob,
        safe=forward,
        reduce_chain_axis_names=reduce_chain_axis_names,
    )
    x_mean = _mix_in_state_mean(emp_x_mean, x_mean)
    return x - x_mean

  def _dot_product_part(x, p, shard_axes=None):
    event_axes = ps.range(batch_ndims, ps.rank(x))
    return distribute_lib.reduce_sum(x * p, event_axes, shard_axes)

  def _dot_product(x):
    dot_products = _map_structure_up_to_with_axes(
        x,
        _dot_product_part,
        x,
        direction,
        experimental_shard_axis_names=experimental_shard_axis_names)
    return sum(tf.nest.flatten(dot_products))

  previous_state = tf.nest.map_structure(_center_previous_state, previous_state,
                                         state_mean)
  proposed_state = tf.nest.map_structure(_center_proposed_state, proposed_state,
                                         state_mean)

  previous_proj = _dot_product(previous_state)
  proposed_proj = _dot_product(proposed_state)

  snaper = (
      tf.square(tf.square(proposed_proj) - tf.square(previous_proj)) /
      trajectory_length)
  return snaper


class GradientBasedTrajectoryLengthAdaptation(kernel_base.TransitionKernel):
  """Use gradient ascent to adapt inner kernel's trajectory length.

  This kernel optimizes the continuous trajectory length (aka integration time)
  parameter of Hamiltonian Monte Carlo. It does so by following the gradient of
  a criterion with respect to the trajectory length. The criterion is computed
  via `criterion_fn` with signature `(previous_state, proposed_state,
  accept_prob, trajectory_length) -> criterion`, where both the returned
  values retain the batch dimensions implied by the first three inputs. See
  `chees_criterion` for an example.

  To avoid resonances, this kernel jitters the integration time between 0 and
  the learned trajectory length by default.

  The initial trajectory length is extracted from the inner
  `HamiltonianMonteCarlo` kernel by multiplying the initial step size and
  initial number of leapfrog steps. This (and other algorithmic details) imply
  that the step size must be a scalar.

  In general, adaptation prevents the chain from reaching a stationary
  distribution, so obtaining consistent samples requires `num_adaptation_steps`
  be set to a value [somewhat smaller][1] than the number of burnin steps.
  However, it may sometimes be helpful to set `num_adaptation_steps` to a larger
  value during development in order to inspect the behavior of the chain during
  adaptation.

  Optionally, it is possible to use the improved gradient estimator from [3] by
  setting `use_reverse_estimator` to `True`. This estimator relies on the
  reversibility of HMC proposal to reduce variance and thus improve the
  adaptation speed and reliability. If this is set to `true`, `criterion_fn`
  needs to also take the `forward` argument to distinguish the implied
  integration direction.

  #### Examples

  This implements something similar to ChEES HMC from [2].

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfb = tfp.bijectors
  tfd = tfp.distributions

  target_log_prob_fn = tfd.JointDistributionSequential([
      tfd.Normal(0., 20.),
      tfd.HalfNormal(10.),
  ]).log_prob

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
      num_adaptation_steps=num_adaptation_steps)
  kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
      kernel,
      num_adaptation_steps=num_adaptation_steps,
      reduce_fn=tfp.math.reduce_log_harmonic_mean_exp)
  kernel = tfp.mcmc.TransformedTransitionKernel(
      kernel,
      [tfb.Identity(),
       tfb.Exp()])

  def trace_fn(_, pkr):
    return (
        pkr.inner_results.inner_results.inner_results.accepted_results
        .step_size,
        pkr.inner_results.inner_results.max_trajectory_length,
        pkr.inner_results.inner_results.inner_results.log_accept_ratio,
    )

  # The chain will be stepped for num_results + num_burnin_steps, adapting for
  # the first num_adaptation_steps.
  samples, [step_size, max_trajectory_length, log_accept_ratio] = (
      tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          current_state=[tf.zeros(num_chains),
                         tf.zeros(num_chains)],
          kernel=kernel,
          trace_fn=trace_fn,))

  # ~0.95, because Exp bijector is really bad for HalfNormal. Use Softplus in
  # practice.
  accept_prob = tf.math.exp(tfp.math.reduce_logmeanexp(
      tf.minimum(log_accept_ratio, 0.)))
  ```

  #### References

  [1]: <http://andrewgelman.com/2017/12/15/
        burn-vs-warm-iterative-simulation-algorithms/#comment-627745>

  [2]: Hoffman, M., Radul, A., & Sountsov, P. (2020). An Adaptive MCMC Scheme
       for Setting Trajectory Lengths in Hamiltonian Monte Carlo.
       <https://proceedings.mlr.press/v130/hoffman21a>

  [3]: Riou-Durand, L., Sountsov, P., Vogrinc, J., Margossian, C., Power, S.
       (2023) Adaptive Tuning for Metropolis Adjusted Langevin Trajectories.
       <https://proceedings.mlr.press/v206/riou-durand23a.html>

  """

  def __init__(
      self,
      inner_kernel,
      num_adaptation_steps,
      use_halton_sequence_jitter=True,
      adaptation_rate=0.025,
      jitter_amount=1.,
      criterion_fn=chees_criterion,
      max_leapfrog_steps=1000,
      averaged_sq_grad_adaptation_rate=0.05,
      num_leapfrog_steps_getter_fn=hmc_like_num_leapfrog_steps_getter_fn,
      num_leapfrog_steps_setter_fn=hmc_like_num_leapfrog_steps_setter_fn,
      step_size_getter_fn=hmc_like_step_size_getter_fn,
      initial_velocity_getter_fn=hmc_like_initial_velocity_getter_fn,
      proposed_velocity_getter_fn=hmc_like_proposed_velocity_getter_fn,
      log_accept_prob_getter_fn=hmc_like_log_accept_prob_getter_fn,
      proposed_state_getter_fn=hmc_like_proposed_state_getter_fn,
      use_reverse_estimator=False,
      validate_args=False,
      experimental_shard_axis_names=None,
      experimental_reduce_chain_axis_names=None,
      name=None):
    """Creates the trajectory length adaptation kernel.

    The default setter_fn and the getter_fn callbacks assume that the inner
    kernel produces kernel results structurally the same as the
    `HamiltonianMonteCarlo` kernel (possibly wrapped in some step size
    adaptation kernel).

    Args:
      inner_kernel: `TransitionKernel`-like object.
      num_adaptation_steps: Scalar `int` `Tensor` number of initial steps to
        during which to adjust the trajectory length. This may be greater, less
        than, or equal to the number of burnin steps.
      use_halton_sequence_jitter: Python bool. Whether to use a Halton sequence
        for jittering the trajectory length. This makes the procedure more
        stable than sampling trajectory lengths from a uniform distribution.
      adaptation_rate: Floating point scalar `Tensor`. How rapidly to adapt the
        trajectory length.
      jitter_amount: Floating point scalar `Tensor`. How much to jitter the
        trajectory on the next step. The trajectory length is sampled from `[(1
        - jitter_amount) * max_trajectory_length, max_trajectory_length]`.
      criterion_fn: Callable with `(previous_state, proposed_state, accept_prob)
        -> criterion`. Computes the criterion value.
      max_leapfrog_steps: Int32 scalar `Tensor`. Clips the number of leapfrog
        steps to this value.
      averaged_sq_grad_adaptation_rate: Floating point scalar `Tensor`. How
        rapidly to adapt the running average squared gradient. This is `1 -
        beta_2` from Adam.
      num_leapfrog_steps_getter_fn: A callable with the signature
        `(kernel_results) -> num_leapfrog_steps` where `kernel_results` are the
        results of the `inner_kernel`, and `num_leapfrog_steps` is a floating
        point `Tensor`.
      num_leapfrog_steps_setter_fn: A callable with the signature
        `(kernel_results, new_num_leapfrog_steps) -> new_kernel_results` where
        `kernel_results` are the results of the `inner_kernel`,
        `new_num_leapfrog_steps` is a scalar tensor `Tensor`, and
        `new_kernel_results` are a copy of `kernel_results` with the number of
        leapfrog steps set.
      step_size_getter_fn: A callable with the signature `(kernel_results) ->
        step_size` where `kernel_results` are the results of the `inner_kernel`,
        and `step_size` is a floating point `Tensor`.
      initial_velocity_getter_fn: A callable with the signature
        `(kernel_results) -> initial_velocity` where `kernel_results` are the
        results of the `inner_kernel`, and `initial_velocity` is a (possibly
        nested) floating point `Tensor`. Velocity is the derivative of state
        with respect to trajectory length.
      proposed_velocity_getter_fn: A callable with the signature
        `(kernel_results) -> proposed_velocity` where `kernel_results` are the
        results of the `inner_kernel`, and `proposed_velocity` is a (possibly
        nested) floating point `Tensor`. Velocity is the derivative of state
        with respect to trajectory length.
      log_accept_prob_getter_fn: A callable with the signature `(kernel_results)
        -> log_accept_prob` where `kernel_results` are the results of the
        `inner_kernel`, and `log_accept_prob` is a floating point `Tensor`.
        `log_accept_prob` has shape `[C0, ...., Cb]` with `b > 0`.
      proposed_state_getter_fn: A callable with the signature `(kernel_results)
        -> proposed_state` where `kernel_results` are the results of the
        `inner_kernel`, and `proposed_state` is a (possibly nested) floating
        point `Tensor`.
      use_reverse_estimator: Whether to use an improved estimator to compute
        trajectory length gradients. If `True`, `criterion_fn` needs to take a
        `forward` kwarg.
      validate_args: Python `bool`. When `True` kernel parameters are checked
        for validity. When `False` invalid inputs may silently render incorrect
        outputs.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.
      experimental_reduce_chain_axis_names: A string or list of string names
        indicating how batches of chains are sharded.
      name: Python `str` name prefixed to Ops created by this class. Default:
        'simple_step_size_adaptation'.

    Raises:
      ValueError: If `inner_kernel` contains a `TransformedTransitionKernel` in
        its hierarchy. If you need to use the `TransformedTransitionKernel`,
        place it above this kernel in the hierarchy (see the example in the
        class docstring).
    """
    inner_kernel = mcmc_util.enable_store_parameters_in_results(
        inner_kernel).experimental_with_shard_axes(
            experimental_shard_axis_names)
    _forbid_inner_transformed_kernel(inner_kernel)

    with tf.name_scope(
        mcmc_util.make_name(name, 'gradient_based_trajectory_length_adaptation',
                            '__init__')) as name:
      num_adaptation_steps = tf.convert_to_tensor(
          num_adaptation_steps, dtype=tf.int32, name='num_adaptation_steps')
      max_leapfrog_steps = tf.convert_to_tensor(
          max_leapfrog_steps, dtype=tf.int32, name='max_leapfrog_steps')

    self._parameters = dict(
        inner_kernel=inner_kernel,
        num_adaptation_steps=num_adaptation_steps,
        use_halton_sequence_jitter=use_halton_sequence_jitter,
        adaptation_rate=adaptation_rate,
        jitter_amount=jitter_amount,
        criterion_fn=criterion_fn,
        max_leapfrog_steps=max_leapfrog_steps,
        averaged_sq_grad_adaptation_rate=averaged_sq_grad_adaptation_rate,
        num_leapfrog_steps_getter_fn=num_leapfrog_steps_getter_fn,
        num_leapfrog_steps_setter_fn=num_leapfrog_steps_setter_fn,
        step_size_getter_fn=step_size_getter_fn,
        initial_velocity_getter_fn=initial_velocity_getter_fn,
        proposed_velocity_getter_fn=proposed_velocity_getter_fn,
        log_accept_prob_getter_fn=log_accept_prob_getter_fn,
        proposed_state_getter_fn=hmc_like_proposed_state_getter_fn,
        use_reverse_estimator=use_reverse_estimator,
        validate_args=validate_args,
        experimental_shard_axis_names=experimental_shard_axis_names,
        experimental_reduce_chain_axis_names=experimental_reduce_chain_axis_names,
        name=name,
    )

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernel']

  @property
  def use_halton_sequence_jitter(self):
    return self._parameters['use_halton_sequence_jitter']

  @property
  def num_adaptation_steps(self):
    return self._parameters['num_adaptation_steps']

  def criterion_fn(self, previous_state, proposed_state, accept_prob,
                   trajectory_length, forward=True):
    kwargs = {}
    if self.experimental_reduce_chain_axis_names is not None:
      kwargs['experimental_reduce_chain_axis_names'] = (
          self.experimental_reduce_chain_axis_names)
    if self.experimental_shard_axis_names is not None:
      kwargs['experimental_shard_axis_names'] = (
          self.experimental_shard_axis_names)
    if self.use_reverse_estimator:
      kwargs['forward'] = forward
    return self._parameters['criterion_fn'](previous_state, proposed_state,
                                            accept_prob, trajectory_length,
                                            **kwargs)

  @property
  def max_leapfrog_steps(self):
    return self._parameters['max_leapfrog_steps']

  @property
  def averaged_sq_grad_adaptation_rate(self):
    return self._parameters['averaged_sq_grad_adaptation_rate']

  def num_leapfrog_steps_getter_fn(self, kernel_results):
    return self._parameters['num_leapfrog_steps_getter_fn'](kernel_results)

  def num_leapfrog_steps_setter_fn(self, kernel_results,
                                   new_num_leapfrog_steps):
    return self._parameters['num_leapfrog_steps_setter_fn'](
        kernel_results, new_num_leapfrog_steps)

  def step_size_getter_fn(self, kernel_results):
    return self._parameters['step_size_getter_fn'](kernel_results)

  def initial_velocity_getter_fn(self, kernel_results):
    return self._parameters['initial_velocity_getter_fn'](kernel_results)

  def proposed_velocity_getter_fn(self, kernel_results):
    return self._parameters['proposed_velocity_getter_fn'](kernel_results)

  def log_accept_prob_getter_fn(self, kernel_results):
    return self._parameters['log_accept_prob_getter_fn'](kernel_results)

  def proposed_state_getter_fn(self, kernel_results):
    return self._parameters['proposed_state_getter_fn'](kernel_results)

  @property
  def use_reverse_estimator(self):
    return self._parameters['use_reverse_estimator']

  @property
  def validate_args(self):
    return self._parameters['validate_args']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(
        mcmc_util.make_name(self.name,
                            'gradient_based_trajectory_length_adaptation',
                            'one_step')):

      jitter_seed, inner_seed = samplers.split_seed(seed)

      dtype = previous_kernel_results.adaptation_rate.dtype
      current_state = tf.nest.map_structure(
          lambda x: tf.convert_to_tensor(x, dtype=dtype), current_state)
      step_f = tf.cast(previous_kernel_results.step, dtype)
      if self.use_halton_sequence_jitter:
        trajectory_jitter = _halton_sequence(step_f)
      else:
        trajectory_jitter = samplers.uniform((), seed=jitter_seed, dtype=dtype)

      jitter_amount = previous_kernel_results.jitter_amount
      trajectory_jitter = (
          trajectory_jitter * jitter_amount + (1. - jitter_amount))

      adapting = previous_kernel_results.step < self.num_adaptation_steps
      max_trajectory_length = tf.where(
          adapting, previous_kernel_results.max_trajectory_length,
          previous_kernel_results.averaged_max_trajectory_length)
      jittered_trajectory_length = (max_trajectory_length * trajectory_jitter)

      step_size = _ensure_step_size_is_scalar(
          self.step_size_getter_fn(previous_kernel_results), self.validate_args)
      num_leapfrog_steps = tf.cast(
          tf.maximum(
              tf.ones([], dtype),
              tf.math.ceil(jittered_trajectory_length / step_size)), tf.int32)

      previous_kernel_results_with_jitter = self.num_leapfrog_steps_setter_fn(
          previous_kernel_results, num_leapfrog_steps)

      new_state, new_inner_results = self.inner_kernel.one_step(
          current_state, previous_kernel_results_with_jitter.inner_results,
          inner_seed)

      initial_velocity = self.initial_velocity_getter_fn(new_inner_results)
      proposed_state = self.proposed_state_getter_fn(new_inner_results)
      proposed_velocity = self.proposed_velocity_getter_fn(new_inner_results)
      accept_prob = tf.exp(self.log_accept_prob_getter_fn(new_inner_results))

      new_kernel_results = _update_trajectory_grad(
          previous_kernel_results_with_jitter,
          previous_state=current_state,
          proposed_state=proposed_state,
          proposed_velocity=proposed_velocity,
          initial_velocity=initial_velocity,
          trajectory_jitter=trajectory_jitter,
          accept_prob=accept_prob,
          step_size=step_size,
          criterion_fn=self.criterion_fn,
          max_leapfrog_steps=self.max_leapfrog_steps,
          experimental_shard_axis_names=self.experimental_shard_axis_names,
          reduce_chain_axis_names=self.experimental_reduce_chain_axis_names,
          use_reverse_estimator=self.use_reverse_estimator)

      # Undo the effect of adaptation if we're not in the burnin phase. We keep
      # the criterion, however, as that's a diagnostic. We also keep the
      # leapfrog steps setting, as that's an effect of jitter (and also doubles
      # as a diagnostic).
      criterion = new_kernel_results.criterion
      new_kernel_results = mcmc_util.choose(
          adapting, new_kernel_results, previous_kernel_results_with_jitter)

      new_kernel_results = new_kernel_results._replace(
          inner_results=new_inner_results,
          step=previous_kernel_results.step + 1,
          criterion=criterion,
          seed=seed)

      return new_state, new_kernel_results

  def bootstrap_results(self, init_state):
    with tf.name_scope(
        mcmc_util.make_name(self.name,
                            'gradient_based_trajectory_length_adaptation',
                            'bootstrap_results')):
      inner_results = self.inner_kernel.bootstrap_results(init_state)
      dtype = self.log_accept_prob_getter_fn(inner_results).dtype
      init_state = tf.nest.map_structure(
          lambda x: tf.convert_to_tensor(x, dtype=dtype), init_state)
      step_size = _ensure_step_size_is_scalar(
          self.step_size_getter_fn(inner_results), self.validate_args)
      init_max_trajectory_length = (
          step_size *
          tf.cast(self.num_leapfrog_steps_getter_fn(inner_results), dtype))
      results = GradientBasedTrajectoryLengthAdaptationResults(
          inner_results=inner_results,
          max_trajectory_length=init_max_trajectory_length,
          step=tf.zeros([], tf.int32),
          adaptation_rate=tf.convert_to_tensor(
              self.parameters['adaptation_rate'], dtype,
              name='adaptation_rate'),
          jitter_amount=tf.convert_to_tensor(
              self.parameters['jitter_amount'], dtype, name='jitter_amount'),
          averaged_sq_grad=tf.zeros([], dtype),
          averaged_sq_grad_adaptation_rate=tf.convert_to_tensor(
              self.averaged_sq_grad_adaptation_rate,
              dtype,
              name='averaged_sq_grad_adaptation_rate'),
          averaged_max_trajectory_length=tf.zeros([], dtype),
          criterion=tf.zeros_like(
              self.log_accept_prob_getter_fn(inner_results)),
          seed=samplers.zeros_seed(),
      )
      return results

  @property
  def is_calibrated(self):
    return self.inner_kernel.is_calibrated

  @property
  def experimental_shard_axis_names(self):
    return self._parameters['experimental_shard_axis_names']

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(experimental_shard_axis_names=shard_axis_names)

  @property
  def experimental_reduce_chain_axis_names(self):
    return self._parameters['experimental_reduce_chain_axis_names']


def _forbid_inner_transformed_kernel(inner_kernel):
  """Forbids inner kernel from containing `TransformedTransitionKernel`."""
  # TODO(b/169898277): The issue is that the proposed_velocity must be in the
  # same space as the chain state, and TransformedTransitionKernel breaks that
  # invariant.
  while hasattr(inner_kernel,
                'parameters') and 'inner_kernel' in inner_kernel.parameters:
    if isinstance(inner_kernel, transformed_kernel.TransformedTransitionKernel):
      raise ValueError(
          'The inner kernel cannot contain a `TransformedTransitionKernel`. '
          'Please place the `TransformedTransitionKernel` above this kernel '
          'in the hierarchy (see the docstring example of '
          '`GradientBasedTrajectoryLengthAdaptation` kernel.)')
    inner_kernel = inner_kernel.parameters['inner_kernel']


def _ensure_step_size_is_scalar(step_size, validate_args):
  """Ensures that the step size is a scalar `Tensor`."""
  if tf.nest.is_nested(step_size):
    raise ValueError('Step size must be a scalar. Got: {}'.format(step_size))
  rank = ps.rank(step_size)
  rank_ = tf.get_static_value(rank)
  if rank_ is not None:
    if rank_ != 0:
      raise ValueError('Step size must be a scalar. Got: {}'.format(step_size))
  elif validate_args:
    with tf.control_dependencies(
        [assert_util.assert_rank(step_size, 0, 'Step size must be a scalar.')]):
      return tf.identity(step_size)
  return step_size


def _halton_sequence(i, max_bits=MAX_HALTON_SEQUENCE_BITS):
  bit_masks = 2**tf.range(max_bits, dtype=i.dtype)
  return tf.einsum('i,i->', tf.math.mod((i + 1) // bit_masks, 2),
                   0.5 / bit_masks)


def _update_trajectory_grad(previous_kernel_results,
                            previous_state,
                            initial_velocity,
                            proposed_state,
                            proposed_velocity,
                            trajectory_jitter,
                            accept_prob,
                            step_size,
                            criterion_fn,
                            max_leapfrog_steps,
                            use_reverse_estimator,
                            experimental_shard_axis_names=None,
                            reduce_chain_axis_names=None):
  """Updates the trajectory length."""

  # Compute criterion grads.
  def leapfrog_action(dt):
    fwd_start_end_vel = [
        (True, previous_state, proposed_state, proposed_velocity)
    ]
    if use_reverse_estimator:
      fwd_start_end_vel.append((
          False,
          proposed_state,
          previous_state,
          tf.nest.map_structure(lambda x: -x, initial_velocity),
      ))

    # This represents the effect on the criterion value as the state follows
    # the proposed velocity. This implicitly assumes an identity mass matrix.
    def adjust_state(x, v, shard_axes=None):
      broadcasted_dt = distribute_lib.pbroadcast(
          bu.left_justified_expand_dims_like(dt, v), shard_axes)
      return x + broadcasted_dt * v

    criterion_vals = []
    for forward, start, end, vel in fwd_start_end_vel:
      adjusted_end = _map_structure_up_to_with_axes(
          end,
          adjust_state,
          end,
          vel,
          experimental_shard_axis_names=experimental_shard_axis_names)
      criterion_val = criterion_fn(
          previous_state=start,
          proposed_state=adjusted_end,
          accept_prob=accept_prob,
          # We add the step size here because we effectively do `floor(traj +
          # step_size) / step_size` when computing the number of leapfrog steps.
          trajectory_length=(
              trajectory_jitter * previous_kernel_results.max_trajectory_length
              + step_size
              + dt
          ),
          forward=forward,
      )
      criterion_vals.append(criterion_val)
    return tf.reduce_mean(criterion_vals, axis=0)

  criterion, trajectory_grad = gradient.value_and_gradient(
      leapfrog_action, tf.zeros_like(accept_prob))
  trajectory_grad *= trajectory_jitter

  # Weight by acceptance probability.
  reduce_chain_axis_names = distribute_lib.canonicalize_named_axis(
      reduce_chain_axis_names)
  trajectory_grad = tf.where(accept_prob > 1e-4, trajectory_grad, 0.)
  trajectory_grad = tf.where(
      tf.math.is_finite(trajectory_grad), trajectory_grad, 0.)
  trajectory_grad = (
      _reduce_sum_with_axes(trajectory_grad * accept_prob, None,
                            reduce_chain_axis_names) /
      _reduce_sum_with_axes(accept_prob + 1e-20, None, reduce_chain_axis_names))

  # Compute Adam/RMSProp step size.
  dtype = previous_kernel_results.adaptation_rate.dtype
  iteration_f = tf.cast(previous_kernel_results.step, dtype) + 1.
  averaged_sq_grad_adaptation_rate = (
      previous_kernel_results.averaged_sq_grad_adaptation_rate)
  new_averaged_sq_grad = ((1 - averaged_sq_grad_adaptation_rate) *
                          previous_kernel_results.averaged_sq_grad +
                          averaged_sq_grad_adaptation_rate * trajectory_grad**2)
  adjusted_averaged_sq_grad = new_averaged_sq_grad / (
      1. - (1 - averaged_sq_grad_adaptation_rate)**iteration_f)
  trajectory_step_size = (
      previous_kernel_results.adaptation_rate /
      tf.sqrt(adjusted_averaged_sq_grad + 1e-20))

  # Apply the gradient. Clip absolute value to ~log(2)/2.
  log_update = tf.clip_by_value(trajectory_step_size * trajectory_grad, -0.35,
                                0.35)
  new_max_trajectory_length = (
      previous_kernel_results.max_trajectory_length * tf.exp(log_update)
  )

  # Iterate averaging.
  average_weight = iteration_f**(-0.5)
  new_averaged_max_trajectory_length = tf.exp(
      average_weight * tf.math.log(new_max_trajectory_length) +
      (1 - average_weight) *
      tf.math.log(1e-10 +
                  previous_kernel_results.averaged_max_trajectory_length))

  # Clip the maximum trajectory length.
  new_max_trajectory_length = _clip_max_trajectory_length(
      new_max_trajectory_length, step_size,
      previous_kernel_results.adaptation_rate, max_leapfrog_steps)

  return previous_kernel_results._replace(
      criterion=criterion,
      max_trajectory_length=new_max_trajectory_length,
      averaged_sq_grad=new_averaged_sq_grad,
      averaged_max_trajectory_length=new_averaged_max_trajectory_length)


def _clip_max_trajectory_length(max_trajectory_length, step_size,
                                trajectory_adaptation_rate, max_leapfrog_steps):
  return tf.where(
      trajectory_adaptation_rate > 0,
      tf.clip_by_value(
          max_trajectory_length, 0.,
          step_size * tf.cast(max_leapfrog_steps, max_trajectory_length.dtype)),
      max_trajectory_length)


def _check_at_least_two_chains(accept_prob, reduce_chain_axis_names,
                               validate_args, message):
  """Checks that the number of chains is at least 2."""
  # Number of total chains is local batch size * distributed axis size
  local_axis_size = ps.size(accept_prob)
  distributed_axis_size = int(
      ps.reduce_prod(
          [distribute_lib.get_axis_size(a) for a in reduce_chain_axis_names]))
  num_chains = local_axis_size * distributed_axis_size
  num_chains_ = tf.get_static_value(num_chains)
  if num_chains_ is not None:
    if num_chains_ < 2:
      raise ValueError('{} Got: {}'.format(message, num_chains_))
  elif validate_args:
    with tf.control_dependencies(
        [assert_util.assert_greater_equal(num_chains, 2, message)]):
      accept_prob = tf.identity(accept_prob)
  return accept_prob
