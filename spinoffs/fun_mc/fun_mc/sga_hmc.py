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
"""Implementation of stochastic gradient ascent hamiltonian monte carlo.

SGA-HMC is generalized version of ChEES-HMC described in [1]

#### References

[1]: Hoffman, M., Radul, A., & Sountsov, P. (2020). An Adaptive MCMC Scheme
     for Setting Trajectory Lengths in Hamiltonian Monte Carlo. In
     preparation.
"""

from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Tuple  # pylint: disable=unused-import

from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc

tf = backend.tf
tfp = backend.tfp
util = backend.util
distribute_lib = backend.distribute_lib

__all__ = [
    'chees_criterion',
    'chees_per_grad_criterion',
    'ChEESPerGradExtra',
    'hamiltonian_monte_carlo_with_state_grads_step',
    'HamiltonianMonteCarloWithStateGradsExtra',
]


class HamiltonianMonteCarloWithStateGradsExtra(NamedTuple):
  """Extra outputs for 'hamiltonian_monte_carlo_with_state_grads_step'."""
  hmc_extra: 'fun_mc.HamiltonianMonteCarloExtra'
  num_integrator_steps: 'fun_mc.IntTensor'
  proposed_state: 'fun_mc.State'


@util.named_call
def hamiltonian_monte_carlo_with_state_grads_step(
    hmc_state: 'fun_mc.HamiltonianMonteCarloState',
    trajectory_length: 'fun_mc.FloatTensor',
    scalar_step_size: 'fun_mc.FloatTensor',
    step_size_scale: 'fun_mc.FloatNest' = 1.,
    named_axis: 'Optional[fun_mc.StringNest]' = None,
    **hmc_kwargs
) -> ('Tuple[fun_mc.HamiltonianMonteCarloState, '
      'HamiltonianMonteCarloWithStateGradsExtra]'):
  """Hamiltonian Monte Carlo (HMC) step with gradients for proposed state.

  This acts as a `fun_mc.hamiltonian_monte_carlo_step`, where the
  `num_integrator_steps` is defined as `ceil(trajectory_length /
  scalar_step_size)` and `step_size` is defined as `scalar_step_size *
  step_size_scale`. The main feature of this function is that it propagates the
  gradients from `hmc_with_state_grads_extra.proposed_state` to
  `trajectory_length` (these are the only gradients propagated at the moment).
  This feature can be used to do gradient-based optimization of
  `trajectory_length` based on criteria that depend on the `proposed_state`
  (e.g. [1]).

  Args:
    hmc_state: `fun_mc.HamiltonianMonteCarloState`.
    trajectory_length: Trajectory length used by HMC.
    scalar_step_size: Scalar step size (used to compute the number of leapfrog
      steps).
    step_size_scale: Step size scale, structure broadcastable to the
      `hmc_state.state`.
    named_axis: Named axes of the state. Same structure as `hmc_state.state`.
    **hmc_kwargs: Passed to `fun_mc.hamiltonian_monte_carlo_step`.

  Returns:
    hmc_state: `fun_mc.HamiltonianMonteCarloState`.
    hmc_with_grads_extra: Extra outputs.

  #### References

  [1]: Hoffman, M., Radul, A., & Sountsov, P. (2021). An Adaptive MCMC Scheme
       for Setting Trajectory Lengths in Hamiltonian Monte Carlo.
       http://proceedings.mlr.press/v130/hoffman21a.html
  """

  @tf.custom_gradient
  def hmc(trajectory_length):
    trajectory_length = tf.convert_to_tensor(trajectory_length)
    num_integrator_steps = tf.cast(
        tf.math.ceil(trajectory_length / scalar_step_size), tf.int32)
    # In case something goes negative.
    num_integrator_steps = tf.maximum(1, num_integrator_steps)
    new_hmc_state, hmc_extra = fun_mc.hamiltonian_monte_carlo_step(
        hmc_state,
        num_integrator_steps=num_integrator_steps,
        step_size=util.map_tree(lambda s: s * scalar_step_size,
                                step_size_scale),
        named_axis=named_axis,
        **hmc_kwargs)
    hmc_with_grads_extra = HamiltonianMonteCarloWithStateGradsExtra(
        proposed_state=hmc_extra.proposed_hmc_state.state,
        hmc_extra=hmc_extra,
        num_integrator_steps=num_integrator_steps)
    res = (new_hmc_state, hmc_with_grads_extra)

    def grad(*grads):
      grads = util.unflatten_tree(res, util.flatten_tree(grads))

      step_size_scale_bc = fun_mc.maybe_broadcast_structure(
          step_size_scale, hmc_extra.integrator_extra.momentum_grads)

      # We wish to compute `grads^T @
      # jacobian(proposed_state(trajectory_length))`.
      #
      # The Jacobian is known from from Hamilton's equations:
      #
      # dx / dt = dK(v) / dv
      #
      # where `x` is the state, `v` is the momentum and `K` is the kinetic
      # energy. Since `step_size_scale` rescales momentum, we the right hand
      # side of that expression is `momentum_grads * step_size_scale` by the
      # chain rule. Since the Jacobian in question has 1 row, the
      # vector-Jacobian product is simply the dot product.
      state_grads = util.map_tree(lambda s, m, g: s * m * g, step_size_scale_bc,
                                  hmc_extra.integrator_extra.momentum_grads,
                                  grads[1].proposed_state)

      def do_sum(x, named_axis):
        return distribute_lib.reduce_sum(
            x, list(range(len(trajectory_length.shape), len(x.shape))),
            named_axis)

      if named_axis is None:
        named_axis_bc = util.map_tree(lambda _: [], state_grads)
      else:
        named_axis_bc = named_axis

      return sum(
          util.flatten_tree(
              util.map_tree_up_to(state_grads, do_sum, state_grads,
                                  named_axis_bc)))

    return res, grad

  return hmc(trajectory_length)


@util.named_call
def chees_criterion(
    previous_state: 'fun_mc.State',
    proposed_state: 'fun_mc.State',
    accept_prob: 'fun_mc.FloatTensor',
    trajectory_length: 'Optional[fun_mc.FloatTensor]' = None,
    state_mean: 'Optional[fun_mc.State]' = None,
    state_mean_weight: 'fun_mc.FloatNest' = 0.,
    named_axis: 'Optional[fun_mc.StringNest]' = None,
    chain_named_axis: 'Optional[fun_mc.StringNest]' = None,
) -> 'Tuple[fun_mc.FloatTensor, fun_mc.FloatTensor]':
  """The ChEES criterion from [1].

  ChEES stands for Change in the Estimator of the Expected Square.

  ```None
  ChEES = 1/4 E[(||x' - E[x]||**2 - ||x - E[x]||**2)**2],
  ```

  where `x` is the previous chain state, `x'` is the next chain state, and
  `||.||` is the L2 norm. Both expectations are with respect to the chain's
  stationary distribution. In practice, the inner expectation is replaced by the
  empirical mean across chains optionally averaged with a provided `state_mean`
  (weighted by `state_mean_weight`).

  This can be thought of as the standard expected squared jump distance (ESJD)
  criterion, except that the jump distance is computed in the space of centered
  squared L2 norms. It is also possible to relate ChEES to ESS computed in the
  same space if the true autocorrelation function of the centered squared L2
  norm follows a certain functional form.

  ChEES in this implementation is scaled by a normalized
  acceptance probability, so as to discard contributions from bad proposals.

  Unlike ChEES, regular ESJD is maximized by perfectly anticorrelated proposals,
  which can give excellent mean estimates but terrible variance estimates;
  maximizing ChEES should give good estimates across a wider range of types of
  posterior expectations.

  Args:
    previous_state: (Possibly nested) floating point `Tensor`. The previous
      state of the MCMC chain.
    proposed_state: (Possibly nested) floating point `Tensor`. The proposed
      state of the MCMC chain.
    accept_prob: Floating `Tensor`. Probability of acceping the proposed state.
    trajectory_length: Ignored.
    state_mean: (Possibly nested) floating point `Tensor`. Optional estimate of
      the MCMC chain mean.
    state_mean_weight: Floating point `Tensor`. Used to weight `state_mean` with
      the mean computed by averaging across the previous/proposed state. Setting
      it to effectively uses `state_mean` as the only source of the MCMCM chain
      mean.
    named_axis: Named axes of the state. Same structure as `previous_state`.
    chain_named_axis: Named axes of the MCMC chain that the criterion is to be
      averaged over.

  Returns:
    chees: The value of the ChEES criterion.
    per_chain_chees: The value of the ChEES criterion per chain.

  #### References

  [1]: Hoffman, M., Radul, A., & Sountsov, P. (2020). An Adaptive MCMC Scheme
       for Setting Trajectory Lengths in Hamiltonian Monte Carlo. In
       preparation.

  """
  del trajectory_length
  batch_ndims = len(accept_prob.shape)
  batch_axes = tuple(range(batch_ndims))
  no_state_mean = object()
  if state_mean is None:
    state_mean = fun_mc.maybe_broadcast_structure(no_state_mean, previous_state)
  state_mean_weight = fun_mc.maybe_broadcast_structure(state_mean_weight,
                                                       previous_state)
  if named_axis is None:
    named_axis_bc = util.map_tree(lambda _: [], previous_state)
  else:
    named_axis_bc = named_axis

  if chain_named_axis is None:
    chain_named_axis = []

  def _center_previous_state(x, mx, mw):
    x_center = distribute_lib.reduce_mean(
        x, axis=batch_axes, named_axis=chain_named_axis)
    if mx is not no_state_mean:
      x_center = x_center * (1 - mw) + mx * mw
    # The empirical mean here is a stand-in for the true mean, so we drop the
    # gradient that flows through this term.
    return x - tf.stop_gradient(x_center)

  def _center_proposed_state(x, mx, mw):
    expand_shape = list(accept_prob.shape) + [1] * (
        len(x.shape) - len(accept_prob.shape))
    expanded_accept_prob = tf.reshape(accept_prob, expand_shape)

    # Weight the proposed state by the acceptance probability. The goal here is
    # to get a reliable diagnostic of the underlying dynamics, rather than
    # incorporating the effect of the MetropolisHastings correction.

    # accept_prob is zero when x is NaN, but we still want to sanitize such
    # values.
    x_safe = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    # If all accept_prob's are zero, the x_center will have a nonsense value,
    # but we'll set the overall criterion to zero in this case, so it's fine.
    x_center = (
        distribute_lib.reduce_sum(
            expanded_accept_prob * x_safe,
            axis=batch_axes,
            named_axis=chain_named_axis) /
        (distribute_lib.reduce_sum(
            expanded_accept_prob, axis=batch_axes, named_axis=chain_named_axis)
         + 1e-20))
    if mx is not no_state_mean:
      x_center = x_center * (1 - mw) + mx * mw
    # The empirical mean here is a stand-in for the true mean, so we drop the
    # gradient that flows through this term.
    return x - tf.stop_gradient(x_center)

  def _sum_event_part(x, named_axis):
    event_axes = tuple(range(batch_ndims, len(x.shape)))
    return distribute_lib.reduce_sum(x, axis=event_axes, named_axis=named_axis)

  def _sum_event(x):
    return sum(
        util.flatten_tree(
            util.map_tree_up_to(
                x,
                _sum_event_part,
                x,
                named_axis_bc,
            )))

  def _square(x):
    return util.map_tree(tf.square, x)

  def _sub(x, y):
    return util.map_tree(lambda x, y: x - y, x, y)

  previous_state = util.map_tree(_center_previous_state, previous_state,
                                 state_mean, state_mean_weight)
  proposed_state = util.map_tree(_center_proposed_state, proposed_state,
                                 state_mean, state_mean_weight)
  chees = 0.25 * tf.square(
      _sum_event(_sub(_square(proposed_state), _square(previous_state))))

  # Zero-out per-chain ChEES values where acceptance probability is low. Those
  # values are probably not reflective of the underlying dynamics.
  chees = tf.where(accept_prob > 1e-4, chees, 0.)
  accept_prob = accept_prob / distribute_lib.reduce_sum(
      accept_prob + 1e-20, named_axis=chain_named_axis)
  chees = chees * accept_prob

  return distribute_lib.reduce_mean(chees, named_axis=chain_named_axis), chees


class ChEESPerGradExtra(NamedTuple):
  chees: 'fun_mc.FloatTensor'
  per_chain_chees: 'fun_mc.FloatTensor'
  per_chain_chees_per_grad: 'fun_mc.FloatTensor'


@util.named_call
def chees_per_grad_criterion(
    previous_state: 'fun_mc.State',
    proposed_state: 'fun_mc.State',
    accept_prob: 'fun_mc.FloatTensor',
    trajectory_length: 'fun_mc.FloatTensor',
    power: 'fun_mc.FloatTensor' = 1.,
    state_mean: 'Optional[fun_mc.State]' = None,
    state_mean_weight: 'fun_mc.FloatNest' = 0.,
    named_axis: 'Optional[fun_mc.StringNest]' = None,
    chain_named_axis: 'Optional[fun_mc.StringNest]' = None,
) -> 'Tuple[fun_mc.FloatTensor, ChEESPerGradExtra]':
  """ChEES per gradient criterion.

  This criterion is computed as:

  ```none
  ChEES/grad = 1/4 E[(||x' - E[x]||**2 - ||x - E[x]||**2)**2 /
    trajectory_length**power].
  ```

  For constant step sizes, `trajectory_length` is proportional to the number of
  gradients evaluated, meaning that this criterion trades off maximizing ChEES
  and using more computation to do so. In practice, this mitigates an issue with
  ChEES where it is periodic and has spurious maxima at prohibitively large
  trajectories that can nonetheless be found by gradient ascent with an
  aggressive learning rate. This criterion penalizes those maxima, making them
  less likely to be found.

  Args:
    previous_state: (Possibly nested) floating point `Tensor`. The previous
      state of the MCMC chain.
    proposed_state: (Possibly nested) floating point `Tensor`. The proposed
      state of the MCMC chain.
    accept_prob: Floating `Tensor`. Probability of acceping the proposed state.
    trajectory_length: Trajectory length associated with the transition from
      `previous_state` to `proposed_state`.
    power: Floating `Tensor`. Used to scale the `trajectory_length` term.
    state_mean: (Possibly nested) floating point `Tensor`. Optional estimate of
      the MCMC chain mean.
    state_mean_weight: Floating point `Tensor`. Used to weight `state_mean` with
      the mean computed by averaging across the previous/proposed state. Setting
      it to effectively uses `state_mean` as the only source of the MCMCM chain
      mean.
    named_axis: Named axes of the state. Same structure as `previous_state`.
    chain_named_axis: Named axes of the MCMC chain that the criterion is to be
      averaged over.

  Returns:
    chees: The value of the ChEES per grad criterion.
    chees_per_grad_extra: `ChEESPerGradExtra`.
  """
  chees, per_chain_chees = chees_criterion(
      previous_state=previous_state,
      proposed_state=proposed_state,
      accept_prob=accept_prob,
      state_mean=state_mean,
      state_mean_weight=state_mean_weight,
      named_axis=named_axis,
      chain_named_axis=chain_named_axis)
  per_chain_chees_per_grad = per_chain_chees / distribute_lib.pbroadcast(
      trajectory_length**power, chain_named_axis)
  extra = ChEESPerGradExtra(
      chees=chees,
      per_chain_chees=per_chain_chees,
      per_chain_chees_per_grad=per_chain_chees_per_grad,
  )
  return distribute_lib.reduce_mean(
      per_chain_chees_per_grad, named_axis=chain_named_axis), extra
