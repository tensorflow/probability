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
