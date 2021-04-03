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
"""FunMC API."""

from fun_mc import prefab
from fun_mc import util_tfp
from fun_mc.fun_mc_lib import adam_init
from fun_mc.fun_mc_lib import adam_step
from fun_mc.fun_mc_lib import AdamExtra
from fun_mc.fun_mc_lib import AdamState
from fun_mc.fun_mc_lib import blanes_3_stage_step
from fun_mc.fun_mc_lib import blanes_4_stage_step
from fun_mc.fun_mc_lib import call_fn
from fun_mc.fun_mc_lib import call_potential_fn
from fun_mc.fun_mc_lib import call_potential_fn_with_grads
from fun_mc.fun_mc_lib import call_transition_operator
from fun_mc.fun_mc_lib import call_transport_map
from fun_mc.fun_mc_lib import call_transport_map_with_ldj
from fun_mc.fun_mc_lib import choose
from fun_mc.fun_mc_lib import gaussian_momentum_sample
from fun_mc.fun_mc_lib import gaussian_proposal
from fun_mc.fun_mc_lib import gradient_descent_init
from fun_mc.fun_mc_lib import gradient_descent_step
from fun_mc.fun_mc_lib import GradientDescentExtra
from fun_mc.fun_mc_lib import GradientDescentState
from fun_mc.fun_mc_lib import hamiltonian_integrator
from fun_mc.fun_mc_lib import hamiltonian_monte_carlo_init
from fun_mc.fun_mc_lib import hamiltonian_monte_carlo_step
from fun_mc.fun_mc_lib import HamiltonianMonteCarloExtra
from fun_mc.fun_mc_lib import HamiltonianMonteCarloState
from fun_mc.fun_mc_lib import IntegratorExtras
from fun_mc.fun_mc_lib import IntegratorState
from fun_mc.fun_mc_lib import IntegratorStep
from fun_mc.fun_mc_lib import IntegratorStepState
from fun_mc.fun_mc_lib import leapfrog_step
from fun_mc.fun_mc_lib import make_gaussian_kinetic_energy_fn
from fun_mc.fun_mc_lib import make_surrogate_loss_fn
from fun_mc.fun_mc_lib import maximal_reflection_coupling_proposal
from fun_mc.fun_mc_lib import MaximalReflectiveCouplingProposalExtra
from fun_mc.fun_mc_lib import maybe_broadcast_structure
from fun_mc.fun_mc_lib import mclachlan_optimal_4th_order_step
from fun_mc.fun_mc_lib import metropolis_hastings_step
from fun_mc.fun_mc_lib import MetropolisHastingsExtra
from fun_mc.fun_mc_lib import potential_scale_reduction_extract
from fun_mc.fun_mc_lib import potential_scale_reduction_init
from fun_mc.fun_mc_lib import potential_scale_reduction_step
from fun_mc.fun_mc_lib import PotentialFn
from fun_mc.fun_mc_lib import PotentialScaleReductionState
from fun_mc.fun_mc_lib import random_walk_metropolis_init
from fun_mc.fun_mc_lib import random_walk_metropolis_step
from fun_mc.fun_mc_lib import RandomWalkMetropolisExtra
from fun_mc.fun_mc_lib import RandomWalkMetropolisState
from fun_mc.fun_mc_lib import recover_state_from_args
from fun_mc.fun_mc_lib import reparameterize_potential_fn
from fun_mc.fun_mc_lib import running_approximate_auto_covariance_init
from fun_mc.fun_mc_lib import running_approximate_auto_covariance_step
from fun_mc.fun_mc_lib import running_covariance_init
from fun_mc.fun_mc_lib import running_covariance_step
from fun_mc.fun_mc_lib import running_mean_init
from fun_mc.fun_mc_lib import running_mean_step
from fun_mc.fun_mc_lib import running_variance_init
from fun_mc.fun_mc_lib import running_variance_step
from fun_mc.fun_mc_lib import RunningApproximateAutoCovarianceState
from fun_mc.fun_mc_lib import RunningCovarianceState
from fun_mc.fun_mc_lib import RunningMeanState
from fun_mc.fun_mc_lib import RunningVarianceState
from fun_mc.fun_mc_lib import ruth4_step
from fun_mc.fun_mc_lib import sign_adaptation
from fun_mc.fun_mc_lib import simple_dual_averages_init
from fun_mc.fun_mc_lib import simple_dual_averages_step
from fun_mc.fun_mc_lib import SimpleDualAveragesExtra
from fun_mc.fun_mc_lib import SimpleDualAveragesState
from fun_mc.fun_mc_lib import splitting_integrator_step
from fun_mc.fun_mc_lib import State
from fun_mc.fun_mc_lib import trace
from fun_mc.fun_mc_lib import transform_log_prob_fn
from fun_mc.fun_mc_lib import TransitionOperator
from fun_mc.fun_mc_lib import TransportMap

__all__ = [
    'adam_init',
    'adam_step',
    'AdamExtra',
    'AdamState',
    'blanes_3_stage_step',
    'blanes_4_stage_step',
    'call_fn',
    'call_potential_fn',
    'call_potential_fn_with_grads',
    'call_transition_operator',
    'call_transport_map',
    'call_transport_map_with_ldj',
    'choose',
    'gaussian_momentum_sample',
    'gaussian_proposal',
    'gradient_descent_init',
    'gradient_descent_step',
    'GradientDescentExtra',
    'GradientDescentState',
    'hamiltonian_integrator',
    'hamiltonian_monte_carlo_init',
    'hamiltonian_monte_carlo_step',
    'HamiltonianMonteCarloExtra',
    'HamiltonianMonteCarloState',
    'IntegratorExtras',
    'IntegratorState',
    'IntegratorStep',
    'IntegratorStepState',
    'leapfrog_step',
    'make_gaussian_kinetic_energy_fn',
    'make_surrogate_loss_fn',
    'maximal_reflection_coupling_proposal',
    'MaximalReflectiveCouplingProposalExtra',
    'maybe_broadcast_structure',
    'mclachlan_optimal_4th_order_step',
    'metropolis_hastings_step',
    'MetropolisHastingsExtra',
    'potential_scale_reduction_extract',
    'potential_scale_reduction_init',
    'potential_scale_reduction_step',
    'PotentialFn',
    'PotentialScaleReductionState',
    'prefab',
    'random_walk_metropolis_init',
    'random_walk_metropolis_step',
    'RandomWalkMetropolisExtra',
    'RandomWalkMetropolisState',
    'reparameterize_potential_fn',
    'recover_state_from_args',
    'running_approximate_auto_covariance_init',
    'running_approximate_auto_covariance_step',
    'running_covariance_init',
    'running_covariance_step',
    'running_mean_init',
    'running_mean_step',
    'running_variance_init',
    'running_variance_step',
    'RunningApproximateAutoCovarianceState',
    'RunningCovarianceState',
    'RunningMeanState',
    'RunningVarianceState',
    'ruth4_step',
    'sign_adaptation',
    'simple_dual_averages_init',
    'simple_dual_averages_step',
    'SimpleDualAveragesExtra',
    'SimpleDualAveragesState',
    'splitting_integrator_step',
    'State',
    'trace',
    'transform_log_prob_fn',
    'TransitionOperator',
    'TransportMap',
    'util_tfp',
]

