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
"""Functional MCMC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from discussion.fun_mcmc import prefab
from discussion.fun_mcmc import util_tfp
from discussion.fun_mcmc.backend import get_backend
from discussion.fun_mcmc.backend import JAX
from discussion.fun_mcmc.backend import MANUAL_TRANSFORMS
from discussion.fun_mcmc.backend import set_backend
from discussion.fun_mcmc.backend import TENSORFLOW
from discussion.fun_mcmc.fun_mcmc_lib import adam_init
from discussion.fun_mcmc.fun_mcmc_lib import adam_step
from discussion.fun_mcmc.fun_mcmc_lib import AdamExtra
from discussion.fun_mcmc.fun_mcmc_lib import AdamState
from discussion.fun_mcmc.fun_mcmc_lib import blanes_3_stage_step
from discussion.fun_mcmc.fun_mcmc_lib import blanes_4_stage_step
from discussion.fun_mcmc.fun_mcmc_lib import call_fn
from discussion.fun_mcmc.fun_mcmc_lib import call_potential_fn
from discussion.fun_mcmc.fun_mcmc_lib import call_potential_fn_with_grads
from discussion.fun_mcmc.fun_mcmc_lib import call_transition_operator
from discussion.fun_mcmc.fun_mcmc_lib import call_transport_map
from discussion.fun_mcmc.fun_mcmc_lib import call_transport_map_with_ldj
from discussion.fun_mcmc.fun_mcmc_lib import gaussian_momentum_sample
from discussion.fun_mcmc.fun_mcmc_lib import gradient_descent_step
from discussion.fun_mcmc.fun_mcmc_lib import GradientDescentExtra
from discussion.fun_mcmc.fun_mcmc_lib import GradientDescentState
from discussion.fun_mcmc.fun_mcmc_lib import hamiltonian_integrator
from discussion.fun_mcmc.fun_mcmc_lib import hamiltonian_monte_carlo
from discussion.fun_mcmc.fun_mcmc_lib import hamiltonian_monte_carlo_init
from discussion.fun_mcmc.fun_mcmc_lib import HamiltonianMonteCarloExtra
from discussion.fun_mcmc.fun_mcmc_lib import HamiltonianMonteCarloState
from discussion.fun_mcmc.fun_mcmc_lib import IntegratorExtras
from discussion.fun_mcmc.fun_mcmc_lib import IntegratorState
from discussion.fun_mcmc.fun_mcmc_lib import IntegratorStep
from discussion.fun_mcmc.fun_mcmc_lib import IntegratorStepState
from discussion.fun_mcmc.fun_mcmc_lib import leapfrog_step
from discussion.fun_mcmc.fun_mcmc_lib import make_gaussian_kinetic_energy_fn
from discussion.fun_mcmc.fun_mcmc_lib import make_surrogate_loss_fn
from discussion.fun_mcmc.fun_mcmc_lib import maybe_broadcast_structure
from discussion.fun_mcmc.fun_mcmc_lib import mclachlan_optimal_4th_order_step
from discussion.fun_mcmc.fun_mcmc_lib import metropolis_hastings_step
from discussion.fun_mcmc.fun_mcmc_lib import MetropolisHastingsExtra
from discussion.fun_mcmc.fun_mcmc_lib import potential_scale_reduction_extract
from discussion.fun_mcmc.fun_mcmc_lib import potential_scale_reduction_init
from discussion.fun_mcmc.fun_mcmc_lib import potential_scale_reduction_step
from discussion.fun_mcmc.fun_mcmc_lib import PotentialFn
from discussion.fun_mcmc.fun_mcmc_lib import PotentialScaleReductionState
from discussion.fun_mcmc.fun_mcmc_lib import random_walk_metropolis
from discussion.fun_mcmc.fun_mcmc_lib import random_walk_metropolis_init
from discussion.fun_mcmc.fun_mcmc_lib import RandomWalkMetropolisExtra
from discussion.fun_mcmc.fun_mcmc_lib import RandomWalkMetropolisState
from discussion.fun_mcmc.fun_mcmc_lib import reparameterize_potential_fn
from discussion.fun_mcmc.fun_mcmc_lib import running_approximate_auto_covariance_init
from discussion.fun_mcmc.fun_mcmc_lib import running_approximate_auto_covariance_step
from discussion.fun_mcmc.fun_mcmc_lib import running_covariance_init
from discussion.fun_mcmc.fun_mcmc_lib import running_covariance_step
from discussion.fun_mcmc.fun_mcmc_lib import running_mean_init
from discussion.fun_mcmc.fun_mcmc_lib import running_mean_step
from discussion.fun_mcmc.fun_mcmc_lib import running_variance_init
from discussion.fun_mcmc.fun_mcmc_lib import running_variance_step
from discussion.fun_mcmc.fun_mcmc_lib import RunningApproximateAutoCovarianceState
from discussion.fun_mcmc.fun_mcmc_lib import RunningCovarianceState
from discussion.fun_mcmc.fun_mcmc_lib import RunningMeanState
from discussion.fun_mcmc.fun_mcmc_lib import RunningVarianceState
from discussion.fun_mcmc.fun_mcmc_lib import ruth4_step
from discussion.fun_mcmc.fun_mcmc_lib import sign_adaptation
from discussion.fun_mcmc.fun_mcmc_lib import simple_dual_averages_init
from discussion.fun_mcmc.fun_mcmc_lib import simple_dual_averages_step
from discussion.fun_mcmc.fun_mcmc_lib import SimpleDualAveragesExtra
from discussion.fun_mcmc.fun_mcmc_lib import SimpleDualAveragesState
from discussion.fun_mcmc.fun_mcmc_lib import splitting_integrator_step
from discussion.fun_mcmc.fun_mcmc_lib import State
from discussion.fun_mcmc.fun_mcmc_lib import trace
from discussion.fun_mcmc.fun_mcmc_lib import transform_log_prob_fn
from discussion.fun_mcmc.fun_mcmc_lib import TransitionOperator
from discussion.fun_mcmc.fun_mcmc_lib import TransportMap

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
    'gaussian_momentum_sample',
    'get_backend',
    'gradient_descent_step',
    'GradientDescentExtra',
    'GradientDescentState',
    'hamiltonian_integrator',
    'hamiltonian_monte_carlo',
    'hamiltonian_monte_carlo_init',
    'HamiltonianMonteCarloExtra',
    'HamiltonianMonteCarloState',
    'IntegratorExtras',
    'IntegratorState',
    'IntegratorStep',
    'IntegratorStepState',
    'JAX',
    'leapfrog_step',
    'make_gaussian_kinetic_energy_fn',
    'make_surrogate_loss_fn',
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
    'random_walk_metropolis',
    'random_walk_metropolis_init',
    'RandomWalkMetropolisExtra',
    'RandomWalkMetropolisState',
    'reparameterize_potential_fn',
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
    'set_backend',
    'sign_adaptation',
    'simple_dual_averages_init',
    'simple_dual_averages_step',
    'SimpleDualAveragesExtra',
    'SimpleDualAveragesState',
    'splitting_integrator_step',
    'State',
    'TENSORFLOW',
    'trace',
    'transform_log_prob_fn',
    'TransitionOperator',
    'TransportMap',
    'util_tfp',
]
