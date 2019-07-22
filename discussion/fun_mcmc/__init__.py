# Copyright 2019 The TensorFlow Probability Authors.
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

from discussion.fun_mcmc.fun_mcmc_lib import adam_init
from discussion.fun_mcmc.fun_mcmc_lib import adam_step
from discussion.fun_mcmc.fun_mcmc_lib import AdamExtra
from discussion.fun_mcmc.fun_mcmc_lib import AdamState
from discussion.fun_mcmc.fun_mcmc_lib import blanes_3_stage_step
from discussion.fun_mcmc.fun_mcmc_lib import blanes_4_stage_step
from discussion.fun_mcmc.fun_mcmc_lib import call_and_grads
from discussion.fun_mcmc.fun_mcmc_lib import call_fn
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
from discussion.fun_mcmc.fun_mcmc_lib import maybe_broadcast_structure
from discussion.fun_mcmc.fun_mcmc_lib import mclachlan_optimal_4th_order_step
from discussion.fun_mcmc.fun_mcmc_lib import metropolis_hastings_step
from discussion.fun_mcmc.fun_mcmc_lib import PotentialFn
from discussion.fun_mcmc.fun_mcmc_lib import ruth4_step
from discussion.fun_mcmc.fun_mcmc_lib import sign_adaptation
from discussion.fun_mcmc.fun_mcmc_lib import spliting_integrator_step
from discussion.fun_mcmc.fun_mcmc_lib import State
from discussion.fun_mcmc.fun_mcmc_lib import trace
from discussion.fun_mcmc.fun_mcmc_lib import transform_log_prob_fn
from discussion.fun_mcmc.fun_mcmc_lib import transition_kernel_wrapper
from discussion.fun_mcmc.fun_mcmc_lib import TransitionOperator

__all__ = [
    'adam_init',
    'adam_step',
    'AdamExtra',
    'AdamState',
    'blanes_3_stage_step',
    'blanes_4_stage_step',
    'call_and_grads',
    'call_fn',
    'gaussian_momentum_sample',
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
    'leapfrog_step',
    'make_gaussian_kinetic_energy_fn',
    'maybe_broadcast_structure',
    'mclachlan_optimal_4th_order_step',
    'metropolis_hastings_step',
    'PotentialFn',
    'ruth4_step',
    'sign_adaptation',
    'spliting_integrator_step',
    'State',
    'trace',
    'transform_log_prob_fn',
    'transition_kernel_wrapper',
    'TransitionOperator',
]
