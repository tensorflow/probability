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
"""Functional MCMC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from experimental.fun_mcmc.fun_mcmc_lib import blanes_3_stage_step
from experimental.fun_mcmc.fun_mcmc_lib import blanes_4_stage_step
from experimental.fun_mcmc.fun_mcmc_lib import call_and_grads
from experimental.fun_mcmc.fun_mcmc_lib import call_fn
from experimental.fun_mcmc.fun_mcmc_lib import hamiltonian_monte_carlo
from experimental.fun_mcmc.fun_mcmc_lib import HamiltonianMonteCarloExtra
from experimental.fun_mcmc.fun_mcmc_lib import HamiltonianMonteCarloState
from experimental.fun_mcmc.fun_mcmc_lib import IntegratorStepState
from experimental.fun_mcmc.fun_mcmc_lib import leapfrog_step
from experimental.fun_mcmc.fun_mcmc_lib import maybe_broadcast_structure
from experimental.fun_mcmc.fun_mcmc_lib import metropolis_hastings_step
from experimental.fun_mcmc.fun_mcmc_lib import PotentialFn
from experimental.fun_mcmc.fun_mcmc_lib import ruth4_step
from experimental.fun_mcmc.fun_mcmc_lib import sign_adaptation
from experimental.fun_mcmc.fun_mcmc_lib import State
from experimental.fun_mcmc.fun_mcmc_lib import symmetric_spliting_integrator_step
from experimental.fun_mcmc.fun_mcmc_lib import trace
from experimental.fun_mcmc.fun_mcmc_lib import transform_log_prob_fn
from experimental.fun_mcmc.fun_mcmc_lib import transition_kernel_wrapper
from experimental.fun_mcmc.fun_mcmc_lib import TransitionOperator

__all__ = [
    'blanes_3_stage_step',
    'blanes_4_stage_step',
    'call_and_grads',
    'call_fn',
    'hamiltonian_monte_carlo',
    'HamiltonianMonteCarloExtra',
    'HamiltonianMonteCarloState',
    'IntegratorStepState',
    'leapfrog_step',
    'maybe_broadcast_structure',
    'metropolis_hastings_step',
    'PotentialFn',
    'ruth4_step',
    'sign_adaptation',
    'State',
    'symmetric_spliting_integrator_step',
    'trace',
    'transform_log_prob_fn',
    'transition_kernel_wrapper',
    'TransitionOperator',
]
