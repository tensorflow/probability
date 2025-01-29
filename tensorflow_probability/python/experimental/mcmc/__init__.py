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
"""TensorFlow Probability experimental MCMC package."""

from tensorflow_probability.python.experimental.mcmc.covariance_reducer import CovarianceReducer
from tensorflow_probability.python.experimental.mcmc.covariance_reducer import VarianceReducer
from tensorflow_probability.python.experimental.mcmc.diagonal_mass_matrix_adaptation import DiagonalMassMatrixAdaptation
from tensorflow_probability.python.experimental.mcmc.elliptical_slice_sampler import EllipticalSliceSampler
from tensorflow_probability.python.experimental.mcmc.expectations_reducer import ExpectationsReducer
from tensorflow_probability.python.experimental.mcmc.gradient_based_trajectory_length_adaptation import chees_criterion
from tensorflow_probability.python.experimental.mcmc.gradient_based_trajectory_length_adaptation import chees_rate_criterion
from tensorflow_probability.python.experimental.mcmc.gradient_based_trajectory_length_adaptation import GradientBasedTrajectoryLengthAdaptation
from tensorflow_probability.python.experimental.mcmc.gradient_based_trajectory_length_adaptation import GradientBasedTrajectoryLengthAdaptationResults
from tensorflow_probability.python.experimental.mcmc.gradient_based_trajectory_length_adaptation import snaper_criterion
from tensorflow_probability.python.experimental.mcmc.initialization import init_near_unconstrained_zero
from tensorflow_probability.python.experimental.mcmc.initialization import retry_init
from tensorflow_probability.python.experimental.mcmc.kernel_builder import KernelBuilder
from tensorflow_probability.python.experimental.mcmc.kernel_outputs import KernelOutputs
from tensorflow_probability.python.experimental.mcmc.nuts_autobatching import NoUTurnSampler
from tensorflow_probability.python.experimental.mcmc.particle_filter import infer_trajectories
from tensorflow_probability.python.experimental.mcmc.particle_filter import particle_filter
from tensorflow_probability.python.experimental.mcmc.particle_filter import reconstruct_trajectories
from tensorflow_probability.python.experimental.mcmc.particle_filter_augmentation import augment_prior_with_state_history
from tensorflow_probability.python.experimental.mcmc.particle_filter_augmentation import augment_with_observation_history
from tensorflow_probability.python.experimental.mcmc.particle_filter_augmentation import augment_with_state_history
from tensorflow_probability.python.experimental.mcmc.particle_filter_augmentation import StateWithHistory
from tensorflow_probability.python.experimental.mcmc.potential_scale_reduction_reducer import PotentialScaleReductionReducer
from tensorflow_probability.python.experimental.mcmc.preconditioned_hmc import PreconditionedHamiltonianMonteCarlo
from tensorflow_probability.python.experimental.mcmc.preconditioned_nuts import PreconditionedNoUTurnSampler
from tensorflow_probability.python.experimental.mcmc.progress_bar_reducer import make_tqdm_progress_bar_fn
from tensorflow_probability.python.experimental.mcmc.progress_bar_reducer import ProgressBarReducer
from tensorflow_probability.python.experimental.mcmc.reducer import Reducer
from tensorflow_probability.python.experimental.mcmc.sample import sample_chain
from tensorflow_probability.python.experimental.mcmc.sample_discarding_kernel import SampleDiscardingKernel
from tensorflow_probability.python.experimental.mcmc.sample_fold import sample_chain_with_burnin
from tensorflow_probability.python.experimental.mcmc.sample_fold import sample_fold
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import default_make_hmc_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import gen_make_hmc_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import gen_make_transform_hmc_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import make_rwmh_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import sample_sequential_monte_carlo
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import simple_heuristic_tuning
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import ess_below_threshold
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import log_ess_from_log_weights
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import SequentialMonteCarlo
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import SequentialMonteCarloResults
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import WeightedParticles
from tensorflow_probability.python.experimental.mcmc.sharded import Sharded
from tensorflow_probability.python.experimental.mcmc.snaper_hmc import sample_snaper_hmc
from tensorflow_probability.python.experimental.mcmc.snaper_hmc import SampleSNAPERHamiltonianMonteCarloResults
from tensorflow_probability.python.experimental.mcmc.snaper_hmc import SNAPERHamiltonianMonteCarlo
from tensorflow_probability.python.experimental.mcmc.snaper_hmc import SNAPERHamiltonianMonteCarloResults
from tensorflow_probability.python.experimental.mcmc.step import step_kernel
from tensorflow_probability.python.experimental.mcmc.thermodynamic_integrals import remc_thermodynamic_integrals
from tensorflow_probability.python.experimental.mcmc.thinning_kernel import ThinningKernel
from tensorflow_probability.python.experimental.mcmc.tracing_reducer import TracingReducer
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_deterministic_minimum_error
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_independent
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_stratified
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_systematic
from tensorflow_probability.python.experimental.mcmc.windowed_sampling import default_hmc_trace_fn
from tensorflow_probability.python.experimental.mcmc.windowed_sampling import default_nuts_trace_fn
from tensorflow_probability.python.experimental.mcmc.windowed_sampling import windowed_adaptive_hmc
from tensorflow_probability.python.experimental.mcmc.windowed_sampling import windowed_adaptive_nuts
from tensorflow_probability.python.experimental.mcmc.with_reductions import WithReductions
from tensorflow_probability.python.experimental.mcmc.with_reductions import WithReductionsKernelResults


__all__ = [
    'CovarianceReducer',
    'DiagonalMassMatrixAdaptation',
    'EllipticalSliceSampler',
    'ExpectationsReducer',
    'GradientBasedTrajectoryLengthAdaptation',
    'GradientBasedTrajectoryLengthAdaptationResults',
    'KernelBuilder',
    'KernelOutputs',
    'NoUTurnSampler',
    'PotentialScaleReductionReducer',
    'PreconditionedHamiltonianMonteCarlo',
    'PreconditionedNoUTurnSampler',
    'ProgressBarReducer',
    'Reducer',
    'SNAPERHamiltonianMonteCarlo',
    'SNAPERHamiltonianMonteCarloResults',
    'SampleDiscardingKernel',
    'SampleSNAPERHamiltonianMonteCarloResults',
    'SequentialMonteCarlo',
    'SequentialMonteCarloResults',
    'Sharded',
    'StateWithHistory',
    'ThinningKernel',
    'TracingReducer',
    'VarianceReducer',
    'WeightedParticles',
    'WithReductions',
    'WithReductionsKernelResults',
    'augment_prior_with_state_history',
    'augment_with_observation_history',
    'augment_with_state_history',
    'chees_criterion',
    'chees_rate_criterion',
    'default_make_hmc_kernel_fn',
    'log_ess_from_log_weights',
    'ess_below_threshold',
    'gen_make_hmc_kernel_fn',
    'gen_make_transform_hmc_kernel_fn',
    'infer_trajectories',
    'init_near_unconstrained_zero',
    'make_rwmh_kernel_fn',
    'make_tqdm_progress_bar_fn',
    'particle_filter',
    'reconstruct_trajectories',
    'remc_thermodynamic_integrals',
    'resample_deterministic_minimum_error',
    'resample_independent',
    'resample_stratified',
    'resample_systematic',
    'retry_init',
    'sample_chain',
    'sample_chain_with_burnin',
    'sample_fold',
    'sample_sequential_monte_carlo',
    'sample_snaper_hmc',
    'simple_heuristic_tuning',
    'snaper_criterion',
    'step_kernel',
    'default_hmc_trace_fn',
    'default_nuts_trace_fn',
    'windowed_adaptive_hmc',
    'windowed_adaptive_nuts',
]
