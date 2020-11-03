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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.experimental.mcmc.covariance_reducer import CovarianceReducer
from tensorflow_probability.python.experimental.mcmc.covariance_reducer import VarianceReducer
from tensorflow_probability.python.experimental.mcmc.elliptical_slice_sampler import EllipticalSliceSampler
from tensorflow_probability.python.experimental.mcmc.expectations_reducer import ExpectationsReducer
from tensorflow_probability.python.experimental.mcmc.gradient_based_trajectory_length_adaptation import chees_criterion
from tensorflow_probability.python.experimental.mcmc.gradient_based_trajectory_length_adaptation import GradientBasedTrajectoryLengthAdaptation
from tensorflow_probability.python.experimental.mcmc.gradient_based_trajectory_length_adaptation import GradientBasedTrajectoryLengthAdaptationResults
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
from tensorflow_probability.python.experimental.mcmc.progress_bar_reducer import make_tqdm_progress_bar_fn
from tensorflow_probability.python.experimental.mcmc.progress_bar_reducer import ProgressBarReducer
from tensorflow_probability.python.experimental.mcmc.reducer import Reducer
from tensorflow_probability.python.experimental.mcmc.sample import step_kernel
from tensorflow_probability.python.experimental.mcmc.sample_discarding_kernel import SampleDiscardingKernel
from tensorflow_probability.python.experimental.mcmc.sample_fold import sample_chain
from tensorflow_probability.python.experimental.mcmc.sample_fold import sample_fold
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import default_make_hmc_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import gen_make_hmc_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import gen_make_transform_hmc_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import make_rwmh_kernel_fn
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import sample_sequential_monte_carlo
from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import simple_heuristic_tuning
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import ess_below_threshold
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import SequentialMonteCarlo
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import SequentialMonteCarloResults
from tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel import WeightedParticles
from tensorflow_probability.python.experimental.mcmc.tracing_reducer import TracingReducer
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_deterministic_minimum_error
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_independent
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_stratified
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_systematic
from tensorflow_probability.python.experimental.mcmc.with_reductions import WithReductions
from tensorflow_probability.python.experimental.mcmc.with_reductions import WithReductionsKernelResults


__all__ = [
    'augment_prior_with_state_history',
    'augment_with_observation_history',
    'augment_with_state_history',
    'chees_criterion',
    'CovarianceReducer',
    'default_make_hmc_kernel_fn',
    'EllipticalSliceSampler',
    'ess_below_threshold',
    'ExpectationsReducer',
    'gen_make_hmc_kernel_fn',
    'gen_make_transform_hmc_kernel_fn',
    'GradientBasedTrajectoryLengthAdaptation',
    'GradientBasedTrajectoryLengthAdaptationResults',
    'infer_trajectories',
    'KernelBuilder',
    'KernelOutputs',
    'make_rwmh_kernel_fn',
    'make_tqdm_progress_bar_fn',
    'NoUTurnSampler',
    'particle_filter',
    'PotentialScaleReductionReducer',
    'PreconditionedHamiltonianMonteCarlo',
    'ProgressBarReducer',
    'reconstruct_trajectories',
    'Reducer',
    'resample_deterministic_minimum_error',
    'resample_independent',
    'resample_stratified',
    'resample_systematic',
    'sample_chain',
    'sample_fold',
    'sample_sequential_monte_carlo',
    'SampleDiscardingKernel',
    'SequentialMonteCarlo',
    'SequentialMonteCarloResults',
    'simple_heuristic_tuning',
    'StateWithHistory',
    'step_kernel',
    'TracingReducer',
    'VarianceReducer',
    'WeightedParticles',
    'WithReductions',
    'WithReductionsKernelResults',
]
