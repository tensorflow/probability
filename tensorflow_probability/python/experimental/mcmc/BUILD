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
# Description:
#   An implementation of the No U-Turn Sampler.
#   Achieves batch execution across chains by using
#   //tensorflow_probability/python/internal/auto_batching
#   internally.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "mcmc",
    srcs = ["__init__.py"],
    numpy_omit_deps = [
        ":preconditioned_hmc",
    ],
    substrates_omit_deps = [
        ":elliptical_slice_sampler",
        ":kernel_builder",
        ":kernel_outputs",
        ":nuts_autobatching",
        ":particle_filter_augmentation",
        ":progress_bar_reducer",
    ],
    deps = [
        ":covariance_reducer",
        ":diagonal_mass_matrix_adaptation",
        ":elliptical_slice_sampler",
        ":expectations_reducer",
        ":gradient_based_trajectory_length_adaptation",
        ":initialization",
        ":kernel_builder",
        ":kernel_outputs",
        ":nuts_autobatching",
        ":particle_filter",
        ":particle_filter_augmentation",
        ":potential_scale_reduction_reducer",
        ":preconditioned_hmc",
        ":preconditioned_nuts",
        ":preconditioning_utils",
        ":progress_bar_reducer",
        ":reducer",
        ":sample",
        ":sample_discarding_kernel",
        ":sample_fold",
        ":sample_sequential_monte_carlo",
        ":sequential_monte_carlo_kernel",
        ":sharded",
        ":snaper_hmc",
        ":step",
        ":thermodynamic_integrals",
        ":thinning_kernel",
        ":tracing_reducer",
        ":weighted_resampling",
        ":windowed_sampling",
        ":with_reductions",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "initialization",
    srcs = ["initialization.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/bijectors:joint_map",
        "//tensorflow_probability/python/bijectors:restructure",
        "//tensorflow_probability/python/distributions:batch_broadcast",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/experimental/distribute",
    ],
)

multi_substrate_py_test(
    name = "initialization_test",
    size = "medium",
    srcs = ["initialization_test.py"],
    deps = [
        ":initialization",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:pad",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/distributions:half_cauchy",
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:multinomial",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/experimental/distribute",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "kernel_builder",
    srcs = ["kernel_builder.py"],
    deps = [
        ":kernel_outputs",
        ":preconditioned_hmc",
        ":progress_bar_reducer",
        ":sample_discarding_kernel",
        ":step",
        ":tracing_reducer",
        ":with_reductions",
        "//tensorflow_probability/python/mcmc:dual_averaging_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:langevin",
        "//tensorflow_probability/python/mcmc:nuts",
        "//tensorflow_probability/python/mcmc:random_walk_metropolis",
        "//tensorflow_probability/python/mcmc:replica_exchange_mc",
        "//tensorflow_probability/python/mcmc:simple_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

py_test(
    name = "kernel_builder_test",
    size = "small",
    srcs = ["kernel_builder_test.py"],
    deps = [
        ":kernel_builder",
        ":sample_discarding_kernel",
        ":with_reductions",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:nuts",
        "//tensorflow_probability/python/mcmc:simple_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
    ],
)

py_library(
    name = "kernel_outputs",
    srcs = ["kernel_outputs.py"],
    deps = [
        ":tracing_reducer",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

py_test(
    name = "kernel_outputs_test",
    size = "small",
    srcs = ["kernel_outputs_test.py"],
    deps = [
        ":kernel_builder",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "diagonal_mass_matrix_adaptation",
    srcs = ["diagonal_mass_matrix_adaptation.py"],
    deps = [
        ":preconditioning_utils",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/stats:sample_stats",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "diagonal_mass_matrix_adaptation_test",
    size = "large",
    timeout = "long",
    srcs = ["diagonal_mass_matrix_adaptation_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 3,
    deps = [
        ":diagonal_mass_matrix_adaptation",
        ":preconditioned_hmc",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/experimental/stats:sample_stats",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:sample",
    ],
)

multi_substrate_py_library(
    name = "preconditioning_utils",
    srcs = ["preconditioning_utils.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:reshape",
        "//tensorflow_probability/python/distributions:batch_broadcast",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/experimental/distribute:joint_distribution",
        "//tensorflow_probability/python/experimental/distribute:sharded",
        "//tensorflow_probability/python/experimental/distributions:mvn_precision_factor_linop",
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_library(
    name = "preconditioned_nuts",
    srcs = ["preconditioned_nuts.py"],
    deps = [
        ":preconditioning_utils",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal:leapfrog_integrator",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "pnuts_test",
    size = "medium",
    srcs = ["pnuts_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 10,
    deps = [
        ":preconditioned_nuts",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:half_normal",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_linear_operator",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/distributions:student_t",
        "//tensorflow_probability/python/distributions:wishart",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc:diagnostic",
        "//tensorflow_probability/python/mcmc:dual_averaging_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:sample",
        "//tensorflow_probability/python/mcmc:simple_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
        "//tensorflow_probability/python/stats:sample_stats",
    ],
)

multi_substrate_py_library(
    name = "preconditioned_hmc",
    srcs = ["preconditioned_hmc.py"],
    deps = [
        ":preconditioning_utils",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:metropolis_hastings",
        "//tensorflow_probability/python/mcmc/internal:leapfrog_integrator",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "preconditioned_hmc_test",
    size = "large",
    timeout = "long",
    srcs = ["preconditioned_hmc_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 3,
    deps = [
        ":preconditioned_hmc",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:reshape",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_linear_operator",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:wishart",
        "//tensorflow_probability/python/experimental/distributions:mvn_precision_factor_linop",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/internal:auto_composite_tensor",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/mcmc:diagnostic",
        "//tensorflow_probability/python/mcmc:dual_averaging_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:sample",
        "//tensorflow_probability/python/mcmc:simple_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
        "//tensorflow_probability/python/mcmc/internal:leapfrog_integrator",
        "//tensorflow_probability/python/stats:sample_stats",
    ],
)

py_library(
    name = "progress_bar_reducer",
    srcs = ["progress_bar_reducer.py"],
    deps = [
        ":reducer",
        # tensorflow dep,
        # tqdm dep,
    ],
)

py_test(
    name = "progress_bar_reducer_test",
    size = "small",
    srcs = ["progress_bar_reducer_test.py"],
    deps = [
        ":progress_bar_reducer",
        ":sample_fold",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "elliptical_slice_sampler",
    srcs = ["elliptical_slice_sampler.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

py_test(
    name = "elliptical_slice_sampler_test",
    size = "large",
    timeout = "long",
    srcs = ["elliptical_slice_sampler_test.py"],
    shard_count = 6,
    deps = [
        ":elliptical_slice_sampler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:sample",
    ],
)

py_library(
    name = "nuts_autobatching",
    srcs = ["nuts_autobatching.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/random",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

py_library(
    name = "nuts_autobatching_testlib",
    testonly = 1,
    srcs = ["nuts_autobatching_test.py"],
    deps = [
        ":nuts_autobatching",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:scale_matvec_tril",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/distributions:wishart",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:sample",
    ],
)

py_test(
    name = "nuts_autobatching_test",
    size = "medium",
    srcs = ["nuts_autobatching_test.py"],
    shard_count = 7,
    tags = ["nozapfhahn"],
    deps = [
        ":nuts_autobatching_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "nuts_autobatching_test_gpu",
    srcs = ["nuts_autobatching_test.py"],
    main = "nuts_autobatching_test.py",
    shard_count = 7,
    tags = [
        "notap",
        "nozapfhahn",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":nuts_autobatching_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "nuts_autobatching_xla_test_cpu",
    size = "large",
    srcs = ["nuts_autobatching_xla_test.py"],
    args = ["--test_device=cpu"],
    main = "nuts_autobatching_xla_test.py",
    shard_count = 1,
    tags = [
        "nozapfhahn",
    ],
    deps = [
        ":nuts_autobatching",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/experimental/auto_batching:tf_backend",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:nuts",
        "//tensorflow_probability/python/mcmc:sample",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_test(
    name = "nuts_autobatching_xla_test_gpu",
    size = "large",
    srcs = ["nuts_autobatching_xla_test.py"],
    args = ["--test_device=gpu"],
    main = "nuts_autobatching_xla_test.py",
    shard_count = 1,
    tags = [
        "nozapfhahn",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":nuts_autobatching",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/experimental/auto_batching:tf_backend",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:nuts",
        "//tensorflow_probability/python/mcmc:sample",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "particle_filter",
    srcs = ["particle_filter.py"],
    deps = [
        ":sequential_monte_carlo_kernel",
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/internal:loop_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "particle_filter_test",
    size = "large",
    srcs = ["particle_filter_test.py"],
    numpy_tags = ["notap"],
    shard_count = 3,
    deps = [
        ":particle_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:linear_gaussian_ssm",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:hidden_markov_model",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_library(
    name = "particle_filter_augmentation",
    srcs = ["particle_filter_augmentation.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:joint_distribution_util",
    ],
)

py_test(
    name = "particle_filter_augmentation_test",
    size = "medium",
    srcs = ["particle_filter_augmentation_test.py"],
    deps = [
        ":particle_filter",
        ":particle_filter_augmentation",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sample_sequential_monte_carlo",
    srcs = ["sample_sequential_monte_carlo.py"],
    deps = [
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:random_walk_metropolis",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "sample_sequential_monte_carlo_test",
    size = "medium",
    timeout = "long",
    srcs = ["sample_sequential_monte_carlo_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 4,
    deps = [
        ":sample_sequential_monte_carlo",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:half_normal",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:mixture",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/experimental/mcmc:sequential_monte_carlo_kernel",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:nuts",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "sequential_monte_carlo_kernel",
    srcs = ["sequential_monte_carlo_kernel.py"],
    deps = [
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "sequential_monte_carlo_kernel_test",
    size = "medium",
    srcs = ["sequential_monte_carlo_kernel_test.py"],
    deps = [
        ":sequential_monte_carlo_kernel",
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "weighted_resampling",
    srcs = ["weighted_resampling.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:exponential",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "weighted_resampling_test",
    size = "medium",
    srcs = ["weighted_resampling_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 2,
    deps = [
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:chi2",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "step",
    srcs = ["step.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:loop_util",
    ],
)

multi_substrate_py_test(
    name = "step_test",
    size = "small",
    srcs = ["step_test.py"],
    shard_count = 5,
    deps = [
        ":step",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:sample",
    ],
)

multi_substrate_py_library(
    name = "reducer",
    srcs = ["reducer.py"],
    deps = [
        # six dep,
    ],
)

multi_substrate_py_library(
    name = "covariance_reducer",
    srcs = ["covariance_reducer.py"],
    deps = [
        ":reducer",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/stats",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "covariance_reducer_test",
    size = "small",
    srcs = ["covariance_reducer_test.py"],
    deps = [
        ":covariance_reducer",
        ":step",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "with_reductions",
    srcs = ["with_reductions.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "with_reductions_test",
    size = "medium",
    srcs = ["with_reductions_test.py"],
    shard_count = 2,
    deps = [
        ":covariance_reducer",
        ":step",
        ":with_reductions",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:sample",
        "//tensorflow_probability/python/mcmc:simple_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
    ],
)

multi_substrate_py_library(
    name = "sample",
    srcs = ["sample.py"],
    deps = [
        ":step",
        ":tracing_reducer",
        ":with_reductions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "sample_test",
    size = "small",
    srcs = ["sample_test.py"],
    deps = [
        ":sample",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sample_fold",
    srcs = ["sample_fold.py"],
    deps = [
        ":sample",
        ":sample_discarding_kernel",
        ":step",
        ":thinning_kernel",
        ":tracing_reducer",
        ":with_reductions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "sample_fold_test",
    size = "small",
    srcs = ["sample_fold_test.py"],
    shard_count = 5,
    deps = [
        ":covariance_reducer",
        ":expectations_reducer",
        ":sample_fold",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:hmc",
    ],
)

multi_substrate_py_library(
    name = "sample_discarding_kernel",
    srcs = ["sample_discarding_kernel.py"],
    deps = [
        ":step",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "sample_discarding_kernel_test",
    size = "small",
    srcs = ["sample_discarding_kernel_test.py"],
    deps = [
        ":covariance_reducer",
        ":sample_discarding_kernel",
        ":with_reductions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "thermodynamic_integrals",
    srcs = ["thermodynamic_integrals.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "thermodynamic_integrals_test",
    size = "medium",
    srcs = ["thermodynamic_integrals_test.py"],
    deps = [
        ":thermodynamic_integrals",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:diagnostic",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:replica_exchange_mc",
        "//tensorflow_probability/python/mcmc:sample",
    ],
)

multi_substrate_py_library(
    name = "thinning_kernel",
    srcs = ["thinning_kernel.py"],
    deps = [
        ":step",
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "thinning_kernel_test",
    size = "small",
    srcs = ["thinning_kernel_test.py"],
    deps = [
        ":covariance_reducer",
        ":thinning_kernel",
        ":with_reductions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "tracing_reducer",
    srcs = ["tracing_reducer.py"],
    deps = [
        ":reducer",
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "tracing_reducer_test",
    size = "small",
    srcs = ["tracing_reducer_test.py"],
    deps = [
        ":sample_fold",
        ":tracing_reducer",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "expectations_reducer",
    srcs = ["expectations_reducer.py"],
    deps = [
        ":reducer",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/stats:sample_stats",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "expectations_reducer_test",
    srcs = ["expectations_reducer_test.py"],
    shard_count = 5,
    deps = [
        ":expectations_reducer",
        ":step",
        ":with_reductions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "potential_scale_reduction_reducer",
    srcs = ["potential_scale_reduction_reducer.py"],
    deps = [
        ":reducer",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/stats",
    ],
)

multi_substrate_py_test(
    name = "potential_scale_reduction_reducer_test",
    size = "small",
    srcs = ["potential_scale_reduction_reducer_test.py"],
    deps = [
        ":potential_scale_reduction_reducer",
        ":sample_fold",
        ":tracing_reducer",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:diagnostic",
        "//tensorflow_probability/python/mcmc:hmc",
    ],
)

multi_substrate_py_library(
    name = "gradient_based_trajectory_length_adaptation",
    srcs = ["gradient_based_trajectory_length_adaptation.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc:simple_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "gradient_based_trajectory_length_adaptation_test",
    size = "large",
    srcs = ["gradient_based_trajectory_length_adaptation_test.py"],
    disabled_substrates = ["numpy"],  # Mostly gradient-based tests.
    shard_count = 3,
    deps = [
        ":gradient_based_trajectory_length_adaptation",
        ":preconditioned_hmc",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/distributions:half_normal",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc:dual_averaging_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:sample",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
    ],
)

multi_substrate_py_library(
    name = "snaper_hmc",
    srcs = ["snaper_hmc.py"],
    deps = [
        ":gradient_based_trajectory_length_adaptation",
        ":preconditioned_hmc",
        ":preconditioning_utils",
        ":reducer",
        ":sample_discarding_kernel",
        ":sharded",
        ":thinning_kernel",
        ":with_reductions",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/bijectors:joint_map",
        "//tensorflow_probability/python/bijectors:restructure",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:loop_util",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc:dual_averaging_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "snaper_hmc_test",
    size = "large",
    srcs = ["snaper_hmc_test.py"],
    disabled_substrates = ["numpy"],  # Mostly gradient-based tests.
    shard_count = 6,
    deps = [
        ":potential_scale_reduction_reducer",
        ":reducer",
        ":sharded",
        ":snaper_hmc",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:exponential",
        "//tensorflow_probability/python/distributions:gaussian_process",
        "//tensorflow_probability/python/distributions:half_cauchy",
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:lognormal",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/experimental/distribute:joint_distribution",
        "//tensorflow_probability/python/experimental/distribute:sharded",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/math/psd_kernels:matern",
        "//tensorflow_probability/python/mcmc:dual_averaging_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:sample",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "windowed_sampling",
    srcs = ["windowed_sampling.py"],
    deps = [
        ":diagonal_mass_matrix_adaptation",
        ":initialization",
        ":preconditioned_hmc",
        ":preconditioned_nuts",
        ":preconditioning_utils",
        ":sharded",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:joint_map",
        "//tensorflow_probability/python/bijectors:reshape",
        "//tensorflow_probability/python/bijectors:restructure",
        "//tensorflow_probability/python/experimental/stats:sample_stats",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc:dual_averaging_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:sample",
    ],
)

multi_substrate_py_test(
    name = "windowed_sampling_test",
    size = "large",
    srcs = ["windowed_sampling_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 12,
    tags = [
        "no-oss-ci",  # Prohibitively slow.
        "requires-mem:16g",
    ],
    deps = [
        ":windowed_sampling",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:pad",
        "//tensorflow_probability/python/distributions:autoregressive",
        "//tensorflow_probability/python/distributions:beta_binomial",
        "//tensorflow_probability/python/distributions:dirichlet",
        "//tensorflow_probability/python/distributions:half_normal",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:multinomial",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/experimental/distribute",
        "//tensorflow_probability/python/internal:callable_util",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sharded",
    srcs = ["sharded.py"],
    deps = [
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/mcmc:kernel",
    ],
)

multi_substrate_py_test(
    name = "sharded_test",
    size = "small",
    srcs = ["sharded_test.py"],
    deps = [
        ":sharded",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:kernel",
    ],
)
