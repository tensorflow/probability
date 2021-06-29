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

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "mcmc",
    srcs = ["__init__.py"],
    numpy_omit_deps = [
        ":preconditioned_hmc",
    ],
    srcs_version = "PY3",
    substrates_omit_deps = [
        ":covariance_reducer",
        ":elliptical_slice_sampler",
        ":expectations_reducer",
        ":kernel_builder",
        ":kernel_outputs",
        ":nuts_autobatching",
        ":particle_filter_augmentation",
        ":potential_scale_reduction_reducer",
        ":progress_bar_reducer",
        ":reducer",
        ":sample",
        ":step",
        ":sample_discarding_kernel",
        ":sample_fold",
        ":thinning_kernel",
        ":tracing_reducer",
        ":with_reductions",
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
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
    ],
)

multi_substrate_py_test(
    name = "initialization_test",
    srcs = ["initialization_test.py"],
    srcs_version = "PY3",
    deps = [
        ":initialization",
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "kernel_builder",
    srcs = ["kernel_builder.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":kernel_builder",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "diagonal_mass_matrix_adaptation",
    srcs = ["diagonal_mass_matrix_adaptation.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "preconditioning_utils",
    srcs = ["preconditioning_utils.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:reshape",
        "//tensorflow_probability/python/distributions:batch_broadcast",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/experimental/distribute:joint_distribution",
        "//tensorflow_probability/python/experimental/distribute:sharded",
        "//tensorflow_probability/python/experimental/distributions:mvn_precision_factor_linop",
        "//tensorflow_probability/python/internal:auto_composite_tensor",
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
    srcs_version = "PY3",
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
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "preconditioned_hmc",
    srcs = ["preconditioned_hmc.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "progress_bar_reducer",
    srcs = ["progress_bar_reducer.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "elliptical_slice_sampler",
    srcs = ["elliptical_slice_sampler.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    shard_count = 6,
    srcs_version = "PY3",
    deps = [
        ":elliptical_slice_sampler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "nuts_autobatching",
    srcs = ["nuts_autobatching.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/math",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/random",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

py_library(
    name = "nuts_autobatching_testlib",
    testonly = 1,
    srcs = ["nuts_autobatching_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "nuts_autobatching_test",
    size = "medium",
    srcs = ["nuts_autobatching_test.py"],
    python_version = "PY3",
    shard_count = 7,
    srcs_version = "PY3",
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
    python_version = "PY3",
    shard_count = 7,
    srcs_version = "PY3",
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
    python_version = "PY3",
    shard_count = 1,
    srcs_version = "PY3",
    tags = [
        "nozapfhahn",
    ],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_test(
    name = "nuts_autobatching_xla_test_gpu",
    size = "large",
    srcs = ["nuts_autobatching_xla_test.py"],
    args = ["--test_device=gpu"],
    main = "nuts_autobatching_xla_test.py",
    python_version = "PY3",
    shard_count = 1,
    srcs_version = "PY3",
    tags = [
        "nozapfhahn",
        "requires-gpu-nvidia",
    ],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "particle_filter",
    srcs = ["particle_filter.py"],
    srcs_version = "PY3",
    deps = [
        ":sequential_monte_carlo_kernel",
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:docstring_util",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":particle_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_library(
    name = "particle_filter_augmentation",
    srcs = ["particle_filter_augmentation.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":particle_filter_augmentation",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sample_sequential_monte_carlo",
    srcs = ["sample_sequential_monte_carlo.py"],
    srcs_version = "PY3",
    deps = [
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
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
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "sequential_monte_carlo_kernel",
    srcs = ["sequential_monte_carlo_kernel.py"],
    srcs_version = "PY3",
    deps = [
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "sequential_monte_carlo_kernel_test",
    size = "medium",
    srcs = ["sequential_monte_carlo_kernel_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "weighted_resampling",
    srcs = ["weighted_resampling.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "weighted_resampling_test",
    size = "medium",
    srcs = ["weighted_resampling_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 2,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_library(
    name = "step",
    srcs = ["step.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

py_test(
    name = "step_test",
    size = "small",
    srcs = ["step_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "reducer",
    srcs = ["reducer.py"],
    srcs_version = "PY3",
    deps = [
        # six dep,
    ],
)

py_library(
    name = "covariance_reducer",
    srcs = ["covariance_reducer.py"],
    srcs_version = "PY3",
    deps = [
        ":reducer",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/stats",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

py_test(
    name = "covariance_reducer_test",
    size = "small",
    srcs = ["covariance_reducer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":covariance_reducer",
        ":step",
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "with_reductions",
    srcs = ["with_reductions.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

py_test(
    name = "with_reductions_test",
    srcs = ["with_reductions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "sample",
    srcs = ["sample.py"],
    srcs_version = "PY3",
    deps = [
        ":step",
        ":tracing_reducer",
        ":with_reductions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

py_test(
    name = "sample_test",
    size = "small",
    srcs = ["sample_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":step",
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "sample_fold",
    srcs = ["sample_fold.py"],
    srcs_version = "PY3",
    deps = [
        ":sample_discarding_kernel",
        ":step",
        ":thinning_kernel",
        ":tracing_reducer",
        ":with_reductions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

py_test(
    name = "sample_fold_test",
    size = "small",
    srcs = ["sample_fold_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "sample_discarding_kernel",
    srcs = ["sample_discarding_kernel.py"],
    srcs_version = "PY3",
    deps = [
        ":step",
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

py_test(
    name = "sample_discarding_kernel_test",
    size = "small",
    srcs = ["sample_discarding_kernel_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "thermodynamic_integrals",
    srcs = ["thermodynamic_integrals.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "thermodynamic_integrals_test",
    size = "medium",
    srcs = ["thermodynamic_integrals_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":thermodynamic_integrals",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "thinning_kernel",
    srcs = ["thinning_kernel.py"],
    srcs_version = "PY3",
    deps = [
        ":step",
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

py_test(
    name = "thinning_kernel_test",
    size = "small",
    srcs = ["thinning_kernel_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":thinning_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "tracing_reducer",
    srcs = ["tracing_reducer.py"],
    srcs_version = "PY3",
    deps = [
        ":reducer",
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

py_test(
    name = "tracing_reducer_test",
    size = "small",
    srcs = ["tracing_reducer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "expectations_reducer",
    srcs = ["expectations_reducer.py"],
    srcs_version = "PY3",
    deps = [
        ":reducer",
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

py_test(
    name = "expectations_reducer_test",
    srcs = ["expectations_reducer_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "potential_scale_reduction_reducer",
    srcs = ["potential_scale_reduction_reducer.py"],
    srcs_version = "PY3",
    deps = [
        ":reducer",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/stats",
    ],
)

py_test(
    name = "potential_scale_reduction_reducer_test",
    size = "small",
    srcs = ["potential_scale_reduction_reducer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc/internal:test_fixtures",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "gradient_based_trajectory_length_adaptation",
    srcs = ["gradient_based_trajectory_length_adaptation.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "windowed_sampling",
    srcs = ["windowed_sampling.py"],
    srcs_version = "PY3",
    deps = [
        ":diagonal_mass_matrix_adaptation",
        ":initialization",
        ":preconditioned_hmc",
        ":preconditioned_nuts",
        ":preconditioning_utils",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:joint_map",
        "//tensorflow_probability/python/bijectors:reshape",
        "//tensorflow_probability/python/bijectors:restructure",
        "//tensorflow_probability/python/experimental/stats:sample_stats",
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
    python_version = "PY3",
    shard_count = 12,
    srcs_version = "PY3",
    tags = [
        "no-oss-ci",  # Prohibitively slow.
        "requires-mem:16g",
    ],
    deps = [
        ":windowed_sampling",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sharded",
    srcs = ["sharded.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sharded",
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:kernel",
    ],
)
