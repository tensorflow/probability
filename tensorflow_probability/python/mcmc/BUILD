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
#   MCMC methods, diagnostics, and related utilities.

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
        # gradient-based kernels
        ":hmc",
        ":langevin",
        ":nuts",
    ],
    deps = [
        ":diagnostic",
        ":dual_averaging_step_size_adaptation",
        ":hmc",
        ":kernel",
        ":langevin",
        ":metropolis_hastings",
        ":nuts",
        ":random_walk_metropolis",
        ":replica_exchange_mc",
        ":sample",
        ":sample_annealed_importance",
        ":sample_halton_sequence",
        ":simple_step_size_adaptation",
        ":slice_sampler_kernel",
        ":transformed_kernel",
    ],
)

multi_substrate_py_library(
    name = "diagnostic",
    srcs = ["diagnostic.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/stats",
    ],
)

multi_substrate_py_test(
    name = "diagnostic_test",
    srcs = ["diagnostic_test.py"],
    jax_size = "medium",
    deps = [
        ":diagnostic",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "dual_averaging_step_size_adaptation",
    srcs = ["dual_averaging_step_size_adaptation.py"],
    deps = [
        ":kernel",
        ":simple_step_size_adaptation",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "dual_averaging_step_size_adaptation_test",
    size = "medium",
    srcs = ["dual_averaging_step_size_adaptation_test.py"],
    numpy_tags = ["notap"],
    shard_count = 4,
    deps = [
        ":dual_averaging_step_size_adaptation",
        ":hmc",
        ":kernel",
        ":nuts",
        ":sample",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/experimental/mcmc:preconditioned_nuts",
        "//tensorflow_probability/python/experimental/mcmc:sharded",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "hmc",
    srcs = ["hmc.py"],
    deps = [
        ":kernel",
        ":metropolis_hastings",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/mcmc/internal:leapfrog_integrator",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "hmc_test",
    size = "medium",
    timeout = "long",
    srcs = ["hmc_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 6,
    deps = [
        ":hmc",
        ":sample",
        ":simple_step_size_adaptation",
        ":transformed_kernel",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/util:deferred_tensor",
    ],
)

multi_substrate_py_library(
    name = "nuts",
    srcs = ["nuts.py"],
    deps = [
        ":kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc/internal:leapfrog_integrator",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_library(
    name = "nuts_testlib",
    testonly = 1,
    deps = [
        ":diagnostic",
        ":dual_averaging_step_size_adaptation",
        ":nuts",
        ":sample",
        ":transformed_kernel",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:half_normal",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/distributions:sigmoid_beta",
        "//tensorflow_probability/python/distributions:student_t",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/stats:sample_stats",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_test(
    name = "nuts_test",
    size = "large",
    srcs = ["nuts_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 7,
    tags = ["nozapfhahn"],
    deps = [
        ":nuts_testlib",
        # tensorflow dep,
    ],
)

py_test(
    name = "nuts_test_gpu",
    size = "large",
    srcs = ["nuts_test.py"],
    main = "nuts_test.py",
    shard_count = 7,
    tags = [
        "notap",
        "nozapfhahn",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":nuts_testlib",
    ],
)

multi_substrate_py_library(
    name = "langevin",
    srcs = ["langevin.py"],
    deps = [
        ":kernel",
        ":metropolis_hastings",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:diag_jacobian",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "langevin_test",
    size = "medium",
    timeout = "long",
    srcs = ["langevin_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 3,
    deps = [
        ":langevin",
        ":sample",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "kernel",
    srcs = ["kernel.py"],
    deps = [
        # six dep,
    ],
)

multi_substrate_py_library(
    name = "metropolis_hastings",
    srcs = ["metropolis_hastings.py"],
    deps = [
        ":kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "metropolis_hastings_test",
    srcs = ["metropolis_hastings_test.py"],
    deps = [
        ":kernel",
        ":metropolis_hastings",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "random_walk_metropolis",
    srcs = ["random_walk_metropolis.py"],
    deps = [
        ":kernel",
        ":metropolis_hastings",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "random_walk_metropolis_test",
    size = "medium",
    srcs = ["random_walk_metropolis_test.py"],
    shard_count = 2,
    deps = [
        ":random_walk_metropolis",
        ":sample",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:cauchy",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "replica_exchange_mc",
    srcs = ["replica_exchange_mc.py"],
    deps = [
        ":kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:broadcast_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "replica_exchange_mc_test",
    size = "medium",
    timeout = "long",
    srcs = ["replica_exchange_mc_test.py"],
    shard_count = 11,
    deps = [
        ":diagnostic",
        ":hmc",
        ":random_walk_metropolis",
        ":replica_exchange_mc",
        ":sample",
        ":simple_step_size_adaptation",
        # absl/logging dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/stats:sample_stats",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "sample",
    srcs = ["sample.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:loop_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "sample_test",
    size = "medium",
    timeout = "long",
    srcs = ["sample_test.py"],
    shard_count = 3,
    deps = [
        ":hmc",
        ":kernel",
        ":nuts",
        ":random_walk_metropolis",
        ":sample",
        ":simple_step_size_adaptation",
        ":transformed_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:pad",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:exponential",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:multinomial",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/experimental/mcmc:diagonal_mass_matrix_adaptation",
        "//tensorflow_probability/python/experimental/mcmc:preconditioned_hmc",
        "//tensorflow_probability/python/experimental/stats:sample_stats",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/random:random_ops",
    ],
)

multi_substrate_py_library(
    name = "sample_annealed_importance",
    srcs = ["sample_annealed_importance.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "sample_annealed_importance_test",
    size = "medium",
    srcs = ["sample_annealed_importance_test.py"],
    numpy_tags = ["notap"],
    shard_count = 3,
    deps = [
        ":hmc",
        ":sample_annealed_importance",
        ":transformed_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sample_halton_sequence",
    srcs = ["sample_halton_sequence.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "sample_halton_sequence_test",
    size = "medium",
    timeout = "long",
    srcs = ["sample_halton_sequence_test.py"],
    numpy_size = "large",
    numpy_tags = [
        "requires-mem:16g",
        "no-oss-ci",
    ],
    deps = [
        ":sample_halton_sequence",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:monte_carlo",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "slice_sampler_kernel",
    srcs = ["slice_sampler_kernel.py"],
    deps = [
        ":kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/mcmc/internal:slice_sampler_utils",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/random:random_ops",
    ],
)

multi_substrate_py_test(
    name = "slice_sampler_test",
    size = "medium",
    timeout = "long",
    srcs = ["slice_sampler_test.py"],
    numpy_tags = ["notap"],  # too slow.
    shard_count = 6,
    deps = [
        ":random_walk_metropolis",
        ":sample",
        ":slice_sampler_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc/internal",
        "//tensorflow_probability/python/stats:sample_stats",
    ],
)

multi_substrate_py_library(
    name = "transformed_kernel",
    srcs = ["transformed_kernel.py"],
    deps = [
        ":kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "transformed_kernel_test",
    size = "medium",
    timeout = "long",
    srcs = ["transformed_kernel_test.py"],
    shard_count = 6,
    deps = [
        ":hmc",
        ":kernel",
        ":langevin",
        ":random_walk_metropolis",
        ":sample",
        ":simple_step_size_adaptation",
        ":transformed_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:cholesky_lkj",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "eight_schools_hmc",
    srcs = ["eight_schools_hmc.py"],
    deps = [
        ":hmc",
        ":sample",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:normal",
    ],
)

# bazel run -c opt :eight_schools_hmc_eager_test -- --benchmarks=.
py_test(
    name = "eight_schools_hmc_eager_test",
    size = "medium",  # Might run > 1 minute.
    srcs = ["eight_schools_hmc_eager_test.py"],
    deps = [
        ":eight_schools_hmc",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

# bazel run -c opt :eight_schools_hmc_graph_test -- --benchmarks=.
py_test(
    name = "eight_schools_hmc_graph_test",
    size = "medium",  # Might run > 1 minute.
    srcs = ["eight_schools_hmc_graph_test.py"],
    deps = [
        ":eight_schools_hmc",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "simple_step_size_adaptation",
    srcs = ["simple_step_size_adaptation.py"],
    deps = [
        ":kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:unnest",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

multi_substrate_py_test(
    name = "simple_step_size_adaptation_test",
    size = "medium",
    srcs = ["simple_step_size_adaptation_test.py"],
    deps = [
        ":hmc",
        ":kernel",
        ":sample",
        ":simple_step_size_adaptation",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/experimental/mcmc:sharded",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:generic",
    ],
)
