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
# tfp.experimental distributions library.

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
    name = "bijectors",
    srcs = ["__init__.py"],
    deps = [
        ":distribution_bijectors",
        ":highway_flow",
        ":scalar_function_with_inferred_inverse",
        ":sharded",
    ],
)

multi_substrate_py_library(
    name = "scalar_function_with_inferred_inverse",
    srcs = ["scalar_function_with_inferred_inverse.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/internal:custom_gradient",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/math:root_search",
    ],
)

multi_substrate_py_test(
    name = "scalar_function_with_inferred_inverse_test",
    size = "medium",
    srcs = ["scalar_function_with_inferred_inverse_test.py"],
    jax_size = "medium",
    deps = [
        ":scalar_function_with_inferred_inverse",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:power",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:student_t",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "distribution_bijectors",
    srcs = ["distribution_bijectors.py"],
    deps = [
        ":scalar_function_with_inferred_inverse",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:inline",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:normal_cdf",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:joint_distribution",
        "//tensorflow_probability/python/distributions:markov_chain",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/distributions:uniform",
    ],
)

multi_substrate_py_test(
    name = "distribution_bijectors_test",
    size = "medium",
    srcs = ["distribution_bijectors_test.py"],
    shard_count = 5,
    deps = [
        ":distribution_bijectors",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:exponential",
        "//tensorflow_probability/python/distributions:hypothesis_testlib",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:markov_chain",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/experimental/vi:automatic_structured_vi",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/mcmc:dual_averaging_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:nuts",
        "//tensorflow_probability/python/mcmc:sample",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
        "//tensorflow_probability/python/vi:optimization",
    ],
)

multi_substrate_py_library(
    name = "highway_flow",
    srcs = ["highway_flow.py"],
    deps = [
        ":scalar_function_with_inferred_inverse",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/bijectors:fill_scale_tril",
        "//tensorflow_probability/python/bijectors:fill_triangular",
        "//tensorflow_probability/python/bijectors:pad",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/bijectors:transform_diagonal",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/util:deferred_tensor",
    ],
)

multi_substrate_py_test(
    name = "highway_flow_test",
    size = "medium",
    srcs = ["highway_flow_test.py"],
    disabled_substrates = ["numpy"],
    jax_size = "medium",
    deps = [
        ":highway_flow",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector_test_util",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "sharded",
    srcs = ["sharded.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:log_prob_ratio",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:samplers",
    ],
)

multi_substrate_py_test(
    name = "sharded_test",
    size = "medium",
    srcs = ["sharded_test.py"],
    disabled_substrates = ["numpy"],
    deps = [
        ":sharded",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/experimental/distribute:sharded",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)
