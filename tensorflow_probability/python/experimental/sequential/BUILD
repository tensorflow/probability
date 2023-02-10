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
# Description:
#   Tools for sequential inference/filtering.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    default_applicable_licenses = ["//tensorflow_probability:license"],
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "sequential",
    srcs = ["__init__.py"],
    deps = [
        ":ensemble_adjustment_kalman_filter",
        ":ensemble_kalman_filter",
        ":extended_kalman_filter",
        ":iterated_filter",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "extended_kalman_filter",
    srcs = ["extended_kalman_filter.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:linear_gaussian_ssm",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_library(
    name = "ensemble_kalman_filter",
    srcs = ["ensemble_kalman_filter.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_linear_operator",
        "//tensorflow_probability/python/distributions:mvn_low_rank_update_linear_operator_covariance",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "ensemble_adjustment_kalman_filter",
    srcs = ["ensemble_adjustment_kalman_filter.py"],
    deps = [
        ":ensemble_kalman_filter",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_test(
    name = "extended_kalman_filter_test",
    size = "large",
    timeout = "long",
    srcs = ["extended_kalman_filter_test.py"],

    # shard_count = 6,
    deps = [
        ":extended_kalman_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "ensemble_kalman_filter_test",
    size = "medium",
    srcs = ["ensemble_kalman_filter_test.py"],
    shard_count = 3,
    deps = [
        ":ensemble_kalman_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/stats:sample_stats",
    ],
)

multi_substrate_py_test(
    name = "ensemble_adjustment_kalman_filter_test",
    size = "medium",
    srcs = ["ensemble_adjustment_kalman_filter_test.py"],
    deps = [
        ":ensemble_adjustment_kalman_filter",
        ":ensemble_kalman_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "iterated_filter",
    srcs = ["iterated_filter.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/experimental/mcmc:particle_filter",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "iterated_filter_test",
    size = "medium",
    srcs = ["iterated_filter_test.py"],
    jax_size = "large",
    shard_count = 3,
    deps = [
        ":iterated_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
