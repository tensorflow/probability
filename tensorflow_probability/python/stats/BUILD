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
#   Statistical Ops.

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
    name = "stats",
    srcs = ["__init__.py"],
    jax_omit_deps = [
        ":ranking",
    ],
    deps = [
        ":calibration",
        ":kendalls_tau",
        ":leave_one_out",
        ":moving_stats",
        ":quantiles",
        ":ranking",
        ":sample_stats",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "leave_one_out",
    srcs = ["leave_one_out.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:generic",
    ],
)

multi_substrate_py_test(
    name = "leave_one_out_test",
    size = "small",
    srcs = ["leave_one_out_test.py"],
    jax_size = "medium",
    deps = [
        ":leave_one_out",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "moving_stats",
    srcs = ["moving_stats.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "moving_stats_test",
    size = "medium",
    srcs = ["moving_stats_test.py"],
    jax_tags = ["notap"],
    deps = [
        ":moving_stats",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "ranking",
    srcs = ["ranking.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "ranking_test",
    size = "medium",
    srcs = ["ranking_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        ":quantiles",
        ":ranking",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:truncated_normal",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sample_stats",
    srcs = ["sample_stats.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:generic",
    ],
)

multi_substrate_py_test(
    name = "sample_stats_test",
    size = "small",
    srcs = ["sample_stats_test.py"],
    shard_count = 5,
    deps = [
        ":sample_stats",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_library(
    name = "quantiles",
    srcs = ["quantiles.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "quantiles_test",
    size = "medium",
    srcs = ["quantiles_test.py"],
    numpy_tags = ["notap"],
    deps = [
        ":quantiles",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "calibration",
    srcs = ["calibration.py"],
    deps = [
        ":quantiles",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:generic",
    ],
)

multi_substrate_py_test(
    name = "calibration_test",
    size = "large",
    srcs = ["calibration_test.py"],
    numpy_tags = ["notap"],
    deps = [
        ":calibration",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "kendalls_tau",
    srcs = ["kendalls_tau.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "kendalls_tau_test",
    size = "medium",
    srcs = ["kendalls_tau_test.py"],
    jax_tags = ["notap"],
    deps = [
        ":kendalls_tau",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

exports_files(
    [
        "__init__.py",
        "calibration.py",
        "kendalls_tau.py",
        "leave_one_out.py",
        "moving_stats.py",
        "quantiles.py",
        "ranking.py",
        "sample_stats.py",
    ],
    visibility = ["//tensorflow_probability:__subpackages__"],
)
