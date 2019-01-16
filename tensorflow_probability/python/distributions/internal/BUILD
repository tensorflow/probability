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
#   Internal helper libraries for distributions.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "internal",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":moving_stats",
        ":statistical_testing",
    ],
)

py_library(
    name = "moving_stats",
    srcs = ["moving_stats.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "moving_stats_test",
    size = "small",
    srcs = ["moving_stats_test.py"],
    tags = ["nomsan"],  # disable to avoid false positives from scipy.
    deps = [
        ":moving_stats",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "statistical_testing",
    srcs = ["statistical_testing.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_test(
    name = "statistical_testing_test",
    size = "medium",
    srcs = ["statistical_testing_test.py"],
    shard_count = 4,
    deps = [
        ":statistical_testing",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "correlation_matrix_volumes_py",
    srcs = [
        "correlation_matrix_volumes_lib.py",
    ],
    deps = [
        # numpy dep,
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_binary(
    name = "correlation_matrix_volumes",
    srcs = [
        "correlation_matrix_volumes.py",
    ],
    deps = [
        ":correlation_matrix_volumes_py",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "correlation_matrix_volumes_test",
    size = "medium",
    srcs = ["correlation_matrix_volumes_test.py"],
    shard_count = 4,
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":correlation_matrix_volumes_py",
        ":statistical_testing",
        # numpy dep,
    ],
)
