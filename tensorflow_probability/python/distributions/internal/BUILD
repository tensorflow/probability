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
    name = "init",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "moving_stats",
    srcs = ["moving_stats.py"],
    deps = [
        ":init",
        # tensorflow dep,
    ],
)

py_test(
    name = "moving_stats_test",
    size = "medium",
    srcs = ["moving_stats_test.py"],
    tags = [
        "nomsan",  # disable to avoid false positives from scipy.
    ],
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
        ":init",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:seed_stream",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
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
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "correlation_matrix_volumes_py",
    srcs = [
        "correlation_matrix_volumes_lib.py",
    ],
    deps = [
        ":init",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_binary(
    name = "correlation_matrix_volumes",
    srcs = ["correlation_matrix_volumes.py"],
    python_version = "PY2",
    deps = [
        ":correlation_matrix_volumes_py",
        ":init",
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
        # tensorflow dep,
    ],
)

py_library(
    name = "slicing",
    srcs = ["slicing.py"],
    deps = [
        ":init",
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "slicing_test",
    srcs = ["slicing_test.py"],
    deps = [
        ":slicing",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)
