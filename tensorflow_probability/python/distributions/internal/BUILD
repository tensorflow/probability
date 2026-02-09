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

# Placeholder: py_binary
# Placeholder: py_library
# Placeholder: py_test
load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

multi_substrate_py_library(
    name = "init",
    srcs = ["__init__.py"],
)

multi_substrate_py_library(
    name = "statistical_testing",
    srcs = ["statistical_testing.py"],
    deps = [
        ":init",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "statistical_testing_test",
    size = "medium",
    srcs = ["statistical_testing_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    shard_count = 4,
    deps = [
        ":statistical_testing",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "stochastic_process_util",
    srcs = ["stochastic_process_util.py"],
    deps = [
        ":init",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

py_library(
    name = "correlation_matrix_volumes_lib",
#     testonly = 1,  # DisableOnExport
    srcs = [
        "correlation_matrix_volumes_lib.py",
    ],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":init",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:empirical_statistical_testing",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:linalg",
    ],
)

py_binary(
    name = "correlation_matrix_volumes",
#     testonly = 1,  # DisableOnExport
    srcs = ["correlation_matrix_volumes.py"],
    strict_deps = False,
    deps = [
        ":correlation_matrix_volumes_lib",
        ":init",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "correlation_matrix_volumes_test",
    size = "small",
    srcs = ["correlation_matrix_volumes_test.py"],
    shard_count = 4,
    strict_deps = False,
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":correlation_matrix_volumes_lib",
        ":statistical_testing",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)
