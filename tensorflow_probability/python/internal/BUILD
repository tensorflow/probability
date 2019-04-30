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
#   Internal utilities for TensorFlow probability.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//energy/plasma_debugger/util/tfp_internal:__subpackages__",
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "internal",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "assert_util",
    srcs = ["assert_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "dtype_util",
    srcs = ["dtype_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "dtype_util_test",
    size = "small",
    srcs = ["dtype_util_test.py"],
    deps = [
        ":dtype_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "distribution_util",
    srcs = ["distribution_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":assert_util",
        ":dtype_util",
        ":prefer_static",
        ":reparameterization",
        ":tensorshape_util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "distribution_util_test",
    size = "small",
    srcs = ["distribution_util_test.py"],
    deps = [
        ":distribution_util",
        ":tensorshape_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/math:gradient",
    ],
)

py_library(
    name = "monte_carlo",
    srcs = ["monte_carlo.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "nest_util",
    srcs = ["nest_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "nest_util_test",
    size = "small",
    srcs = ["nest_util_test.py"],
    deps = [
        ":nest_util",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "prefer_static",
    srcs = ["prefer_static.py"],
    deps = [
        ":tensorshape_util",
        # decorator dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "prefer_static_test",
    srcs = ["prefer_static_test.py"],
    deps = [
        ":prefer_static",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "reparameterization",
    srcs = ["reparameterization.py"],
)

py_library(
    name = "special_math",
    srcs = ["special_math.py"],
    deps = [
        ":dtype_util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "special_math_test",
    srcs = ["special_math_test.py"],
    deps = [
        ":special_math",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/math:gradient",
    ],
)

py_library(
    name = "tensorshape_util",
    srcs = ["tensorshape_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "test_case",
    testonly = 1,
    srcs = ["test_case.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "test_case_test",
    size = "small",
    srcs = ["test_case_test.py"],
    deps = [
        ":dtype_util",
        ":test_case",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtype_util",
        # absl/flags dep,
        # hypothesis dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:seed_stream",
    ],
)

py_test(
    name = "test_util_test",
    size = "small",
    srcs = ["test_util_test.py"],
    deps = [
        ":test_util",
        # absl/flags dep,
        # absl/testing:flagsaver dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:seed_stream",
    ],
)
