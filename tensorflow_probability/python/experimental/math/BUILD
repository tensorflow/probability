# Copyright 2021 The TensorFlow Probability Authors.
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
#   Experimental math.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "math",
    srcs = ["__init__.py"],
    deps = [
        ":manual_special_functions",
    ],
)

multi_substrate_py_library(
    name = "manual_special_functions",
    srcs = ["manual_special_functions.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:custom_gradient",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_test(
    name = "manual_special_functions_test",
    srcs = ["manual_special_functions_test.py"],
    tags = [
        "tf1-broken",
    ],
    deps = [
        ":manual_special_functions",
        # mock dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

exports_files(
    glob(["**/*.py"]),
    visibility = ["//tensorflow_probability:__subpackages__"],
)
