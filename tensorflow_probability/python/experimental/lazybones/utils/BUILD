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
#   Utilities for lazybones symbolic execution/analysis.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":special_methods",
        ":utils_impl",
        ":weak_container",
    ],
)

py_library(
    name = "special_methods",
    srcs = ["special_methods.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "utils_impl",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_probability/python/experimental/lazybones:deferred",
    ],
)

# py2and3
py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    deps = [
        # absl/testing:absltest dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "weak_container",
    srcs = ["weak_container.py"],
    srcs_version = "PY2AND3",
)
