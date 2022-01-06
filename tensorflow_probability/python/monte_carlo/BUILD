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
#   Support for Monte Carlo expectations

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

multi_substrate_py_library(
    name = "monte_carlo",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":expectation",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "expectation",
    srcs = ["expectation.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:monte_carlo",
    ],
)

multi_substrate_py_test(
    name = "expectation_test",
    size = "small",
    srcs = ["expectation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
