# Copyright 2019 The TensorFlow Probability Authors.
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
#   Convergence criteria for optimizations.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "convergence_criteria",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convergence_criterion",
        ":loss_not_decreasing",
        ":successive_gradients_are_uncorrelated",
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "convergence_criterion",
    srcs = ["convergence_criterion.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "loss_not_decreasing",
    srcs = ["loss_not_decreasing.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convergence_criterion",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "successive_gradients_are_uncorrelated",
    srcs = ["successive_gradients_are_uncorrelated.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convergence_criterion",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "loss_not_decreasing_test",
    size = "medium",
    srcs = ["loss_not_decreasing_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "successive_gradients_are_uncorrelated_test",
    size = "medium",
    srcs = ["successive_gradients_are_uncorrelated_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
