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
    name = "convergence_criteria",
    srcs = ["__init__.py"],
    deps = [
        ":convergence_criterion",
        ":loss_not_decreasing",
        ":successive_gradients_are_uncorrelated",
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "convergence_criterion",
    srcs = ["convergence_criterion.py"],
    deps = [
        # six dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "loss_not_decreasing",
    srcs = ["loss_not_decreasing.py"],
    deps = [
        ":convergence_criterion",
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "successive_gradients_are_uncorrelated",
    srcs = ["successive_gradients_are_uncorrelated.py"],
    deps = [
        ":convergence_criterion",
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "loss_not_decreasing_test",
    size = "medium",
    srcs = ["loss_not_decreasing_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        ":loss_not_decreasing",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "successive_gradients_are_uncorrelated_test",
    size = "medium",
    srcs = ["successive_gradients_are_uncorrelated_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        ":successive_gradients_are_uncorrelated",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/util:deferred_tensor",
        "//tensorflow_probability/python/vi:csiszar_divergence",
    ],
)
