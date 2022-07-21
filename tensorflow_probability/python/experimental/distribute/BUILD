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
#   Contains utilities for writing distributed TFP code.
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
    name = "distribute",
    srcs = ["__init__.py"],
    deps = [
        ":joint_distribution",
        ":sharded",
        "//tensorflow_probability/python/internal:distribute_lib",
    ],
)

multi_substrate_py_library(
    name = "sharded",
    srcs = ["sharded.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:log_prob_ratio",
        "//tensorflow_probability/python/experimental/bijectors:sharded",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:samplers",
    ],
)

multi_substrate_py_test(
    name = "diagonal_mass_matrix_adaptation_test",
    size = "large",
    srcs = ["diagonal_mass_matrix_adaptation_test.py"],
    disabled_substrates = ["numpy"],
    deps = [
        ":joint_distribution",
        ":sharded",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution",
    srcs = ["joint_distribution.py"],
    deps = [
        ":sharded",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/distributions:log_prob_ratio",
        "//tensorflow_probability/python/internal:auto_composite_tensor",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:samplers",
    ],
)

multi_substrate_py_test(
    name = "sharded_test",
    size = "medium",
    srcs = ["sharded_test.py"],
    disabled_substrates = ["numpy"],
    deps = [
        ":sharded",
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "joint_distribution_test",
    size = "large",
    srcs = ["joint_distribution_test.py"],
    disabled_substrates = ["numpy"],
    deps = [
        ":joint_distribution",
        ":sharded",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
