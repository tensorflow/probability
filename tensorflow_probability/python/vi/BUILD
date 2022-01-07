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
#   Methods and objectives for variational inference.

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
    name = "vi",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    substrates_omit_deps = [
        ":optimization",
        ":mutual_information",
    ],
    deps = [
        ":csiszar_divergence",
        ":mutual_information",
        ":optimization",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "csiszar_divergence",
    srcs = ["csiszar_divergence.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/monte_carlo",
        "//tensorflow_probability/python/stats:leave_one_out",
    ],
)

multi_substrate_py_test(
    name = "csiszar_divergence_test",
    size = "medium",
    srcs = ["csiszar_divergence_test.py"],
    jax_size = "medium",
    numpy_size = "medium",
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "mutual_information",
    srcs = ["mutual_information.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

py_test(
    name = "mutual_information_test",
    size = "small",
    srcs = ["mutual_information_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "optimization",
    srcs = ["optimization.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/monte_carlo",
    ],
)

py_test(
    name = "optimization_test",
    size = "medium",
    srcs = ["optimization_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
