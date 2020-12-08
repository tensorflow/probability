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

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_library(
    name = "distribute",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":distribute_lib",
        ":joint_distribution",
        ":sharded",
    ],
)

py_library(
    name = "distribute_lib",
    srcs = ["distribute_lib.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "sharded",
    srcs = ["sharded.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
    ],
)

py_library(
    name = "joint_distribution",
    srcs = ["joint_distribution.py"],
    deps = [
        ":distribute_lib",
        ":sharded",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
    ],
)

py_test(
    name = "sharded_test",
    srcs = ["sharded_test.py"],
    python_version = "PY3",
    deps = [
        ":sharded",
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "joint_distribution_test",
    srcs = ["joint_distribution_test.py"],
    python_version = "PY3",
    deps = [
        ":joint_distribution",
        ":sharded",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "distribute_lib_test",
    srcs = ["distribute_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":distribute_lib",
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
