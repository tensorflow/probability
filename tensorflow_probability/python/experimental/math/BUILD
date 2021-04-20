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
#   TensorFlow Probability experimental utility functions.

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "math",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":no_pivot_ldl",
    ],
)

py_library(
    name = "no_pivot_ldl",
    srcs = ["no_pivot_ldl.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "no_pivot_ldl_test",
    srcs = ["no_pivot_ldl_test.py"],
    python_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/math:no_pivot_ldl",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
