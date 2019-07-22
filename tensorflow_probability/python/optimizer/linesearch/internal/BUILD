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
# Internal implementation details of line search optimization methods

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "internal",
    srcs = ["__init__.py"],
    deps = [
        ":hager_zhang_lib",
    ],
)

py_library(
    name = "hager_zhang_lib",
    srcs = ["hager_zhang_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "hager_zhang_lib_test",
    srcs = ["hager_zhang_lib_test.py"],
    deps = [
        ":hager_zhang_lib",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/math:gradient",
    ],
)
