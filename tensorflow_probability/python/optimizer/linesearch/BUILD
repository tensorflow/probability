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
# Line search optimization methods

load("//tensorflow_probability/python:build_defs.bzl", "multi_substrate_py_test", "multi_substrate_py_library")

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "linesearch",
    srcs = ["__init__.py"],
    deps = [
        ":hager_zhang",
        "//tensorflow_probability/python/optimizer/linesearch/internal",
    ],
)

multi_substrate_py_library(
    name = "hager_zhang",
    srcs = ["hager_zhang.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "hager_zhang_test",
    size = "small",
    srcs = ["hager_zhang_test.py"],
    numpy_tags = ["notap"],
    shard_count = 5,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
