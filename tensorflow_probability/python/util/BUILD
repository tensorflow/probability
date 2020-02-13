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
#   TensorFlow Probability utility functions.

# [internal] load python3.bzl
load("//tensorflow_probability/python:build_defs.bzl", "multi_substrate_py_test", "multi_substrate_py_library")

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "util",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":deferred_tensor",
        ":seed_stream",
    ],
)

multi_substrate_py_library(
    name = "deferred_tensor",
    srcs = ["deferred_tensor.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:name_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

py_test(
    name = "deferred_tensor_test",
    size = "small",
    srcs = ["deferred_tensor_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "seed_stream",
    srcs = ["seed_stream.py"],
)

multi_substrate_py_test(
    name = "seed_stream_test",
    size = "small",
    srcs = ["seed_stream_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

exports_files(
    [
        "seed_stream.py",
        "seed_stream_test.py",
    ],
    visibility = ["//tensorflow_probability:__subpackages__"],
)
