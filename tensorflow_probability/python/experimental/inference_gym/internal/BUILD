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
#   Internal utilities for the inference gym.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_test",
    "multi_substrate_py_library",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
        "//tools/inference_gym_ground_truth:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "internal",
    srcs = ["__init__.py"],
    deps = [
        ":array_to_source",
        ":data",
    ],
)

multi_substrate_py_library(
    name = "array_to_source",
    srcs = ["array_to_source.py"],
    deps = [
        # numpy dep,
    ],
)

multi_substrate_py_test(
    name = "array_to_source_test",
    srcs = ["array_to_source_test.py"],
    deps = [
        ":array_to_source",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal/datasets",
    ],
)

multi_substrate_py_test(
    name = "data_test",
    srcs = ["data_test.py"],
    deps = [
        ":data",
        ":test_util",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
    ],
)

multi_substrate_py_library(
    name = "ground_truth_encoding",
    srcs = ["ground_truth_encoding.py"],
    deps = [
        ":array_to_source",
    ],
)

multi_substrate_py_test(
    name = "ground_truth_encoding_test",
    srcs = ["ground_truth_encoding_test.py"],
    deps = [
        ":ground_truth_encoding",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    deps = [
        ":test_util",
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
