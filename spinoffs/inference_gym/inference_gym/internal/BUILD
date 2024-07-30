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

# [internal] load pytype.bzl (pytype_library, pytype_strict_library)
# [internal] load strict.bzl
# Placeholder: py_library

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
        # tensorflow_probability/opensource/tools/inference_gym_ground_truth:__subpackages__ dep,
        "//inference_gym:__subpackages__",
    ],
)

licenses(["notice"])

# pytype_strict
py_library(
    name = "internal",
    srcs = ["__init__.py"],
    deps = [
        ":array_to_source",
        ":data",
    ],
)

# pytype_strict
py_library(
    name = "array_to_source",
    srcs = ["array_to_source.py"],
    deps = [
        # numpy dep,
    ],
)

# py_strict
py_test(
    name = "array_to_source_test",
    srcs = ["array_to_source_test.py"],
    deps = [
        ":array_to_source",
        # tensorflow_probability/python/internal:test_util dep,
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        # numpy dep,
        "//inference_gym/internal/datasets",
    ],
)

# py_strict
py_test(
    name = "data_test",
    srcs = ["data_test.py"],
    deps = [
        ":data",
        ":test_util",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        # tensorflow_probability/python/internal:test_util dep,
        # tensorflow/compiler/jit dep,
    ],
)

# pytype_strict
py_library(
    name = "ground_truth_encoding",
    srcs = ["ground_truth_encoding.py"],
    deps = [
        ":array_to_source",
    ],
)

# py_strict
py_test(
    name = "ground_truth_encoding_test",
    srcs = ["ground_truth_encoding_test.py"],
    deps = [
        ":ground_truth_encoding",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow_probability/python/internal:test_util dep,
    ],
)

# Not strict due to the special .jax etc dependencies.
# pytype
py_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.py"],
    deps = [
        # tensorflow dep,
        # tensorflow_probability dep,
        # tensorflow_probability/python/internal:test_util dep,
        # tensorflow_probability/python/internal:test_util.jax dep,
        # tensorflow_probability/python/internal:test_util.jax.raw dep,
        # tensorflow_probability/python/internal:test_util.numpy dep,
        # tensorflow_probability/python/internal:test_util.numpy.raw dep,
        "//inference_gym:using_jax",
        "//inference_gym:using_numpy",
        "//inference_gym:using_tensorflow",
        # tensorflow/compiler/jit dep,
    ],
)

# py_strict
py_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    deps = [
        ":test_util",
        # absl/logging dep,
        # jax dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/targets",
    ],
)
