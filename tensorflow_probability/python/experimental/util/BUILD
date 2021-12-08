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
#   TensorFlow Probability experimental utility functions.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "util",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    substrates_omit_deps = [
        ":composite_tensor",
        ":deferred_module",
        ":trainable",
    ],
    deps = [
        ":composite_tensor",
        ":deferred_module",
        ":jit_public_methods",
        ":trainable",
    ],
)

py_library(
    name = "deferred_module",
    srcs = ["deferred_module.py"],
    srcs_version = "PY3",
    deps = [
        ":special_methods",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:nest_util",
    ],
)

py_test(
    name = "deferred_module_test",
    size = "small",
    srcs = ["deferred_module_test.py"],
    python_version = "PY3",
    deps = [
        ":deferred_module",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_library(
    name = "jit_public_methods",
    srcs = ["jit_public_methods.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:kullback_leibler",
    ],
)

multi_substrate_py_test(
    name = "jit_public_methods_test",
    size = "small",
    srcs = ["jit_public_methods_test.py"],
    disabled_substrates = ["numpy"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

py_library(
    name = "special_methods",
    srcs = ["special_methods.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "trainable",
    srcs = ["trainable.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/util:deferred_tensor",
    ],
)

py_test(
    name = "trainable_test",
    srcs = ["trainable_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "composite_tensor",
    srcs = ["composite_tensor.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
    ],
)

py_test(
    name = "composite_tensor_test",
    srcs = ["composite_tensor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
