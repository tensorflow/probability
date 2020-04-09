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
#   API-unstable code that is part of the TFP package.

# [internal] load python3.bzl
load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    substrates_omit_deps = [
        ":composite_tensor",
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/experimental/edward2",
        "//tensorflow_probability/python/experimental/linalg",
        "//tensorflow_probability/python/experimental/marginalize",
        "//tensorflow_probability/python/experimental/mcmc",
        "//tensorflow_probability/python/experimental/nn",
        "//tensorflow_probability/python/experimental/sequential",
        "//tensorflow_probability/python/experimental/sts_gibbs",
        "//tensorflow_probability/python/experimental/substrates",
        "//tensorflow_probability/python/experimental/vi",
    ],
    deps = [
        ":composite_tensor",
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/experimental/edward2",
        "//tensorflow_probability/python/experimental/inference_gym",
        "//tensorflow_probability/python/experimental/linalg",
        "//tensorflow_probability/python/experimental/marginalize",
        "//tensorflow_probability/python/experimental/mcmc",
        "//tensorflow_probability/python/experimental/nn",
        "//tensorflow_probability/python/experimental/sequential",
        "//tensorflow_probability/python/experimental/sts_gibbs",
        "//tensorflow_probability/python/experimental/substrates",
        "//tensorflow_probability/python/experimental/vi",
    ],
)

py_library(
    name = "composite_tensor",
    srcs = ["composite_tensor.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
    ],
)

py_test(
    name = "composite_tensor_test",
    srcs = ["composite_tensor_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
