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

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    numpy_omit_deps = [
        "//tensorflow_probability/python/experimental/distribute",
        "//tensorflow_probability/python/experimental/stats",
    ],
    srcs_version = "PY3",
    substrates_omit_deps = [
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/experimental/linalg",
        "//tensorflow_probability/python/experimental/marginalize",
        "//tensorflow_probability/python/experimental/nn",
        "//tensorflow_probability/python/experimental/sequential",
        "//tensorflow_probability/python/experimental/sts_gibbs",
        "//tensorflow_probability/python/experimental/substrates",
        "//tensorflow_probability/python/experimental/vi",
        "//tensorflow_probability/python/internal:auto_composite_tensor",
        "//tensorflow_probability/python/experimental/util:composite_tensor",
    ],
    deps = [
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/experimental/bijectors",
        "//tensorflow_probability/python/experimental/distribute",
        "//tensorflow_probability/python/experimental/distributions",
        "//tensorflow_probability/python/experimental/linalg",
        "//tensorflow_probability/python/experimental/marginalize",
        "//tensorflow_probability/python/experimental/math",
        "//tensorflow_probability/python/experimental/mcmc",
        "//tensorflow_probability/python/experimental/nn",
        "//tensorflow_probability/python/experimental/parallel_filter",
        "//tensorflow_probability/python/experimental/sequential",
        "//tensorflow_probability/python/experimental/stats",
        "//tensorflow_probability/python/experimental/sts_gibbs",
        "//tensorflow_probability/python/experimental/substrates",
        "//tensorflow_probability/python/experimental/tangent_spaces",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/experimental/util:composite_tensor",
        "//tensorflow_probability/python/experimental/vi",
        "//tensorflow_probability/python/internal:all_util",
        "//tensorflow_probability/python/internal:auto_composite_tensor",
    ],
)
