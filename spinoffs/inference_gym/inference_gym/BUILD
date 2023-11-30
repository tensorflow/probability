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
#   A package for target densities and benchmarking of inference algorithms
#   against the same.

# Placeholder: py_library
# [internal] load pytype.bzl (pytype_strict_library)

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
        "//inference_gym:__subpackages__",
    ],
)

licenses(["notice"])

# This target is here for convenience. Link to one of the `using_*` targets to
# minimize the number of dependencies.
# pytype_strict
py_library(
    name = "inference_gym",
    visibility = ["//visibility:public"],
    deps = [
        ":using_jax",
        ":using_numpy",
        ":using_tensorflow",
    ],
)

py_library(
    name = "using_numpy",
    srcs = ["using_numpy.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":backend_numpy",
        "//inference_gym/backends:rewrite",
        "//inference_gym/targets",
        # tensorflow_probability/substrates:numpy dep,
    ],
)

py_library(
    name = "using_jax",
    srcs = ["using_jax.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":backend_jax",
        # jax dep,
        "//inference_gym/backends:rewrite",
        "//inference_gym/targets",
        # tensorflow_probability/substrates:jax dep,
    ],
)

py_library(
    name = "using_tensorflow",
    srcs = ["using_tensorflow.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":backend_tensorflow",
        # tensorflow dep,
        # tensorflow_probability dep,
        "//inference_gym/backends:rewrite",
        "//inference_gym/targets",
    ],
)

# pytype_strict
py_library(
    name = "backend_numpy",
    srcs = ["dynamic/backend_numpy/__init__.py"],
)

# pytype_strict
py_library(
    name = "backend_jax",
    srcs = ["dynamic/backend_jax/__init__.py"],
)

# pytype_strict
py_library(
    name = "backend_tensorflow",
    srcs = ["dynamic/backend_tensorflow/__init__.py"],
)
