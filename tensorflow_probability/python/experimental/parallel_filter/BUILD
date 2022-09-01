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
#   Parallel implementations of Bayesian filtering and smoothing algorithms.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

multi_substrate_py_library(
    name = "parallel_filter",
    srcs = ["__init__.py"],
    deps = [
        ":parallel_kalman_filter_lib",
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "parallel_kalman_filter_lib",
    srcs = ["parallel_kalman_filter_lib.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/math:scan_associative",
    ],
)

multi_substrate_py_test(
    name = "parallel_kalman_filter_test",
    size = "large",
    srcs = ["parallel_kalman_filter_test.py"],
    jax_size = "large",
    deps = [
        ":parallel_kalman_filter_lib",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)
