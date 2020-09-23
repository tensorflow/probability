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
#   MCMC methods, diagnostics, and related utilities.

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

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "distributions",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":mvn_inverse_scale_linop",
    ],
)

multi_substrate_py_library(
    name = "mvn_inverse_scale_linop",
    srcs = ["mvn_inverse_scale_linop.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_test(
    name = "mvn_inverse_scale_linop_test",
    size = "medium",
    timeout = "long",
    srcs = ["mvn_inverse_scale_linop_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
