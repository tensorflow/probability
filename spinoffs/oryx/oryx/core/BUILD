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
# [internal] load pytype.bzl (pytype_strict_library)
# [internal] load strict.bzl

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# pytype_strict
py_library(
    name = "core",
    srcs = ["__init__.py"],
    deps = [
        ":primitive",
        ":pytree",
        ":serialize",
        ":trace_util",
        "//oryx/core/interpreters",
        "//oryx/core/interpreters:harvest",
        "//oryx/core/interpreters:log_prob",
        "//oryx/core/interpreters:unzip",
        "//oryx/core/interpreters/inverse:core",
        "//oryx/core/interpreters/inverse:custom_inverse",
        "//oryx/core/ppl",
        "//oryx/core/state",
    ],
)

# pytype_strict
py_library(
    name = "serialize",
    srcs = ["serialize.py"],
)

# pytype_strict
py_library(
    name = "pytree",
    srcs = ["pytree.py"],
    deps = [
        # jax dep,
    ],
)

# pytype_strict
py_library(
    name = "primitive",
    srcs = ["primitive.py"],
    deps = [
        ":trace_util",
        # jax dep,
    ],
)

# pytype_strict
py_library(
    name = "trace_util",
    srcs = ["trace_util.py"],
    deps = [
        # jax dep,
    ],
)

# pytype_strict
py_library(
    name = "kwargs_util",
    srcs = ["kwargs_util.py"],
    deps = [
        # jax dep,
    ],
)

# py_strict
py_test(
    name = "serialize_test",
    srcs = ["serialize_test.py"],
    deps = [
        ":serialize",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
        "//oryx/experimental/nn",
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "kwargs_util_test",
    srcs = ["kwargs_util_test.py"],
    deps = [
        ":kwargs_util",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/core/interpreters/inverse:custom_inverse",
    ],
)
