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
# [internal] load strict.bzl
# [internal] load pytype.bzl (pytype_strict_library)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# pytype_strict
py_library(
    name = "interpreters",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":harvest",
        ":log_prob",
        ":propagate",
        ":unzip",
        "//oryx/core/interpreters/inverse",
    ],
)

# pytype_strict
py_library(
    name = "unzip",
    srcs = ["unzip.py"],
    srcs_version = "PY3",
    deps = [
        ":harvest",
        # jax dep,
        # numpy dep,
        "//oryx/core:trace_util",
    ],
)

# pytype_strict
py_library(
    name = "propagate",
    srcs = ["propagate.py"],
    srcs_version = "PY3",
    deps = [
        ":harvest",
        # jax dep,
        "//oryx/core:pytree",
    ],
)

# pytype_strict
py_library(
    name = "harvest",
    srcs = ["harvest.py"],
    srcs_version = "PY3",
    deps = [
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/core:trace_util",
    ],
)

# pytype_strict
py_library(
    name = "log_prob",
    srcs = ["log_prob.py"],
    srcs_version = "PY3",
    deps = [
        ":propagate",
        # jax dep,
        "//oryx/core:trace_util",
        "//oryx/core/interpreters/inverse",
    ],
)

# py_strict
py_test(
    name = "unzip_test",
    srcs = ["unzip_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":harvest",
        ":unzip",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//oryx/core:trace_util",
        "//oryx/core/state",
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "propagate_test",
    srcs = ["propagate_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":propagate",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//oryx/core:trace_util",
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "harvest_test",
    srcs = ["harvest_test.py"],
    python_version = "PY3",
    deps = [
        ":harvest",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//oryx/core:trace_util",
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "log_prob_test",
    srcs = ["log_prob_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":log_prob",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/bijectors",
        "//oryx/core",
        "//oryx/core/state",
        "//oryx/distributions",
        "//oryx/internal:test_util",
    ],
)
