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
    name = "nn",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":combinator",
        ":convolution",
        ":core",
        ":function",
        ":normalization",
        ":pooling",
        ":reshape",
    ],
)

# pytype_strict
py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # jax dep,
        # jax:stax dep,
        "//oryx/core/state",
    ],
)

# pytype_strict
py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        # jax dep,
        # jax:stax dep,
        "//oryx/core:kwargs_util",
        "//oryx/core:primitive",
        "//oryx/core/interpreters:unzip",
        "//oryx/core/interpreters/inverse:core",
        "//oryx/core/state",
    ],
)

# pytype_strict
py_library(
    name = "combinator",
    srcs = ["combinator.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # jax dep,
        "//oryx/core/state",
    ],
)

# pytype_strict
py_library(
    name = "convolution",
    srcs = ["convolution.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # jax dep,
        # jax:stax dep,
        "//oryx/core/state",
    ],
)

# pytype_strict
py_library(
    name = "normalization",
    srcs = ["normalization.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # jax dep,
        # jax:stax dep,
        "//oryx/core:kwargs_util",
    ],
)

# pytype_strict
py_library(
    name = "pooling",
    srcs = ["pooling.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # jax dep,
        "//oryx/core/state",
    ],
)

# pytype_strict
py_library(
    name = "reshape",
    srcs = ["reshape.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # jax dep,
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    python_version = "PY3",
    deps = [
        ":base",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "core_test",
    srcs = ["core_test.py"],
    python_version = "PY3",
    deps = [
        ":core",
        ":nn",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # jax:stax dep,
        # numpy dep,
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "combinator_test",
    srcs = ["combinator_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":combinator",
        ":convolution",
        ":core",
        ":normalization",
        ":reshape",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "convolution_test",
    srcs = ["convolution_test.py"],
    python_version = "PY3",
    deps = [
        ":convolution",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "normalization_test",
    srcs = ["normalization_test.py"],
    python_version = "PY3",
    deps = [
        ":normalization",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "pooling_test",
    srcs = ["pooling_test.py"],
    python_version = "PY3",
    deps = [
        ":pooling",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "reshape_test",
    srcs = ["reshape_test.py"],
    python_version = "PY3",
    deps = [
        ":reshape",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/core/state",
    ],
)

# pytype_strict
py_library(
    name = "function",
    srcs = ["function.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "function_test",
    srcs = ["function_test.py"],
    python_version = "PY3",
    deps = [
        ":nn",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
    ],
)
