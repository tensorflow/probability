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
    deps = [
        ":base",
        # jax dep,
        # jax:stax dep,
        "//oryx/core:primitive",
        "//oryx/core/state",
    ],
)

# pytype_strict
py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        # jax dep,
        # jax:stax dep,
        "//oryx/core:kwargs_util",
        "//oryx/core:primitive",
        "//oryx/core/interpreters/inverse:core",
        "//oryx/core/state",
    ],
)

# pytype_strict
py_library(
    name = "combinator",
    srcs = ["combinator.py"],
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
    deps = [
        ":base",
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    deps = [
        ":base",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/core/state",
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "core_test",
    srcs = ["core_test.py"],
    deps = [
        ":core",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # jax:stax dep,
        # numpy dep,
        "//oryx/core/state",
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "combinator_test",
    srcs = ["combinator_test.py"],
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
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "convolution_test",
    srcs = ["convolution_test.py"],
    deps = [
        ":convolution",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        "//oryx/core/state",
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "normalization_test",
    srcs = ["normalization_test.py"],
    deps = [
        ":normalization",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "pooling_test",
    srcs = ["pooling_test.py"],
    deps = [
        ":pooling",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "reshape_test",
    srcs = ["reshape_test.py"],
    deps = [
        ":reshape",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/core/state",
        "//oryx/internal:test_util",
    ],
)

# pytype_strict
py_library(
    name = "function",
    srcs = ["function.py"],
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
    deps = [
        ":nn",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
        "//oryx/internal:test_util",
    ],
)
