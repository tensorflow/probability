# Copyright 2021 The TensorFlow Probability Authors.
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
# Mini-library for pattern matching and term rewriting
# [internal] load pytype.bzl (pytype_strict_library)
# [internal] load strict.bzl

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# pytype_strict
py_library(
    name = "matching",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":jax_rewrite",
        ":matcher",
        ":rules",
    ],
)

# pytype_strict
py_library(
    name = "matcher",
    srcs = ["matcher.py"],
    srcs_version = "PY3",
    deps = [
        # dataclasses dep,
    ],
)

# pytype_strict
py_library(
    name = "rules",
    srcs = ["rules.py"],
    srcs_version = "PY3",
    deps = [":matcher"],
)

# pytype_strict
py_library(
    name = "jax_rewrite",
    srcs = ["jax_rewrite.py"],
    srcs_version = "PY3",
    deps = [
        ":matcher",
        ":rules",
        # dataclasses dep,
        # jax dep,
        # numpy dep,
        "//oryx/core:trace_util",
    ],
)

# py_strict
py_test(
    name = "matcher_test",
    srcs = ["matcher_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":matcher",
        # absl/testing:absltest dep,
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "rules_test",
    srcs = ["rules_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":matcher",
        ":rules",
        # absl/testing:absltest dep,
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "jax_rewrite_test",
    srcs = ["jax_rewrite_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":jax_rewrite",
        ":matcher",
        ":rules",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/internal:test_util",
    ],
)
