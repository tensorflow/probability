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
    name = "inverse",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":core",
        ":custom_inverse",
        ":rules",
        ":slice",
    ],
)

# pytype_strict
py_library(
    name = "slice",
    srcs = ["slice.py"],
    srcs_version = "PY3",
    deps = [
        # dataclasses dep,
        # jax dep,
        "//oryx/core:pytree",
    ],
)

# pytype_strict
py_library(
    name = "rules",
    srcs = ["rules.py"],
    srcs_version = "PY3",
    deps = [
        ":core",
        ":custom_inverse",
        ":slice",
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/core/interpreters:harvest",
    ],
)

# pytype_strict
py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        ":slice",
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/core:trace_util",
        "//oryx/core/interpreters:harvest",
        "//oryx/core/interpreters:propagate",
    ],
)

# pytype_strict
py_library(
    name = "custom_inverse",
    srcs = ["custom_inverse.py"],
    srcs_version = "PY3",
    deps = [
        ":core",
        ":slice",
        # jax dep,
        "//oryx/core:kwargs_util",
        "//oryx/core:primitive",
    ],
)

# py_strict
py_test(
    name = "inverse_test",
    srcs = ["inverse_test.py"],
    python_version = "PY3",
    deps = [
        ":core",
        ":rules",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/interpreters:harvest",
        "//oryx/internal:test_util",
        # tensorflow_probability/substrates:jax dep,
    ],
)

# py_strict
py_test(
    name = "slice_test",
    srcs = ["slice_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":slice",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/internal:test_util",
    ],
)

# py_strict
py_test(
    name = "custom_inverse_test",
    srcs = ["custom_inverse_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":core",
        ":custom_inverse",
        ":rules",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/internal:test_util",
    ],
)
