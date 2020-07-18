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
    name = "mcmc",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":kernels",
        ":utils",
    ],
)

# pytype_strict
py_library(
    name = "kernels",
    srcs = ["kernels.py"],
    srcs_version = "PY3",
    deps = [
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/core/interpreters:harvest",
        "//oryx/core/ppl",
        "//oryx/core/state",
        "//oryx/distributions",
    ],
)

# pytype_strict
py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        # jax dep,
        "//oryx/core/interpreters/inverse:core",
        "//oryx/core/ppl",
    ],
)

# py_strict
py_test(
    name = "kernels_test",
    srcs = ["kernels_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":kernels",
        ":utils",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/interpreters:harvest",
        # [tfp_jax dependency],
    ],
)
