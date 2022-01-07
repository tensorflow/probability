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
    name = "optimizers",
    srcs = ["__init__.py"],
    deps = [
        ":optix",
    ],
)

# pytype_strict
py_library(
    name = "optix",
    srcs = ["optix.py"],
    visibility = ["//visibility:private"],
    deps = [
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/core/state",
    ],
)

# py_strict
py_test(
    name = "optix_test",
    srcs = ["optix_test.py"],
    deps = [
        ":optix",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/state",
        "//oryx/internal:test_util",
    ],
)
