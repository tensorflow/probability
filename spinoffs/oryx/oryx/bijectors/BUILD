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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# pytype_strict
py_library(
    name = "bijectors",
    srcs = ["__init__.py"],
    deps = [
        ":bijector_extensions",
        # tensorflow_probability/substrates:jax dep,
    ],
)

# pytype_strict
py_library(
    name = "bijector_extensions",
    srcs = ["bijector_extensions.py"],
    deps = [
        # jax dep,
        # six dep,
        "//oryx/core:primitive",
        "//oryx/core/interpreters/inverse",
        "//oryx/core/interpreters/inverse:slice",
        # tensorflow_probability/substrates:jax dep,
    ],
)

# py_strict
py_test(
    name = "bijector_extensions_test",
    srcs = ["bijector_extensions_test.py"],
    deps = [
        ":bijectors",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core",
        "//oryx/internal:test_util",
    ],
)
