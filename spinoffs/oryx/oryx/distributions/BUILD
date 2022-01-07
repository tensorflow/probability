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
    name = "distributions",
    srcs = ["__init__.py"],
    deps = [
        ":distribution_extensions",
        # tensorflow_probability/substrates:jax dep,
    ],
)

# pytype_strict
py_library(
    name = "distribution_extensions",
    srcs = ["distribution_extensions.py"],
    deps = [
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/core:trace_util",
        "//oryx/core/interpreters:harvest",
        "//oryx/core/interpreters:log_prob",
        "//oryx/core/interpreters:unzip",
        "//oryx/core/interpreters/inverse",
        "//oryx/core/ppl",
        # tensorflow_probability/substrates:jax dep,
    ],
)

# py_strict
py_test(
    name = "distribution_extensions_test",
    srcs = ["distribution_extensions_test.py"],
    deps = [
        ":distributions",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "//oryx/core",
        "//oryx/core/ppl",
        "//oryx/internal:test_util",
        # tensorflow_probability/substrates:jax dep,
    ],
)
