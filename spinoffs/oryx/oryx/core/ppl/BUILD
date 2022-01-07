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
    name = "ppl",
    srcs = ["__init__.py"],
    deps = [
        ":effect_handler",
        ":transformations",
    ],
)

# pytype_strict
py_library(
    name = "transformations",
    srcs = ["transformations.py"],
    deps = [
        ":plate_util",
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/core/interpreters:harvest",
        "//oryx/core/interpreters:log_prob",
    ],
)

# pytype_strict
py_library(
    name = "plate_util",
    srcs = ["plate_util.py"],
    deps = [
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/core/interpreters:log_prob",
        "//oryx/core/interpreters:propagate",
    ],
)

# pytype_strict
py_library(
    name = "effect_handler",
    srcs = ["effect_handler.py"],
    deps = [
        # jax dep,
        "//oryx/core:trace_util",
    ],
)

# py_strict
py_test(
    name = "transformations_test",
    srcs = ["transformations_test.py"],
    deps = [
        ":transformations",
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        "//oryx/core/interpreters:log_prob",
        "//oryx/internal:test_util",
        # tensorflow_probability/substrates:jax dep,
    ],
)

# py_strict
py_test(
    name = "effect_handler_test",
    srcs = ["effect_handler_test.py"],
    deps = [
        ":effect_handler",
        # absl/testing:absltest dep,
        # jax dep,
        "//oryx/core:primitive",
        "//oryx/internal:test_util",
    ],
)
