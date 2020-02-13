# Copyright 2019 The TensorFlow Probability Authors.
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
# Tests for the backend integration.

# [internal] load python3.bzl

# py_test rules are rewritten to py3_test on export to OSS, for reasons of
# reversibility of import<->export transforms. This is only necessary until we
# eliminate all py2and3_test targets, after which everything can safely use
# py_test, interanlly and externally, and no rewrites will be needed.
# @unused  <-- satisfy buildifier errors
load("//tensorflow_probability/python:build_defs.bzl", "py3_test")

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_test(
    name = "numpy_integration_test",
    size = "small",
    srcs = ["numpy_integration_test.py"],
    tags = ["tfp_numpy"],
    deps = [
        "//tensorflow_probability",
    ],
)

py3_test(
    name = "jax_integration_test",
    srcs = ["jax_integration_test.py"],
    python_version = "PY3",
    tags = ["tfp_jax"],
    deps = [
        # jax dep,
        "//tensorflow_probability",
    ],
)

py_binary(
    name = "rewrite",
    srcs = ["rewrite.py"],
    python_version = "PY3",
    deps = [
        # absl:app dep,
    ],
)
