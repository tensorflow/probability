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
# Description:
#   Internal utilities for TensorFlow probability.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
#         "//tensorflow_probability/google:friends",  # DisableOnExport
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "internal",
    srcs = ["__init__.py"],
    deps = [
        ":test_fixtures",
    ],
)

multi_substrate_py_library(
    name = "test_fixtures",
    srcs = ["test_fixtures.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/mcmc:reducer",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/random:random_ops",
    ],
)
