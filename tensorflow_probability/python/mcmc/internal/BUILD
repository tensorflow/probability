# Copyright 2018 The TensorFlow Probability Authors.
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
#   MCMC methods, diagnostics, and related utilities.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "internal",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":leapfrog_integrator",
        ":slice_sampler_utils",
        ":util",
    ],
)

py_library(
    name = "slice_sampler_utils",
    srcs = ["slice_sampler_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    deps = [
        ":util",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "leapfrog_integrator",
    srcs = ["leapfrog_integrator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        # tensorflow dep,
    ],
)

py_test(
    name = "leapfrog_integrator_test",
    size = "small",
    srcs = ["leapfrog_integrator_test.py"],
    deps = [
        ":leapfrog_integrator",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)
