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
#   TensorFlow Probability ODE solvers.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
    ],
)

py_library(
    name = "bdf",
    srcs = ["bdf.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":bdf_util",
        ":util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "bdf_util",
    srcs = ["bdf_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "bdf_util_test",
    size = "small",
    srcs = ["bdf_util_test.py"],
    deps = [
        ":bdf",
        ":bdf_util",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "ode",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":bdf",
    ],
)

py_test(
    name = "ode_test",
    size = "large",
    srcs = ["ode_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # tensorflow dep,
    ],
)
