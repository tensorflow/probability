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
#   TensorFlow Eager kernel for one No U-Turn Sampler trajectory.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_binary(
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [":logistic_regression_lib"],
)

py_library(
    name = "logistic_regression_lib",
    srcs = ["logistic_regression.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":no_u_turn_sampler",
        # matplotlib dep,
        # numpy dep,
        # sklearn dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "logistic_regression_test",
    size = "medium",
    srcs = ["logistic_regression.py"],
    args = [
        "--fake_data",
        "--max_steps=3",
        "--skip_plots",
    ],
    main = "logistic_regression.py",
    srcs_version = "PY2AND3",
    tags = [
        "nozapfhahn",
    ],
    deps = [
        ":logistic_regression_lib",
    ],
)

py_library(
    name = "no_u_turn_sampler",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nuts",
    ],
)

py_library(
    name = "nuts",
    srcs = ["nuts.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "nuts_test",
    srcs = ["nuts_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":no_u_turn_sampler",
        # absl/flags dep,
        # absl/testing:parameterized dep,
        # matplotlib dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)
