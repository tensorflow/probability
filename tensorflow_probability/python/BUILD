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
#   Tensorflow Probability implements probabilistic and statistical
#   methods, including modeling and Bayesian inference. APIs here are
#   meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_library(
    name = "monte_carlo",
    srcs = [
        "monte_carlo.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "monte_carlo_test",
    size = "small",
    srcs = ["tests/monte_carlo_test.py"],
    deps = [
        ":monte_carlo",
        # numpy dep,
        # tensorflow dep,
    ],
)
