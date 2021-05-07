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
licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "losses",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":losses_lib",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

py_library(
    name = "losses_lib",
    srcs = ["losses.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/experimental/nn:layers",
        "//tensorflow_probability/python/experimental/nn:variational_base",
        "//tensorflow_probability/python/internal:all_util",
        "//tensorflow_probability/python/monte_carlo:expectation",
    ],
)
