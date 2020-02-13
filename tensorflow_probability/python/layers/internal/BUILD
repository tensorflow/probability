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
#   Internal helper libraries for layers.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "internal",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":distribution_tensor_coercible",
        ":tensor_tuple",
    ],
)

py_library(
    name = "distribution_tensor_coercible",
    srcs = ["distribution_tensor_coercible.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/util",
    ],
)

py_test(
    name = "distribution_tensor_coercible_test",
    size = "small",
    srcs = ["distribution_tensor_coercible_test.py"],
    deps = [
        ":distribution_tensor_coercible",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "tensor_tuple",
    srcs = ["tensor_tuple.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "tensor_tuple_test",
    size = "small",
    srcs = ["tensor_tuple_test.py"],
    deps = [
        ":tensor_tuple",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)
