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
#   Internal helper libraries for distributions.

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
        ":util",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        ":util",
        # numpy dep,
        # tensorflow dep,
    ],
)
