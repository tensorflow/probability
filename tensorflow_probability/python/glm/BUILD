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
#   Generalized Linear Model specification, fitting, and related utilities.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "glm",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":family",
        ":fisher_scoring",
        ":proximal_hessian",
    ],
)

py_library(
    name = "fisher_scoring",
    srcs = ["fisher_scoring.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:linalg",
    ],
)

py_test(
    name = "fisher_scoring_test",
    size = "small",
    srcs = ["fisher_scoring_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "proximal_hessian",
    srcs = ["proximal_hessian.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fisher_scoring",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math:linalg",
    ],
)

py_test(
    name = "proximal_hessian_test",
    size = "medium",
    srcs = ["proximal_hessian_test.py"],
    shard_count = 4,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "family",
    srcs = ["family.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:name_util",
        "//tensorflow_probability/python/util:deferred_tensor",
    ],
)

py_test(
    name = "family_test",
    size = "small",
    srcs = ["family_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
