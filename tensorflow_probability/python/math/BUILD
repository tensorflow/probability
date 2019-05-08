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
#   TensorFlow Probability general math functions.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "math",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":custom_gradient",
        ":diag_jacobian",
        ":gradient",
        ":interpolation",
        ":linalg",
        ":numeric",
        ":random_ops",
        ":root_search",
        ":sparse",
        "//tensorflow_probability/python/math/ode",
    ],
)

py_library(
    name = "custom_gradient",
    srcs = [
        "custom_gradient.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "custom_gradient_test",
    size = "small",
    srcs = ["custom_gradient_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "gradient",
    srcs = [
        "gradient.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "gradient_test",
    size = "small",
    srcs = ["gradient_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "interpolation",
    srcs = [
        "interpolation.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "interpolation_test",
    size = "small",
    srcs = ["interpolation_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "linalg",
    srcs = [
        "linalg.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

py_test(
    name = "linalg_test",
    size = "small",
    srcs = ["linalg_test.py"],
    shard_count = 5,
    deps = [
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "numeric",
    srcs = [
        "numeric.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "numeric_test",
    srcs = [
        "numeric_test.py",
    ],
    deps = [
        ":gradient",
        ":numeric",
        # absl/testing:parameterized dep,
        # numpy dep,
        "//tensorflow_probability/python/internal:test_case",
    ],
)

py_library(
    name = "random_ops",
    srcs = [
        "random_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "random_ops_test",
    size = "small",
    srcs = ["random_ops_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "diag_jacobian",
    srcs = [
        "diag_jacobian.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gradient",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "diag_jacobian_test",
    size = "small",
    srcs = ["diag_jacobian_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "root_search",
    srcs = [
        "root_search.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "root_search_test",
    size = "medium",
    srcs = ["root_search_test.py"],
    deps = [
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "sparse",
    srcs = [
        "sparse.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "sparse_test",
    size = "small",
    srcs = ["sparse_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)
