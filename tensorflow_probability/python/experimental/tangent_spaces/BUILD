# Copyright 2021 The TensorFlow Probability Authors.
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

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "tangent_spaces",
    srcs = ["__init__.py"],
    deps = [
        ":simplex",
        ":spaces",
        ":spherical",
        ":symmetric_matrix",
    ],
)

multi_substrate_py_library(
    name = "spaces_test_util",
    testonly = True,
    srcs = ["spaces_test_util.py"],
    deps = [
        ":spaces",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "spaces",
    srcs = ["spaces.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "spaces_test",
    size = "medium",
    srcs = ["spaces_test.py"],
    shard_count = 3,
    tags = [
        "tf1-broken",
    ],
    deps = [
        ":spaces",
        ":spaces_test_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:reshape",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:scale_matvec_tril",
        "//tensorflow_probability/python/bijectors:transform_diagonal",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "simplex",
    srcs = ["simplex.py"],
    deps = [
        ":spaces",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/linalg:linear_operator_row_block",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "simplex_test",
    size = "medium",
    srcs = ["simplex_test.py"],
    shard_count = 3,
    tags = [
        "tf1-broken",
    ],
    deps = [
        ":simplex",
        ":spaces_test_util",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:scale_matvec_tril",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "spherical",
    srcs = ["spherical.py"],
    deps = [
        ":spaces",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/linalg:linear_operator_row_block",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "spherical_test",
    size = "medium",
    srcs = ["spherical_test.py"],
    shard_count = 3,
    tags = [
        "tf1-broken",
    ],
    deps = [
        ":spaces_test_util",
        ":spherical",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:scale_matvec_tril",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "symmetric_matrix",
    srcs = ["symmetric_matrix.py"],
    deps = [
        ":spaces",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:linalg",
    ],
)

multi_substrate_py_test(
    name = "symmetric_matrix_test",
    size = "medium",
    srcs = ["symmetric_matrix_test.py"],
    shard_count = 3,
    tags = [
        "tf1-broken",
    ],
    deps = [
        ":spaces_test_util",
        ":symmetric_matrix",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:reshape",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:linalg",
    ],
)
