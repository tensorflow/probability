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
load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_test",
    "multi_substrate_py_library",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "math",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":custom_gradient",
        ":diag_jacobian",
        ":generic",
        ":gradient",
        ":interpolation",
        ":linalg",
        ":minimize",
        ":numeric",
        ":random_ops",
        ":root_search",
        ":scan_associative",
        ":sparse",
        ":special",
        "//tensorflow_probability/python/math/ode",
        "//tensorflow_probability/python/math/psd_kernels",
    ],
)

multi_substrate_py_library(
    name = "custom_gradient",
    srcs = [
        "custom_gradient.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_test(
    name = "custom_gradient_test",
    size = "small",
    srcs = ["custom_gradient_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "generic",
    srcs = [
        "generic.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "generic_test",
    size = "small",
    srcs = ["generic_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "gradient",
    srcs = [
        "gradient.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "gradient_test",
    size = "small",
    srcs = ["gradient_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "interpolation",
    srcs = [
        "interpolation.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "interpolation_test",
    size = "small",
    srcs = ["interpolation_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "linalg",
    srcs = [
        "linalg.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "linalg_test",
    size = "medium",
    srcs = ["linalg_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    shard_count = 5,
    tags = ["hypothesis"],
    deps = [
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "minimize",
    srcs = [
        "minimize.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
    ],
)

py_test(
    name = "minimize_test",
    size = "small",
    srcs = ["minimize_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "numeric",
    srcs = [
        "numeric.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_test(
    name = "numeric_test",
    size = "small",
    srcs = [
        "numeric_test.py",
    ],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "random_ops",
    srcs = [
        "random_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:samplers",
    ],
)

multi_substrate_py_test(
    name = "random_ops_test",
    size = "small",
    srcs = ["random_ops_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "diag_jacobian",
    srcs = [
        "diag_jacobian.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gradient",
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "diag_jacobian_test",
    size = "small",
    srcs = ["diag_jacobian_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "root_search",
    srcs = [
        "root_search.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_test(
    name = "root_search_test",
    size = "small",
    srcs = ["root_search_test.py"],
    deps = [
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "sparse",
    srcs = [
        "sparse.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
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
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "scan_associative",
    srcs = ["scan_associative.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "scan_associative_test",
    size = "medium",
    srcs = ["scan_associative_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":scan_associative",
        # numpy dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "special",
    srcs = [
        "special.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "special_test",
    size = "small",
    srcs = ["special_test.py"],
    jax_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

exports_files(
    [
        "generic.py",
        "generic_test.py",
        "gradient.py",
        "gradient_test.py",
        "linalg.py",
        "numeric.py",
        "special.py",
    ],
    visibility = ["//tensorflow_probability:__subpackages__"],
)
