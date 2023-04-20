# Copyright 2019 The TensorFlow Probability Authors.
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
#   Experimental linear algebra tools.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

multi_substrate_py_library(
    name = "linalg",
    srcs = ["__init__.py"],
    numpy_omit_deps = [
        ":linear_operator_interpolated_psd_kernel",
    ],
    substrates_omit_deps = [
        ":linear_operator_psd_kernel",
        ":no_pivot_ldl",
    ],
    deps = [
        ":linear_operator_interpolated_psd_kernel",
        ":linear_operator_psd_kernel",
        ":linear_operator_unitary",
        ":no_pivot_ldl",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "linear_operator_interpolated_psd_kernel",
    srcs = ["linear_operator_interpolated_psd_kernel.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/math:interpolation",
    ],
)

multi_substrate_py_test(
    name = "linear_operator_interpolated_psd_kernel_test",
    size = "large",
    srcs = ["linear_operator_interpolated_psd_kernel_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 8,
    deps = [
        ":linear_operator_interpolated_psd_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:matern",
        "//tensorflow_probability/python/math/psd_kernels:polynomial",
        # tensorflow/compiler/jit dep,
    ],
)

py_library(
    name = "linear_operator_psd_kernel",
    srcs = ["linear_operator_psd_kernel.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

py_test(
    name = "linear_operator_psd_kernel_test",
    size = "small",
    srcs = ["linear_operator_psd_kernel_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/linalg:linear_operator_psd_kernel",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
        "//tensorflow_probability/python/math/psd_kernels:polynomial",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_library(
    name = "linear_operator_row_block",
    srcs = ["linear_operator_row_block.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "linear_operator_row_block_test",
    srcs = ["linear_operator_row_block_test.py"],
    python_version = "PY3",
    deps = [
        ":linear_operator_row_block",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "linear_operator_unitary",
    srcs = ["linear_operator_unitary.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_test(
    name = "linear_operator_unitary_test",
    size = "small",
    srcs = ["linear_operator_unitary_test.py"],
    deps = [
        ":linear_operator_unitary",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "no_pivot_ldl",
    srcs = ["no_pivot_ldl.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "no_pivot_ldl_test",
    srcs = ["no_pivot_ldl_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/linalg:no_pivot_ldl",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)
