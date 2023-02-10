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
# tfp.experimental PSD kernels library.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow_probability:license"],
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "psd_kernels",
    srcs = ["__init__.py"],
    deps = [
        ":additive_kernel",
        ":multitask_kernel",
    ],
)

multi_substrate_py_library(
    name = "additive_kernel",
    srcs = ["additive_kernel.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math/psd_kernels:positive_semidefinite_kernel",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "additive_kernel_test",
    size = "medium",
    srcs = ["additive_kernel_test.py"],
    shard_count = 4,
    tags = ["tf1-broken"],
    deps = [
        ":additive_kernel",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)

multi_substrate_py_library(
    name = "multitask_kernel",
    srcs = ["multitask_kernel.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/math/psd_kernels:positive_semidefinite_kernel",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "multitask_kernel_test",
    size = "medium",
    srcs = ["multitask_kernel_test.py"],
    deps = [
        ":multitask_kernel",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)
