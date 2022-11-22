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

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

multi_substrate_py_library(
    name = "internal",
    srcs = ["__init__.py"],
    deps = [
        ":test_util",
        ":util",
    ],
)

multi_substrate_py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math/psd_kernels:positive_semidefinite_kernel",
    ],
)

multi_substrate_py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    numpy_tags = ["notap"],
    deps = [
        ":util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_test(
    name = "test_util_test",
    size = "small",
    srcs = ["test_util_test.py"],
    deps = [
        ":test_util",
        # numpy dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)
