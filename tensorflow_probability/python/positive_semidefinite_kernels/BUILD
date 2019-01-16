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
# Library for representation of positive-semidefinite kernel functions.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_library(
    name = "positive_semidefinite_kernels",
    srcs = ["__init__.py"],
    deps = [
        ":exp_sin_squared",
        ":exponentiated_quadratic",
        ":matern",
        ":polynomial",
        ":positive_semidefinite_kernel",
        "//tensorflow_probability/python/positive_semidefinite_kernels/internal",
    ],
)

py_library(
    name = "positive_semidefinite_kernel",
    srcs = ["positive_semidefinite_kernel.py"],
    deps = [
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "positive_semidefinite_kernel_test",
    size = "small",
    srcs = ["positive_semidefinite_kernel_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/positive_semidefinite_kernels/internal:util",
    ],
)

py_library(
    name = "exponentiated_quadratic",
    srcs = ["exponentiated_quadratic.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/positive_semidefinite_kernels/internal:util",
    ],
)

py_test(
    name = "exponentiated_quadratic_test",
    srcs = ["exponentiated_quadratic_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "exp_sin_squared",
    srcs = ["exp_sin_squared.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/positive_semidefinite_kernels/internal:util",
    ],
)

py_test(
    name = "exp_sin_squared_test",
    srcs = ["exp_sin_squared_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "matern",
    srcs = ["matern.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/positive_semidefinite_kernels/internal:util",
    ],
)

py_test(
    name = "matern_test",
    srcs = ["matern_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "rational_quadratic",
    srcs = ["rational_quadratic.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/positive_semidefinite_kernels/internal:util",
    ],
)

py_test(
    name = "rational_quadratic_test",
    srcs = ["rational_quadratic_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "polynomial",
    srcs = ["polynomial.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/positive_semidefinite_kernels/internal:util",
    ],
)

py_test(
    name = "polynomial_test",
    srcs = ["polynomial_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)
