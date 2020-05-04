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
load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_test",
    "multi_substrate_py_library",
)

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

multi_substrate_py_library(
    name = "psd_kernels",
    srcs = ["__init__.py"],
    deps = [
        ":exp_sin_squared",
        ":exponentiated_quadratic",
        ":feature_scaled",
        ":feature_transformed",
        ":kumaraswamy_transformed",
        ":matern",
        ":polynomial",
        ":positive_semidefinite_kernel",
        ":rational_quadratic",
        ":schur_complement",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math/psd_kernels/internal",
    ],
)

multi_substrate_py_library(
    name = "positive_semidefinite_kernel",
    srcs = ["positive_semidefinite_kernel.py"],
    deps = [
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "positive_semidefinite_kernel_test",
    size = "small",
    srcs = ["positive_semidefinite_kernel_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_library(
    name = "exponentiated_quadratic",
    srcs = ["exponentiated_quadratic.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "exponentiated_quadratic_test",
    size = "small",
    srcs = ["exponentiated_quadratic_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "exp_sin_squared",
    srcs = ["exp_sin_squared.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "exp_sin_squared_test",
    size = "small",
    srcs = ["exp_sin_squared_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "matern",
    srcs = ["matern.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "matern_test",
    size = "small",
    srcs = ["matern_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "rational_quadratic",
    srcs = ["rational_quadratic.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "rational_quadratic_test",
    size = "small",
    srcs = ["rational_quadratic_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "polynomial",
    srcs = ["polynomial.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "polynomial_test",
    size = "small",
    srcs = ["polynomial_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "schur_complement",
    srcs = ["schur_complement.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:cholesky_outer_product",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "schur_complement_test",
    size = "medium",
    srcs = ["schur_complement_test.py"],
    jax_size = "large",
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "feature_scaled",
    srcs = ["feature_scaled.py"],
    deps = [
        ":feature_transformed",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "feature_scaled_test",
    size = "small",
    srcs = ["feature_scaled_test.py"],
    jax_size = "medium",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "feature_transformed",
    srcs = ["feature_transformed.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "feature_transformed_test",
    size = "medium",
    srcs = ["feature_transformed_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "kumaraswamy_transformed",
    srcs = ["kumaraswamy_transformed.py"],
    deps = [
        ":feature_transformed",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:kumaraswamy_cdf",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "kumaraswamy_transformed_test",
    size = "medium",
    srcs = ["kumaraswamy_transformed_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "hypothesis_testlib",
    testonly = True,
    srcs = ["hypothesis_testlib.py"],
    deps = [
        # hypothesis dep,
        # numpy dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

py_test(
    name = "psd_kernel_properties_test",
    size = "medium",
    srcs = ["psd_kernel_properties_test.py"],
    shard_count = 6,
    tags = ["hypothesis"],
    deps = [
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:hypothesis_testlib",
    ],
)
