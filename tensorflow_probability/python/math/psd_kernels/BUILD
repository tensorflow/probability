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

# Placeholder: py_library
# Placeholder: py_test
load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = ["//tensorflow_probability:__subpackages__"],
)

multi_substrate_py_library(
    name = "psd_kernels",
    srcs = ["__init__.py"],
    deps = [
        ":changepoint",
        ":exp_sin_squared",
        ":exponential_curve",
        ":exponentiated_quadratic",
        ":feature_scaled",
        ":feature_transformed",
        ":gamma_exponential",
        ":kumaraswamy_transformed",
        ":matern",
        ":parabolic",
        ":pointwise_exponential",
        ":polynomial",
        ":positive_semidefinite_kernel",
        ":rational_quadratic",
        ":schur_complement",
        ":spectral_mixture",
        "//tensorflow_probability/python/internal:all_util",
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
        "//tensorflow_probability/python/internal:auto_composite_tensor",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:slicing",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "positive_semidefinite_kernel_test",
    size = "small",
    srcs = ["positive_semidefinite_kernel_test.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:auto_composite_tensor",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_library(
    name = "changepoint",
    srcs = ["changepoint.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:ascending",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "changepoint_test",
    size = "small",
    srcs = ["changepoint_test.py"],
    jax_size = "medium",
    deps = [
        ":changepoint",
        ":exponentiated_quadratic",
        ":matern",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "exponentiated_quadratic",
    srcs = ["exponentiated_quadratic.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "exponentiated_quadratic_test",
    size = "small",
    srcs = ["exponentiated_quadratic_test.py"],
    deps = [
        ":exponentiated_quadratic",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
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
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "exp_sin_squared_test",
    size = "small",
    srcs = ["exp_sin_squared_test.py"],
    deps = [
        ":exp_sin_squared",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "gamma_exponential",
    srcs = ["gamma_exponential.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "gamma_exponential_test",
    size = "small",
    srcs = ["gamma_exponential_test.py"],
    deps = [
        ":exponentiated_quadratic",
        ":gamma_exponential",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
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
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/math:bessel",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "matern_test",
    size = "medium",
    srcs = ["matern_test.py"],
    jax_size = "medium",
    shard_count = 4,
    deps = [
        ":matern",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "parabolic",
    srcs = ["parabolic.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "parabolic_test",
    size = "small",
    srcs = ["parabolic_test.py"],
    deps = [
        ":parabolic",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
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
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "rational_quadratic_test",
    size = "small",
    srcs = ["rational_quadratic_test.py"],
    deps = [
        ":rational_quadratic",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "polynomial",
    srcs = ["polynomial.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
        "//tensorflow_probability/python/util",
    ],
)

multi_substrate_py_test(
    name = "polynomial_test",
    size = "small",
    srcs = ["polynomial_test.py"],
    deps = [
        ":polynomial",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
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
        "//tensorflow_probability/python/distributions:cholesky_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "schur_complement_test",
    size = "medium",
    srcs = ["schur_complement_test.py"],
    jax_size = "large",
    deps = [
        ":exponentiated_quadratic",
        ":schur_complement",
        # absl/testing:parameterized dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/math/psd_kernels/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "spectral_mixture",
    srcs = ["spectral_mixture.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "spectral_mixture_test",
    size = "small",
    srcs = ["spectral_mixture_test.py"],
    jax_size = "medium",
    deps = [
        ":spectral_mixture",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "feature_scaled",
    srcs = ["feature_scaled.py"],
    deps = [
        ":feature_transformed",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "feature_scaled_test",
    size = "small",
    srcs = ["feature_scaled_test.py"],
    jax_size = "medium",
    deps = [
        ":exponentiated_quadratic",
        ":feature_scaled",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
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
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "feature_transformed_test",
    size = "medium",
    srcs = ["feature_transformed_test.py"],
    deps = [
        ":exponentiated_quadratic",
        ":feature_transformed",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:scale_matvec_diag",
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
        ":exponentiated_quadratic",
        ":kumaraswamy_transformed",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "hypothesis_testlib",
    testonly = True,
    srcs = ["hypothesis_testlib.py"],
    strict_deps = False,
    deps = [
        ":psd_kernels",
        # hypothesis dep,
        # numpy dep,
        "//tensorflow_probability/python/experimental/distributions:marginal_fns",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

py_test(
    name = "psd_kernel_properties_test",
    size = "medium",
    srcs = ["psd_kernel_properties_test.py"],
    shard_count = 20,
    tags = ["hypothesis"],
    deps = [
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:hypothesis_testlib",
    ],
)

multi_substrate_py_library(
    name = "pointwise_exponential",
    srcs = ["pointwise_exponential.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:parameter_properties",
    ],
)

multi_substrate_py_test(
    name = "pointwise_exponential_test",
    srcs = ["pointwise_exponential_test.py"],
    deps = [
        ":matern",
        ":parabolic",
        ":pointwise_exponential",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "exponential_curve",
    srcs = ["exponential_curve.py"],
    deps = [
        ":positive_semidefinite_kernel",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "exponential_curve_test",
    size = "medium",
    srcs = ["exponential_curve_test.py"],
    deps = [
        ":exponential_curve",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)
