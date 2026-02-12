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

# Placeholder: py_library
# Placeholder: py_test
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
    name = "math",
    srcs = ["__init__.py"],
    substrates_omit_deps = [
        ":sparse",
    ],
    deps = [
        ":bessel",
        ":custom_gradient",
        ":diag_jacobian",
        ":generic",
        ":gradient",
        ":gram_schmidt",
        ":hypergeometric",
        ":integration",
        ":interpolation",
        ":linalg",
        ":minimize",
        ":numeric",
        ":root_search",
        ":scan_associative",
        ":sparse",
        ":special",
        "//tensorflow_probability/python/internal:all_util",
        "//tensorflow_probability/python/math/ode",
        "//tensorflow_probability/python/math/psd_kernels",
    ],
)

multi_substrate_py_library(
    name = "bessel",
    srcs = [
        "bessel.py",
    ],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "bessel_test",
    size = "medium",
    srcs = ["bessel_test.py"],
    shard_count = 7,
    deps = [
        ":bessel",
        ":gradient",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "custom_gradient",
    srcs = [
        "custom_gradient.py",
    ],
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
        "gradient",
        ":custom_gradient",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "gram_schmidt",
    srcs = [
        "gram_schmidt.py",
    ],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "gram_schmidt_test",
    size = "small",
    srcs = ["gram_schmidt_test.py"],
    deps = [
        ":gram_schmidt",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_library(
    name = "generic",
    srcs = ["generic.py"],
    deps = [
        ":scan_associative",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:custom_gradient",
        "//tensorflow_probability/python/internal:distribute_lib",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:variadic_reduce",
    ],
)

multi_substrate_py_test(
    name = "generic_test",
    size = "medium",
    srcs = ["generic_test.py"],
    shard_count = 3,
    deps = [
        ":generic",
        ":gradient",
        ":linalg",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:distribute_test_lib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "gradient",
    srcs = [
        "gradient.py",
    ],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_test(
    name = "gradient_test",
    size = "small",
    srcs = ["gradient_test.py"],
    deps = [
        ":gradient",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "hypergeometric",
    srcs = [
        "hypergeometric.py",
    ],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "hypergeometric_test",
    size = "medium",
    srcs = ["hypergeometric_test.py"],
    jax_size = "large",
    shard_count = 3,
    deps = [
        # absl/testing:parameterized dep,
        # mpmath dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:hypergeometric",
    ],
)

multi_substrate_py_library(
    name = "integration",
    srcs = [
        "integration.py",
    ],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "integration_test",
    srcs = ["integration_test.py"],
    deps = [
        ":integration",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_library(
    name = "interpolation",
    srcs = [
        "interpolation.py",
    ],
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
    deps = [
        ":gradient",
        ":interpolation",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "linalg",
    srcs = [
        "linalg.py",
    ],
    deps = [
        ":generic",
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
    shard_count = 6,
    tags = ["hypothesis"],
    deps = [
        ":gradient",
        ":linalg",
        ":math",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:fill_scale_tril",
        "//tensorflow_probability/python/experimental/linalg",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:matern",
    ],
)

multi_substrate_py_library(
    name = "minimize",
    srcs = [
        "minimize.py",
    ],
    deps = [
        ":gradient",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:loop_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tf_keras",
    ],
)

multi_substrate_py_test(
    name = "minimize_test",
    size = "small",
    srcs = ["minimize_test.py"],
    numpy_tags = ["notap"],
    deps = [
        ":minimize",
        # numpy dep,
        # optax dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/internal:tf_keras",
        "//tensorflow_probability/python/optimizer",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "numeric",
    srcs = [
        "numeric.py",
    ],
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
        ":gradient",
        ":numeric",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "diag_jacobian",
    srcs = [
        "diag_jacobian.py",
    ],
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
    deps = [
        ":diag_jacobian",
        ":gradient",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "root_search",
    srcs = [
        "root_search.py",
    ],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:callable_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "root_search_test",
    size = "small",
    srcs = ["root_search_test.py"],
    deps = [
        ":root_search",
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:special_math",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "sparse",
    srcs = [
        "sparse.py",
    ],
#     strict_deps = False,  # DisableOnExport
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_test(
    name = "sparse_test",
    size = "small",
    srcs = ["sparse_test.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":sparse",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "scan_associative",
    srcs = ["scan_associative.py"],
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
    shard_count = 4,
    deps = [
        ":gradient",
        ":scan_associative",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "special",
    srcs = [
        "special.py",
    ],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:custom_gradient",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
    ],
)

multi_substrate_py_test(
    name = "special_test",
    size = "medium",
    srcs = ["special_test.py"],
    shard_count = 10,
    deps = [
        ":gradient",
        ":special",
        # absl/testing:parameterized dep,
        # mpmath dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:half_cauchy",
        "//tensorflow_probability/python/distributions:half_normal",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
