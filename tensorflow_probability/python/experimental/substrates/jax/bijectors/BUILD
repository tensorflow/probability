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
# Bijectors with the JAX backend.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

FILENAMES = [
    "absolute_value",
    "affine_linear_operator",
    "affine",
    "affine_scalar",
    "batch_normalization",
    "bijector",
    "blockwise",
    "categorical_to_discrete",
    "chain",
    "cholesky_outer_product",
    "cholesky_to_inv_cholesky",
    "correlation_cholesky",
    "cumsum",
    "discrete_cosine_transform",
    "expm1",
    "exp",
    "fill_triangular",
    "gumbel",
    "identity",
    "inline",
    "invert",
    "iterated_sigmoid_centered",
    "kumaraswamy",
    # "masked_autoregressive",
    "matrix_inverse_tril",
    # "matveclu",
    "normal_cdf",
    "ordered",
    "permute",
    "power_transform",
    # "real_nvp",
    "rational_quadratic_spline",
    "reciprocal",
    "reshape",
    "scale_tril",
    "sigmoid",
    "sinh_arcsinh",
    "softfloor",
    "softmax_centered",
    "softplus",
    "softsign",
    "square",
    "tanh",
    "transform_diagonal",
    "transpose",
    "weibull",
]

TEST_FILENAMES = [filename + "_test" for filename in FILENAMES]

TAP_BLACKLIST = ["{}_test".format(f) for f in (
    "affine",
    "batch_normalization",
    "bijector",
    "categorical_to_discrete",
    "correlation_cholesky",
    "discrete_cosine_transform",
    "permute",
    "rational_quadratic_spline",
    "reshape",
    "sinh_arcsinh",
)]

[genrule(
    name = "rewrite_{}".format(filename),
    srcs = ["//tensorflow_probability/python/bijectors:{}.py".format(filename)],
    outs = ["{}.py".format(filename)],
    cmd = "$(location //tensorflow_probability/python/experimental/substrates/numpy:rewrite) $(SRCS) --numpy_to_jax > $@",
    tools = ["//tensorflow_probability/python/experimental/substrates/numpy:rewrite"],
) for filename in [
    "__init__",
    "bijector_test_util",
    "hypothesis_testlib",
] + FILENAMES + TEST_FILENAMES]

DEPS = {
    "correlation_cholesky_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/distributions:lkj",
    ],
    "hypothesis_testlib": ["//tensorflow_probability"],
    "invert_test": ["//tensorflow_probability/python/experimental/substrates/jax"],
    "ordered_test": ["//tensorflow_probability"],
    "transpose_test": ["//tensorflow_probability/python/experimental/substrates/jax"],
}

[py_library(
    name = filename,
    srcs = ["{}.py".format(filename)],
    deps = [
        "//tensorflow_probability/python/experimental/substrates/jax/internal",
        "//tensorflow_probability/python/internal/backend/jax",
    ] + DEPS.get(filename, []),
) for filename in ["hypothesis_testlib"] + FILENAMES]

py_library(
    name = "bijector_test_util",
    srcs = ["bijector_test_util.py"],
    deps = [
        "//tensorflow_probability/python/experimental/substrates/jax/distributions:independent",
        "//tensorflow_probability/python/experimental/substrates/jax/distributions:uniform",
    ],
)

[py_test(
    name = filename,
    size = "medium",
    srcs = ["{}.py".format(filename)],
    tags = (["notap"] if filename in TAP_BLACKLIST else []) + ["tfp_jax"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        ":hypothesis_testlib",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # jax dep,
        # mock dep,
        # numpy dep,
        # scipy dep,
        # six dep,
        "//tensorflow_probability/python/experimental/substrates/jax/distributions:deprecated_linalg",
        "//tensorflow_probability/python/experimental/substrates/jax/internal:assert_util",
        "//tensorflow_probability/python/experimental/substrates/jax/internal:distribution_util",
        "//tensorflow_probability/python/experimental/substrates/jax/internal:dtype_util",
        "//tensorflow_probability/python/experimental/substrates/jax/internal:hypothesis_testlib",
        "//tensorflow_probability/python/experimental/substrates/jax/internal:prefer_static",
        "//tensorflow_probability/python/experimental/substrates/jax/internal:test_case",
        "//tensorflow_probability/python/experimental/substrates/jax/internal:test_util",
        "//tensorflow_probability/python/internal:name_util",
        "//tensorflow_probability/python/internal/backend/jax",
    ] + DEPS.get(filename, []),
) for filename in TEST_FILENAMES]

py_library(
    name = "bijectors",
    srcs = ["__init__.py"],
    deps = [":{}".format(filename) for filename in FILENAMES],
)
