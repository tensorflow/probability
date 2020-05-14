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
#   Contains ops for bijectors.
#   APIs here are meant to evolve over time.

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

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "bijectors",
    srcs = ["__init__.py"],
    substrates_omit_deps = [
        ":real_nvp",
        ":scale_matvec_lu",
    ],
    deps = [
        ":absolute_value",
        ":affine",
        ":affine_linear_operator",
        ":affine_scalar",
        ":batch_normalization",
        ":bijector",
        ":blockwise",
        ":categorical_to_discrete",
        ":chain",
        ":cholesky_outer_product",
        ":cholesky_to_inv_cholesky",
        ":correlation_cholesky",
        ":cumsum",
        ":discrete_cosine_transform",
        ":exp",
        ":expm1",
        ":ffjord",
        ":fill_scale_tril",
        ":fill_triangular",
        ":frechet_cdf",
        ":generalized_pareto",
        ":gumbel_cdf",
        ":identity",
        ":inline",
        ":invert",
        ":iterated_sigmoid_centered",
        ":kumaraswamy_cdf",
        ":lambertw_transform",
        ":masked_autoregressive",
        ":matrix_inverse_tril",
        ":normal_cdf",
        ":ordered",
        ":pad",
        ":permute",
        ":power_transform",
        ":rational_quadratic_spline",
        ":real_nvp",
        ":reciprocal",
        ":reshape",
        ":scale",
        ":scale_matvec_diag",
        ":scale_matvec_linear_operator",
        ":scale_matvec_lu",
        ":scale_matvec_tril",
        ":shift",
        ":sigmoid",
        ":sinh",
        ":sinh_arcsinh",
        ":soft_clip",
        ":softfloor",
        ":softmax_centered",
        ":softplus",
        ":softsign",
        ":square",
        ":tanh",
        ":transform_diagonal",
        ":transpose",
        ":weibull_cdf",
    ],
)

multi_substrate_py_library(
    name = "bijector",
    srcs = [
        "bijector.py",
        "chain.py",
    ],
    deps = [
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:cache_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:name_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

py_test(
    name = "bijector_properties_test",
    size = "medium",
    srcs = ["bijector_properties_test.py"],
    shard_count = 20,
    tags = ["hypothesis"],
    deps = [
        ":categorical_to_discrete",
        ":hypothesis_testlib",
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
    ],
)

multi_substrate_py_library(
    name = "absolute_value",
    srcs = ["absolute_value.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

multi_substrate_py_library(
    name = "affine_linear_operator",
    srcs = ["affine_linear_operator.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "affine_scalar",
    srcs = ["affine_scalar.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "affine",
    srcs = ["affine.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "scale",
    srcs = ["scale.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "shift",
    srcs = ["shift.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "scale_matvec_linear_operator",
    srcs = ["scale_matvec_linear_operator.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "scale_matvec_diag",
    srcs = ["scale_matvec_diag.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "scale_matvec_tril",
    srcs = ["scale_matvec_tril.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "batch_normalization",
    srcs = ["batch_normalization.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "blockwise",
    srcs = ["blockwise.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "categorical_to_discrete",
    srcs = ["categorical_to_discrete.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "chain",
    srcs = [],  # chain.py is already a src for :bijector
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "cholesky_outer_product",
    srcs = ["cholesky_outer_product.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "cholesky_to_inv_cholesky",
    srcs = ["cholesky_to_inv_cholesky.py"],
    deps = [
        ":bijector",
        ":cholesky_outer_product",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "correlation_cholesky",
    srcs = ["correlation_cholesky.py"],
    deps = [
        ":bijector",
        ":fill_triangular",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "cumsum",
    srcs = ["cumsum.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_library(
    name = "discrete_cosine_transform",
    srcs = ["discrete_cosine_transform.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
#         "//tensorflow_probability/google:platform_google",  # DisableOnExport
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "exp",
    srcs = ["exp.py"],
    deps = [
        ":invert",
        ":power_transform",
    ],
)

multi_substrate_py_library(
    name = "expm1",
    srcs = ["expm1.py"],
    deps = [
        ":bijector",
        ":invert",
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "ffjord",
    srcs = ["ffjord.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:cache_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "fill_triangular",
    srcs = ["fill_triangular.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:linalg",
    ],
)

multi_substrate_py_library(
    name = "frechet_cdf",
    srcs = ["frechet_cdf.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "generalized_pareto",
    srcs = ["generalized_pareto.py"],
    deps = [
        ":bijector",
        ":chain",
        ":shift",
        ":sigmoid",
        ":softplus",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "gumbel_cdf",
    srcs = ["gumbel_cdf.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "identity",
    srcs = ["identity.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "inline",
    srcs = ["inline.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "invert",
    srcs = ["invert.py"],
    deps = [
        ":bijector",
    ],
)

multi_substrate_py_library(
    name = "kumaraswamy_cdf",
    srcs = ["kumaraswamy_cdf.py"],
    deps = [
        ":bijector",
        ":invert",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "lambertw_transform",
    srcs = ["lambertw_transform.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "masked_autoregressive",
    srcs = ["masked_autoregressive.py"],
    deps = [
        ":affine_scalar",
        ":bijector",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:numeric",
    ],
)

multi_substrate_py_library(
    name = "matrix_inverse_tril",
    srcs = ["matrix_inverse_tril.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "scale_matvec_lu",
    srcs = ["scale_matvec_lu.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:linalg",
    ],
)

multi_substrate_py_library(
    name = "normal_cdf",
    srcs = ["normal_cdf.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:special_math",
    ],
)

multi_substrate_py_library(
    name = "ordered",
    srcs = ["ordered.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

multi_substrate_py_library(
    name = "pad",
    srcs = ["pad.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "permute",
    srcs = ["permute.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "power_transform",
    srcs = ["power_transform.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
    ],
)

multi_substrate_py_library(
    name = "rational_quadratic_spline",
    srcs = ["rational_quadratic_spline.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util",
    ],
)

py_library(
    name = "real_nvp",
    srcs = ["real_nvp.py"],
    deps = [
        ":affine_scalar",
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "reciprocal",
    srcs = ["reciprocal.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

multi_substrate_py_library(
    name = "reshape",
    srcs = ["reshape.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "fill_scale_tril",
    srcs = ["fill_scale_tril.py"],
    deps = [
        ":affine_scalar",
        ":chain",
        ":fill_triangular",
        ":softplus",
        ":transform_diagonal",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "sigmoid",
    srcs = ["sigmoid.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "sinh",
    srcs = ["sinh.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "sinh_arcsinh",
    srcs = ["sinh_arcsinh.py"],
    deps = [
        ":bijector",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "soft_clip",
    srcs = ["soft_clip.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "softfloor",
    srcs = ["softfloor.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "softmax_centered",
    srcs = ["softmax_centered.py"],
    deps = [
        ":bijector",
        ":pad",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_library(
    name = "softplus",
    srcs = ["softplus.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "softsign",
    srcs = ["softsign.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "square",
    srcs = ["square.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

multi_substrate_py_library(
    name = "iterated_sigmoid_centered",
    srcs = ["iterated_sigmoid_centered.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "tanh",
    srcs = ["tanh.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "transform_diagonal",
    srcs = ["transform_diagonal.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "transpose",
    srcs = ["transpose.py"],
    deps = [
        ":bijector",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "weibull_cdf",
    srcs = ["weibull_cdf.py"],
    deps = [
        ":bijector",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "bijector_test_util",
    testonly = 1,
    srcs = ["bijector_test_util.py"],
    deps = [
        ":reshape",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "hypothesis_testlib",
    testonly = 1,
    srcs = ["hypothesis_testlib.py"],
    deps = [
        # absl/logging dep,
        # hypothesis dep,
        # six dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "bijector_test",
    size = "small",
    srcs = ["bijector_test.py"],
    numpy_tags = ["notap"],
    deps = [
        ":bijectors",
        # mock dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "bijector_composition_test",
    size = "small",
    srcs = ["bijector_composition_test.py"],
    deps = [
        # mock dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "absolute_value_test",
    size = "small",
    srcs = ["absolute_value_test.py"],
    deps = [
        ":bijectors",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "affine_test",
    size = "small",
    srcs = ["affine_test.py"],
    shard_count = 5,
    tags = ["noasan"],  # times out b/63678675
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "affine_linear_operator_test",
    size = "small",
    srcs = ["affine_linear_operator_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "affine_scalar_test",
    size = "small",
    srcs = ["affine_scalar_test.py"],
    deps = [
        ":bijector_test_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "scale_test",
    size = "small",
    srcs = ["scale_test.py"],
    deps = [
        ":bijector_test_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "shift_test",
    size = "small",
    srcs = ["shift_test.py"],
    deps = [
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "scale_matvec_diag_test",
    size = "small",
    srcs = ["scale_matvec_diag_test.py"],
    deps = [
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "scale_matvec_tril_test",
    size = "small",
    srcs = ["scale_matvec_tril_test.py"],
    deps = [
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "scale_matvec_linear_operator_test",
    size = "small",
    srcs = ["scale_matvec_linear_operator_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "batch_normalization_test",
    size = "small",
    srcs = ["batch_normalization_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "blockwise_test",
    size = "small",
    srcs = ["blockwise_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "categorical_to_discrete_test",
    size = "small",
    srcs = ["categorical_to_discrete_test.py"],
    deps = [
        ":bijector_test_util",
        ":categorical_to_discrete",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "chain_test",
    size = "small",
    srcs = ["chain_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "cholesky_outer_product_test",
    size = "small",
    srcs = ["cholesky_outer_product_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "cholesky_to_inv_cholesky_test",
    size = "small",
    srcs = ["cholesky_to_inv_cholesky_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "correlation_cholesky_test",
    size = "medium",
    srcs = ["correlation_cholesky_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:cholesky_lkj",
        "//tensorflow_probability/python/distributions:lkj",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:sample",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
    ],
)

multi_substrate_py_test(
    name = "cumsum_test",
    size = "small",
    srcs = ["cumsum_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "discrete_cosine_transform_test",
    size = "small",
    srcs = ["discrete_cosine_transform_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "exp_test",
    size = "small",
    srcs = ["exp_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "expm1_test",
    size = "small",
    srcs = ["expm1_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "ffjord_test",
    size = "medium",
    srcs = ["ffjord_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "fill_triangular_test",
    size = "small",
    srcs = ["fill_triangular_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "frechet_cdf_test",
    size = "small",
    srcs = ["frechet_cdf_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "generalized_pareto_test",
    size = "small",
    srcs = ["generalized_pareto_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "gumbel_cdf_test",
    size = "small",
    srcs = ["gumbel_cdf_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "identity_test",
    size = "small",
    srcs = ["identity_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "inline_test",
    size = "small",
    srcs = ["inline_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "invert_test",
    size = "small",
    srcs = ["invert_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "kumaraswamy_cdf_test",
    size = "small",
    srcs = ["kumaraswamy_cdf_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "lambertw_transform_test",
    size = "small",
    srcs = ["lambertw_transform_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        ":lambertw_transform",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "masked_autoregressive_test",
    size = "small",
    timeout = "moderate",
    srcs = ["masked_autoregressive_test.py"],
    deps = [
        ":bijectors",
        ":masked_autoregressive",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_test(
    name = "matrix_inverse_tril_test",
    size = "small",
    srcs = ["matrix_inverse_tril_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "scale_matvec_lu_test",
    size = "small",
    srcs = ["scale_matvec_lu_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "normal_cdf_test",
    size = "small",
    srcs = ["normal_cdf_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "ordered_test",
    size = "small",
    srcs = ["ordered_test.py"],
    deps = [
        ":bijector_test_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "pad_test",
    size = "small",
    srcs = ["pad_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "permute_test",
    size = "small",
    srcs = ["permute_test.py"],
    numpy_tags = ["notap"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "power_transform_test",
    size = "small",
    srcs = ["power_transform_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "real_nvp_test",
    size = "small",
    srcs = ["real_nvp_test.py"],
    deps = [
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "rational_quadratic_spline_test",
    size = "medium",
    srcs = ["rational_quadratic_spline_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    tags = ["hypothesis"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        ":hypothesis_testlib",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "reciprocal_test",
    size = "small",
    srcs = ["reciprocal_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "reshape_test",
    size = "small",
    srcs = ["reshape_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "fill_scale_tril_test",
    size = "small",
    srcs = ["fill_scale_tril_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "sigmoid_test",
    size = "small",
    srcs = ["sigmoid_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_test(
    name = "sinh_test",
    size = "small",
    srcs = ["sinh_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "sinh_arcsinh_test",
    size = "small",
    srcs = ["sinh_arcsinh_test.py"],
    tags = ["no_windows"],  # TODO: needs investigation on Windows.
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "soft_clip_test",
    size = "small",
    srcs = ["soft_clip_test.py"],
    numpy_tags = ["notap"],  # TODO(b/149350565): remove this.
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "softfloor_test",
    size = "small",
    srcs = ["softfloor_test.py"],
    jax_size = "medium",
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "softmax_centered_test",
    size = "small",
    srcs = ["softmax_centered_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "softplus_test",
    size = "small",
    srcs = ["softplus_test.py"],
    jax_size = "medium",
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_test(
    name = "softsign_test",
    size = "small",
    srcs = ["softsign_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "square_test",
    size = "small",
    srcs = ["square_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "iterated_sigmoid_centered_test",
    size = "small",
    srcs = ["iterated_sigmoid_centered_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_test(
    name = "tanh_test",
    size = "small",
    srcs = ["tanh_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "transform_diagonal_test",
    size = "medium",
    srcs = ["transform_diagonal_test.py"],
    tags = ["hypothesis"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        ":hypothesis_testlib",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "transpose_test",
    size = "small",
    srcs = ["transpose_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "weibull_cdf_test",
    size = "small",
    srcs = ["weibull_cdf_test.py"],
    deps = [
        ":bijector_test_util",
        ":bijectors",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

exports_files(
    [
        "__init__.py",
        "absolute_value.py",
        "absolute_value_test.py",
        "affine.py",
        "affine.py",
        "affine_linear_operator.py",
        "affine_linear_operator_test.py",
        "affine_scalar.py",
        "affine_scalar_test.py",
        "batch_normalization.py",
        "bijector.py",
        "bijector_test_util.py",
        "blockwise.py",
        "blockwise_test.py",
        "categorical_to_discrete.py",
        "chain.py",
        "chain_test.py",
        "cholesky_outer_product.py",
        "cholesky_outer_product_test.py",
        "cholesky_to_inv_cholesky.py",
        "cholesky_to_inv_cholesky_test.py",
        "correlation_cholesky.py",
        "cumsum.py",
        "cumsum_test.py",
        "discrete_cosine_transform.py",
        "exp.py",
        "exp_test.py",
        "expm1.py",
        "expm1_test.py",
        "fill_scale_tril.py",
        "fill_scale_tril_test.py",
        "fill_triangular.py",
        "fill_triangular_test.py",
        "generalized_pareto.py",
        "generalized_pareto_test.py",
        "gumbel.py",
        "gumbel_cdf.py",
        "gumbel_cdf_test.py",
        "hypothesis_testlib.py",
        "identity.py",
        "identity_test.py",
        "inline.py",
        "inline_test.py",
        "invert.py",
        "invert_test.py",
        "iterated_sigmoid_centered.py",
        "iterated_sigmoid_centered_test.py",
        "kumaraswamy.py",
        "kumaraswamy.py",
        "kumaraswamy_cdf.py",
        "kumaraswamy_cdf_test.py",
        "lambertw_transform.py",
        "lambertw_transform_test.py",
        "matrix_inverse_tril.py",
        "matrix_inverse_tril_test.py",
        "normal_cdf.py",
        "normal_cdf_test.py",
        "ordered.py",
        "ordered_test.py",
        "pad.py",
        "pad_test.py",
        "permute.py",
        "power_transform.py",
        "power_transform_test.py",
        "rational_quadratic_spline.py",
        "reciprocal.py",
        "reciprocal_test.py",
        "reshape.py",
        "reshape.py",
        "reshape_test.py",
        "scale.py",
        "scale_matvec_diag.py",
        "scale_matvec_linear_operator.py",
        "scale_matvec_tril.py",
        "shift.py",
        "sigmoid.py",
        "sigmoid_test.py",
        "sinh.py",
        "sinh_test.py",
        "sinh_arcsinh.py",
        "sinh_arcsinh_test.py",
        "soft_clip.py",
        "softfloor.py",
        "soft_clip_test.py",
        "softfloor_test.py",
        "softmax_centered.py",
        "softmax_centered_test.py",
        "softplus.py",
        "softplus_test.py",
        "softsign.py",
        "softsign_test.py",
        "square.py",
        "square_test.py",
        "tanh.py",
        "tanh_test.py",
        "transform_diagonal.py",
        "transform_diagonal_test.py",
        "transpose.py",
        "transpose_test.py",
        "weibull.py",
        "weibull_cdf.py",
        "weibull_cdf_test.py",
    ],
    visibility = ["//tensorflow_probability:__subpackages__"],
)

exports_files(
    ["ffjord.py"],
    visibility = ["//tensorflow_probability:__subpackages__"],
)
