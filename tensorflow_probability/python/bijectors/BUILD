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

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "bijectors",
    srcs = glob(
        ["*.py"],
        exclude = glob([
            "*_test.py",
        ]) + ["shape.py"],
    ),
    srcs_version = "PY2AND3",
    deps = [
        ":shape",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "shape",
    srcs = ["shape.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "shape_test",
    size = "medium",
    srcs = ["shape_test.py"],
    deps = [
        ":shape",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "conditional_bijector_test",
    size = "small",
    srcs = ["conditional_bijector_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "absolute_value_test",
    size = "small",
    srcs = ["absolute_value_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "affine_test",
    size = "large",
    srcs = ["affine_test.py"],
    shard_count = 5,
    tags = ["noasan"],  # times out b/63678675
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "affine_scalar_test",
    size = "small",
    srcs = ["affine_scalar_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "affine_linear_operator_test",
    size = "small",
    srcs = ["affine_linear_operator_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "batch_normalization_test",
    size = "small",
    srcs = ["batch_normalization_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "chain_test",
    size = "small",
    srcs = ["chain_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "cholesky_outer_product_test",
    size = "small",
    srcs = ["cholesky_outer_product_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "exp_test",
    size = "small",
    srcs = ["exp_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "fill_triangular_test",
    size = "small",
    srcs = ["fill_triangular_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "gumbel_test",
    size = "small",
    srcs = ["gumbel_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "inline_test",
    size = "small",
    srcs = ["inline_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "invert_test",
    size = "small",
    srcs = ["invert_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "kumaraswamy_bijector_test",
    size = "small",
    srcs = ["kumaraswamy_bijector_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "masked_autoregressive_test",
    size = "small",
    srcs = ["masked_autoregressive_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "matrix_inverse_tril_test",
    size = "small",
    srcs = ["matrix_inverse_tril_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "real_nvp_test",
    size = "small",
    srcs = ["real_nvp_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "permute_test",
    size = "small",
    srcs = ["permute_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "power_transform_test",
    size = "small",
    srcs = ["power_transform_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "reshape_test",
    size = "medium",
    srcs = ["reshape_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "scale_tril_test",
    size = "medium",
    srcs = ["scale_tril_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "sigmoid_test",
    size = "small",
    srcs = ["sigmoid_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

# Tests for SinhArcSinh bijector.  The file name has the extra "_bijector" to
# avoid BUILD rule name conflicts with the distribution by the same name.
py_test(
    name = "sinh_arcsinh_bijector_test",
    size = "small",
    srcs = ["sinh_arcsinh_bijector_test.py"],
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "softmax_centered_test",
    size = "small",
    srcs = ["softmax_centered_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "softplus_test",
    size = "small",
    srcs = ["softplus_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "softsign_test",
    size = "small",
    srcs = ["softsign_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "square_test",
    size = "small",
    srcs = ["square_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "transform_diagonal_test",
    size = "small",
    srcs = ["transform_diagonal_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "weibull_test",
    size = "small",
    srcs = ["weibull_test.py"],
    deps = [
        ":bijectors",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)
