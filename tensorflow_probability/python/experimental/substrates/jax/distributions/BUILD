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
# Distributions with the JAX backend.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

FILENAMES = [
    "autoregressive",
    "batch_reshape",
    "bernoulli",
    "beta",
    "binomial",
    "blockwise",
    "categorical",
    "cauchy",
    "chi2",
    "chi",
    "cholesky_lkj",
    "deterministic",
    "dirichlet_multinomial",
    "dirichlet",
    "distribution",
    "doublesided_maxwell",
    "empirical",
    "exponential",
    "finite_discrete",
    "gamma_gamma",
    "gamma",
    "gaussian_process",
    "gaussian_process_regression_model",
    "generalized_pareto",
    "geometric",
    "gumbel",
    "half_cauchy",
    "half_normal",
    "hidden_markov_model",
    "horseshoe",
    "independent",
    "inverse_gamma",
    "inverse_gaussian",
    "joint_distribution_coroutine",
    "joint_distribution_named",
    "joint_distribution",
    "joint_distribution_sequential",
    "kullback_leibler",
    "kumaraswamy",
    "laplace",
    "linear_gaussian_ssm",
    "lkj",
    "logistic",
    "lognormal",
    "logitnormal",
    "mixture",
    "mixture_same_family",
    "multinomial",
    "multivariate_student_t",
    "mvn_diag_plus_low_rank",
    "mvn_diag",
    "mvn_full_covariance",
    "mvn_linear_operator",
    "mvn_tril",
    "negative_binomial",
    "normal_conjugate_posteriors",
    "normal",
    "onehot_categorical",
    "pareto",
    "pert",
    "pixel_cnn",
    "plackett_luce",
    "poisson_lognormal",
    "poisson",
    "probit_bernoulli",
    "quantized_distribution",
    "relaxed_bernoulli",
    "relaxed_onehot_categorical",
    "sample",
    "sinh_arcsinh",
    "student_t_process",
    "student_t",
    "transformed_distribution",
    "triangular",
    "truncated_normal",
    "uniform",
    "variational_gaussian_process",
    "vector_diffeomixture",
    "vector_exponential_diag",
    "vector_exponential_linear_operator",
    "vector_laplace_diag",
    "vector_laplace_linear_operator",
    "vector_sinh_arcsinh_diag",
    "von_mises_fisher",
    "von_mises",
    "wishart",
    "zipf",
]

NO_TESTS = ("joint_distribution", "vector_exponential_linear_operator", "vector_laplace_linear_operator")

TEST_FILENAMES = [filename + "_test" for filename in FILENAMES if filename not in NO_TESTS]

TEST_SIZES = {
    "cholesky_lkj_test": "large",
}

TAP_WHITELIST = ["{}_test".format(filename) for filename in (
    "cauchy",
    "cholesky_lkj",
    "exponential",
    "gamma_gamma",
    "gaussian_process",
    "gaussian_process_regression_model",
    "geometric",
    "gumbel",
    "half_cauchy",
    "inverse_gaussian",
    "kullback_leibler",
    "kumaraswamy",
    "laplace",
    "logitnormal",
    "lognormal",
    "mvn_diag_plus_low_rank",
    "mvn_full_covariance",
    "relaxed_bernoulli",
    "sinh_arcsinh",
    "sample",
    "student_t_process",
    "variational_gaussian_process",
    "vector_exponential_diag",
    "vector_laplace_diag",
    "vector_sinh_arcsinh_diag",
)]

[genrule(
    name = "rewrite_{}".format(filename),
    srcs = ["//tensorflow_probability/python/distributions:{}.py".format(filename)],
    outs = ["{}.py".format(filename)],
    cmd = "$(location //tensorflow_probability/python/experimental/substrates/meta:rewrite) $(SRCS) --numpy_to_jax > $@",
    tools = ["//tensorflow_probability/python/experimental/substrates/meta:rewrite"],
) for filename in FILENAMES + TEST_FILENAMES]

genrule(
    name = "_init_prefix",
    srcs = ["//tensorflow_probability/python/distributions:__init__.py"],
    outs = ["_init_prefix.py"],
    cmd = "$(location //tensorflow_probability/python/experimental/substrates/meta:rewrite) $(SRCS) --numpy_to_jax > $@",
    tools = ["//tensorflow_probability/python/experimental/substrates/meta:rewrite"],
)

genrule(
    name = "__init__",
    srcs = [
        ":_init_prefix.py",
        "_init_suffix.py",
    ],
    outs = ["__init__.py"],
    cmd = "cat $(SRCS) > $@",
)

COMMON_DEPS = {
    "distribution": [":kullback_leibler"],
    "kullback_leibler": [],
}

DEPS = {
    "lkj": [":beta"],
    "cholesky_lkj": [":lkj"],
    "chi": [
        ":chi2",
        ":transformed_distribution",
        "//tensorflow_probability/python/experimental/substrates/jax/bijectors:invert",
        "//tensorflow_probability/python/experimental/substrates/jax/bijectors:square",
    ],
    "von_mises_fisher": [":beta"],
    "von_mises": [":normal"],
    "generalized_pareto_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/internal:hypothesis_testlib",
    ],
    "independent_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/internal:hypothesis_testlib",
    ],
    "linear_gaussian_ssm_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/internal:hypothesis_testlib",
    ],
    "lkj_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/distributions/internal:statistical_testing",
    ],
    "mixture_same_family_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/internal:hypothesis_testlib",
    ],
    "mixture_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/internal:hypothesis_testlib",
    ],
    "pert_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/distributions/internal:statistical_testing",
    ],
    "transformed_distribution_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/internal:hypothesis_testlib",
    ],
    "triangular_test": [
        "//tensorflow_probability/python/experimental/substrates/jax/distributions/internal:statistical_testing",
    ],
}

[py_library(
    name = filename,
    srcs = ["{}.py".format(filename)],
    deps = (COMMON_DEPS.get(
        filename,
        [":{}".format(k) for k in COMMON_DEPS.keys()],
    ) + DEPS.get(filename, []) + [
        "//tensorflow_probability/python/internal/backend/jax",
        "//tensorflow_probability/python/experimental/substrates/jax/bijectors",
        "//tensorflow_probability/python/experimental/substrates/jax/internal",
        "//tensorflow_probability/python/experimental/substrates/jax/math",
        "//tensorflow_probability/python/experimental/substrates/jax/distributions/internal",
    ]),
) for filename in FILENAMES]

[py_test(
    name = filename,
    size = TEST_SIZES.get(filename, "medium"),
    srcs = ["{}.py".format(filename)],
    tags = ([] if filename in TAP_WHITELIST else ["notap"]) + ["tfp_jax"],
    deps = DEPS.get(filename, []) + [
        ":distributions",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # jax dep,
        # numpy dep,
        # scipy dep,
        # six dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/substrates/jax/internal:test_util",
    ],
) for filename in TEST_FILENAMES]

py_library(
    name = "distributions",
    srcs = ["__init__.py"],
    deps = [":{}".format(filename) for filename in FILENAMES] + [
        "//tensorflow_probability/python/experimental/substrates/jax/util:seed_stream",
    ],
)
