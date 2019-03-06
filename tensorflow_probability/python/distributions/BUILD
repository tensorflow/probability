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
#   Contains ops for statistical distributions (with pdf, cdf, sample, etc...).
#   APIs here are meant to evolve over time.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "distributions",
    srcs = ["__init__.py"],
    deps = [
        ":autoregressive",
        ":batch_reshape",
        ":bernoulli",
        ":beta",
        ":binomial",
        ":blockwise",
        ":categorical",
        ":cauchy",
        ":chi",
        ":chi2",
        ":conditional_distribution",
        ":conditional_transformed_distribution",
        ":deterministic",
        ":dirichlet",
        ":dirichlet_multinomial",
        ":distribution",
        ":empirical",
        ":exponential",
        ":gamma",
        ":gamma_gamma",
        ":gaussian_process",
        ":gaussian_process_regression_model",
        ":geometric",
        ":gumbel",
        ":half_cauchy",
        ":half_normal",
        ":hidden_markov_model",
        ":horseshoe",
        ":independent",
        ":inverse_gamma",
        ":inverse_gaussian",
        ":kullback_leibler",
        ":kumaraswamy",
        ":laplace",
        ":linear_gaussian_ssm",
        ":lkj",
        ":logistic",
        ":lognormal",
        ":mixture",
        ":mixture_same_family",
        ":multinomial",
        ":multivariate_student_t",
        ":mvn_diag",
        ":mvn_diag_plus_low_rank",
        ":mvn_full_covariance",
        ":mvn_linear_operator",
        ":mvn_tril",
        ":negative_binomial",
        ":normal",
        ":normal_conjugate_posteriors",
        ":onehot_categorical",
        ":pareto",
        ":poisson",
        ":poisson_lognormal",
        ":quantized_distribution",
        ":relaxed_bernoulli",
        ":relaxed_onehot_categorical",
        ":sample_stats",
        ":seed_stream",
        ":sinh_arcsinh",
        ":student_t",
        ":student_t_process",
        ":transformed_distribution",
        ":triangular",
        ":truncated_normal",
        ":uniform",
        ":variational_gaussian_process",
        ":vector_diffeomixture",
        ":vector_exponential_diag",
        ":vector_laplace_diag",
        ":vector_sinh_arcsinh_diag",
        ":von_mises",
        ":von_mises_fisher",
        ":wishart",
        ":zipf",
        "//tensorflow_probability/python/distributions/internal",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "autoregressive",
    srcs = ["autoregressive.py"],
    deps = [
        ":distribution",
        ":seed_stream",
        # tensorflow dep,
    ],
)

py_library(
    name = "batch_reshape",
    srcs = ["batch_reshape.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "bernoulli",
    srcs = ["bernoulli.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "beta",
    srcs = ["beta.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "binomial",
    srcs = ["binomial.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "blockwise",
    srcs = ["blockwise.py"],
    deps = [
        ":distribution",
        ":seed_stream",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "categorical",
    srcs = ["categorical.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "cauchy",
    srcs = ["cauchy.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "chi",
    srcs = ["chi.py"],
    deps = [
        ":chi2",
        ":kullback_leibler",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:square",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "chi2",
    srcs = ["chi2.py"],
    deps = [
        ":gamma",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "conditional_distribution",
    srcs = ["conditional_distribution.py"],
    deps = [
        ":distribution",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "conditional_transformed_distribution",
    srcs = ["conditional_transformed_distribution.py"],
    deps = [
        ":conditional_distribution",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "deterministic",
    srcs = ["deterministic.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "dirichlet",
    srcs = ["dirichlet.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "dirichlet_multinomial",
    srcs = ["dirichlet_multinomial.py"],
    deps = [
        ":distribution",
        ":multinomial",
        ":seed_stream",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "distribution",
    srcs = ["distribution.py"],
    deps = [
        ":kullback_leibler",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions/internal:slicing",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "empirical",
    srcs = ["empirical.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "exponential",
    srcs = ["exponential.py"],
    deps = [
        ":gamma",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "gamma",
    srcs = ["gamma.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "gamma_gamma",
    srcs = ["gamma_gamma.py"],
    deps = [
        ":distribution",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "gaussian_process",
    srcs = ["gaussian_process.py"],
    deps = [
        ":mvn_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "gaussian_process_regression_model",
    srcs = ["gaussian_process_regression_model.py"],
    deps = [
        ":mvn_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "geometric",
    srcs = ["geometric.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "gumbel",
    srcs = ["gumbel.py"],
    deps = [
        ":transformed_distribution",
        ":uniform",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:gumbel",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "half_cauchy",
    srcs = ["half_cauchy.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "half_normal",
    srcs = ["half_normal.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
    ],
)

py_library(
    name = "hidden_markov_model",
    srcs = ["hidden_markov_model.py"],
    deps = [
        ":categorical",
        ":distribution",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "horseshoe",
    srcs = ["horseshoe.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "independent",
    srcs = ["independent.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "inverse_gamma",
    srcs = ["inverse_gamma.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "inverse_gaussian",
    srcs = ["inverse_gaussian.py"],
    deps = [
        ":distribution",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
    ],
)

py_library(
    name = "kullback_leibler",
    srcs = ["kullback_leibler.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "kumaraswamy",
    srcs = ["kumaraswamy.py"],
    deps = [
        ":transformed_distribution",
        ":uniform",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:kumaraswamy",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "laplace",
    srcs = ["laplace.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
    ],
)

py_library(
    name = "linear_gaussian_ssm",
    srcs = ["linear_gaussian_ssm.py"],
    deps = [
        ":distribution",
        ":mvn_tril",
        ":seed_stream",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "lkj",
    srcs = ["lkj.py"],
    deps = [
        ":beta",
        ":distribution",
        ":normal",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:test_case",
    ],
)

py_library(
    name = "logistic",
    srcs = ["logistic.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "lognormal",
    srcs = ["lognormal.py"],
    deps = [
        ":normal",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "mixture",
    srcs = ["mixture.py"],
    deps = [
        ":categorical",
        ":distribution",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "mixture_same_family",
    srcs = ["mixture_same_family.py"],
    deps = [
        ":distribution",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "mvn_diag_plus_low_rank",
    srcs = ["mvn_diag_plus_low_rank.py"],
    deps = [
        ":mvn_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "mvn_diag",
    srcs = ["mvn_diag.py"],
    deps = [
        ":mvn_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "mvn_full_covariance",
    srcs = ["mvn_full_covariance.py"],
    deps = [
        ":mvn_tril",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "mvn_linear_operator",
    srcs = ["mvn_linear_operator.py"],
    deps = [
        ":kullback_leibler",
        ":normal",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine_linear_operator",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "mvn_tril",
    srcs = ["mvn_tril.py"],
    deps = [
        ":mvn_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "multivariate_student_t",
    srcs = ["multivariate_student_t.py"],
    deps = [
        ":chi2",
        ":distribution",
        ":mvn_linear_operator",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/math",
    ],
)

py_library(
    name = "multinomial",
    srcs = ["multinomial.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "negative_binomial",
    srcs = ["negative_binomial.py"],
    deps = [
        ":distribution",
        ":seed_stream",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "normal",
    srcs = ["normal.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
    ],
)

py_library(
    name = "normal_conjugate_posteriors",
    srcs = ["normal_conjugate_posteriors.py"],
    deps = [
        ":normal",
        # tensorflow dep,
    ],
)

py_library(
    name = "onehot_categorical",
    srcs = ["onehot_categorical.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "pareto",
    srcs = ["pareto.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "poisson",
    srcs = ["poisson.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "poisson_lognormal",
    srcs = ["poisson_lognormal.py"],
    deps = [
        ":categorical",
        ":distribution",
        ":normal",
        ":poisson",
        ":seed_stream",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "quantized_distribution",
    srcs = ["quantized_distribution.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "relaxed_bernoulli",
    srcs = ["relaxed_bernoulli.py"],
    deps = [
        ":logistic",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "relaxed_onehot_categorical",
    srcs = ["relaxed_onehot_categorical.py"],
    deps = [
        ":distribution",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "sample_stats",
    srcs = ["sample_stats.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/stats",
    ],
)

py_library(
    name = "seed_stream",
    srcs = ["seed_stream.py"],
)

py_library(
    name = "sinh_arcsinh",
    srcs = ["sinh_arcsinh.py"],
    deps = [
        ":normal",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine_scalar",
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:sinh_arcsinh",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "student_t",
    srcs = ["student_t.py"],
    deps = [
        ":distribution",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "student_t_process",
    srcs = ["student_t_process.py"],
    deps = [
        ":multivariate_student_t",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "triangular",
    srcs = ["triangular.py"],
    deps = [
        ":distribution",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "transformed_distribution",
    srcs = ["transformed_distribution.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "truncated_normal",
    srcs = ["truncated_normal.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
    ],
)

py_library(
    name = "uniform",
    srcs = ["uniform.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "variational_gaussian_process",
    srcs = ["variational_gaussian_process.py"],
    deps = [
        ":gaussian_process",
        ":kullback_leibler",
        ":mvn_linear_operator",
        ":normal",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "vector_diffeomixture",
    srcs = ["vector_diffeomixture.py"],
    deps = [
        ":categorical",
        ":distribution",
        ":normal",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine_linear_operator",
        "//tensorflow_probability/python/bijectors:softmax_centered",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "vector_exponential_diag",
    srcs = ["vector_exponential_diag.py"],
    deps = [
        ":vector_exponential_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_exponential_linear_operator",
    srcs = ["vector_exponential_linear_operator.py"],
    deps = [
        ":exponential",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine_linear_operator",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_laplace_diag",
    srcs = ["vector_laplace_diag.py"],
    deps = [
        ":vector_laplace_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "vector_laplace_linear_operator",
    srcs = ["vector_laplace_linear_operator.py"],
    deps = [
        ":laplace",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine_linear_operator",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_sinh_arcsinh_diag",
    srcs = ["vector_sinh_arcsinh_diag.py"],
    deps = [
        ":normal",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine",
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:sinh_arcsinh",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "vector_student_t",
    srcs = ["vector_student_t.py"],
    deps = [
        ":student_t",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "von_mises",
    srcs = ["von_mises.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":normal",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/math:gradient",
    ],
)

py_library(
    name = "von_mises_fisher",
    srcs = ["von_mises_fisher.py"],
    deps = [
        ":beta",
        ":distribution",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "wishart",
    srcs = ["wishart.py"],
    deps = [
        ":distribution",
        ":seed_stream",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
    ],
)

py_library(
    name = "zipf",
    srcs = ["zipf.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_test(
    name = "autoregressive_test",
    size = "small",
    srcs = ["autoregressive_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "batch_reshape_test",
    size = "small",
    srcs = ["batch_reshape_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "bernoulli_test",
    srcs = ["bernoulli_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "beta_test",
    srcs = ["beta_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "binomial_test",
    size = "small",
    srcs = ["binomial_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "blockwise_test",
    srcs = ["blockwise_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "categorical_test",
    srcs = ["categorical_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "cauchy_test",
    size = "medium",
    srcs = ["cauchy_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
    ],
)

py_test(
    name = "chi_test",
    srcs = ["chi_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "chi2_test",
    srcs = ["chi2_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "conditional_distribution_test",
    size = "small",
    srcs = [
        "conditional_distribution_test.py",
        "distribution_test.py",
    ],
    tags = ["no_pip"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "conditional_transformed_distribution_test",
    size = "medium",
    srcs = [
        "conditional_transformed_distribution_test.py",
        "transformed_distribution_test.py",
    ],
    tags = [
        "no_pip",
    ],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "deterministic_test",
    size = "small",
    srcs = ["deterministic_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "dirichlet_multinomial_test",
    srcs = ["dirichlet_multinomial_test.py"],
    shard_count = 4,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "dirichlet_test",
    srcs = ["dirichlet_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "distribution_test",
    size = "small",
    srcs = ["distribution_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "empirical_test",
    srcs = ["empirical_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "exponential_test",
    srcs = ["exponential_test.py"],
    deps = [
        ":exponential",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "gamma_test",
    srcs = ["gamma_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "gamma_gamma_test",
    size = "small",
    srcs = ["gamma_gamma_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "gaussian_process_test",
    size = "small",
    srcs = ["gaussian_process_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/positive_semidefinite_kernels",
    ],
)

py_test(
    name = "gaussian_process_regression_model_test",
    size = "small",
    srcs = ["gaussian_process_regression_model_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/positive_semidefinite_kernels",
    ],
)

py_test(
    name = "geometric_test",
    size = "small",
    srcs = ["geometric_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "gumbel_test",
    size = "small",
    srcs = ["gumbel_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "half_cauchy_test",
    size = "medium",
    srcs = ["half_cauchy_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "half_normal_test",
    size = "medium",
    srcs = ["half_normal_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "hidden_markov_model_test",
    size = "medium",
    srcs = ["hidden_markov_model_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "horseshoe_test",
    size = "medium",
    srcs = ["horseshoe_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "independent_test",
    size = "small",
    srcs = ["independent_test.py"],
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "inverse_gamma_test",
    srcs = ["inverse_gamma_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "inverse_gaussian_test",
    srcs = ["inverse_gaussian_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "kullback_leibler_test",
    srcs = ["kullback_leibler_test.py"],
    deps = [
        ":bernoulli",
        ":beta",
        ":categorical",
        ":deterministic",
        ":dirichlet",
        ":gamma",
        ":kullback_leibler",
        ":normal",
        # tensorflow dep,
    ],
)

py_test(
    name = "kumaraswamy_test",
    srcs = ["kumaraswamy_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "laplace_test",
    srcs = ["laplace_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "linear_gaussian_ssm_test",
    size = "large",
    srcs = ["linear_gaussian_ssm_test.py"],
    shard_count = 4,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "lkj_test",
    size = "medium",
    srcs = [
        "lkj_test.py",
    ],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "logistic_test",
    size = "small",
    srcs = ["logistic_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "lognormal_test",
    size = "small",
    srcs = ["lognormal_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "mixture_test",
    size = "medium",
    srcs = ["mixture_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "mixture_same_family_test",
    srcs = ["mixture_same_family_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "multinomial_test",
    srcs = ["multinomial_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "multivariate_student_t_test",
    srcs = ["multivariate_student_t_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "mvn_diag_plus_low_rank_test",
    size = "medium",
    srcs = ["mvn_diag_plus_low_rank_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "mvn_diag_test",
    size = "small",
    srcs = ["mvn_diag_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "mvn_full_covariance_test",
    size = "small",
    srcs = ["mvn_full_covariance_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "mvn_linear_operator_test",
    size = "small",
    srcs = ["mvn_linear_operator_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "mvn_tril_test",
    size = "medium",
    srcs = ["mvn_tril_test.py"],
    shard_count = 5,
    tags = ["nomsan"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "negative_binomial_test",
    size = "small",
    srcs = ["negative_binomial_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "normal_test",
    srcs = ["normal_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "normal_conjugate_posteriors_test",
    size = "small",
    srcs = ["normal_conjugate_posteriors_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "onehot_categorical_test",
    size = "small",
    srcs = ["onehot_categorical_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "pareto_test",
    size = "small",
    srcs = ["pareto_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "poisson_test",
    size = "small",
    srcs = ["poisson_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "poisson_lognormal_test",
    size = "medium",
    srcs = ["poisson_lognormal_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "quantized_distribution_test",
    size = "medium",
    srcs = ["quantized_distribution_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "relaxed_bernoulli_test",
    size = "small",
    srcs = ["relaxed_bernoulli_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "relaxed_onehot_categorical_test",
    size = "small",
    srcs = ["relaxed_onehot_categorical_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "sample_stats_test",
    size = "medium",
    srcs = ["sample_stats_test.py"],
    tags = ["nomsan"],  # disable to avoid false positives from scipy.
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "seed_stream_test",
    size = "small",
    srcs = ["seed_stream_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "sinh_arcsinh_test",
    size = "small",
    srcs = ["sinh_arcsinh_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "student_t_test",
    srcs = ["student_t_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "student_t_process_test",
    size = "small",
    srcs = ["student_t_process_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/positive_semidefinite_kernels",
    ],
)

py_test(
    name = "transformed_distribution_test",
    size = "medium",
    srcs = ["transformed_distribution_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "triangular_test",
    srcs = ["triangular_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "truncated_normal_test",
    size = "large",
    srcs = ["truncated_normal_test.py"],
    shard_count = 10,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "uniform_test",
    srcs = ["uniform_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "variational_gaussian_process_test",
    size = "small",
    srcs = ["variational_gaussian_process_test.py"],
    deps = [
        #":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "vector_diffeomixture_test",
    size = "medium",
    srcs = ["vector_diffeomixture_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "vector_exponential_diag_test",
    size = "medium",
    srcs = ["vector_exponential_diag_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "vector_laplace_diag_test",
    size = "medium",
    srcs = ["vector_laplace_diag_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "vector_sinh_arcsinh_diag_test",
    size = "medium",
    srcs = ["vector_sinh_arcsinh_diag_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "vector_student_t_test",
    size = "medium",
    srcs = ["vector_student_t_test.py"],
    deps = [
        ":vector_student_t",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "von_mises_fisher_test",
    size = "medium",
    srcs = ["von_mises_fisher_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "von_mises_test",
    size = "medium",
    srcs = ["von_mises_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "wishart_test",
    size = "small",
    srcs = ["wishart_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "zipf_test",
    size = "large",
    srcs = ["zipf_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
