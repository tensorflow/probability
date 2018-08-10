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
        ":binomial",
        ":cauchy",
        ":chi2",
        ":conditional_distribution",
        ":conditional_transformed_distribution",
        ":deterministic",
        ":gamma_gamma",
        ":gaussian_process",
        ":gaussian_process_regression_model",
        ":geometric",
        ":gumbel",
        ":half_normal",
        ":independent",
        ":inverse_gamma",
        ":kumaraswamy",
        ":linear_gaussian_ssm",
        ":lkj",
        ":logistic",
        ":lognormal",
        ":mixture",
        ":mixture_same_family",
        ":mvn_diag",
        ":mvn_diag_plus_low_rank",
        ":mvn_full_covariance",
        ":mvn_linear_operator",
        ":mvn_tril",
        ":negative_binomial",
        ":normal_conjugate_posteriors",
        ":onehot_categorical",
        ":poisson",
        ":poisson_lognormal",
        ":quantized_distribution",
        ":relaxed_bernoulli",
        ":relaxed_onehot_categorical",
        ":sample_stats",
        ":seed_stream",
        ":sinh_arcsinh",
        ":truncated_normal",
        ":vector_diffeomixture",
        ":vector_exponential_diag",
        ":vector_exponential_linear_operator",
        ":vector_laplace_diag",
        ":vector_laplace_linear_operator",
        ":vector_sinh_arcsinh_diag",
        ":vector_student_t",
        ":von_mises_fisher",
        ":wishart",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions/internal",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "autoregressive",
    srcs = ["autoregressive.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
    ],
)

py_library(
    name = "batch_reshape",
    srcs = ["batch_reshape.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "binomial",
    srcs = ["binomial.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "cauchy",
    srcs = ["cauchy.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "chi2",
    srcs = ["chi2.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "conditional_distribution",
    srcs = ["conditional_distribution.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "conditional_transformed_distribution",
    srcs = ["conditional_transformed_distribution.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "deterministic",
    srcs = ["deterministic.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "gamma_gamma",
    srcs = ["gamma_gamma.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "gaussian_process",
    srcs = ["gaussian_process.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "gaussian_process_regression_model",
    srcs = ["gaussian_process_regression_model.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "geometric",
    srcs = ["geometric.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "gumbel",
    srcs = ["gumbel.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "half_normal",
    srcs = ["half_normal.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "independent",
    srcs = ["independent.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "inverse_gamma",
    srcs = ["inverse_gamma.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "kumaraswamy",
    srcs = ["kumaraswamy.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "linear_gaussian_ssm",
    srcs = ["linear_gaussian_ssm.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "lkj",
    srcs = ["lkj.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "logistic",
    srcs = ["logistic.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "lognormal",
    srcs = ["lognormal.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
    ],
)

py_library(
    name = "mixture",
    srcs = ["mixture.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "mixture_same_family",
    srcs = ["mixture_same_family.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "mvn_diag_plus_low_rank",
    srcs = ["mvn_diag_plus_low_rank.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "mvn_diag",
    srcs = ["mvn_diag.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "mvn_full_covariance",
    srcs = ["mvn_full_covariance.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "mvn_linear_operator",
    srcs = ["mvn_linear_operator.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "mvn_tril",
    srcs = ["mvn_tril.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "negative_binomial",
    srcs = ["negative_binomial.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "normal_conjugate_posteriors",
    srcs = ["normal_conjugate_posteriors.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "onehot_categorical",
    srcs = ["onehot_categorical.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "poisson_lognormal",
    srcs = ["poisson_lognormal.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "poisson",
    srcs = ["poisson.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "quantized_distribution",
    srcs = ["quantized_distribution.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
    ],
)

py_library(
    name = "relaxed_bernoulli",
    srcs = ["relaxed_bernoulli.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
    ],
)

py_library(
    name = "relaxed_onehot_categorical",
    srcs = ["relaxed_onehot_categorical.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
    ],
)

py_library(
    name = "sample_stats",
    srcs = ["sample_stats.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "seed_stream",
    srcs = ["seed_stream.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "sinh_arcsinh",
    srcs = ["sinh_arcsinh.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "truncated_normal",
    srcs = ["truncated_normal.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_diffeomixture",
    srcs = ["vector_diffeomixture.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_exponential_diag",
    srcs = ["vector_exponential_diag.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_exponential_linear_operator",
    srcs = ["vector_exponential_linear_operator.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_laplace_diag",
    srcs = ["vector_laplace_diag.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_laplace_linear_operator",
    srcs = ["vector_laplace_linear_operator.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_sinh_arcsinh_diag",
    srcs = ["vector_sinh_arcsinh_diag.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "vector_student_t",
    srcs = ["vector_student_t.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_library(
    name = "von_mises_fisher",
    srcs = ["von_mises_fisher.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "wishart",
    srcs = ["wishart.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

py_test(
    name = "autoregressive_test",
    size = "small",
    srcs = ["autoregressive_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "batch_reshape_test",
    size = "small",
    srcs = ["batch_reshape_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "binomial_test",
    size = "small",
    srcs = ["binomial_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "cauchy_test",
    size = "medium",
    srcs = ["cauchy_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "chi2_test",
    srcs = ["chi2_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
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
        ":distributions",
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "conditional_transformed_distribution_test",
    size = "medium",
    srcs = [
        "conditional_transformed_distribution_test.py",
        "transformed_distribution_test.py",
    ],
    tags = ["no_pip"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "deterministic_test",
    size = "small",
    srcs = ["deterministic_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "distribution_test",
    size = "small",
    srcs = ["distribution_test.py"],
    deps = [
        ":distributions",
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "gamma_gamma_test",
    size = "small",
    srcs = ["gamma_gamma_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "gaussian_process_test",
    size = "small",
    srcs = ["gaussian_process_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "gaussian_process_regression_model_test",
    size = "small",
    srcs = ["gaussian_process_regression_model_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "geometric_test",
    size = "small",
    srcs = ["geometric_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "half_normal_test",
    size = "medium",
    srcs = ["half_normal_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "independent_test",
    size = "small",
    srcs = ["independent_test.py"],
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "inverse_gamma_test",
    srcs = ["inverse_gamma_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "kumaraswamy_test",
    srcs = ["kumaraswamy_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "linear_gaussian_ssm_test",
    size = "medium",
    srcs = ["linear_gaussian_ssm_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "logistic_test",
    size = "small",
    srcs = ["logistic_test.py"],
    deps = [
        ":distributions",
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "lognormal_test",
    size = "small",
    srcs = ["lognormal_test.py"],
    deps = [
        ":distributions",
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "mixture_test",
    size = "medium",
    srcs = ["mixture_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "mixture_same_family_test",
    size = "small",
    srcs = ["mixture_same_family_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "mvn_diag_plus_low_rank_test",
    size = "medium",
    srcs = ["mvn_diag_plus_low_rank_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "mvn_diag_test",
    size = "small",
    srcs = ["mvn_diag_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "mvn_full_covariance_test",
    size = "small",
    srcs = ["mvn_full_covariance_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "mvn_linear_operator_test",
    size = "small",
    srcs = ["mvn_linear_operator_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "mvn_tril_test",
    size = "small",
    srcs = ["mvn_tril_test.py"],
    tags = ["nomsan"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "negative_binomial_test",
    size = "small",
    srcs = ["negative_binomial_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "normal_conjugate_posteriors_test",
    size = "small",
    srcs = ["normal_conjugate_posteriors_test.py"],
    deps = [
        ":distributions",
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "onehot_categorical_test",
    size = "small",
    srcs = ["onehot_categorical_test.py"],
    deps = [
        ":distributions",
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "poisson_test",
    size = "small",
    srcs = ["poisson_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "poisson_lognormal_test",
    size = "medium",
    srcs = ["poisson_lognormal_test.py"],
    deps = [
        ":distributions",
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
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "relaxed_bernoulli_test",
    size = "small",
    srcs = ["relaxed_bernoulli_test.py"],
    deps = [
        ":distributions",
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "relaxed_onehot_categorical_test",
    size = "small",
    srcs = ["relaxed_onehot_categorical_test.py"],
    deps = [
        ":distributions",
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
        ":distributions",
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
        ":distributions",
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
        ":distributions",
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "sinh_arcsinh_test",
    size = "small",
    srcs = ["sinh_arcsinh_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "transformed_distribution_test",
    size = "medium",
    srcs = ["transformed_distribution_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "truncated_normal_test",
    size = "large",
    srcs = ["truncated_normal_test.py"],
    shard_count = 10,
    deps = [
        ":distributions",
        # absl/testing:parameterized dep,
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
        ":distributions",
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
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "vector_laplace_diag_test",
    size = "medium",
    srcs = ["vector_laplace_diag_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "vector_sinh_arcsinh_diag_test",
    size = "medium",
    srcs = ["vector_sinh_arcsinh_diag_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "von_mises_fisher_test",
    size = "medium",
    srcs = ["von_mises_fisher_test.py"],
    deps = [
        ":distributions",
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
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "wishart_test",
    size = "small",
    srcs = ["wishart_test.py"],
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)
