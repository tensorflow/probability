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
    srcs = glob(
        [
            "*.py",
        ],
        exclude = glob([
            "*_test.py",
        ]),
    ),
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions/internal:moving_stats",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:distribution_util",
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
    name = "truncated_normal_test",
    size = "large",
    srcs = ["truncated_normal_test.py"],
    deps = [
        ":distributions",
        # absl/testing:parameterized dep,
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
