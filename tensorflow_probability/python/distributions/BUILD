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
load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_test",
    "multi_substrate_py_library",
    "py3_test",  # @unused
)

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "distributions",
    srcs = ["__init__.py"],
    deps = [
        ":autoregressive",
        ":batch_reshape",
        ":bernoulli",
        ":beta",
        ":beta_binomial",
        ":binomial",
        ":blockwise",
        ":categorical",
        ":cauchy",
        ":chi",
        ":chi2",
        ":cholesky_lkj",
        ":continuous_bernoulli",
        ":deterministic",
        ":dirichlet",
        ":dirichlet_multinomial",
        ":distribution",
        ":doublesided_maxwell",
        ":empirical",
        ":exponential",
        ":finite_discrete",
        ":gamma",
        ":gamma_gamma",
        ":gaussian_process",
        ":gaussian_process_regression_model",
        ":generalized_pareto",
        ":geometric",
        ":gumbel",
        ":half_cauchy",
        ":half_normal",
        ":half_student_t",
        ":hidden_markov_model",
        ":horseshoe",
        ":independent",
        ":inverse_gamma",
        ":inverse_gaussian",
        ":johnson_su",
        ":joint_distribution",
        ":joint_distribution_auto_batched",
        ":joint_distribution_coroutine",
        ":joint_distribution_named",
        ":joint_distribution_sample_path_mixin",
        ":joint_distribution_sequential",
        ":joint_distribution_util",
        ":kullback_leibler",
        ":kumaraswamy",
        ":lambertw_f",
        ":laplace",
        ":linear_gaussian_ssm",
        ":lkj",
        ":logistic",
        ":logitnormal",
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
        ":ordered_logistic",
        ":pareto",
        ":pert",
        ":pixel_cnn",
        ":plackett_luce",
        ":poisson",
        ":poisson_lognormal",
        ":probit_bernoulli",
        ":quantized_distribution",
        ":relaxed_bernoulli",
        ":relaxed_onehot_categorical",
        ":sample",
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
        ":von_mises",
        ":von_mises_fisher",
        ":wishart",
        ":zipf",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/math:linalg",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "autoregressive",
    srcs = ["autoregressive.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "batch_reshape",
    srcs = ["batch_reshape.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "bernoulli",
    srcs = ["bernoulli.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "beta",
    srcs = ["beta.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:special",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "beta_binomial",
    srcs = ["beta_binomial.py"],
    deps = [
        ":beta",
        ":binomial",
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:special",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "binomial",
    srcs = ["binomial.py"],
    deps = [
        ":distribution",
        ":exponential",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:batched_rejection_sampler",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:implementation_selection",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "blockwise",
    srcs = ["blockwise.py"],
    deps = [
        ":distribution",
        ":joint_distribution_sequential",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "categorical",
    srcs = ["categorical.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "cauchy",
    srcs = ["cauchy.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "chi",
    srcs = ["chi.py"],
    deps = [
        ":chi2",
        ":kullback_leibler",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/bijectors:square",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:special",
    ],
)

multi_substrate_py_library(
    name = "chi2",
    srcs = ["chi2.py"],
    deps = [
        ":distribution",
        ":gamma",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:special",
    ],
)

multi_substrate_py_library(
    name = "cholesky_lkj",
    srcs = ["cholesky_lkj.py"],
    deps = [
        ":distribution",
        ":lkj",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:correlation_cholesky",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:special",
    ],
)

multi_substrate_py_library(
    name = "continuous_bernoulli",
    srcs = ["continuous_bernoulli.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "deterministic",
    srcs = ["deterministic.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "dirichlet",
    srcs = ["dirichlet.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softmax_centered",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "dirichlet_multinomial",
    srcs = ["dirichlet_multinomial.py"],
    deps = [
        ":distribution",
        ":multinomial",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "distribution",
    srcs = ["distribution.py"],
    deps = [
        ":kullback_leibler",
        # decorator dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions/internal:slicing",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:name_util",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "doublesided_maxwell",
    srcs = ["doublesided_maxwell.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "empirical",
    srcs = ["empirical.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "exponential",
    srcs = ["exponential.py"],
    deps = [
        ":gamma",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "finite_discrete",
    srcs = ["finite_discrete.py"],
    deps = [
        ":categorical",
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "gamma",
    srcs = ["gamma.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:batched_rejection_sampler",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "gamma_gamma",
    srcs = ["gamma_gamma.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "gaussian_process",
    srcs = ["gaussian_process.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":mvn_linear_operator",
        ":normal",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "gaussian_process_regression_model",
    srcs = ["gaussian_process_regression_model.py"],
    deps = [
        ":gaussian_process",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math/psd_kernels",
        "//tensorflow_probability/python/util",
    ],
)

multi_substrate_py_library(
    name = "generalized_pareto",
    srcs = ["generalized_pareto.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:generalized_pareto",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "geometric",
    srcs = ["geometric.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "gumbel",
    srcs = ["gumbel.py"],
    deps = [
        ":kullback_leibler",
        ":transformed_distribution",
        ":uniform",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:gumbel_cdf",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "half_cauchy",
    srcs = ["half_cauchy.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "half_normal",
    srcs = ["half_normal.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "hidden_markov_model",
    srcs = ["hidden_markov_model.py"],
    deps = [
        ":categorical",
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "horseshoe",
    srcs = ["horseshoe.py"],
    deps = [
        ":distribution",
        ":half_cauchy",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "hypothesis_testlib",
    testonly = 1,
    srcs = ["hypothesis_testlib.py"],
    deps = [
        # absl/logging dep,
        # hypothesis dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/bijectors:hypothesis_testlib",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "independent",
    srcs = ["independent.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "inverse_gamma",
    srcs = ["inverse_gamma.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:reciprocal",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "inverse_gaussian",
    srcs = ["inverse_gaussian.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "johnson_su",
    srcs = ["johnson_su.py"],
    deps = [
        ":normal",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/bijectors:sinh",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution",
    srcs = ["joint_distribution.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution_auto_batched",
    srcs = ["joint_distribution_auto_batched.py"],
    deps = [
        ":joint_distribution_coroutine",
        ":joint_distribution_named",
        ":joint_distribution_sample_path_mixin",
        ":joint_distribution_sequential",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution_coroutine",
    srcs = ["joint_distribution_coroutine.py"],
    deps = [
        ":joint_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution_named",
    srcs = ["joint_distribution_named.py"],
    deps = [
        ":joint_distribution_sequential",
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution_sample_path_mixin",
    srcs = ["joint_distribution_sample_path_mixin.py"],
    deps = [
        ":joint_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution_sequential",
    srcs = ["joint_distribution_sequential.py"],
    deps = [
        ":joint_distribution",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution_util",
    srcs = ["joint_distribution_util.py"],
    deps = [
        ":joint_distribution_named",
        ":joint_distribution_sequential",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

multi_substrate_py_library(
    name = "kullback_leibler",
    srcs = ["kullback_leibler.py"],
    deps = [
        # six dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "kumaraswamy",
    srcs = ["kumaraswamy.py"],
    deps = [
        ":transformed_distribution",
        ":uniform",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:kumaraswamy_cdf",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "laplace",
    srcs = ["laplace.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "lambertw_f",
    srcs = ["lambertw_f.py"],
    deps = [
        ":normal",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/internal:assert_util",
    ],
)

multi_substrate_py_library(
    name = "linear_gaussian_ssm",
    srcs = ["linear_gaussian_ssm.py"],
    deps = [
        ":distribution",
        ":independent",
        ":mvn_tril",
        ":normal",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "lkj",
    srcs = ["lkj.py"],
    deps = [
        ":beta",
        ":distribution",
        ":normal",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:cholesky_outer_product",
        "//tensorflow_probability/python/bijectors:correlation_cholesky",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math:special",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "logistic",
    srcs = ["logistic.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "lognormal",
    srcs = ["lognormal.py"],
    deps = [
        ":normal",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/internal:assert_util",
    ],
)

multi_substrate_py_library(
    name = "logitnormal",
    srcs = ["logitnormal.py"],
    deps = [
        ":normal",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
    ],
)

multi_substrate_py_library(
    name = "mixture",
    srcs = ["mixture.py"],
    deps = [
        ":categorical",
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "mixture_same_family",
    srcs = ["mixture_same_family.py"],
    deps = [
        ":distribution",
        ":independent",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "mvn_diag_plus_low_rank",
    srcs = ["mvn_diag_plus_low_rank.py"],
    deps = [
        ":mvn_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "mvn_diag",
    srcs = ["mvn_diag.py"],
    deps = [
        ":mvn_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
    ],
)

multi_substrate_py_library(
    name = "mvn_full_covariance",
    srcs = ["mvn_full_covariance.py"],
    deps = [
        ":mvn_tril",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

multi_substrate_py_library(
    name = "mvn_linear_operator",
    srcs = ["mvn_linear_operator.py"],
    deps = [
        ":kullback_leibler",
        ":normal",
        ":sample",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:scale_matvec_linear_operator",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "mvn_tril",
    srcs = ["mvn_tril.py"],
    deps = [
        ":mvn_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "multivariate_student_t",
    srcs = ["multivariate_student_t.py"],
    deps = [
        ":chi2",
        ":distribution",
        ":mvn_linear_operator",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "multinomial",
    srcs = ["multinomial.py"],
    deps = [
        ":categorical",
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "negative_binomial",
    srcs = ["negative_binomial.py"],
    deps = [
        ":distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "normal",
    srcs = ["normal.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "normal_conjugate_posteriors",
    srcs = ["normal_conjugate_posteriors.py"],
    deps = [
        ":normal",
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "onehot_categorical",
    srcs = ["onehot_categorical.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "ordered_logistic",
    srcs = ["ordered_logistic.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math",
    ],
)

multi_substrate_py_library(
    name = "pareto",
    srcs = ["pareto.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/util:deferred_tensor",
    ],
)

multi_substrate_py_library(
    name = "pert",
    srcs = ["pert.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "pixel_cnn",
    srcs = ["pixel_cnn.py"],
    deps = [
        ":categorical",
        ":distribution",
        ":independent",
        ":logistic",
        ":mixture_same_family",
        ":quantized_distribution",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/layers:weight_norm",
    ],
)

multi_substrate_py_library(
    name = "plackett_luce",
    srcs = ["plackett_luce.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "poisson",
    srcs = ["poisson.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "poisson_lognormal",
    srcs = ["poisson_lognormal.py"],
    deps = [
        ":categorical",
        ":distribution",
        ":normal",
        ":poisson",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "probit_bernoulli",
    srcs = ["probit_bernoulli.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "quantized_distribution",
    srcs = ["quantized_distribution.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "relaxed_bernoulli",
    srcs = ["relaxed_bernoulli.py"],
    deps = [
        ":distribution",
        ":logistic",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "relaxed_onehot_categorical",
    srcs = ["relaxed_onehot_categorical.py"],
    deps = [
        ":distribution",
        ":transformed_distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:exp",
        "//tensorflow_probability/python/bijectors:softmax_centered",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "sample",
    srcs = ["sample.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "sinh_arcsinh",
    srcs = ["sinh_arcsinh.py"],
    deps = [
        ":normal",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine_scalar",
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/bijectors:sinh_arcsinh",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "half_student_t",
    srcs = ["half_student_t.py"],
    deps = [
        ":distribution",
        ":student_t",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:special",
    ],
)

multi_substrate_py_library(
    name = "student_t",
    srcs = ["student_t.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:numeric",
        "//tensorflow_probability/python/math:special",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "student_t_process",
    srcs = ["student_t_process.py"],
    deps = [
        ":distribution",
        ":multivariate_student_t",
        ":student_t",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "triangular",
    srcs = ["triangular.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "transformed_distribution",
    srcs = ["transformed_distribution.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "truncated_normal",
    srcs = ["truncated_normal.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:special_math",
    ],
)

multi_substrate_py_library(
    name = "uniform",
    srcs = ["uniform.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "variational_gaussian_process",
    srcs = ["variational_gaussian_process.py"],
    deps = [
        ":gaussian_process",
        ":independent",
        ":kullback_leibler",
        ":mvn_linear_operator",
        ":normal",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math/psd_kernels:positive_semidefinite_kernel",
    ],
)

multi_substrate_py_library(
    name = "vector_diffeomixture",
    srcs = ["vector_diffeomixture.py"],
    deps = [
        ":categorical",
        ":distribution",
        ":normal",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine_linear_operator",
        "//tensorflow_probability/python/bijectors:softmax_centered",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "vector_exponential_diag",
    srcs = ["vector_exponential_diag.py"],
    deps = [
        ":vector_exponential_linear_operator",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_library(
    name = "vector_exponential_linear_operator",
    srcs = ["vector_exponential_linear_operator.py"],
    deps = [
        ":exponential",
        ":sample",
        ":transformed_distribution",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:affine_linear_operator",
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:scale_matvec_linear_operator",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_library(
    name = "von_mises",
    srcs = ["von_mises.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":normal",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:scale",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/bijectors:sigmoid",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "von_mises_fisher",
    srcs = ["von_mises_fisher.py"],
    deps = [
        ":beta",
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:softmax_centered",
        "//tensorflow_probability/python/bijectors:square",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "wishart",
    srcs = ["wishart.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:cholesky_outer_product",
        "//tensorflow_probability/python/bijectors:fill_scale_tril",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/bijectors:transform_diagonal",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "zipf",
    srcs = ["zipf.py"],
    deps = [
        ":distribution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "autoregressive_test",
    srcs = ["autoregressive_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "batch_reshape_test",
    srcs = ["batch_reshape_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/bijectors:bijector_test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "bernoulli_test",
    srcs = ["bernoulli_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "beta_test",
    srcs = ["beta_test.py"],
    jax_size = "medium",
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "beta_binomial_test",
    size = "medium",
    srcs = ["beta_binomial_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "binomial_test",
    size = "medium",
    srcs = ["binomial_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

py_test(
    name = "binomial_test_gpu",
    size = "medium",
    srcs = ["binomial_test.py"],
    main = "binomial_test.py",
    tags = ["requires-gpu-nvidia"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_test(
    name = "blockwise_test",
    srcs = ["blockwise_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "categorical_test",
    srcs = ["categorical_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "cauchy_test",
    srcs = ["cauchy_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
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

multi_substrate_py_test(
    name = "chi2_test",
    srcs = ["chi2_test.py"],
    jax_size = "medium",
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "cholesky_lkj_test",
    size = "medium",
    srcs = [
        "cholesky_lkj_test.py",
    ],
    jax_size = "large",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "continuous_bernoulli_test",
    size = "medium",
    srcs = ["continuous_bernoulli_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "deterministic_test",
    srcs = ["deterministic_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "dirichlet_multinomial_test",
    size = "medium",
    srcs = ["dirichlet_multinomial_test.py"],
    numpy_tags = ["notap"],
    shard_count = 4,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "dirichlet_test",
    srcs = ["dirichlet_test.py"],
    jax_size = "medium",
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "distribution_test",
    srcs = ["distribution_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "doublesided_maxwell_test",
    srcs = ["doublesided_maxwell_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "empirical_test",
    srcs = ["empirical_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "exponential_test",
    srcs = ["exponential_test.py"],
    deps = [
        ":exponential",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "finite_discrete_test",
    srcs = ["finite_discrete_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "gamma_test",
    size = "medium",
    srcs = ["gamma_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "gamma_gamma_test",
    srcs = ["gamma_gamma_test.py"],
    jax_size = "medium",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "gaussian_process_test",
    srcs = ["gaussian_process_test.py"],
    jax_size = "medium",  # Due to GaussianProcessDynamicTest.testShapes
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels",
    ],
)

multi_substrate_py_test(
    name = "gaussian_process_regression_model_test",
    srcs = ["gaussian_process_regression_model_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    deps = [
        ":distributions",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels",
    ],
)

multi_substrate_py_test(
    name = "generalized_pareto_test",
    size = "medium",
    srcs = ["generalized_pareto_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    tags = ["hypothesis"],
    deps = [
        # hypothesis dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "geometric_test",
    srcs = ["geometric_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "gumbel_test",
    srcs = ["gumbel_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "half_cauchy_test",
    srcs = ["half_cauchy_test.py"],
    numpy_tags = ["notap"],
    shard_count = 5,
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "half_normal_test",
    srcs = ["half_normal_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "hidden_markov_model_test",
    size = "medium",
    srcs = ["hidden_markov_model_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    shard_count = 4,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "horseshoe_test",
    size = "medium",
    srcs = ["horseshoe_test.py"],
    jax_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "independent_test",
    srcs = ["independent_test.py"],
    numpy_tags = ["notap"],
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    deps = [
        # hypothesis dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "inverse_gamma_test",
    size = "medium",
    srcs = ["inverse_gamma_test.py"],
    jax_size = "large",
    shard_count = 5,
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "inverse_gaussian_test",
    size = "medium",
    srcs = ["inverse_gaussian_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "kullback_leibler_test",
    srcs = ["kullback_leibler_test.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":normal",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "johnson_su_test",
    size = "medium",
    srcs = ["johnson_su_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "joint_distribution_auto_batched_test",
    size = "medium",
    srcs = ["joint_distribution_auto_batched_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "joint_distribution_coroutine_test",
    size = "medium",
    srcs = ["joint_distribution_coroutine_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "joint_distribution_named_test",
    srcs = ["joint_distribution_named_test.py"],
    jax_size = "medium",
    deps = [
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "joint_distribution_sequential_test",
    size = "medium",
    srcs = ["joint_distribution_sequential_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "joint_distribution_util_test",
    size = "medium",
    srcs = ["joint_distribution_util_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "kumaraswamy_test",
    srcs = ["kumaraswamy_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "laplace_test",
    size = "medium",
    srcs = ["laplace_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "lambertw_f_test",
    srcs = ["lambertw_f_test.py"],
    numpy_tags = ["notap"],
    deps = [
        ":lambertw_f",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "linear_gaussian_ssm_test",
    size = "medium",
    srcs = ["linear_gaussian_ssm_test.py"],
    jax_size = "large",
    numpy_tags = ["notap"],
    shard_count = 5,
    deps = [
        ":linear_gaussian_ssm",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "lkj_test",
    size = "medium",
    srcs = ["lkj_test.py"],
    jax_size = "large",
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    shard_count = 3,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "logistic_test",
    srcs = ["logistic_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "lognormal_test",
    srcs = ["lognormal_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "logitnormal_test",
    srcs = ["logitnormal_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "mixture_test",
    size = "medium",
    srcs = ["mixture_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # hypothesis dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "mixture_same_family_test",
    size = "medium",
    srcs = ["mixture_same_family_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "multinomial_test",
    size = "medium",
    srcs = ["multinomial_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    shard_count = 5,
    deps = [
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions:hypothesis_testlib",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "multivariate_student_t_test",
    size = "medium",
    srcs = ["multivariate_student_t_test.py"],
    numpy_tags = ["notap"],
    shard_count = 3,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "mvn_diag_plus_low_rank_test",
    srcs = ["mvn_diag_plus_low_rank_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "mvn_diag_test",
    srcs = ["mvn_diag_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "mvn_full_covariance_test",
    srcs = ["mvn_full_covariance_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "mvn_linear_operator_test",
    srcs = ["mvn_linear_operator_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "mvn_tril_test",
    size = "medium",
    srcs = ["mvn_tril_test.py"],
    numpy_tags = ["notap"],
    shard_count = 5,
    tags = ["nomsan"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "negative_binomial_test",
    srcs = ["negative_binomial_test.py"],
    jax_size = "medium",
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "normal_test",
    srcs = ["normal_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "normal_conjugate_posteriors_test",
    srcs = ["normal_conjugate_posteriors_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "onehot_categorical_test",
    srcs = ["onehot_categorical_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "ordered_logistic_test",
    srcs = ["ordered_logistic_test.py"],
    jax_size = "medium",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "pareto_test",
    size = "medium",
    srcs = ["pareto_test.py"],
    shard_count = 3,
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "pert_test",
    srcs = ["pert_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "plackett_luce_test",
    srcs = ["plackett_luce_test.py"],
    deps = [
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "pixel_cnn_test",
    size = "large",
    srcs = ["pixel_cnn_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    shard_count = 8,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "poisson_test",
    size = "medium",
    srcs = ["poisson_test.py"],
    jax_size = "medium",
    shard_count = 2,
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "poisson_lognormal_test",
    size = "medium",
    srcs = ["poisson_lognormal_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "probit_bernoulli_test",
    srcs = ["probit_bernoulli_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:special_math",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "quantized_distribution_test",
    srcs = ["quantized_distribution_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "relaxed_bernoulli_test",
    srcs = ["relaxed_bernoulli_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "relaxed_onehot_categorical_test",
    srcs = ["relaxed_onehot_categorical_test.py"],
    jax_size = "medium",
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "sample_test",
    srcs = ["sample_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "sinh_arcsinh_test",
    srcs = ["sinh_arcsinh_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "half_student_t_test",
    size = "medium",
    srcs = ["half_student_t_test.py"],
    jax_size = "medium",
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "student_t_test",
    size = "medium",
    srcs = ["student_t_test.py"],
    jax_size = "medium",
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "student_t_process_test",
    srcs = ["student_t_process_test.py"],
    jax_size = "medium",
    deps = [
        ":distributions",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels",
    ],
)

multi_substrate_py_test(
    name = "transformed_distribution_test",
    size = "medium",
    srcs = ["transformed_distribution_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "triangular_test",
    size = "medium",
    srcs = ["triangular_test.py"],
    jax_size = "large",
    shard_count = 4,
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "truncated_normal_test",
    srcs = ["truncated_normal_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    shard_count = 10,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "uniform_test",
    srcs = ["uniform_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "variational_gaussian_process_test",
    size = "medium",
    srcs = ["variational_gaussian_process_test.py"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
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

multi_substrate_py_test(
    name = "vector_exponential_diag_test",
    srcs = ["vector_exponential_diag_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "von_mises_fisher_test",
    size = "medium",
    srcs = ["von_mises_fisher_test.py"],
    jax_size = "large",
    shard_count = 5,
    deps = [
        ":von_mises_fisher",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "von_mises_test",
    size = "medium",
    srcs = ["von_mises_test.py"],
    jax_size = "large",
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "wishart_test",
    srcs = ["wishart_test.py"],
    jax_size = "large",
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_test(
    name = "zipf_test",
    size = "medium",
    srcs = ["zipf_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "distribution_properties_test",
    size = "large",
    srcs = ["distribution_properties_test.py"],
    shard_count = 40,
    tags = ["hypothesis"],
    deps = [
        ":hypothesis_testlib",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py3_test(
    name = "jax_transformation_test",
    size = "large",
    srcs = ["jax_transformation_test.py"],
    python_version = "PY3",
    shard_count = 40,
    srcs_version = "PY3",
    tags = [
        "hypothesis",
        "notap",
    ],
    deps = [
        ":hypothesis_testlib.jax",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # jax dep,
        # numpy dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib.jax",
        "//tensorflow_probability/python/internal:tensor_util.jax",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util.jax",
    ],
)

py_test(
    name = "stochastic_process_properties_test",
    size = "large",
    srcs = ["stochastic_process_properties_test.py"],
    shard_count = 3,
    tags = ["hypothesis"],
    deps = [
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:hypothesis_testlib",
    ],
)

exports_files(
    [
        "__init__.py",
        "autoregressive.py",
        "batch_reshape.py",
        "bernoulli.py",
        "beta.py",
        "binomial.py",
        "blockwise.py",
        "categorical.py",
        "cauchy.py",
        "cauchy_test.py",
        "chi.py",
        "chi2.py",
        "cholesky_lkj.py",
        "continuous_bernoulli.py",
        "deterministic.py",
        "dirichlet.py",
        "dirichlet_multinomial.py",
        "distribution.py",
        "doublesided_maxwell.py",
        "empirical.py",
        "exponential.py",
        "finite_discrete.py",
        "gamma.py",
        "gamma_gamma.py",
        "gamma_gamma_test.py",
        "generalized_pareto.py",
        "geometric.py",
        "geometric_test.py",
        "gumbel.py",
        "half_cauchy.py",
        "half_normal.py",
        "half_student_t.py",
        "hidden_markov_model.py",
        "horseshoe.py",
        "independent.py",
        "inverse_gamma.py",
        "inverse_gaussian.py",
        "inverse_gaussian_test.py",
        "johnson_su.py",
        "joint_distribution.py",
        "joint_distribution_coroutine.py",
        "joint_distribution_named.py",
        "joint_distribution_sequential.py",
        "kullback_leibler.py",
        "kumaraswamy.py",
        "kumaraswamy_test.py",
        "laplace.py",
        "lambertw_f.py",
        "linear_gaussian_ssm.py",
        "lkj.py",
        "logistic.py",
        "logitnormal.py",
        "lognormal.py",
        "lognormal_test.py",
        "mixture.py",
        "mixture_same_family.py",
        "multinomial.py",
        "multivariate_student_t.py",
        "mvn_diag.py",
        "mvn_diag_plus_low_rank.py",
        "mvn_full_covariance.py",
        "mvn_linear_operator.py",
        "mvn_tril.py",
        "negative_binomial.py",
        "normal.py",
        "normal_conjugate_posteriors.py",
        "onehot_categorical.py",
        "ordered_logistic.py",
        "pareto.py",
        "pert.py",
        "plackett_luce.py",
        "poisson.py",
        "poisson_lognormal.py",
        "probit_bernoulli.py",
        "quantized_distribution.py",
        "relaxed_bernoulli.py",
        "relaxed_onehot_categorical.py",
        "sample.py",
        "sample_test.py",
        "sinh_arcsinh.py",
        "sinh_arcsinh_test.py",
        "student_t.py",
        "transformed_distribution.py",
        "triangular.py",
        "truncated_normal.py",
        "uniform.py",
        "vector_diffeomixture.py",
        "vector_exponential_diag.py",
        "vector_exponential_linear_operator.py",
        "vector_laplace_diag.py",
        "vector_sinh_arcsinh_diag.py",
        "von_mises.py",
        "von_mises_fisher.py",
        "wishart.py",
        "zipf.py",
    ],
    visibility = ["//tensorflow_probability:__subpackages__"],
)
