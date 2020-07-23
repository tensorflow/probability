# Copyright 2020 The TensorFlow Probability Authors.
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
#   A package for target densities.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_test",
    "multi_substrate_py_library",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "targets",
    srcs = ["__init__.py"],
    deps = [
        ":banana",
        ":bayesian_model",
        ":ill_conditioned_gaussian",
        ":item_response_theory",
        ":log_gaussian_cox_process",
        ":logistic_regression",
        ":model",
        ":neals_funnel",
        ":probit_regression",
        ":sparse_logistic_regression",
        ":stochastic_volatility",
        ":vector_model",
        "//tensorflow_probability/python/experimental/inference_gym/targets/ground_truth",  # buildcleaner: keep
    ],
)

multi_substrate_py_library(
    name = "banana",
    srcs = ["banana.py"],
    deps = [
        ":model",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
    ],
)

multi_substrate_py_test(
    name = "banana_test",
    srcs = ["banana_test.py"],
    jax_tags = ["omni_fail"],
    deps = [
        ":banana",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "bayesian_model",
    srcs = ["bayesian_model.py"],
    deps = [
        ":model",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:ground_truth_encoding",
    ],
)

multi_substrate_py_test(
    name = "bayesian_model_test",
    srcs = ["bayesian_model_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "ill_conditioned_gaussian",
    srcs = ["ill_conditioned_gaussian.py"],
    deps = [
        ":model",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
    ],
)

multi_substrate_py_test(
    name = "ill_conditioned_gaussian_test",
    srcs = ["ill_conditioned_gaussian_test.py"],
    deps = [
        ":ill_conditioned_gaussian",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "item_response_theory",
    srcs = ["item_response_theory.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
    ],
)

multi_substrate_py_test(
    name = "item_response_theory_test",
    size = "large",
    srcs = ["item_response_theory_test.py"],
    deps = [
        ":item_response_theory",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "log_gaussian_cox_process",
    srcs = ["log_gaussian_cox_process.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/experimental/inference_gym/internal:data",
        "//tensorflow_probability/python/experimental/inference_gym/targets/ground_truth:synthetic_log_gaussian_cox_process",  # buildcleaner: keep
        "//tensorflow_probability/python/math/psd_kernels",
    ],
)

multi_substrate_py_test(
    name = "log_gaussian_cox_process_test",
    size = "large",
    srcs = ["log_gaussian_cox_process_test.py"],
    deps = [
        ":log_gaussian_cox_process",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/experimental/inference_gym/internal:data",
        "//tensorflow_probability/python/experimental/inference_gym/targets/ground_truth:german_credit_numeric_logistic_regression",  # buildcleaner: keep
    ],
)

multi_substrate_py_test(
    name = "logistic_regression_test",
    size = "large",
    srcs = ["logistic_regression_test.py"],
    deps = [
        ":logistic_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:ground_truth_encoding",
    ],
)

multi_substrate_py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "neals_funnel",
    srcs = ["neals_funnel.py"],
    deps = [
        ":model",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
    ],
)

multi_substrate_py_test(
    name = "neals_funnel_test",
    srcs = ["neals_funnel_test.py"],
    deps = [
        ":neals_funnel",
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "probit_regression",
    srcs = ["probit_regression.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/experimental/inference_gym/internal:data",
        "//tensorflow_probability/python/experimental/inference_gym/targets/ground_truth:german_credit_numeric_probit_regression",  # buildcleaner: keep
    ],
)

multi_substrate_py_test(
    name = "probit_regression_test",
    size = "large",
    srcs = ["probit_regression_test.py"],
    deps = [
        ":probit_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sparse_logistic_regression",
    srcs = ["sparse_logistic_regression.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/experimental/inference_gym/internal:data",
        "//tensorflow_probability/python/experimental/inference_gym/targets/ground_truth:german_credit_numeric_sparse_logistic_regression",  # buildcleaner: keep
    ],
)

multi_substrate_py_test(
    name = "sparse_logistic_regression_test",
    size = "large",
    srcs = ["sparse_logistic_regression_test.py"],
    deps = [
        ":sparse_logistic_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "stochastic_volatility",
    srcs = ["stochastic_volatility.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/experimental/inference_gym/internal:data",
    ],
)

multi_substrate_py_test(
    name = "stochastic_volatility_test",
    size = "large",
    srcs = ["stochastic_volatility_test.py"],
    deps = [
        ":stochastic_volatility",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/inference_gym/internal:test_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "vector_model",
    srcs = ["vector_model.py"],
    deps = [
        ":model",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "vector_model_test",
    srcs = ["vector_model_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
