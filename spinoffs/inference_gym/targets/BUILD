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

# [internal] load pytype.bzl (pytype_strict_library)
# [internal] load strict.bzl

# py_test rules are rewritten to py3_test on export to OSS, for reasons of
# reversibility of import<->export transforms. This is only necessary until we
# eliminate all py2and3_test targets, after which everything can safely use
# py_test, internally and externally, and no rewrites will be needed.
# @unused  <-- satisfy buildifier errors
load("//tensorflow_probability/python:build_defs.bzl", "py3_test")

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
        "//spinoffs/inference_gym:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# pytype_strict
py_library(
    name = "targets",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":banana",
        ":bayesian_model",
        ":brownian_motion",
        ":ill_conditioned_gaussian",
        ":item_response_theory",
        ":log_gaussian_cox_process",
        ":logistic_regression",
        ":model",
        ":neals_funnel",
        ":probit_regression",
        ":radon_contextual_effects",
        ":sparse_logistic_regression",
        ":stochastic_volatility",
        ":vector_model",
        "//spinoffs/inference_gym/targets/ground_truth",  # buildcleaner: keep
    ],
)

py_library(
    name = "banana",
    srcs = ["banana.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        # numpy dep,
    ],
)

# py_strict
py_test(
    name = "banana_test",
    srcs = ["banana_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":banana",
        # tensorflow dep,
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "bayesian_model",
    srcs = ["bayesian_model.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        "//spinoffs/inference_gym/internal:ground_truth_encoding",
    ],
)

# py_strict
py_test(
    name = "bayesian_model_test",
    srcs = ["bayesian_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":targets",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "brownian_motion",
    srcs = ["brownian_motion.py"],
    srcs_version = "PY3",
    deps = [
        ":bayesian_model",
        ":model",
        "//spinoffs/inference_gym/internal:data",
    ],
)

# py_strict
py_test(
    name = "brownian_motion_test",
    size = "large",
    srcs = ["brownian_motion_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        ":brownian_motion",
        # numpy dep,
        # tensorflow dep,
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "ill_conditioned_gaussian",
    srcs = ["ill_conditioned_gaussian.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        # numpy dep,
    ],
)

# py_strict
py_test(
    name = "ill_conditioned_gaussian_test",
    srcs = ["ill_conditioned_gaussian_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ill_conditioned_gaussian",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "item_response_theory",
    srcs = ["item_response_theory.py"],
    srcs_version = "PY3",
    deps = [
        ":bayesian_model",
        ":model",
        # numpy dep,
    ],
)

# py_strict
py_test(
    name = "item_response_theory_test",
    size = "large",
    srcs = ["item_response_theory_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        ":item_response_theory",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//spinoffs/inference_gym/internal:test_util",
        "//spinoffs/inference_gym/targets/ground_truth:synthetic_item_response_theory",  # buildcleaner: keep
    ],
)

py_library(
    name = "log_gaussian_cox_process",
    srcs = ["log_gaussian_cox_process.py"],
    srcs_version = "PY3",
    deps = [
        ":bayesian_model",
        ":model",
        # tensorflow dep,
        "//spinoffs/inference_gym/internal:data",
        "//spinoffs/inference_gym/targets/ground_truth:synthetic_log_gaussian_cox_process",  # buildcleaner: keep
    ],
)

# py_strict
py_test(
    name = "log_gaussian_cox_process_test",
    size = "large",
    srcs = ["log_gaussian_cox_process_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        ":log_gaussian_cox_process",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    srcs_version = "PY3",
    deps = [
        ":bayesian_model",
        ":model",
        "//spinoffs/inference_gym/internal:data",
        "//spinoffs/inference_gym/targets/ground_truth:german_credit_numeric_logistic_regression",  # buildcleaner: keep
    ],
)

# py_strict
py_test(
    name = "logistic_regression_test",
    size = "large",
    srcs = ["logistic_regression_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        ":logistic_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    deps = [
        "//spinoffs/inference_gym/internal:ground_truth_encoding",
    ],
)

# Not strict due to the TF internal nest import.
py3_test(
    name = "model_test",
    srcs = ["model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":targets",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "neals_funnel",
    srcs = ["neals_funnel.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        # numpy dep,
    ],
)

# py_strict
py_test(
    name = "neals_funnel_test",
    srcs = ["neals_funnel_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":neals_funnel",
        # tensorflow dep,
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "probit_regression",
    srcs = ["probit_regression.py"],
    srcs_version = "PY3",
    deps = [
        ":bayesian_model",
        ":model",
        "//spinoffs/inference_gym/internal:data",
        "//spinoffs/inference_gym/targets/ground_truth:german_credit_numeric_probit_regression",  # buildcleaner: keep
    ],
)

# py_strict
py_test(
    name = "probit_regression_test",
    size = "large",
    srcs = ["probit_regression_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        ":probit_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "radon_contextual_effects",
    srcs = ["radon_contextual_effects.py"],
    srcs_version = "PY3",
    deps = [
        ":bayesian_model",
        ":model",
        "//spinoffs/inference_gym/internal:data",
        "//spinoffs/inference_gym/targets/ground_truth:radon_contextual_effects_minnesota",  # buildcleaner: keep
    ],
)

# py_strict
py_test(
    name = "radon_contextual_effects_test",
    srcs = ["radon_contextual_effects_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        ":radon_contextual_effects",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//tensorflow_probability/python/internal:test_util",
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "sparse_logistic_regression",
    srcs = ["sparse_logistic_regression.py"],
    srcs_version = "PY3",
    deps = [
        ":bayesian_model",
        ":model",
        "//spinoffs/inference_gym/internal:data",
        "//spinoffs/inference_gym/targets/ground_truth:german_credit_numeric_sparse_logistic_regression",  # buildcleaner: keep
    ],
)

# py_strict
py_test(
    name = "sparse_logistic_regression_test",
    size = "large",
    srcs = ["sparse_logistic_regression_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        ":sparse_logistic_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "stochastic_volatility",
    srcs = ["stochastic_volatility.py"],
    srcs_version = "PY3",
    deps = [
        ":bayesian_model",
        ":model",
        "//spinoffs/inference_gym/internal:data",
    ],
)

# py_strict
py_test(
    name = "stochastic_volatility_test",
    size = "large",
    srcs = ["stochastic_volatility_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        ":stochastic_volatility",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//spinoffs/inference_gym/internal:test_util",
    ],
)

py_library(
    name = "vector_model",
    srcs = ["vector_model.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
    ],
)

# py_strict
py_test(
    name = "vector_model_test",
    srcs = ["vector_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":targets",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//spinoffs/inference_gym/internal:test_util",
    ],
)
