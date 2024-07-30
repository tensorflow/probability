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
# Placeholder: py_library
# Placeholder: py_test

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
        "//inference_gym:__subpackages__",
    ],
)

licenses(["notice"])

# pytype_strict
py_library(
    name = "targets",
    srcs = ["__init__.py"],
    deps = [
        ":banana",
        ":bayesian_model",
        ":brownian_motion",
        ":eight_schools",
        ":ill_conditioned_gaussian",
        ":item_response_theory",
        ":log_gaussian_cox_process",
        ":logistic_regression",
        ":lorenz_system",
        ":model",
        ":neals_funnel",
        ":non_identifiable_quartic",
        ":plasma_spectroscopy",
        ":probit_regression",
        ":radon_contextual_effects",
        ":sparse_logistic_regression",
        ":stochastic_volatility",
        ":vector_model",
        ":vectorized_stochastic_volatility",
        "//inference_gym/backends:util",
    ],
)

py_library(
    name = "banana",
    srcs = ["banana.py"],
    deps = [
        ":model",
        # numpy dep,
    ],
)

# py_strict
py_test(
    name = "banana_test",
    srcs = ["banana_test.py"],
    deps = [
        ":banana",
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "bayesian_model",
    srcs = ["bayesian_model.py"],
    deps = [
        ":model",
        "//inference_gym/internal:ground_truth_encoding",
    ],
)

# py_strict
py_test(
    name = "bayesian_model_test",
    srcs = ["bayesian_model_test.py"],
    deps = [
        ":targets",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "brownian_motion",
    srcs = ["brownian_motion.py"],
    deps = [
        ":bayesian_model",
        ":model",
        "//inference_gym/internal:data",
        "//inference_gym/targets/ground_truth:brownian_motion_missing_middle_observations",
        "//inference_gym/targets/ground_truth:brownian_motion_unknown_scales_missing_middle_observations",
    ],
)

# py_strict
py_test(
    name = "brownian_motion_test",
    size = "large",
    srcs = ["brownian_motion_test.py"],
    shard_count = 3,
    deps = [
        ":brownian_motion",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "eight_schools",
    srcs = ["eight_schools.py"],
    deps = [
        ":model",
        # numpy dep,
        "//inference_gym/targets/ground_truth:eight_schools",
    ],
)

# py_strict
py_test(
    name = "eight_schools_test",
    srcs = ["eight_schools_test.py"],
    deps = [
        ":eight_schools",
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "ill_conditioned_gaussian",
    srcs = ["ill_conditioned_gaussian.py"],
    deps = [
        ":model",
        # numpy dep,
    ],
)

# py_strict
py_test(
    name = "ill_conditioned_gaussian_test",
    srcs = ["ill_conditioned_gaussian_test.py"],
    deps = [
        ":ill_conditioned_gaussian",
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "item_response_theory",
    srcs = ["item_response_theory.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # numpy dep,
        "//inference_gym/targets/ground_truth:synthetic_item_response_theory",
    ],
)

# py_strict
py_test(
    name = "item_response_theory_test",
    size = "large",
    srcs = ["item_response_theory_test.py"],
    shard_count = 3,
    deps = [
        ":item_response_theory",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "log_gaussian_cox_process",
    srcs = ["log_gaussian_cox_process.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # tensorflow dep,
        "//inference_gym/internal:data",
        "//inference_gym/targets/ground_truth:synthetic_log_gaussian_cox_process",
    ],
)

# py_strict
py_test(
    name = "log_gaussian_cox_process_test",
    size = "large",
    srcs = ["log_gaussian_cox_process_test.py"],
    shard_count = 3,
    deps = [
        ":log_gaussian_cox_process",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    deps = [
        ":bayesian_model",
        ":model",
        "//inference_gym/internal:data",
        "//inference_gym/targets/ground_truth:german_credit_numeric_logistic_regression",
    ],
)

# py_strict
py_test(
    name = "logistic_regression_test",
    size = "large",
    srcs = ["logistic_regression_test.py"],
    shard_count = 3,
    deps = [
        ":logistic_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "lorenz_system",
    srcs = ["lorenz_system.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # numpy dep,
        "//inference_gym/internal:data",
        "//inference_gym/targets/ground_truth:convection_lorenz_bridge",
        "//inference_gym/targets/ground_truth:convection_lorenz_bridge_unknown_scales",
    ],
)

# py_strict
py_test(
    name = "lorenz_system_test",
    size = "large",
    srcs = ["lorenz_system_test.py"],
    shard_count = 3,
    deps = [
        ":lorenz_system",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//inference_gym/internal:ground_truth_encoding",
    ],
)

# Not strict due to the TF internal nest import.
py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":targets",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "neals_funnel",
    srcs = ["neals_funnel.py"],
    deps = [
        ":model",
        # numpy dep,
    ],
)

# py_strict
py_test(
    name = "neals_funnel_test",
    srcs = ["neals_funnel_test.py"],
    deps = [
        ":neals_funnel",
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "non_identifiable_quartic",
    srcs = ["non_identifiable_quartic.py"],
    deps = [
        ":bayesian_model",
        ":model",
    ],
)

# py_strict
py_test(
    name = "non_identifiable_quartic_test",
    srcs = ["non_identifiable_quartic_test.py"],
    deps = [
        ":non_identifiable_quartic",
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "plasma_spectroscopy",
    srcs = ["plasma_spectroscopy.py"],
    deps = [
        ":bayesian_model",
        ":model",
        # numpy dep,
        "//inference_gym/internal:data",
    ],
)

# py_strict
py_test(
    name = "plasma_spectroscopy_test",
    srcs = ["plasma_spectroscopy_test.py"],
    deps = [
        ":plasma_spectroscopy",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "probit_regression",
    srcs = ["probit_regression.py"],
    deps = [
        ":bayesian_model",
        ":model",
        "//inference_gym/internal:data",
        "//inference_gym/targets/ground_truth:german_credit_numeric_probit_regression",
    ],
)

# py_strict
py_test(
    name = "probit_regression_test",
    size = "large",
    srcs = ["probit_regression_test.py"],
    shard_count = 3,
    deps = [
        ":probit_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "radon_contextual_effects",
    srcs = ["radon_contextual_effects.py"],
    deps = [
        ":bayesian_model",
        ":model",
        "//inference_gym/internal:data",
        "//inference_gym/targets/ground_truth:radon_contextual_effects_indiana",
        "//inference_gym/targets/ground_truth:radon_contextual_effects_indiana_halfnormal",
        "//inference_gym/targets/ground_truth:radon_contextual_effects_minnesota",
        "//inference_gym/targets/ground_truth:radon_contextual_effects_minnesota_halfnormal",
    ],
)

# py_strict
py_test(
    name = "radon_contextual_effects_test",
    srcs = ["radon_contextual_effects_test.py"],
    shard_count = 3,
    deps = [
        ":radon_contextual_effects",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "sparse_logistic_regression",
    srcs = ["sparse_logistic_regression.py"],
    deps = [
        ":bayesian_model",
        ":model",
        "//inference_gym/internal:data",
        "//inference_gym/targets/ground_truth:german_credit_numeric_sparse_logistic_regression",
    ],
)

# py_strict
py_test(
    name = "sparse_logistic_regression_test",
    size = "large",
    srcs = ["sparse_logistic_regression_test.py"],
    shard_count = 3,
    deps = [
        ":sparse_logistic_regression",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_datasets dep,  # buildcleaner: keep
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "stochastic_volatility",
    srcs = ["stochastic_volatility.py"],
    deps = [
        ":bayesian_model",
        ":model",
        "//inference_gym/internal:data",
        "//inference_gym/targets/ground_truth:stochastic_volatility_sp500",
        "//inference_gym/targets/ground_truth:stochastic_volatility_sp500_small",
    ],
)

# py_strict
py_test(
    name = "stochastic_volatility_test",
    size = "large",
    srcs = ["stochastic_volatility_test.py"],
    shard_count = 3,
    deps = [
        ":stochastic_volatility",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:data",
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "vector_model",
    srcs = ["vector_model.py"],
    deps = [
        ":model",
    ],
)

# py_strict
py_test(
    name = "vector_model_test",
    srcs = ["vector_model_test.py"],
    deps = [
        ":targets",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)

py_library(
    name = "vectorized_stochastic_volatility",
    srcs = ["vectorized_stochastic_volatility.py"],
    deps = [
        ":bayesian_model",
        ":model",
        "//inference_gym/internal:data",
        "//inference_gym/targets/ground_truth:stochastic_volatility_log_sp500",
        "//inference_gym/targets/ground_truth:stochastic_volatility_log_sp500_small",
        "//inference_gym/targets/ground_truth:stochastic_volatility_sp500",
        "//inference_gym/targets/ground_truth:stochastic_volatility_sp500_small",
    ],
)

# Not strict due to the TF internal nest import.
py_test(
    name = "vectorized_stochastic_volatility_test",
    size = "large",
    srcs = ["vectorized_stochastic_volatility_test.py"],
    shard_count = 3,
    deps = [
        ":stochastic_volatility",
        ":vectorized_stochastic_volatility",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
        # tensorflow_probability/python/internal:test_util dep,
        "//inference_gym/internal:test_util",
    ],
)
