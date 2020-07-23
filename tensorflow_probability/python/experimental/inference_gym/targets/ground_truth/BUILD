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
#   Check in ground truth values.

load(
    "//tensorflow_probability/python:build_defs.bzl",
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
    name = "ground_truth",
    srcs = ["__init__.py"],
    deps = [
        ":german_credit_numeric_logistic_regression",
        ":german_credit_numeric_probit_regression",
        ":german_credit_numeric_sparse_logistic_regression",
        ":stochastic_volatility_sp500",
        ":stochastic_volatility_sp500_small",
        ":synthetic_item_response_theory",
        ":synthetic_log_gaussian_cox_process",
    ],
)

multi_substrate_py_library(
    name = "german_credit_numeric_logistic_regression",
    srcs = ["german_credit_numeric_logistic_regression.py"],
    deps = [
        # numpy dep,
    ],
)

multi_substrate_py_library(
    name = "german_credit_numeric_probit_regression",
    srcs = ["german_credit_numeric_probit_regression.py"],
    deps = [
        # numpy dep,
    ],
)

multi_substrate_py_library(
    name = "german_credit_numeric_sparse_logistic_regression",
    srcs = ["german_credit_numeric_sparse_logistic_regression.py"],
    deps = [
        # numpy dep,
    ],
)

multi_substrate_py_library(
    name = "synthetic_item_response_theory",
    srcs = ["synthetic_item_response_theory.py"],
    deps = [
        # numpy dep,
    ],
)

multi_substrate_py_library(
    name = "synthetic_log_gaussian_cox_process",
    srcs = ["synthetic_log_gaussian_cox_process.py"],
    deps = [
        # numpy dep,
    ],
)

multi_substrate_py_library(
    name = "stochastic_volatility_sp500",
    srcs = ["stochastic_volatility_sp500.py"],
    deps = [
        # numpy dep,
    ],
)

multi_substrate_py_library(
    name = "stochastic_volatility_sp500_small",
    srcs = ["stochastic_volatility_sp500_small.py"],
    deps = [
        # numpy dep,
    ],
)
