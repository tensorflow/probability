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
# Ground truth computation using Stan.
licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "get_ground_truth",
    srcs = ["get_ground_truth.py"],
    python_version = "PY3",
    tags = ["notap"],
    deps = [
        ":targets",
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/inference_gym/internal:ground_truth_encoding",
    ],
)

py_library(
    name = "item_response_theory",
    srcs = ["item_response_theory.py"],
    srcs_version = "PY3",
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "log_gaussian_cox_process",
    srcs = ["log_gaussian_cox_process.py"],
    srcs_version = "PY3",
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    srcs_version = "PY3",
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "stan_model",
    srcs = ["stan_model.py"],
    srcs_version = "PY3",
)

py_library(
    name = "stochastic_volatility",
    srcs = ["stochastic_volatility.py"],
    srcs_version = "PY3",
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "probit_regression",
    srcs = ["probit_regression.py"],
    srcs_version = "PY3",
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "sparse_logistic_regression",
    srcs = ["sparse_logistic_regression.py"],
    srcs_version = "PY3",
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "targets",
    srcs = ["targets.py"],
    srcs_version = "PY3",
    deps = [
        ":item_response_theory",
        ":log_gaussian_cox_process",
        ":logistic_regression",
        ":probit_regression",
        ":sparse_logistic_regression",
        ":stochastic_volatility",
        "//tensorflow_probability/python/experimental/inference_gym/internal:data",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
)
