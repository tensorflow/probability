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

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
        "//inference_gym:__subpackages__",
    ],
)

licenses(["notice"])

# We can't use strict/pytype because `cmdstanpy` is not available internally.
py_library(
    name = "brownian_motion",
    srcs = ["brownian_motion.py"],
    deps = [
        ":stan_model",
        ":util",
        # numpy dep,
    ],
)

py_library(
    name = "eight_schools",
    srcs = ["eight_schools.py"],
    deps = [
        ":stan_model",
        ":util",
        # numpy dep,
    ],
)

py_library(
    name = "item_response_theory",
    srcs = ["item_response_theory.py"],
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "log_gaussian_cox_process",
    srcs = ["log_gaussian_cox_process.py"],
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "lorenz_system",
    srcs = ["lorenz_system.py"],
    deps = [
        ":stan_model",
        ":util",
        # numpy dep,
    ],
)

py_library(
    name = "radon_contextual_effects",
    srcs = ["radon_contextual_effects.py"],
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "radon_contextual_effects_halfnormal",
    srcs = ["radon_contextual_effects_halfnormal.py"],
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "stan_model",
    srcs = ["stan_model.py"],
)

py_library(
    name = "stochastic_volatility",
    srcs = ["stochastic_volatility.py"],
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "probit_regression",
    srcs = ["probit_regression.py"],
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "sparse_logistic_regression",
    srcs = ["sparse_logistic_regression.py"],
    deps = [
        ":stan_model",
        ":util",
    ],
)

py_library(
    name = "targets",
    srcs = ["targets.py"],
    deps = [
        ":brownian_motion",
        ":eight_schools",
        ":item_response_theory",
        ":log_gaussian_cox_process",
        ":logistic_regression",
        ":lorenz_system",
        ":probit_regression",
        ":radon_contextual_effects",
        ":sparse_logistic_regression",
        ":stochastic_volatility",
        "//inference_gym/internal:data",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
)
