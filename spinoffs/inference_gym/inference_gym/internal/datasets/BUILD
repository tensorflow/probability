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
#   Checked in ground truth values.

# [internal] load pytype.bzl (pytype_strict_library)

package(
    default_applicable_licenses = ["//tensorflow_probability:license"],
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
        "//inference_gym:__subpackages__",
    ],
)

licenses(["notice"])

# pytype_strict
py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    deps = [
        ":brownian_motion_missing_middle_observations",
        ":convection_lorenz_bridge",
        ":sp500_closing_prices",
        ":synthetic_item_response_theory",
        ":synthetic_log_gaussian_cox_process",
        ":synthetic_plasma_spectroscopy",
        ":synthetic_plasma_spectroscopy_with_bump",
    ],
)

# pytype_strict
py_library(
    name = "synthetic_item_response_theory",
    srcs = ["synthetic_item_response_theory.py"],
    deps = [
        # numpy dep,
    ],
)

# pytype_strict
py_library(
    name = "sp500_closing_prices",
    srcs = ["sp500_closing_prices.py"],
    deps = [
        # numpy dep,
    ],
)

# pytype_strict
py_library(
    name = "brownian_motion_missing_middle_observations",
    srcs = ["brownian_motion_missing_middle_observations.py"],
    deps = [
        # numpy dep,
    ],
)

# pytype_strict
py_library(
    name = "synthetic_log_gaussian_cox_process",
    srcs = ["synthetic_log_gaussian_cox_process.py"],
    deps = [
        # numpy dep,
    ],
)

# pytype_strict
py_library(
    name = "convection_lorenz_bridge",
    srcs = ["convection_lorenz_bridge.py"],
    deps = [
        # numpy dep,
    ],
)

# pytype_strict
py_library(
    name = "synthetic_plasma_spectroscopy",
    srcs = ["synthetic_plasma_spectroscopy.py"],
    deps = [
        # numpy dep,
    ],
)

# pytype_strict
py_library(
    name = "synthetic_plasma_spectroscopy_with_bump",
    srcs = ["synthetic_plasma_spectroscopy_with_bump.py"],
    deps = [
        # numpy dep,
    ],
)
