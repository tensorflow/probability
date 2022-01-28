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
#   Framework for Bayesian structural time series models

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

multi_substrate_py_library(
    name = "components",
    srcs = ["__init__.py"],
    deps = [
        ":autoregressive",
        ":autoregressive_integrated_moving_average",
        ":autoregressive_moving_average",
        ":dynamic_regression",
        ":local_level",
        ":local_linear_trend",
        ":regression",
        ":seasonal",
        ":semilocal_linear_trend",
        ":smooth_seasonal",
        ":sum",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "autoregressive",
    srcs = ["autoregressive.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "autoregressive_test",
    size = "medium",
    srcs = ["autoregressive_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "autoregressive_integrated_moving_average",
    srcs = ["autoregressive_integrated_moving_average.py"],
    deps = [
        ":autoregressive_moving_average",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "autoregressive_integrated_moving_average_test",
    size = "medium",
    srcs = ["autoregressive_integrated_moving_average_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/sts/components",
    ],
)

multi_substrate_py_library(
    name = "autoregressive_moving_average",
    srcs = ["autoregressive_moving_average.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "autoregressive_moving_average_test",
    size = "medium",
    srcs = ["autoregressive_moving_average_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "dynamic_regression",
    srcs = ["dynamic_regression.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "dynamic_regression_test",
    size = "small",
    srcs = ["dynamic_regression_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "local_level",
    srcs = ["local_level.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "local_level_test",
    size = "small",
    srcs = ["local_level_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "local_linear_trend",
    srcs = ["local_linear_trend.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "local_linear_trend_test",
    size = "small",
    srcs = ["local_linear_trend_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "regression",
    srcs = ["regression.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "regression_test",
    size = "medium",
    srcs = ["regression_test.py"],
    shard_count = 2,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "seasonal",
    srcs = ["seasonal.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "seasonal_test",
    size = "medium",
    srcs = ["seasonal_test.py"],
    shard_count = 6,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "semilocal_linear_trend",
    srcs = ["semilocal_linear_trend.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "semilocal_linear_trend_test",
    size = "small",
    srcs = ["semilocal_linear_trend_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "smooth_seasonal",
    srcs = ["smooth_seasonal.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "smooth_seasonal_test",
    size = "small",
    srcs = ["smooth_seasonal_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "sum",
    srcs = ["sum.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "sum_test",
    size = "medium",
    srcs = ["sum_test.py"],
    shard_count = 2,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
