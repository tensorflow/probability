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

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "sts",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fitting",
        ":forecast",
        ":local_linear_trend",
        ":regression",
        ":seasonal",
        ":structural_time_series",
        ":sum",
    ],
)

py_library(
    name = "fitting",
    srcs = ["fitting.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "fitting_test",
    size = "large",
    srcs = ["fitting_test.py"],
    shard_count = 6,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "forecast",
    srcs = ["forecast.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "forecast_test",
    size = "large",
    srcs = ["forecast_test.py"],
    shard_count = 6,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "local_linear_trend",
    srcs = ["local_linear_trend.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "local_linear_trend_test",
    size = "medium",
    srcs = ["local_linear_trend_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "regression",
    srcs = ["regression.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "regression_test",
    size = "medium",
    srcs = ["regression_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "seasonal",
    srcs = ["seasonal.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "seasonal_test",
    size = "medium",
    srcs = ["seasonal_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "structural_time_series",
    srcs = ["structural_time_series.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "structural_time_series_test",
    size = "medium",
    srcs = ["structural_time_series_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "sum",
    srcs = ["sum.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "sum_test",
    size = "large",
    srcs = ["sum_test.py"],
    shard_count = 2,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)
