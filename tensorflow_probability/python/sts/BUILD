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
    name = "sts",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    substrates_omit_deps = [
        ":decomposition",
        ":default_model",
        ":fitting",
        ":forecast",
        ":holiday_effects",
        ":regularization",
        "//tensorflow_probability/python/sts/anomaly_detection",
    ],
    deps = [
        ":decomposition",
        ":default_model",
        ":fitting",
        ":forecast",
        ":holiday_effects",
        ":regularization",
        ":structural_time_series",
        "//tensorflow_probability/python/internal:all_util",
        "//tensorflow_probability/python/sts/anomaly_detection",
        "//tensorflow_probability/python/sts/components",
    ],
)

py_library(
    name = "decomposition",
    srcs = ["decomposition.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "decomposition_test",
    size = "medium",
    srcs = ["decomposition_test.py"],
    python_version = "PY3",
    shard_count = 6,
    srcs_version = "PY3",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "default_model",
    srcs = ["default_model.py"],
    srcs_version = "PY3",
    deps = [
        ":regularization",
        ":structural_time_series",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/experimental/sts_gibbs",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/sts/components",
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "default_model_test",
    size = "medium",
    srcs = ["default_model_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "fitting",
    srcs = ["fitting.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/sts/internal",
        "//tensorflow_probability/python/util",
    ],
)

py_test(
    name = "fitting_test",
    size = "large",
    srcs = ["fitting_test.py"],
    python_version = "PY3",
    shard_count = 6,
    srcs_version = "PY3",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "forecast",
    srcs = ["forecast.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "forecast_test",
    size = "medium",
    srcs = ["forecast_test.py"],
    python_version = "PY3",
    shard_count = 6,
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "holiday_effects",
    srcs = ["holiday_effects.py"],
    srcs_version = "PY3",
    deps = [
        # holidays dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "holiday_effects_test",
    size = "medium",
    srcs = ["holiday_effects_test.py"],
    python_version = "PY3",
    shard_count = 6,
    srcs_version = "PY3",
    # TODO(gcluo): Add the holidays package to the OSS testing setup.
    tags = ["no-oss-ci"],
    deps = [
        # holidays dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "regularization",
    srcs = ["regularization.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
    ],
)

py_test(
    name = "regularization_test",
    size = "small",
    srcs = ["regularization_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":regularization",
        # numpy dep,
        # pandas dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "structural_time_series",
    srcs = ["structural_time_series.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/sts/internal",
        "//tensorflow_probability/python/util",
    ],
)

multi_substrate_py_test(
    name = "structural_time_series_test",
    size = "medium",
    srcs = ["structural_time_series_test.py"],
    jax_size = "large",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
