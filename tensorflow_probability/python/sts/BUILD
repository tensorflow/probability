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
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

multi_substrate_py_library(
    name = "sts",
    srcs = ["__init__.py"],
    substrates_omit_deps = [
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

multi_substrate_py_library(
    name = "decomposition",
    srcs = ["decomposition.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "decomposition_test",
    size = "medium",
    srcs = ["decomposition_test.py"],
    shard_count = 6,
    deps = [
        ":decomposition",
        ":forecast",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/sts/components:local_linear_trend",
        "//tensorflow_probability/python/sts/components:seasonal",
        "//tensorflow_probability/python/sts/components:sum",
    ],
)

multi_substrate_py_library(
    name = "default_model",
    srcs = ["default_model.py"],
    deps = [
        ":structural_time_series",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:inverse_gamma",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/sts/components:local_linear_trend",
        "//tensorflow_probability/python/sts/components:seasonal",
        "//tensorflow_probability/python/sts/components:sum",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "default_model_test",
    size = "medium",
    srcs = ["default_model_test.py"],
    deps = [
        ":default_model",
        ":fitting",
        ":forecast",
        ":regularization",
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:exponential",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/internal:tf_keras",
        "//tensorflow_probability/python/optimizer",
        "//tensorflow_probability/python/sts/components:local_linear_trend",
        "//tensorflow_probability/python/sts/components:seasonal",
        "//tensorflow_probability/python/sts/components:semilocal_linear_trend",
        "//tensorflow_probability/python/sts/components:sum",
        "//tensorflow_probability/python/vi:optimization",
    ],
)

multi_substrate_py_library(
    name = "fitting",
    srcs = ["fitting.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/vi",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/mcmc:dual_averaging_step_size_adaptation",
        "//tensorflow_probability/python/mcmc:hmc",
        "//tensorflow_probability/python/mcmc:sample",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
        "//tensorflow_probability/python/sts/internal",
        "//tensorflow_probability/python/vi:optimization",
    ],
)

multi_substrate_py_test(
    name = "fitting_test",
    size = "large",
    srcs = ["fitting_test.py"],
    disabled_substrates = ["numpy"],
    shard_count = 6,
    deps = [
        ":fitting",
        # absl/testing:parameterized dep,
        # numpy dep,
        # optax dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/sts/components:local_linear_trend",
        "//tensorflow_probability/python/sts/components:seasonal",
        "//tensorflow_probability/python/sts/components:sum",
        "//tensorflow_probability/python/vi:optimization",
    ],
)

multi_substrate_py_library(
    name = "forecast",
    srcs = ["forecast.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "forecast_test",
    size = "medium",
    srcs = ["forecast_test.py"],
    numpy_tags = ["notap"],
    shard_count = 10,
    deps = [
        ":fitting",
        ":forecast",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/sts/components:seasonal",
        "//tensorflow_probability/python/sts/components:sum",
        "//tensorflow_probability/python/sts/internal:missing_values_util",
    ],
)

multi_substrate_py_library(
    name = "holiday_effects",
    srcs = ["holiday_effects.py"],
    deps = [
        # holidays dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/sts/internal",
    ],
)

multi_substrate_py_test(
    name = "holiday_effects_test",
    size = "medium",
    srcs = ["holiday_effects_test.py"],
    shard_count = 6,

    # TODO(gcluo): Add the holidays package to the OSS testing setup.
    tags = ["no-oss-ci"],
    deps = [
        ":holiday_effects",
        # holidays dep,
        # numpy dep,
        # pandas dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "regularization",
    srcs = ["regularization.py"],
    deps = [
        # numpy dep,
    ],
)

multi_substrate_py_test(
    name = "regularization_test",
    size = "small",
    srcs = ["regularization_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":regularization",
        # numpy dep,
        # pandas dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "structural_time_series",
    srcs = ["structural_time_series.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/sts/internal",
        "//tensorflow_probability/python/util",
    ],
)

multi_substrate_py_test(
    name = "structural_time_series_test",
    size = "large",
    srcs = ["structural_time_series_test.py"],
    jax_size = "large",
    shard_count = 4,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:laplace",
        "//tensorflow_probability/python/distributions:lognormal",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/sts",
    ],
)
