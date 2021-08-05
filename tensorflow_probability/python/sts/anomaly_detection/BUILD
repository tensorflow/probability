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
#   Gibbs sampling for Bayesian structural time series models

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "anomaly_detection",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":anomaly_detection_lib",
    ],
)

py_library(
    name = "anomaly_detection_lib",
    srcs = ["anomaly_detection_lib.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/experimental/sts_gibbs:gibbs_sampler",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math",
        "//tensorflow_probability/python/sts:fitting",
        "//tensorflow_probability/python/sts:forecast",
        "//tensorflow_probability/python/sts:regularization",
        "//tensorflow_probability/python/sts/components",
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "anomaly_detection_test",
    size = "medium",
    srcs = ["anomaly_detection_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:parameterized dep,
        # matplotlib dep,
        # numpy dep,
        # pandas dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/sts_gibbs",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
