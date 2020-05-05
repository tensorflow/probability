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
#   Tools for sequential inference/filtering.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "sequential",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ensemble_adjustment_kalman_filter",
        ":ensemble_kalman_filter",
        ":extended_kalman_filter",
        ":iterated_filter",
        # tensorflow dep,
    ],
)

py_library(
    name = "extended_kalman_filter",
    srcs = ["extended_kalman_filter.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_library(
    name = "ensemble_kalman_filter",
    srcs = ["ensemble_kalman_filter.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_library(
    name = "ensemble_adjustment_kalman_filter",
    srcs = ["ensemble_adjustment_kalman_filter.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:dtype_util",
    ],
)

py_test(
    name = "extended_kalman_filter_test",
    size = "large",
    timeout = "long",
    srcs = ["extended_kalman_filter_test.py"],
    # shard_count = 6,
    deps = [
        ":extended_kalman_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "ensemble_kalman_filter_test",
    size = "medium",
    srcs = ["ensemble_kalman_filter_test.py"],
    deps = [
        ":ensemble_kalman_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "ensemble_adjustment_kalman_filter_test",
    size = "medium",
    srcs = ["ensemble_adjustment_kalman_filter_test.py"],
    deps = [
        ":ensemble_adjustment_kalman_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "iterated_filter",
    srcs = ["iterated_filter.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "iterated_filter_test",
    srcs = ["iterated_filter_test.py"],
    deps = [
        ":iterated_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
