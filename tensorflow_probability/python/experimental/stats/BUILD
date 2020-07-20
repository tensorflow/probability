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
#   Tools for calculating sample statistics.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "stats",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sample_stats",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

py_library(
    name = "sample_stats",
    srcs = ["sample_stats.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/stats",
    ],
)

# py2and3
py_test(
    name = "sample_stats_test",
    size = "small",
    srcs = ["sample_stats_test.py"],
    shard_count = 5,
    deps = [
        ":sample_stats",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
