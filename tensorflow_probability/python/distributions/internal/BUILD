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
#   Internal helper libraries for distributions.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "moving_stats",
    srcs = ["moving_stats.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "statistical_testing",
    srcs = ["statistical_testing.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "moving_stats_test",
    size = "small",
    srcs = ["moving_stats_test.py"],
    tags = ["nomsan"],  # disable to avoid false positives from scipy.
    deps = [
        ":moving_stats",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "statistical_testing_test",
    size = "medium",
    srcs = ["statistical_testing_test.py"],
    shard_count = 4,
    deps = [
        ":statistical_testing",
        # numpy dep,
        # tensorflow dep,
    ],
)
