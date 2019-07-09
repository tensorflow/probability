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
#   An implementation of the No U-Turn Sampler.
#   Achieves batch execution across chains by using
#   //tensorflow_probability/python/internal/auto_batching
#   internally.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "mcmc",
    srcs = [
        "__init__.py",
        "nuts.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal/auto_batching",
        "//tensorflow_probability/python/math",
        "//tensorflow_probability/python/mcmc",
    ],
)

py_library(
    name = "nuts_testlib",
    srcs = ["nuts_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
    ],
)

py_test(
    name = "nuts_test",
    size = "medium",
    srcs = ["nuts_test.py"],
    shard_count = 7,
    srcs_version = "PY2AND3",
    tags = ["nozapfhahn"],
    deps = [
        ":nuts_testlib",
    ],
)

py_test(
    name = "nuts_test_gpu",
    srcs = ["nuts_test.py"],
    main = "nuts_test.py",
    shard_count = 7,
    srcs_version = "PY2AND3",
    tags = [
        "notap",
        "nozapfhahn",
        "requires-gpu-sm35",
    ],
    deps = [":nuts_testlib"],
)
