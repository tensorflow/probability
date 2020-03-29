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

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "sts_gibbs",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":gibbs_sampler",
    ],
)

py_test(
    name = "benchmarks_test",
    size = "medium",
    srcs = ["benchmarks_test.py"],
    tags = ["notap"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/sts_gibbs",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_library(
    name = "gibbs_sampler",
    srcs = ["gibbs_sampler.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/sts/internal",
    ],
)

py_test(
    name = "gibbs_sampler_test",
    size = "medium",
    srcs = ["gibbs_sampler_test.py"],
    shard_count = 2,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/sts_gibbs",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)
