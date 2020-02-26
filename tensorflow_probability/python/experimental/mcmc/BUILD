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
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":elliptical_slice_sampler",
        ":nuts_autobatching",
        # tensorflow dep,
    ],
)

py_library(
    name = "elliptical_slice_sampler",
    srcs = ["elliptical_slice_sampler.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal",
        "//tensorflow_probability/python/util",
    ],
)

py_test(
    name = "elliptical_slice_sampler_test",
    size = "large",
    timeout = "long",
    srcs = ["elliptical_slice_sampler_test.py"],
    shard_count = 6,
    deps = [
        ":elliptical_slice_sampler",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "nuts_autobatching",
    srcs = ["nuts.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/math",
        "//tensorflow_probability/python/mcmc",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

py_library(
    name = "nuts_testlib",
    testonly = 1,
    srcs = ["nuts_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/experimental/auto_batching",
        "//tensorflow_probability/python/internal:test_util",
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
        "//tensorflow_probability/python/internal:test_util",
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
        "requires-gpu-nvidia",
    ],
    deps = [
        ":nuts_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
