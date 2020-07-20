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
        ":particle_filter",
        ":particle_filter_augmentation",
        ":sample",
        ":sample_sequential_monte_carlo",
        ":sequential_monte_carlo_kernel",
        ":weighted_resampling",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:all_util",
    ],
)

py_library(
    name = "elliptical_slice_sampler",
    srcs = ["elliptical_slice_sampler.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/mcmc:kernel",
        "//tensorflow_probability/python/mcmc/internal",
    ],
)

# py2and3
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
        "//tensorflow_probability/python/random",
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

# py2and3
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

# py2and3
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

# py2and3
py_test(
    name = "nuts_xla_test_cpu",
    size = "large",
    srcs = ["nuts_xla_test.py"],
    args = ["--test_device=cpu"],
    main = "nuts_xla_test.py",
    shard_count = 1,
    tags = [
        "nozapfhahn",
    ],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

# py2and3
py_test(
    name = "nuts_xla_test_gpu",
    size = "large",
    srcs = ["nuts_xla_test.py"],
    args = ["--test_device=gpu"],
    main = "nuts_xla_test.py",
    shard_count = 1,
    tags = [
        "nozapfhahn",
        "requires-gpu-nvidia",
    ],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/mcmc",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_library(
    name = "particle_filter",
    srcs = ["particle_filter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sequential_monte_carlo_kernel",
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

# py2and3
py_test(
    name = "particle_filter_test",
    size = "large",
    srcs = ["particle_filter_test.py"],
    deps = [
        ":particle_filter",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_library(
    name = "particle_filter_augmentation",
    srcs = ["particle_filter_augmentation.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

# py2and3
py_test(
    name = "particle_filter_augmentation_test",
    size = "medium",
    srcs = ["particle_filter_augmentation_test.py"],
    deps = [
        ":particle_filter_augmentation",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "sample_sequential_monte_carlo",
    srcs = ["sample_sequential_monte_carlo.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

# py2and3
py_test(
    name = "sample_sequential_monte_carlo_test",
    size = "medium",
    timeout = "long",
    srcs = ["sample_sequential_monte_carlo_test.py"],
    shard_count = 2,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_library(
    name = "sequential_monte_carlo_kernel",
    srcs = ["sequential_monte_carlo_kernel.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":weighted_resampling",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/mcmc/internal:util",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

# py2and3
py_test(
    name = "sequential_monte_carlo_kernel_test",
    size = "medium",
    srcs = ["sequential_monte_carlo_kernel_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/distributions/internal:statistical_testing",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_library(
    name = "weighted_resampling",
    srcs = ["weighted_resampling.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:docstring_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

# py2and3
py_test(
    name = "weighted_resampling_test",
    size = "medium",
    srcs = ["weighted_resampling_test.py"],
    shard_count = 2,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_library(
    name = "sample",
    srcs = ["sample.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

# py2and3
py_test(
    name = "sample_test",
    size = "small",
    srcs = ["sample_test.py"],
    shard_count = 5,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
