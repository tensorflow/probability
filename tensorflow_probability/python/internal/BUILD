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
#   Internal utilities for TensorFlow probability.

# [internal] load python3.bzl
load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_test",
    "multi_substrate_py_library",
)

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
#         "//tensorflow_probability/google:friends",  # DisableOnExport
    ],
)

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "internal",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
)

multi_substrate_py_library(
    name = "assert_util",
    srcs = ["assert_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "batched_rejection_sampler",
    srcs = ["batched_rejection_sampler.py"],
    deps = [
        ":samplers",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "batched_rejection_sampler_test",
    size = "medium",
    srcs = ["batched_rejection_sampler_test.py"],
    deps = [
        ":batched_rejection_sampler",
        ":samplers",
        ":test_util",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

multi_substrate_py_library(
    name = "cache_util",
    srcs = ["cache_util.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "cache_util_test",
    size = "small",
    srcs = ["cache_util_test.py"],
    deps = [
        ":cache_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "docstring_util",
    srcs = [
        "docstring_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "docstring_util_test",
    size = "small",
    srcs = ["docstring_util_test.py"],
    deps = [
        ":docstring_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "dtype_util",
    srcs = ["dtype_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "dtype_util_test",
    size = "small",
    srcs = ["dtype_util_test.py"],
    deps = [
        ":dtype_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "distribution_util",
    srcs = ["distribution_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":assert_util",
        ":dtype_util",
        ":prefer_static",
        ":reparameterization",
        ":tensor_util",
        ":tensorshape_util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "distribution_util_test",
    size = "small",
    srcs = ["distribution_util_test.py"],
    deps = [
        ":distribution_util",
        ":tensorshape_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "implementation_selection",
    srcs = ["implementation_selection.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy",
    ],
)

multi_substrate_py_test(
    name = "implementation_selection_test",
    srcs = ["implementation_selection_test.py"],
    deps = [
        ":implementation_selection",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

py_test(
    name = "implementation_selection_test_gpu",
    srcs = ["implementation_selection_test.py"],
    main = "implementation_selection_test.py",
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":implementation_selection",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

py_library(
    name = "monte_carlo",
    srcs = ["monte_carlo.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "name_util",
    srcs = ["name_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "nest_util",
    srcs = ["nest_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "nest_util_test",
    size = "small",
    srcs = ["nest_util_test.py"],
    deps = [
        ":nest_util",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "prefer_static",
    srcs = ["prefer_static.py"],
    deps = [
        ":dtype_util",
        ":tensorshape_util",
        # decorator dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy",
    ],
)

multi_substrate_py_test(
    name = "prefer_static_test",
    size = "small",
    srcs = ["prefer_static_test.py"],
    deps = [
        ":prefer_static",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "reparameterization",
    srcs = ["reparameterization.py"],
)

multi_substrate_py_library(
    name = "samplers",
    srcs = ["samplers.py"],
    deps = [
        ":dtype_util",
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "samplers_test",
    size = "small",
    srcs = ["samplers_test.py"],
    deps = [
        ":samplers",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "special_math",
    srcs = ["special_math.py"],
    deps = [
        ":dtype_util",
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "special_math_test",
    size = "small",
    srcs = ["special_math_test.py"],
    numpy_tags = ["notap"],
    deps = [
        ":special_math",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "tensor_util",
    srcs = ["tensor_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "tensor_util_test",
    size = "small",
    srcs = ["tensor_util_test.py"],
    deps = [
        ":tensor_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "tensorshape_util",
    srcs = ["tensorshape_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "tensorshape_util_test",
    size = "small",
    srcs = ["tensorshape_util_test.py"],
    deps = [
        ":tensorshape_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "vectorization_util",
    srcs = ["vectorization_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "hypothesis_testlib",
    testonly = 1,
    srcs = ["hypothesis_testlib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensorshape_util",
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/util",
    ],
)

py_test(
    name = "hypothesis_testlib_test",
    size = "small",
    srcs = ["hypothesis_testlib_test.py"],
    shard_count = 5,
    srcs_version = "PY2AND3",
    deps = [
        ":hypothesis_testlib",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtype_util",
        ":test_combinations",
        # absl/flags dep,
        # hypothesis dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal/backend/numpy",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "test_util_test",
    size = "small",
    srcs = ["test_util_test.py"],
    deps = [
        ":test_combinations",
        ":test_util",
        # absl/flags dep,
        # absl/testing:flagsaver dep,
        # tensorflow dep,
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_library(
    name = "test_combinations",
    testonly = 1,
    srcs = ["test_combinations.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl/testing:parameterized dep,
        # six dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "test_combinations_test",
    size = "small",
    srcs = ["test_combinations_test.py"],
    deps = [
        ":test_combinations",
        ":test_util",
        # absl/testing:parameterized dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "vectorization_util_test",
    size = "small",
    srcs = ["vectorization_util_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        ":test_util",
        ":vectorization_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

exports_files(
    [
        "assert_util.py",
        "batched_rejection_sampler.py",
        "distribution_util.py",
        "dtype_util.py",
        "dtype_util_test.py",
        "hypothesis_testlib.py",
        "prefer_static.py",
        "prefer_static_test.py",
        "special_math.py",
        "special_math_test.py",
        "tensor_util.py",
        "test_combinations.py",
        "test_util.py",
        "test_util_test.py",
    ],
    visibility = [
        "//tensorflow_probability:__subpackages__",
#         "//tensorflow_probability/google:friends",  # DisableOnExport
    ],
)
