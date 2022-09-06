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

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)
# [internal] load pytype.bzl (pytype_strict_test)

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
#         "//tensorflow_probability/google:friends",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "internal",
    srcs = ["__init__.py"],
    deps = [
        ":test_util",  # EnableOnExport
    ],
)

py_library(
    name = "all_util",
    srcs = ["all_util.py"],
    deps = [],
)

multi_substrate_py_library(
    name = "auto_composite_tensor",
    srcs = ["auto_composite_tensor.py"],
    deps = [
        # tensorflow dep,
    ],
)

# pytype_strict
py_test(
    name = "auto_composite_tensor_test",
    srcs = ["auto_composite_tensor_test.py"],
    data = glob(["testdata/auto_composite_tensor/**"]),
    deps = [
        ":auto_composite_tensor",
        ":tensor_util",
        ":test_util",
        # absl/flags dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

multi_substrate_py_library(
    name = "assert_util",
    srcs = ["assert_util.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "batch_shape_lib",
    srcs = ["batch_shape_lib.py"],
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "batch_shape_lib_test",
    size = "small",
    srcs = ["batch_shape_lib_test.py"],
    deps = [
        ":test_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

multi_substrate_py_library(
    name = "distribute_lib",
    srcs = ["distribute_lib.py"],
    deps = [
        ":custom_gradient",
        ":prefer_static",
        ":samplers",
        # tensorflow dep,
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_test(
    name = "distribute_lib_test",
    size = "medium",
    srcs = ["distribute_lib_test.py"],
    disabled_substrates = ["numpy"],
    deps = [
        ":distribute_lib",
        ":distribute_test_lib",
        ":test_util",
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

multi_substrate_py_library(
    name = "distribute_test_lib",
    testonly = 1,
    srcs = ["distribute_test_lib.py"],
    deps = [
        ":distribution_util",
        ":test_util",
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "batched_rejection_sampler",
    srcs = ["batched_rejection_sampler.py"],
    deps = [
        ":broadcast_util",
        ":samplers",
        # numpy dep,  # For np.nan
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
    name = "broadcast_util",
    srcs = ["broadcast_util.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "broadcast_util_test",
    srcs = ["broadcast_util_test.py"],
    deps = [
        ":broadcast_util",
        ":test_util",
        # tensorflow dep,
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

multi_substrate_py_library(
    name = "callable_util",
    srcs = ["callable_util.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "callable_util_test",
    size = "small",
    srcs = ["callable_util_test.py"],
    deps = [
        ":callable_util",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "custom_gradient",
    srcs = ["custom_gradient.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "custom_gradient_test",
    srcs = ["custom_gradient_test.py"],
    deps = [
        ":custom_gradient",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
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
    deps = [
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
        # tensorflow dep,
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

multi_substrate_py_library(
    name = "loop_util",
    srcs = ["loop_util.py"],
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "loop_util_test",
    srcs = ["loop_util_test.py"],
    deps = [
        ":auto_composite_tensor",
        ":loop_util",
        ":prefer_static",
        ":tensorshape_util",
        ":test_util",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/math",
        # tensorflow/compiler/jit dep,
    ],
)

py_library(
    name = "lazy_loader",
    srcs = ["lazy_loader.py"],
)

multi_substrate_py_library(
    name = "monte_carlo",
    srcs = ["monte_carlo.py"],
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "name_util",
    srcs = ["name_util.py"],
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "nest_util",
    srcs = ["nest_util.py"],
    deps = [
        ":prefer_static",
        # numpy dep,
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
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "numerics_testing",
    srcs = ["numerics_testing.py"],
    deps = [
        ":dtype_util",
        # absl/logging dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_test(
    name = "numerics_testing_test",
    size = "small",
    srcs = ["numerics_testing_test.py"],
    deps = [
        ":numerics_testing",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "parameter_properties",
    srcs = ["parameter_properties.py"],
    deps = ["//tensorflow_probability/python/bijectors:identity"],
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
        ":prefer_static",
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

multi_substrate_py_test(
    name = "samplers_rbg_test",
    size = "small",
    srcs = ["samplers_test.py"],
    args = ["--test_tfp_jax_prng=rbg"],
    disabled_substrates = [
        "tf",
        "numpy",
    ],
    main = "samplers_test.py",
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
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
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

multi_substrate_py_library(
    name = "tensorshape_util",
    srcs = ["tensorshape_util.py"],
    deps = [
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
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
    name = "trainable_state_util",
    srcs = ["trainable_state_util.py"],
    deps = [
        ":samplers",
        ":structural_tuple",
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "trainable_state_util_test",
    srcs = ["trainable_state_util_test.py"],
    jax_size = "medium",
    numpy_tags = ["notap"],
    deps = [
        # optax dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/internal:trainable_state_util",
    ],
)

multi_substrate_py_library(
    name = "vectorization_util",
    srcs = ["vectorization_util.py"],
    deps = [
        ":samplers",
        # tensorflow dep,
        "//tensorflow_probability/python/util",
    ],
)

multi_substrate_py_library(
    name = "hypothesis_testlib",
#     testonly = 1,  # DisableOnExport
    srcs = ["hypothesis_testlib.py"],
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
    name = "slicing",
    srcs = ["slicing.py"],
    deps = [
        ":prefer_static",
        ":tensorshape_util",
        # six dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "slicing_test",
    size = "small",
    srcs = ["slicing_test.py"],
    deps = [
        ":slicing",
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors",
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:test_util",
        # tensorflow/compiler/jit dep,
    ],
)

multi_substrate_py_library(
    name = "structural_tuple",
    srcs = ["structural_tuple.py"],
)

multi_substrate_py_test(
    name = "structural_tuple_test",
    srcs = ["structural_tuple_test.py"],
    deps = [
        ":nest_util",
        ":structural_tuple",
        ":test_util",
        # tensorflow dep,
    ],
)

multi_substrate_py_library(
    name = "test_util",
#     testonly = 1,  # DisableOnExport
    srcs = ["test_util.py"],
    deps = [
        ":dtype_util",
        ":samplers",
        ":test_combinations",
        # absl/flags dep,
        # hypothesis dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/internal:empirical_statistical_testing",
        "//tensorflow_probability/python/internal/backend/numpy",
        "//tensorflow_probability/python/util:deferred_tensor",
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
    name = "empirical_statistical_testing",
#     testonly = 1,  # DisableOnExport
    srcs = ["empirical_statistical_testing.py"],
    deps = [
        # numpy dep,
        # scipy dep,
    ],
)

multi_substrate_py_test(
    name = "empirical_statistical_testing_test",
    srcs = ["empirical_statistical_testing_test.py"],
    deps = [
        ":empirical_statistical_testing",
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "test_combinations",
#     testonly = 1,  # DisableOnExport
    srcs = ["test_combinations.py"],
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
    size = "medium",
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

multi_substrate_py_library(
    name = "unnest",
    srcs = ["unnest.py"],
    deps = [],
)

multi_substrate_py_test(
    name = "unnest_test",
    size = "small",
    srcs = ["unnest_test.py"],
    deps = [
        ":unnest",
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "variadic_reduce",
    srcs = [
        "variadic_reduce.py",
    ],
    deps = [
        ":implementation_selection",
        ":prefer_static",
        ":tensorshape_util",
        # numpy dep,
        # tensorflow dep,
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
        "slicing.py",
        "slicing_test.py",
        "special_math.py",
        "special_math_test.py",
        "tensor_util.py",
        "test_combinations.py",
        "test_util.py",
        "test_util_test.py",
        "unnest.py",
        "unnest_test.py",
        "vectorization_util.py",
    ],
    visibility = [
        "//tensorflow_probability:__subpackages__",
#         "//tensorflow_probability/google:friends",  # DisableOnExport
    ],
)
