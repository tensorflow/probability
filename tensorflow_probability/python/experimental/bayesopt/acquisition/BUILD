# Copyright 2023 The TensorFlow Probability Authors.
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
#   Experimental TFP bayesopt package.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "acquisition",
    srcs = ["__init__.py"],
    deps = [
        ":acquisition_function",
        ":expected_improvement",
        ":max_value_entropy_search",
        ":probability_of_improvement",
        ":upper_confidence_bound",
        ":weighted_power_scalarization",
    ],
)

multi_substrate_py_library(
    name = "acquisition_function",
    srcs = ["acquisition_function.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_test(
    name = "acquisition_function_test",
    srcs = ["acquisition_function_test.py"],
    deps = [
        ":acquisition_function",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "expected_improvement",
    srcs = ["expected_improvement.py"],
    deps = [
        ":acquisition_function",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:student_t",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:samplers",
    ],
)

multi_substrate_py_test(
    name = "expected_improvement_test",
    srcs = ["expected_improvement_test.py"],
    jax_size = "large",
    shard_count = 2,
    deps = [
        ":expected_improvement",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:gaussian_process_regression_model",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:student_t",
        "//tensorflow_probability/python/distributions:student_t_process_regression_model",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)

multi_substrate_py_library(
    name = "max_value_entropy_search",
    srcs = ["max_value_entropy_search.py"],
    deps = [
        ":acquisition_function",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:gumbel",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:root_search",
        "//tensorflow_probability/python/math:special",
        "//tensorflow_probability/python/mcmc:sample_halton_sequence",
    ],
)

multi_substrate_py_test(
    name = "max_value_entropy_search_test",
    srcs = ["max_value_entropy_search_test.py"],
    shard_count = 3,
    deps = [
        ":max_value_entropy_search",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:gaussian_process_regression_model",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)

multi_substrate_py_library(
    name = "probability_of_improvement",
    srcs = ["probability_of_improvement.py"],
    deps = [
        ":acquisition_function",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:samplers",
    ],
)

multi_substrate_py_test(
    name = "probability_of_improvement_test",
    srcs = ["probability_of_improvement_test.py"],
    deps = [
        ":probability_of_improvement",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:gaussian_process_regression_model",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)

multi_substrate_py_library(
    name = "upper_confidence_bound",
    srcs = ["upper_confidence_bound.py"],
    deps = [
        ":acquisition_function",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:samplers",
    ],
)

multi_substrate_py_test(
    name = "upper_confidence_bound_test",
    srcs = ["upper_confidence_bound_test.py"],
    shard_count = 2,
    deps = [
        ":upper_confidence_bound",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:gaussian_process_regression_model",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)

multi_substrate_py_library(
    name = "weighted_power_scalarization",
    srcs = ["weighted_power_scalarization.py"],
    deps = [
        ":acquisition_function",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/experimental/distributions:multitask_gaussian_process",
        "//tensorflow_probability/python/experimental/distributions:multitask_gaussian_process_regression_model",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:tensor_util",
    ],
)

multi_substrate_py_test(
    name = "weighted_power_scalarization_test",
    srcs = ["weighted_power_scalarization_test.py"],
    deps = [
        ":expected_improvement",
        ":upper_confidence_bound",
        ":weighted_power_scalarization",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:gaussian_process",
        "//tensorflow_probability/python/experimental/distributions:multitask_gaussian_process",
        "//tensorflow_probability/python/experimental/psd_kernels:multitask_kernel",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)
