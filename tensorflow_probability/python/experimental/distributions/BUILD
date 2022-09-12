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
# tfp.experimental distributions library.

load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "distributions",
    srcs = ["__init__.py"],
    deps = [
        ":importance_resample",
        ":increment_log_prob",
        ":joint_distribution_pinned",
        ":marginal_fns",
        ":multitask_gaussian_process",
        ":multitask_gaussian_process_regression_model",
        ":mvn_precision_factor_linop",
        "//tensorflow_probability/python/distributions:inflated",
        "//tensorflow_probability/python/distributions:log_prob_ratio",
    ],
)

multi_substrate_py_library(
    name = "importance_resample",
    srcs = ["importance_resample.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/experimental/mcmc:weighted_resampling",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/mcmc/internal:util",
    ],
)

multi_substrate_py_test(
    name = "importance_resample_test",
    size = "medium",
    srcs = ["importance_resample_test.py"],
    jax_size = "medium",
    deps = [
        ":importance_resample",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:mixture_same_family",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:student_t",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/vi:csiszar_divergence",
    ],
)

multi_substrate_py_library(
    name = "increment_log_prob",
    srcs = ["increment_log_prob.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:log_prob_ratio",
        "//tensorflow_probability/python/internal:callable_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "increment_log_prob_test",
    srcs = ["increment_log_prob_test.py"],
    deps = [
        ":increment_log_prob",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution_pinned",
    srcs = ["joint_distribution_pinned.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "joint_distribution_pinned_test",
    size = "large",
    srcs = ["joint_distribution_pinned_test.py"],
    jax_size = "large",
    shard_count = 13,
    deps = [
        ":increment_log_prob",
        ":joint_distribution_pinned",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:pad",
        "//tensorflow_probability/python/distributions:beta",
        "//tensorflow_probability/python/distributions:binomial",
        "//tensorflow_probability/python/distributions:cholesky_lkj",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:multinomial",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:sample",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:structural_tuple",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:generic",
    ],
)

multi_substrate_py_library(
    name = "marginal_fns",
    srcs = ["marginal_fns.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_linear_operator",
    ],
)

multi_substrate_py_test(
    name = "marginal_fns_test",
    srcs = ["marginal_fns_test.py"],
    numpy_tags = ["notap"],
    shard_count = 3,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/distributions:marginal_fns",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
    ],
)

multi_substrate_py_library(
    name = "multitask_gaussian_process",
    srcs = ["multitask_gaussian_process.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:cholesky_util",
        "//tensorflow_probability/python/experimental/linalg:linear_operator_unitary",
        "//tensorflow_probability/python/experimental/psd_kernels:multitask_kernel",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "multitask_gaussian_process_test",
    size = "medium",
    srcs = ["multitask_gaussian_process_test.py"],
    shard_count = 3,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:gaussian_process",
        "//tensorflow_probability/python/experimental/distributions:multitask_gaussian_process",
        "//tensorflow_probability/python/experimental/distributions:multitask_gaussian_process_regression_model",
        "//tensorflow_probability/python/experimental/psd_kernels:multitask_kernel",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)

multi_substrate_py_library(
    name = "multitask_gaussian_process_regression_model",
    srcs = ["multitask_gaussian_process_regression_model.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:cholesky_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/math/psd_kernels/internal:util",
    ],
)

multi_substrate_py_test(
    name = "multitask_gaussian_process_regression_model_test",
    size = "medium",
    srcs = ["multitask_gaussian_process_regression_model_test.py"],
    shard_count = 2,
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:gaussian_process_regression_model",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/experimental",
        "//tensorflow_probability/python/experimental/distributions:multitask_gaussian_process_regression_model",
        "//tensorflow_probability/python/experimental/psd_kernels:multitask_kernel",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
    ],
)

multi_substrate_py_library(
    name = "mvn_precision_factor_linop",
    srcs = ["mvn_precision_factor_linop.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:invert",
        "//tensorflow_probability/python/bijectors:scale_matvec_linear_operator",
        "//tensorflow_probability/python/bijectors:shift",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:parameter_properties",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensor_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

multi_substrate_py_test(
    name = "mvn_precision_factor_linop_test",
    size = "medium",
    timeout = "long",
    srcs = ["mvn_precision_factor_linop_test.py"],
    deps = [
        ":mvn_precision_factor_linop",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:wishart",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_combinations",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/stats:sample_stats",
    ],
)
