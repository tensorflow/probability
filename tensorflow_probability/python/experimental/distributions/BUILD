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
    srcs_version = "PY3",
    substrates_omit_deps = [
        ":marginal_fns",
    ],
    deps = [
        ":increment_log_prob",
        ":joint_density_coroutine",
        ":joint_distribution_pinned",
        ":marginal_fns",
        ":mvn_precision_factor_linop",
        "//tensorflow_probability/python/distributions:log_prob_ratio",
    ],
)

multi_substrate_py_library(
    name = "increment_log_prob",
    srcs = ["increment_log_prob.py"],
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:identity",
    ],
)

multi_substrate_py_library(
    name = "joint_density_coroutine",
    srcs = ["joint_density_coroutine.py"],
    srcs_version = "PY3",
    deps = [
        ":increment_log_prob",
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "joint_density_coroutine_test",
    size = "small",
    srcs = ["joint_density_coroutine_test.py"],
    jax_size = "small",
    shard_count = 1,
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "joint_distribution_pinned",
    srcs = ["joint_distribution_pinned.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "marginal_fns",
    srcs = ["marginal_fns.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_linear_operator",
    ],
)

py_test(
    name = "marginal_fns_test",
    srcs = ["marginal_fns_test.py"],
    python_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "mvn_precision_factor_linop",
    srcs = ["mvn_precision_factor_linop.py"],
    srcs_version = "PY3",
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
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_combinations",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
