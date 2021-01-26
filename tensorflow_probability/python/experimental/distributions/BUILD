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

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "distributions",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":laplace_approximation",
        ":joint_distribution_pinned",
        ":mvn_precision_factor_linop",
        "//tensorflow_probability/python/distributions:log_prob_ratio",
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

multi_substrate_py_library(
    name = "laplace_approximation",
    srcs = ["laplace_approximation.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_probability/python/bijectors:chain",
        "//tensorflow_probability/python/bijectors:joint_map",
        "//tensorflow_probability/python/bijectors:reshape",
        "//tensorflow_probability/python/bijectors:split",
        "//tensorflow_probability/python/distributions:transformed_distribution",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/math",
        "//tensorflow_probability/python/optimizer:bfgs",
        "//tensorflow_probability/python/optimizer:bfgs_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

multi_substrate_py_test(
    name = "laplace_approximation_test",
    size = "large",
    srcs = ["laplace_approximation_test.py"],
    srcs_version = "PY3",
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
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:tensor_util",
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
        "//tensorflow_probability/python/internal:test_util",
    ],
)
