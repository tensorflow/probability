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
#   Generalized Linear Model specification, fitting, and related utilities.

# Placeholder: py_library
# Placeholder: py_test
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
    name = "glm",
    srcs = ["__init__.py"],
    substrates_omit_deps = [":proximal_hessian"],
    deps = [
        ":family",
        ":fisher_scoring",
        ":proximal_hessian",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "fisher_scoring",
    srcs = ["fisher_scoring.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:linalg",
    ],
)

multi_substrate_py_test(
    name = "fisher_scoring_test",
    size = "small",
    srcs = ["fisher_scoring_test.py"],
    deps = [
        ":family",
        ":fisher_scoring",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "proximal_hessian",
    srcs = ["proximal_hessian.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":fisher_scoring",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/math:linalg",
        "//tensorflow_probability/python/optimizer:proximal_hessian_sparse",
    ],
)

py_test(
    name = "proximal_hessian_test",
    size = "medium",
    srcs = ["proximal_hessian_test.py"],
    shard_count = 4,
    deps = [
        ":family",
        ":proximal_hessian",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:uniform",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:sparse",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "family",
    srcs = ["family.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:bijector",
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:binomial",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:lognormal",
        "//tensorflow_probability/python/distributions:negative_binomial",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:name_util",
        "//tensorflow_probability/python/util:deferred_tensor",
    ],
)

multi_substrate_py_test(
    name = "family_test",
    size = "small",
    srcs = ["family_test.py"],
    disabled_substrates = ["numpy"],  # The test uses CustomExponentialFamily, which uses gradients.
    deps = [
        ":family",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:bernoulli",
        "//tensorflow_probability/python/distributions:binomial",
        "//tensorflow_probability/python/distributions:gamma",
        "//tensorflow_probability/python/distributions:lognormal",
        "//tensorflow_probability/python/distributions:negative_binomial",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
