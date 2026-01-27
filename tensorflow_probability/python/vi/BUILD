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
#   Methods and objectives for variational inference.

# Placeholder: py_library
# Placeholder: py_test
load(
    "//tensorflow_probability/python:build_defs.bzl",
    "multi_substrate_py_library",
    "multi_substrate_py_test",
)

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

multi_substrate_py_library(
    name = "vi",
    srcs = ["__init__.py"],
    substrates_omit_deps = [
        ":mutual_information",
    ],
    deps = [
        ":csiszar_divergence",
        ":mutual_information",
        ":optimization",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

multi_substrate_py_library(
    name = "csiszar_divergence",
    srcs = ["csiszar_divergence.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/monte_carlo",
        "//tensorflow_probability/python/stats:leave_one_out",
    ],
)

multi_substrate_py_test(
    name = "csiszar_divergence_test",
    size = "medium",
    srcs = ["csiszar_divergence_test.py"],
    jax_size = "medium",
    numpy_size = "medium",
    shard_count = 5,
    deps = [
        ":csiszar_divergence",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:joint_distribution_named",
        "//tensorflow_probability/python/distributions:joint_distribution_sequential",
        "//tensorflow_probability/python/distributions:kullback_leibler",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:student_t",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/stats:leave_one_out",
    ],
)

py_library(
    name = "mutual_information",
    srcs = ["mutual_information.py"],
    strict_deps = False,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
    ],
)

py_test(
    name = "mutual_information_test",
    size = "small",
    srcs = ["mutual_information_test.py"],
    shard_count = 5,
    deps = [
        ":mutual_information",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "optimization",
    srcs = ["optimization.py"],
    deps = [
        ":csiszar_divergence",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:nest_util",
        "//tensorflow_probability/python/math:minimize",
        "//tensorflow_probability/python/monte_carlo",
    ],
)

multi_substrate_py_test(
    name = "optimization_test",
    size = "medium",
    srcs = ["optimization_test.py"],
    numpy_tags = ["notap"],
    shard_count = 5,
    deps = [
        ":optimization",
        # numpy dep,
        # optax dep,
        # tensorflow dep,
        "//tensorflow_probability/python/bijectors:fill_scale_tril",
        "//tensorflow_probability/python/bijectors:softplus",
        "//tensorflow_probability/python/distributions:deterministic",
        "//tensorflow_probability/python/distributions:gaussian_process",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:joint_distribution_auto_batched",
        "//tensorflow_probability/python/distributions:joint_distribution_coroutine",
        "//tensorflow_probability/python/distributions:lognormal",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/distributions:poisson",
        "//tensorflow_probability/python/experimental/util",
        "//tensorflow_probability/python/internal:samplers",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/internal:tf_keras",
        "//tensorflow_probability/python/math/psd_kernels:exponentiated_quadratic",
        "//tensorflow_probability/python/util:deferred_tensor",
    ],
)
