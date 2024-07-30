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
#   Optimizers.

load("//tensorflow_probability/python:build_defs.bzl", "multi_substrate_py_library", "multi_substrate_py_test")

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])

multi_substrate_py_library(
    name = "optimizer",
    srcs = ["__init__.py"],
    substrates_omit_deps = [
        ":differential_evolution",
        ":proximal_hessian_sparse",
        ":sgld",
        ":variational_sgd",
    ],
    deps = [
        ":bfgs",
        ":differential_evolution",
        ":lbfgs",
        ":nelder_mead",
        ":proximal_hessian_sparse",
        ":sgld",
        ":variational_sgd",
        "//tensorflow_probability/python/internal:all_util",
        "//tensorflow_probability/python/optimizer/convergence_criteria",
        "//tensorflow_probability/python/optimizer/linesearch",
    ],
)

multi_substrate_py_library(
    name = "sgld",
    srcs = ["sgld.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:tf_keras",
        "//tensorflow_probability/python/math:diag_jacobian",
    ],
)

multi_substrate_py_test(
    name = "sgld_test",
    size = "medium",
    srcs = ["sgld_test.py"],
    disabled_substrates = [
        "numpy",
        "jax",
    ],
    deps = [
        ":sgld",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:mvn_tril",
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:diag_jacobian",
    ],
)

multi_substrate_py_library(
    name = "variational_sgd",
    srcs = ["variational_sgd.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:tf_keras",
    ],
)

multi_substrate_py_test(
    name = "variational_sgd_test",
    srcs = ["variational_sgd_test.py"],
    disabled_substrates = [
        "numpy",
        "jax",
    ],
    deps = [
        ":variational_sgd",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "bfgs",
    srcs = ["bfgs.py"],
    deps = [
        ":bfgs_utils",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:distribution_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/optimizer/linesearch",
    ],
)

multi_substrate_py_test(
    name = "bfgs_test",
    size = "medium",
    timeout = "long",
    srcs = ["bfgs_test.py"],
    numpy_tags = ["notap"],
    shard_count = 5,
    deps = [
        ":bfgs",
        ":bfgs_utils",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "lbfgs",
    srcs = ["lbfgs.py"],
    deps = [
        ":bfgs_utils",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "lbfgs_test",
    timeout = "long",
    srcs = ["lbfgs_test.py"],
    numpy_tags = ["notap"],
    shard_count = 5,
    deps = [
        ":bfgs_utils",
        ":lbfgs",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
        "//tensorflow_probability/python/math:gradient",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "bfgs_utils",
    srcs = ["bfgs_utils.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/optimizer/linesearch:hager_zhang",
    ],
)

multi_substrate_py_library(
    name = "nelder_mead",
    srcs = ["nelder_mead.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "nelder_mead_test",
    size = "medium",
    srcs = ["nelder_mead_test.py"],
    shard_count = 3,
    deps = [
        ":nelder_mead",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

multi_substrate_py_library(
    name = "differential_evolution",
    srcs = ["differential_evolution.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:categorical",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)

multi_substrate_py_test(
    name = "differential_evolution_test",
    size = "medium",
    srcs = ["differential_evolution_test.py"],
    disabled_substrates = [
        "numpy",
        "jax",
    ],
    deps = [
        ":differential_evolution",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "proximal_hessian_sparse",
    srcs = ["proximal_hessian_sparse.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
        "//tensorflow_probability/python/math:linalg",
    ],
)

multi_substrate_py_test(
    name = "proximal_hessian_sparse_test",
    srcs = ["proximal_hessian_sparse_test.py"],
    disabled_substrates = [
        "numpy",
        "jax",
    ],
    deps = [
        ":proximal_hessian_sparse",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)
