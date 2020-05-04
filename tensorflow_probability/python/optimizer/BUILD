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

load("//tensorflow_probability/python:build_defs.bzl", "multi_substrate_py_test", "multi_substrate_py_library")

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

multi_substrate_py_library(
    name = "optimizer",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":bfgs",
        ":differential_evolution",
        ":lbfgs",
        ":nelder_mead",
        ":proximal_hessian_sparse",
        ":sgld",
        ":variational_sgd",
        "//tensorflow_probability/python/optimizer/convergence_criteria",
        "//tensorflow_probability/python/optimizer/linesearch",
    ],
)

multi_substrate_py_library(
    name = "sgld",
    srcs = ["sgld.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
    ],
)

multi_substrate_py_test(
    name = "sgld_test",
    size = "medium",
    srcs = ["sgld_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "variational_sgd",
    srcs = ["variational_sgd.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:assert_util",
    ],
)

multi_substrate_py_test(
    name = "variational_sgd_test",
    srcs = ["variational_sgd_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "bfgs",
    srcs = ["bfgs.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":bfgs_utils",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_test(
    name = "bfgs_test",
    size = "medium",
    timeout = "long",
    srcs = ["bfgs_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "lbfgs",
    srcs = ["lbfgs.py"],
    srcs_version = "PY2AND3",
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
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    shard_count = 5,
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "bfgs_utils",
    srcs = ["bfgs_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

multi_substrate_py_library(
    name = "nelder_mead",
    srcs = ["nelder_mead.py"],
    srcs_version = "PY2AND3",
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
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "differential_evolution",
    srcs = ["differential_evolution.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/distributions",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/util",
    ],
)

multi_substrate_py_test(
    name = "differential_evolution_test",
    size = "medium",
    srcs = ["differential_evolution_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

multi_substrate_py_library(
    name = "proximal_hessian_sparse",
    srcs = ["proximal_hessian_sparse.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/math:generic",
    ],
)

multi_substrate_py_test(
    name = "proximal_hessian_sparse_test",
    srcs = ["proximal_hessian_sparse_test.py"],
    jax_tags = ["notap"],
    numpy_tags = ["notap"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
