# Copyright 2019 The TensorFlow Probability Authors.
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
# Description: JAX backend.

# py_test rules are rewritten to py3_test on export to OSS, for reasons of
# reversibility of import<->export transforms. This is only necessary until we
# eliminate all py2and3_test targets, after which everything can safely use
# py_test, interanlly and externally, and no rewrites will be needed.
# @unused  <-- satisfy buildifier errors
load("//tensorflow_probability/python:build_defs.bzl", "py3_test")

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

FILENAMES = [
    "__init__",
    "bitwise",
    "compat",
    "control_flow",
    "debugging",
    "dtype",
    "errors",
    "functional_ops",
    "initializers",
    "keras",
    "keras_layers",
    "linalg",
    "linalg_impl",
    "misc",
    "nn",
    "numpy_array",
    "numpy_logging",
    "numpy_math",
    "numpy_signal",
    "numpy_test",
    "ops",
    "private",
    "random_generators",
    "raw_ops",
    "sets_lib",
    "sparse_lib",
    "tensor_array_ops",
    "tensor_array_ops_test",
    "test_lib",
    "v1",
    "v2",
    "_utils",
] + [
    "adjoint_registrations",
    "cholesky_registrations",
    "inverse_registrations",
    "linear_operator_addition",
    "linear_operator_adjoint",
    "linear_operator_algebra",
    "linear_operator_block_diag",
    "linear_operator_block_lower_triangular",
    "linear_operator_circulant",
    "linear_operator_composition",
    "linear_operator_diag",
    "linear_operator_full_matrix",
    "linear_operator_householder",
    "linear_operator_identity",
    "linear_operator_inversion",
    "linear_operator_kronecker",
    "linear_operator_lower_triangular",
    "linear_operator_low_rank_update",
    "linear_operator",
    "linear_operator_toeplitz",
    "linear_operator_util",
    "linear_operator_zeros",
    "matmul_registrations",
    "registrations_util",
    "solve_registrations",
]

[
    genrule(
        name = "rewrite_{}".format(f),
        srcs = ["//tensorflow_probability/python/internal/backend/numpy:{}.py".format(f)],
        outs = ["{}.py".format(f)],
        cmd = "$(location //tensorflow_probability/python/internal/backend/jax:rewrite) $(SRCS) > $@",
        tools = ["//tensorflow_probability/python/internal/backend/jax:rewrite"],
    )
    for f in FILENAMES
]

py_library(
    name = "jax",
    srcs = [":{}.py".format(f) for f in FILENAMES if "_test" not in f],
    deps = [],  # The JAX dependency is a weak dependency, opt-in by user.
)

py_library(
    name = "jax_testlib",
    testonly = 1,
    srcs = [":{}.py".format(f) for f in FILENAMES if "_test" in f],
    deps = [
        ":jax",
    ],
)

py3_test(
    name = "tensor_array_ops_test",
    srcs = ["tensor_array_ops_test.py"],
    main = "tensor_array_ops_test.py",
    python_version = "PY3",
    tags = ["tfp_jax"],
    deps = [
        ":jax",
        ":jax_testlib",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # jax dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py3_test(
    name = "jax_test",
    srcs = ["numpy_test.py"],
    main = "numpy_test.py",
    python_version = "PY3",
    shard_count = 8,
    tags = ["tfp_jax"],
    deps = [
        ":jax",
        ":jax_testlib",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # jax dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_binary(
    name = "rewrite",
    srcs = ["rewrite.py"],
    python_version = "PY3",
    deps = [
        # absl:app dep,
    ],
)
