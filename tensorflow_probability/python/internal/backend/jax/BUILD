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

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

FILENAMES = [
    "__init__",
    "__internal__",
    "bitwise",
    "compat",
    "composite_tensor",
    "config",
    "control_flow",
    "data_structures",
    "debugging",
    "deprecation",
    "dtype",
    "errors",
    "functional_ops",
    "initializers",
    "numpy_keras",
    "keras_layers",
    "linalg",
    "linalg_impl",
    "misc",
    "nest",
    "nested_structure_coder",
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
    "resource_variable_ops",
    "sets_lib",
    "sparse_lib",
    "tensor_array_ops",
    "tensor_array_ops_test",
    "tensor_spec",
    "test_lib",
    "tf_inspect",
    "type_spec",
    "variables",
    "v1",
    "v2",
    "_utils",
]

GEN_FILENAMES = [
    "gen/__init__",
    "gen/tensor_shape",
    "gen/adjoint_registrations",
    "gen/cholesky_registrations",
    "gen/inverse_registrations",
    "gen/linear_operator_addition",
    "gen/linear_operator_adjoint",
    "gen/linear_operator_algebra",
    "gen/linear_operator_block_diag",
    "gen/linear_operator_block_lower_triangular",
    "gen/linear_operator_full_matrix",
    "gen/linear_operator_circulant",
    "gen/linear_operator_composition",
    "gen/linear_operator_diag",
    "gen/linear_operator_householder",
    "gen/linear_operator_identity",
    "gen/linear_operator_inversion",
    "gen/linear_operator_kronecker",
    "gen/linear_operator_lower_triangular",
    "gen/linear_operator_low_rank_update",
    "gen/linear_operator",
    "gen/linear_operator_toeplitz",
    "gen/linear_operator_util",
    "gen/linear_operator_zeros",
    "gen/matmul_registrations",
    "gen/registrations_util",
    "gen/solve_registrations",
]

[
    genrule(
        name = "rewrite_{}".format(f),
        srcs = ["//tensorflow_probability/python/internal/backend/numpy:{}.py".format(f)],
        outs = ["{}.py".format(f)],
        cmd = "$(location //tensorflow_probability/python/internal/backend/jax:rewrite) $(SRCS) --rewrite_numpy_import=1 > $@",
        tools = ["//tensorflow_probability/python/internal/backend/jax:rewrite"],
    )
    for f in FILENAMES
]

[
    genrule(
        name = "rewrite_{}".format(f),
        srcs = ["//tensorflow_probability/python/internal/backend/numpy:{}.py".format(f)],
        outs = ["{}.py".format(f)],
        cmd = "$(location //tensorflow_probability/python/internal/backend/jax:rewrite) $(SRCS) --rewrite_numpy_import=0 > $@",
        tools = ["//tensorflow_probability/python/internal/backend/jax:rewrite"],
    )
    for f in GEN_FILENAMES
]

py_library(
    name = "jax",
    srcs = [":{}.py".format(f) for f in FILENAMES + GEN_FILENAMES if "_test" not in f],
    srcs_version = "PY3",
    deps = [],  # The JAX dependency is a weak dependency, opt-in by user.
)

py_library(
    name = "jax_testlib",
    testonly = 1,
    srcs = [":{}.py".format(f) for f in FILENAMES + GEN_FILENAMES if "_test" in f],
    srcs_version = "PY3",
    deps = [
        ":jax",
    ],
)

py_test(
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

py_test(
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
    srcs_version = "PY3",
    deps = [
        # absl:app dep,
        # absl/flags dep,
    ],
)
